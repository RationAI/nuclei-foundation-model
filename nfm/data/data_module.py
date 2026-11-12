import atexit
import faulthandler
import os
from functools import partial

import pandas as pd
from hydra.utils import instantiate
from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from nfm.data.packing import Batch, collate_slides, collate_views
from nfm.data.samplers import SourceBalancedRandomSampler


def init_worker(worker_id: int) -> None:
    """Prepares a DataLoader worker.

    Prints native stack traces when the worker crashes. PyTorch replaces the
    crash handlers in each worker, so they are re-enabled here. Workers exit
    without native finalization, where lance sometimes aborts. The workers are
    the unit of parallelism, so BLAS gets one thread: with a thread per core in
    every worker, the small matrix products wait for each other.
    """
    faulthandler.disable()
    faulthandler.enable()
    atexit.register(os._exit, 0)
    threadpool_limits(limits=1)


class DataModule(LightningDataModule):
    def __init__(self, datasets: DictConfig, block_size: int = 128) -> None:
        """Initializes the data module.

        Args:
            datasets: The unlabeled and labeled dataset configurations.
            block_size: Attention block size the views are sorted for.
        """
        super().__init__()
        self.datasets = datasets
        self.block_size = block_size

    def prepare_data(self) -> None:
        for dataset in self.datasets.values():
            if dataset.slides_path.startswith("mlflow-artifacts:"):
                dataset.slides_path = download_artifacts(dataset.slides_path)

    def setup(self, stage: str) -> None:
        labeled = self.datasets.labeled
        slides = pd.read_parquet(labeled.slides_path)
        slides = slides[slides["nuclei_count"] >= 512]
        train, validation = train_test_split(
            slides,
            test_size=labeled.split_size,
            stratify=slides[labeled.split_stratify_col],
        )
        self.train_labeled = instantiate(
            labeled.dataset, slides=train, crop_nuclei=labeled.crop_nuclei
        )
        # Whole slides, one per batch.
        self.validation = instantiate(labeled.dataset, slides=validation)
        slides = pd.read_parquet(self.datasets.unlabeled.slides_path)
        self.train = instantiate(self.datasets.unlabeled.dataset, slides=slides)

    def train_dataloader(self) -> CombinedLoader:
        return CombinedLoader(
            {
                "unlabeled": DataLoader(
                    self.train,
                    batch_size=self.datasets.unlabeled.batch_size,
                    sampler=SourceBalancedRandomSampler(self.train.slides.dataset),
                    drop_last=True,
                    num_workers=self.datasets.unlabeled.num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                    in_order=False,
                    multiprocessing_context="spawn",  # lance is not fork-safe
                    worker_init_fn=init_worker,
                    collate_fn=partial(collate_views, block_size=self.block_size),
                ),
                "labeled": DataLoader(
                    self.train_labeled,
                    batch_size=self.datasets.labeled.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=self.datasets.labeled.num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                    in_order=False,
                    worker_init_fn=init_worker,
                    collate_fn=partial(collate_slides, block_size=self.block_size),
                ),
            },
            mode="max_size_cycle",
        )

    def val_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(
            self.validation,
            batch_size=1,  # a whole slide
            num_workers=self.datasets.labeled.eval_num_workers,
            persistent_workers=True,
            pin_memory=True,
            in_order=False,
            worker_init_fn=init_worker,
            collate_fn=partial(collate_slides, block_size=self.block_size),
        )
