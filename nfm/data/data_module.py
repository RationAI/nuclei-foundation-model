import faulthandler
from functools import partial

import pandas as pd
from hydra.utils import instantiate
from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from nfm.data.packing import Batch, collate_tiles, collate_views
from nfm.data.samplers import SourceBalancedRandomSampler


def enable_faulthandler(worker_id: int) -> None:
    """Print native stack traces when a DataLoader worker crashes.

    PyTorch replaces the crash handlers in each worker, so it is re-enabled here.
    """
    faulthandler.disable()
    faulthandler.enable()


class DataModule(LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        mask_probability: float = 0.5,
        mask_ratio: tuple[float, float] = (0.1, 0.5),
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.mask_probability = mask_probability
        self.mask_ratio = tuple(mask_ratio)

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
            labeled.dataset, slides=train, tiles_per_slide=labeled.tiles_per_slide
        )
        self.validation = instantiate(
            labeled.dataset,
            slides=validation,
            tiles_per_slide=labeled.eval_tiles_per_slide,
            grid=True,
        )
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
                    worker_init_fn=enable_faulthandler,
                    collate_fn=partial(
                        collate_views,
                        mask_probability=self.mask_probability,
                        mask_ratio=self.mask_ratio,
                    ),
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
                    worker_init_fn=enable_faulthandler,
                    collate_fn=collate_tiles,
                ),
            },
            mode="max_size_cycle",
        )

    def val_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(
            self.validation,
            batch_size=self.datasets.labeled.eval_batch_size,
            num_workers=self.datasets.labeled.eval_num_workers,
            persistent_workers=True,
            pin_memory=True,
            in_order=False,
            worker_init_fn=enable_faulthandler,
            collate_fn=collate_tiles,
        )
