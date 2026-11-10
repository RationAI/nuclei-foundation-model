import atexit
import faulthandler
import os

import pandas as pd
from hydra.utils import instantiate
from lightning import LightningDataModule
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from nfm.data.packing import Batch, collate_tiles


def init_worker(worker_id: int) -> None:
    """Prepares a DataLoader worker.

    Prints native stack traces when the worker crashes. PyTorch replaces the
    crash handlers in each worker, so they are re-enabled here. Workers exit
    without native finalization, where native libraries sometimes abort.
    """
    faulthandler.disable()
    faulthandler.enable()
    atexit.register(os._exit, 0)


class DataModule(LightningDataModule):
    """Radboud tiles: augmented random tiles for training, whole slides for validation.

    The split is the one of the self-supervised model for the same seed.
    """

    def __init__(
        self,
        slides_path: str,
        split_size: float,
        split_stratify_col: str,
        split_seed: int,
        dataset: DictConfig,
        augmentation: DictConfig,
        tiles_per_slide: int,
        batch_size: int,
        num_workers: int,
        eval_num_workers: int,
    ) -> None:
        super().__init__()
        self.slides_path = slides_path
        self.split_size = split_size
        self.split_stratify_col = split_stratify_col
        self.split_seed = split_seed
        self.dataset = dataset
        self.augmentation = augmentation
        self.tiles_per_slide = tiles_per_slide
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_num_workers = eval_num_workers

    def prepare_data(self) -> None:
        if self.slides_path.startswith("mlflow-artifacts:"):
            self.slides_path = download_artifacts(self.slides_path)

    def setup(self, stage: str) -> None:
        slides = pd.read_parquet(self.slides_path)
        slides = slides[slides["nuclei_count"] >= 512]
        train, validation = train_test_split(
            slides,
            test_size=self.split_size,
            stratify=slides[self.split_stratify_col],
            random_state=self.split_seed,
        )
        self.train = instantiate(
            self.dataset,
            slides=train,
            tiles_per_slide=self.tiles_per_slide,
            augmentation=self.augmentation,
        )
        # Whole slides, divided into tiles.
        self.validation = instantiate(self.dataset, slides=validation)

    def train_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            in_order=False,
            worker_init_fn=init_worker,
            collate_fn=collate_tiles,
        )

    def val_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(
            self.validation,
            batch_size=1,  # a whole slide
            num_workers=self.eval_num_workers,
            persistent_workers=True,
            pin_memory=True,
            in_order=False,
            worker_init_fn=init_worker,
            collate_fn=collate_tiles,
        )
