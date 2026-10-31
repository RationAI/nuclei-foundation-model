from typing import Any

import pandas as pd
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from nfm.data.samplers import SourceBalancedRandomSampler


def collate_tiles(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat([sample[key] for sample in batch]) for key in batch[0]}


class DataModule(LightningDataModule):
    def __init__(self, datasets: DictConfig) -> None:
        super().__init__()
        self.datasets = datasets

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
                    multiprocessing_context="spawn",  # for lance
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
                    collate_fn=collate_tiles,
                ),
            },
            mode="max_size_cycle",
        )

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return DataLoader(
            self.validation,
            batch_size=self.datasets.labeled.eval_batch_size,
            num_workers=self.datasets.labeled.eval_num_workers,
            persistent_workers=True,
            pin_memory=True,
            in_order=False,
            collate_fn=collate_tiles,
        )
