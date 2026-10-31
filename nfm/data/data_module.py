from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from mlflow.artifacts import download_artifacts
from numpy.typing import NDArray
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.utils.data import DataLoader

from nfm.data.datasets.nuclei_dataset import Sample
from nfm.data.samplers import SourceBalancedRandomSampler
from nfm.data.timing import DataTiming
from nfm.modeling.block_mask import (
    BlockMask,
    block_spatial_sort,
    create_ragged_block_quantized_knn_mask,
)


type Crop = tuple[NDArray[np.float32 | np.float64], NDArray[np.float32] | Tensor]
type Layout = list[tuple[NDArray[np.intp], int]]
type Batch = dict[str, Tensor | BlockMask | list[str]]


def _pack_crops(
    crops: list[Crop],
    block_size: int,
    k: int,
    max_sequence_length: int,
    *,
    timing: DataTiming | None = None,
) -> tuple[Batch, Layout]:
    """Pack crops contiguously at their actual lengths."""
    timing = timing or DataTiming("pack", enabled=False)
    positions, efds, neighbors, layout = [], [], [], []
    offset = 0
    for pos, features in crops:
        n = len(pos)
        with timing.measure("spatial_sort"):
            order = block_spatial_sort(pos, block_size, global_offset=offset)
            pos = np.asarray(pos[order], dtype=np.float32)
        with timing.measure("knn_fit"):
            search = NearestNeighbors(n_neighbors=min(k, n)).fit(pos)
        with timing.measure("knn_query"):
            knn = search.kneighbors(pos, return_distance=False)
        with timing.measure("crop_to_tensors"):
            positions.append(torch.from_numpy(pos))
            efds.append(torch.as_tensor(features)[order].float())
            neighbors.append(
                F.pad(torch.from_numpy(knn), (0, k - knn.shape[1]), value=-1)
            )
        layout.append((order, offset))
        offset += n
    with timing.measure("concatenate_tensors"):
        packed = {"pos": torch.cat(positions), "efds": torch.cat(efds)}
    with timing.measure("build_block_mask"):
        packed["block_mask"] = create_ragged_block_quantized_knn_mask(
            neighbors, block_size, max_sequence_length=max_sequence_length
        )
    return packed, layout


def train_collate_fn(
    batch: list[Sample], block_size: int, k: int, max_sequence_length: int
) -> Batch:
    with DataTiming(
        "unlabeled.collate", slides=[s["slide_id"] for s in batch], samples=len(batch)
    ) as timing:
        crops = [
            (pos, efd)
            for sample in batch
            for pos, efd in zip(sample["pos"], sample["efds"], strict=True)
        ]
        packed, layout = _pack_crops(
            crops, block_size, k, max_sequence_length, timing=timing
        )
        with timing.measure("map_correspondences"):
            pairs = [[], []]
            for b, sample in enumerate(batch):
                for view in range(2):
                    order, offset = layout[2 * b + view]
                    ids = sample["indices"][view][order]
                    lookup = {
                        int(nucleus_id): offset + i for i, nucleus_id in enumerate(ids)
                    }
                    pairs[view].append(
                        torch.tensor(
                            [lookup[int(i)] for i in sample["matched_ids"]],
                            dtype=torch.long,
                        )
                    )
        with timing.measure("assemble_batch"):
            packed.update(
                pair_indices=torch.stack([torch.cat(p) for p in pairs]),
                pair_counts=torch.tensor([len(s["matched_ids"]) for s in batch]),
                slide_ids=[s["slide_id"] for s in batch],
            )
        timing.details.update(
            crops=len(crops),
            crop_nuclei=[len(pos) for pos, _ in crops],
            packed_nuclei=len(packed["pos"]),
            matches=packed["pair_indices"].shape[1],
        )
        return packed


def inference_collate_fn(
    batch: list[Sample], block_size: int, k: int, max_sequence_length: int
) -> Batch:
    packed, layout = _pack_crops(
        [(s["pos"], s["efds"]) for s in batch], block_size, k, max_sequence_length
    )
    labels = torch.cat(
        [
            torch.as_tensor(sample["labels"])[order].reshape(-1, 1)
            for sample, (order, _) in zip(batch, layout, strict=True)
        ]
    )
    packed.update(
        labels=labels.nan_to_num(),
        score=torch.isfinite(labels[:, 0]),
        slide_ids=[s["slide_id"] for s in batch],
    )
    return packed


class DataModule(LightningDataModule):
    def __init__(
        self, datasets: DictConfig, block_size: int, k: int, max_sequence_length: int
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.k = k
        self.datasets = datasets
        self.max_sequence_length = max_sequence_length

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
        # Training keeps the configured fixed-count crops and positive seeding.
        # Validation always reads every nucleus, even when it exceeds the
        # labeled training crop size.
        self.validation = instantiate(
            labeled.dataset, slides=validation, crop_size=None
        )

        train = train[train.nuclei_count >= labeled.dataset.crop_size]
        self.train_labeled = instantiate(labeled.dataset, slides=train)

        slides = pd.read_parquet(self.datasets.unlabeled.slides_path)
        self.train = instantiate(self.datasets.unlabeled.dataset, slides=slides)

    def train_dataloader(self) -> CombinedLoader:
        return CombinedLoader(
            {
                "unlabeled": DataLoader(
                    self.train,
                    batch_size=self.datasets.unlabeled.batch_size,
                    sampler=SourceBalancedRandomSampler(self.train),
                    drop_last=True,
                    num_workers=self.datasets.unlabeled.num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                    in_order=False,
                    multiprocessing_context="spawn",  # for lance
                    collate_fn=partial(
                        train_collate_fn,
                        block_size=self.block_size,
                        k=self.k,
                        max_sequence_length=self.max_sequence_length,
                    ),
                ),
                "labeled": DataLoader(
                    self.train_labeled,
                    batch_size=self.datasets.labeled.batch_size,
                    sampler=instantiate(
                        self.datasets.labeled.sampler, dataset=self.train_labeled
                    ),
                    drop_last=True,
                    num_workers=self.datasets.labeled.num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                    in_order=False,
                    collate_fn=partial(
                        inference_collate_fn,
                        block_size=self.block_size,
                        k=self.k,
                        max_sequence_length=self.max_sequence_length,
                    ),
                ),
            },
            mode="max_size_cycle",
        )

    def val_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(
            self.validation,
            batch_size=1,
            num_workers=self.datasets.labeled.eval_num_workers,
            persistent_workers=True,
            pin_memory=True,
            in_order=False,
            collate_fn=partial(
                inference_collate_fn,
                block_size=self.block_size,
                k=self.k,
                max_sequence_length=self.max_sequence_length,
            ),
        )
