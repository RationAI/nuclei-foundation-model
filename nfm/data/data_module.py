from collections.abc import Iterable
from functools import partial

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.data import DataLoader

from nfm.modeling.block_mask import (
    block_spatial_sort,
    create_ragged_block_quantized_knn_mask,
)


def train_collate_fn(
    batch: list[dict[str, np.ndarray]], block_size: int, k: int, total_seq_len: int
) -> dict[str, Tensor | BlockMask | list[int]]:
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")

    all_pos = []
    all_efds = []
    all_knns = []
    all_indices = []

    current_global_idx = 0
    for b in batch:
        for crop in range(len(b["pos"])):
            sort_indices = block_spatial_sort(
                b["pos"][crop], block_size, global_offset=current_global_idx
            )
            sorted_pos = b["pos"][crop][sort_indices]
            _, knn = nbrs.fit(sorted_pos).kneighbors(sorted_pos)

            all_pos.append(torch.from_numpy(sorted_pos))
            all_knns.append(torch.from_numpy(knn))
            all_efds.append(torch.from_numpy(b["efds"][crop][sort_indices]))
            all_indices.append(torch.from_numpy(b["indices"][crop][sort_indices]))
            current_global_idx += len(sorted_pos)

    return {
        "block_mask": create_ragged_block_quantized_knn_mask(
            all_knns, block_size, None
        ),
        # "pos": _pad_to_seq_len(torch.cat(all_pos), total_seq_len),
        # "efds": _pad_to_seq_len(torch.cat(all_efds), total_seq_len),
        "pos": torch.cat(all_pos),
        "efds": torch.cat(all_efds),
        "all_indices": torch.cat(all_indices),
        "seq_lens": [x for b in batch for x in b["seq_lens"]],
    }


def inference_collate_fn(
    batch: list[dict[str, np.ndarray]],
    block_size: int,
    k: int,
    total_seq_len: int | None = None,
) -> dict[str, Tensor | BlockMask]:
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")

    all_pos = []
    all_efds = []
    all_labels = []
    all_knns = []

    current_global_idx = 0
    for b in batch:
        sort_indices = block_spatial_sort(
            b["pos"], block_size, global_offset=current_global_idx
        )
        sorted_pos = b["pos"][sort_indices]
        _, knn = nbrs.fit(sorted_pos).kneighbors(
            sorted_pos, n_neighbors=min(k, len(sorted_pos))
        )

        all_pos.append(torch.from_numpy(sorted_pos))
        all_knns.append(torch.from_numpy(knn))
        all_efds.append(b["efds"][sort_indices])
        all_labels.append(b["labels"][sort_indices])
        current_global_idx += len(sorted_pos)

    return {
        "block_mask": create_ragged_block_quantized_knn_mask(
            all_knns, block_size, None
        ),
        # "pos": _pad_to_seq_len(torch.cat(all_pos), total_seq_len),
        # "efds": _pad_to_seq_len(torch.cat(all_efds), total_seq_len),
        "pos": torch.cat(all_pos),
        "efds": torch.cat(all_efds),
        "labels": torch.cat(all_labels).float(),
        "seq_lens": torch.tensor([b["seq_len"] for b in batch], dtype=torch.int32),
    }


def _pad_to_seq_len(x: Tensor, total_seq_len: int | None, value: float = 0) -> Tensor:
    if total_seq_len is None:
        return x
    pad_len = total_seq_len - x.shape[0]
    if pad_len == 0:
        return x
    pad_shape = (pad_len, *x.shape[1:])
    pad = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
    return torch.cat((x, pad), dim=0)


class DataModule(LightningDataModule):
    def __init__(self, datasets: DictConfig, block_size: int, k: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.k = k
        self.datasets = datasets

    def prepare_data(self) -> None:
        for dataset in self.datasets.values():
            if dataset.slides_path.startswith("mlflow-artifacts:"):
                dataset.slides_path = download_artifacts(dataset.slides_path)

    def setup(self, stage: str) -> None:
        labeled = self.datasets.labeled
        slides = pd.read_parquet(labeled.slides_path)
        train, validation = train_test_split(
            slides,
            test_size=labeled.split_size,
            stratify=slides[labeled.split_stratify_col],
        )
        self.validation = instantiate(labeled.dataset, slides=validation)
        if stage == "fit":
            if "unlabeled" in self.datasets:
                self.train = instantiate(
                    self.datasets.unlabeled.dataset,
                    slides=pd.read_parquet(self.datasets.unlabeled.slides_path),
                )
            self.train_labeled = instantiate(labeled.dataset, slides=train)

    def train_dataloader(self) -> Iterable[dict[str, Tensor]]:
        cfg = self.datasets.labeled
        sampler = instantiate(cfg.sampler, dataset=self.train_labeled)
        labeled = DataLoader(
            self.train_labeled,
            batch_size=cfg.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0,
            pin_memory=True,
            in_order=False,
            collate_fn=partial(
                inference_collate_fn,
                block_size=self.block_size,
                k=self.k,
                total_seq_len=cfg.total_seq_len,
            ),
        )
        if "unlabeled" not in self.datasets:
            return labeled

        cfg = self.datasets.unlabeled
        sampler = (
            instantiate(cfg.sampler, dataset=self.train)
            if cfg.get("sampler") is not None
            else None
        )
        return CombinedLoader(
            {
                "unlabeled": DataLoader(
                    self.train,
                    batch_size=cfg.batch_size,
                    sampler=sampler,
                    shuffle=sampler is None,
                    drop_last=True,
                    num_workers=cfg.num_workers,
                    persistent_workers=cfg.num_workers > 0,
                    pin_memory=True,
                    in_order=False,
                    collate_fn=partial(
                        train_collate_fn,
                        block_size=self.block_size,
                        k=self.k,
                        total_seq_len=cfg.total_seq_len,
                    ),
                ),
                "labeled": labeled,
            },
            mode="max_size_cycle",
        )

    def val_dataloader(self) -> Iterable[dict[str, Tensor]]:
        cfg = self.datasets.labeled
        return DataLoader(
            self.validation,
            batch_size=1,
            num_workers=cfg.eval_num_workers,
            persistent_workers=cfg.eval_num_workers > 0,
            pin_memory=True,
            in_order=False,
            collate_fn=partial(
                inference_collate_fn, block_size=self.block_size, k=self.k
            ),
        )
