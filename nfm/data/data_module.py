from collections.abc import Iterable
from functools import partial

import numpy as np
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.data import DataLoader

from nfm.modeling.block_mask import (
    block_spatial_sort,
    create_ragged_block_quantized_knn_mask,
)


def train_collate_fn(
    batch: list[dict[str, np.ndarray]], block_size: int, k: int
) -> dict[str, Tensor | BlockMask]:
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")

    all_pos = []
    all_efds = []
    all_g_knns = []
    all_l_knns = []
    l_seq_lens = []

    def _local_block_lengths(global_offset: int, seq_len: int) -> list[int]:
        remaining = seq_len
        block_lengths = []

        offset_in_block = global_offset % block_size
        first_block_len = min(block_size - offset_in_block, remaining)
        block_lengths.append(first_block_len)
        remaining -= first_block_len

        while remaining >= block_size:
            block_lengths.append(block_size)
            remaining -= block_size

        if remaining > 0:
            block_lengths.append(remaining)

        return block_lengths

    current_global_idx = 0
    for b in batch:
        sort_indices = block_spatial_sort(
            b["pos"], block_size, global_offset=current_global_idx
        )
        sorted_pos = b["pos"][sort_indices]
        _, knn = nbrs.fit(sorted_pos).kneighbors(sorted_pos)

        all_pos.append(torch.from_numpy(sorted_pos))
        all_g_knns.append(torch.from_numpy(knn))
        all_l_knns.append(torch.from_numpy(knn[:, :1]))
        all_efds.append(b["efds"][sort_indices])
        l_seq_lens.extend(_local_block_lengths(current_global_idx, len(sorted_pos)))
        current_global_idx += len(sorted_pos)

    return {
        "global_block_mask": create_ragged_block_quantized_knn_mask(
            all_g_knns, block_size
        ),
        "local_block_mask": create_ragged_block_quantized_knn_mask(
            all_l_knns, block_size
        ),
        "pos": torch.cat(all_pos),
        "efds": torch.cat(all_efds),
        "g_seq_lens": torch.tensor([b["seq_len"] for b in batch], dtype=torch.int32),
        "l_seq_lens": torch.tensor(l_seq_lens, dtype=torch.int32),
    }


def inference_collate_fn(
    batch: list[dict[str, np.ndarray]], block_size: int, k: int
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
        _, knn = nbrs.fit(sorted_pos).kneighbors(sorted_pos)

        all_pos.append(torch.from_numpy(sorted_pos))
        all_knns.append(torch.from_numpy(knn))
        all_efds.append(b["efds"][sort_indices])
        all_labels.append(b["labels"][sort_indices])
        current_global_idx += len(sorted_pos)

    return {
        "block_mask": create_ragged_block_quantized_knn_mask(all_knns, block_size),
        "pos": torch.cat(all_pos),
        "efds": torch.cat(all_efds),
        "labels": torch.cat(all_labels).float(),
        "seq_lens": torch.tensor([b["seq_len"] for b in batch], dtype=torch.int32),
    }


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: dict[str, int],
        block_size: int,
        k: int,
        num_workers: dict[str, int],
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size
        self.k = k
        self.num_workers = num_workers
        self.datasets = datasets

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self.train = instantiate(self.datasets["train"])
                self.train_labeled = instantiate(self.datasets["train_labeled"])
            case "test":
                self.test = instantiate(self.datasets["test"])

    def train_dataloader(self) -> Iterable[dict[str, Tensor]]:

        return CombinedLoader(
            {
                "unlabeled": DataLoader(
                    self.train,
                    batch_size=self.batch_size["train"],
                    shuffle=True,
                    drop_last=True,
                    num_workers=self.num_workers["train"],
                    persistent_workers=True,
                    pin_memory=True,
                    in_order=False,
                    collate_fn=partial(
                        train_collate_fn, block_size=self.block_size, k=self.k
                    ),
                ),
                "labeled": DataLoader(
                    self.train_labeled,
                    batch_size=self.batch_size["train_labeled"],
                    shuffle=True,
                    drop_last=True,
                    num_workers=self.num_workers["train_labeled"],
                    persistent_workers=True,
                    pin_memory=True,
                    in_order=False,
                    collate_fn=partial(
                        inference_collate_fn, block_size=self.block_size, k=self.k
                    ),
                ),
            },
            mode="max_size_cycle",
        )
