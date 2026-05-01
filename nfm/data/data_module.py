from collections.abc import Iterable
from copy import deepcopy
from functools import partial

import numpy as np
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.data import DataLoader

from nfm.data.augmentations import Compose
from nfm.modeling.block_mask import (
    block_spatial_sort,
    create_ragged_block_quantized_knn_mask,
)


def train_collate_fn(
    batch: list[dict[str, np.ndarray]],
    block_size: int,
    k: int,
    total_seq_len: int | None = None,
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

    real_seq_len = current_global_idx
    target_seq_len = total_seq_len or real_seq_len
    if target_seq_len < real_seq_len:
        raise ValueError(
            f"total_seq_len ({target_seq_len}) must be >= packed length ({real_seq_len})"
        )

    return {
        "block_mask": create_ragged_block_quantized_knn_mask(
            all_knns, block_size, target_seq_len
        ),
        "pos": _pad_to_seq_len(torch.cat(all_pos), target_seq_len),
        "efds": _pad_to_seq_len(torch.cat(all_efds), target_seq_len),
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
        _, knn = nbrs.fit(sorted_pos).kneighbors(sorted_pos)

        all_pos.append(torch.from_numpy(sorted_pos))
        all_knns.append(torch.from_numpy(knn))
        all_efds.append(b["efds"][sort_indices])
        all_labels.append(b["labels"][sort_indices])
        current_global_idx += len(sorted_pos)

    real_seq_len = current_global_idx
    target_seq_len = total_seq_len or real_seq_len
    if target_seq_len < real_seq_len:
        raise ValueError(
            f"total_seq_len ({target_seq_len}) must be >= packed length ({real_seq_len})"
        )

    return {
        "block_mask": create_ragged_block_quantized_knn_mask(
            all_knns, block_size, target_seq_len
        ),
        "pos": _pad_to_seq_len(torch.cat(all_pos), target_seq_len),
        "efds": _pad_to_seq_len(torch.cat(all_efds), target_seq_len),
        "labels": torch.cat(all_labels).float(),
        "seq_lens": torch.tensor([b["seq_len"] for b in batch], dtype=torch.int32),
    }


def _pad_to_seq_len(x: Tensor, total_seq_len: int, value: float | int = 0) -> Tensor:
    pad_len = total_seq_len - x.shape[0]
    if pad_len == 0:
        return x
    pad_shape = (pad_len, *x.shape[1:])
    pad = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
    return torch.cat((x, pad), dim=0)


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: dict[str, int],
        block_size: int,
        k: int,
        num_workers: dict[str, int],
        total_seq_len: int | None = None,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size
        self.k = k
        self.total_seq_len = total_seq_len
        self.num_workers = num_workers
        self.datasets = datasets

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self.train = instantiate(self.datasets["train"])
                train_labeled = instantiate(self.datasets["train_labeled"])
                self.train_labeled, self.val_labeled = torch.utils.data.random_split(
                    train_labeled, [0.7, 0.3]
                )
                self.val_labeled.dataset = deepcopy(self.val_labeled.dataset)
                self.val_labeled.dataset.augmentation = Compose([])

    def train_dataloader(self) -> Iterable[dict[str, Tensor]]:
        return DataLoader(
            self.train_labeled,
            batch_size=self.batch_size["train_labeled"],
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers["train_labeled"],
            persistent_workers=True,
            pin_memory=True,
            in_order=False,
            collate_fn=partial(
                inference_collate_fn,
                block_size=self.block_size,
                k=self.k,
                total_seq_len=self.total_seq_len,
            ),
        )

    def val_dataloader(self) -> Iterable[dict[str, Tensor]]:
        return DataLoader(
            self.val_labeled,
            batch_size=self.batch_size["train_labeled"],
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers["train_labeled"],
            persistent_workers=True,
            pin_memory=True,
            in_order=False,
            collate_fn=partial(
                inference_collate_fn,
                block_size=self.block_size,
                k=self.k,
                total_seq_len=self.total_seq_len,
            ),
        )

        # return CombinedLoader(
        #     {
        #         "unlabeled": DataLoader(
        #             self.train,
        #             batch_size=self.batch_size["train"],
        #             shuffle=True,
        #             drop_last=True,
        #             num_workers=self.num_workers["train"],
        #             persistent_workers=True,
        #             pin_memory=True,
        #             in_order=False,
        #             collate_fn=partial(
        #                 train_collate_fn, block_size=self.block_size, k=self.k
        #             ),
        #         ),
        #         "labeled": DataLoader(
        #             self.train_labeled,
        #             batch_size=self.batch_size["train_labeled"],
        #             shuffle=True,
        #             drop_last=True,
        #             num_workers=self.num_workers["train_labeled"],
        #             persistent_workers=True,
        #             pin_memory=True,
        #             in_order=False,
        #             collate_fn=partial(
        #                 inference_collate_fn, block_size=self.block_size, k=self.k
        #             ),
        #         ),
        #     },
        #     mode="max_size_cycle",
        # )
