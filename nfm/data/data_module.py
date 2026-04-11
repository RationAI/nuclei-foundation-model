from collections.abc import Iterable

import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from omegaconf import DictConfig
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.data import DataLoader

from nfm.modeling.block_mask import create_ragged_block_quantized_knn_mask


def train_collate_fn(
    batch: list[dict[str, Tensor]],
) -> dict[str, Tensor | BlockMask]:
    g_knn = [b["global_knn"] for b in batch]
    l_knn = [b["local_knn"] for b in batch]
    return {
        "global_block_mask": create_ragged_block_quantized_knn_mask(
            g_knn, block_size=128
        ),
        "local_block_mask": create_ragged_block_quantized_knn_mask(
            l_knn, block_size=128
        ),
        "pos": torch.cat([b["pos"] for b in batch]),
        "efds": torch.cat([b["efds"] for b in batch]),
        "seq_lens": torch.tensor([b["seq_len"] for b in batch], dtype=torch.int32),
    }


def inference_collate_fn(
    batch: list[dict[str, Tensor]],
) -> dict[str, Tensor | BlockMask]:
    knn = [b["knn"] for b in batch]
    return {
        "block_mask": create_ragged_block_quantized_knn_mask(knn, block_size=128),
        "pos": torch.cat([b["pos"] for b in batch]),
        "efds": torch.cat([b["efds"] for b in batch]),
        "labels": torch.cat([b["labels"] for b in batch]).float(),
        "seq_lens": torch.tensor([b["seq_len"] for b in batch], dtype=torch.int32),
    }


class DataModule(LightningDataModule):
    def __init__(
        self, batch_size: int, num_workers: int = 0, **datasets: DictConfig
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
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
        labeled_batch = self.batch_size // 3

        return CombinedLoader(
            {
                "unlabeled": DataLoader(
                    self.train,
                    batch_size=self.batch_size - labeled_batch,
                    shuffle=True,
                    drop_last=True,
                    num_workers=self.num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                    in_order=False,
                    collate_fn=train_collate_fn,
                ),
                "labeled": DataLoader(
                    self.train_labeled,
                    batch_size=labeled_batch,
                    shuffle=True,
                    drop_last=True,
                    num_workers=self.num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                    in_order=False,
                    collate_fn=inference_collate_fn,
                ),
            },
            mode="max_size_cycle",
        )
