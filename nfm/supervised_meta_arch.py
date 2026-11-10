"""Supervised baseline: the nuclei transformer trained on Radboud tile labels.

The tile head on the class token is trained end to end. The nucleus probe stays
detached, as in the self-supervised model, so nucleus labels never reach the
encoder. Validation logs under the same names as the self-supervised model.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)

from nfm.modeling.transformer import NucleiTransformer, cls_indices


def _metrics(prefix: str) -> MetricCollection:
    return MetricCollection(
        {
            "AUROC": BinaryAUROC(),
            "AUPRC": BinaryAveragePrecision(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        },
        prefix=prefix,
    )


class SupervisedMetaArch(LightningModule):
    def __init__(
        self,
        backbone: dict[str, Any],
        lr: float = 5e-4,
        probe_lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.probe_lr = probe_lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.backbone = NucleiTransformer(**backbone)
        dim = self.backbone.norm.normalized_shape[0]
        self.tile_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.nucleus_probe = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.tile_metrics = _metrics("validation/tile/")
        self.nucleus_metrics = _metrics("validation/nucleus/")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Returns the class tokens, [tiles, dim], and nucleus tokens, [nuclei, dim]."""
        tokens = self.backbone(batch["features"], batch["positions"], batch["lengths"])
        cls = cls_indices(batch["lengths"])
        nuclei = torch.ones(len(tokens), dtype=torch.bool, device=tokens.device)
        nuclei[cls] = False
        return tokens[cls], tokens[nuclei]

    def logits(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        cls, nuclei = self(batch)
        tile = self.tile_head(cls).squeeze(1)
        # Nucleus labels must not reach the encoder.
        return tile, self.nucleus_probe(nuclei.detach()).squeeze(1)

    def training_step(self, batch: dict[str, Tensor]) -> Tensor:
        tile_logits, nucleus_logits = self.logits(batch)
        tile = F.binary_cross_entropy_with_logits(tile_logits, batch["labels"])
        nucleus = F.binary_cross_entropy_with_logits(
            nucleus_logits, batch["nucleus_labels"]
        )
        self.log_dict(
            {
                "train/tile_loss": tile,
                "train/nucleus_probe_loss": nucleus,
                "train/tokens": float(len(batch["features"])),
            },
            batch_size=len(tile_logits),
            prog_bar=True,
        )
        return tile + nucleus

    def validation_step(self, batch: dict[str, Tensor]) -> None:
        tile_logits, nucleus_logits = self.logits(batch)
        for name, logits, labels, metrics in (
            ("tile", tile_logits, batch["labels"], self.tile_metrics),
            ("nucleus", nucleus_logits, batch["nucleus_labels"], self.nucleus_metrics),
        ):
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            metrics.update(logits.sigmoid(), labels.long())
            # Named as in the self-supervised model, so runs line up.
            self.log(
                f"validation/{name}/probe_loss",
                loss,
                batch_size=len(logits),
                sync_dist=True,
            )
            self.log_dict(metrics, batch_size=len(logits))

    def configure_optimizers(self) -> dict[str, Any]:
        decay, no_decay = [], []
        for name, parameter in [
            *self.backbone.named_parameters(),
            *self.tile_head.named_parameters(),
        ]:
            # Decay would shrink the RoVE frequencies, norms and special tokens.
            skip = parameter.ndim <= 1 or name.endswith("rope.freqs")
            (no_decay if skip else decay).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "lr": self.lr, "weight_decay": self.weight_decay},
                {"params": no_decay, "lr": self.lr, "weight_decay": 0.0},
                {
                    "params": self.nucleus_probe.parameters(),
                    "lr": self.probe_lr,
                    "weight_decay": 1e-7,
                },
            ],
        )
        total = int(self.trainer.estimated_stepping_batches)

        def schedule(step: int) -> float:
            if step < self.warmup_steps:
                return 0.01 + 0.99 * step / self.warmup_steps
            progress = (step - self.warmup_steps) / (total - self.warmup_steps)
            return 1e-3 + (1 - 1e-3) * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
