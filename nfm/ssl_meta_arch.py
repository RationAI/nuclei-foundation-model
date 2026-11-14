"""LeJEPA on pooled Voronoi cells of nuclei, with detached region and nucleus probes."""

import math
from typing import Any

import torch
import torch.nn.functional as F
from lejepa.multivariate import SlicingUnivariateTest
from lejepa.univariate import EppsPulley
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from torchvision.ops import MLP

from nfm.modeling.pooling import pooling as make_pooling
from nfm.modeling.transformer import NucleiTransformer


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


class SSLMetaArch(LightningModule):
    def __init__(
        self,
        backbone: dict[str, Any],
        pooling: dict[str, Any] | None = None,
        projector_dim: int = 16,
        lamb: float = 0.02,
        lr: float = 5e-4,
        probe_lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.lamb = lamb
        self.lr = lr
        self.probe_lr = probe_lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.backbone = NucleiTransformer(**backbone)
        dim = self.backbone.norm.normalized_shape[0]
        # Part of the encoder: trained by the objective, frozen for the probes.
        self.pooling = make_pooling(dim, **(pooling or {}))
        self.projector = MLP(
            dim, [2048, 2048, projector_dim], norm_layer=nn.BatchNorm1d
        )
        self.sigreg = SlicingUnivariateTest(EppsPulley(n_points=17), num_slices=256)
        # The nucleus tokens are RMS normed by the backbone. The SimPool cells are
        # a generalized mean of shifted tokens, positive and not normalized, so
        # the region probe standardizes them.
        self.region_probe = nn.Sequential(
            nn.LayerNorm(dim, elementwise_affine=False), nn.Linear(dim, 1)
        )
        self.nucleus_probe = nn.Linear(dim, 1)
        self.region_metrics = _metrics("validation/region/")
        self.nucleus_metrics = _metrics("validation/nucleus/")

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        """Returns the cell embeddings, [cells, dim], and nucleus tokens."""
        nuclei = self.backbone(
            batch["features"], batch["positions"], batch["lengths"], batch["neighbors"]
        )
        return self.pooling(nuclei, batch["cells"], batch["cell_count"]), nuclei

    def probe_logits(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        regions, nuclei = self(batch)
        return (
            self.region_probe(regions).squeeze(1),
            self.nucleus_probe(nuclei).squeeze(1),
        )

    def encoder_eval(self, mode: bool = True) -> None:
        """Sets the encoder and the pooling, which belongs to it, to eval mode."""
        self.backbone.train(not mode)
        self.pooling.train(not mode)

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        views = batch["unlabeled"]
        regions, _ = self(views)
        # Cells of different views that hold the same nuclei are paired, and each
        # predicts the center of its pair.
        first, second = views["pairs"]
        embeddings = torch.cat([regions[first], regions[second]])
        proj = self.projector(embeddings).unflatten(0, (2, -1))  # [2, pairs, dim]
        invariance = (proj.mean(0) - proj).square().mean()
        with torch.autocast(self.device.type, enabled=False):
            sigreg = self.sigreg(proj.float())
        lejepa = self.lamb * sigreg + (1 - self.lamb) * invariance

        # Labels must not reach the encoder, nor its batch norm statistics.
        labeled = batch["labeled"]
        self.encoder_eval()
        with torch.no_grad():
            region_features, nucleus_features = self(labeled)
        self.encoder_eval(False)
        region_probe = F.binary_cross_entropy_with_logits(
            self.region_probe(region_features).squeeze(1), labeled["cell_labels"]
        )
        nucleus_probe = F.binary_cross_entropy_with_logits(
            self.nucleus_probe(nucleus_features).squeeze(1), labeled["labels"]
        )

        self.log_dict(
            {
                "train/lejepa_loss": lejepa,
                "train/invariance_loss": invariance,
                "train/sigreg_loss": sigreg,
                "train/tokens": float(len(views["features"])),
                "train/cells": float(len(regions)),
                "train/pairs": float(first.numel()),
                "train/region_probe_loss": region_probe,
                "train/nucleus_probe_loss": nucleus_probe,
                **self.embedding_statistics(embeddings.detach().float()),
            },
            batch_size=views["batch_size"],
            prog_bar=True,
        )
        return lejepa + region_probe + nucleus_probe

    @staticmethod
    def embedding_statistics(embeddings: Tensor) -> dict[str, Tensor]:
        """Spread of the paired cell embeddings, comparable across encoders.

        The std is of the L2-normalized embeddings, scaled so that isotropic ones
        give 1. The effective rank is the exponential of the entropy of the
        normalized singular values of the centered embeddings.
        """
        embeddings = embeddings[:4096]  # plenty for the spread, and much cheaper
        normalized = F.normalize(embeddings, dim=1) * embeddings.size(1) ** 0.5
        singular = torch.linalg.svdvals(embeddings - embeddings.mean(0))
        p = singular / singular.sum()
        return {
            "train/embedding_std": normalized.std(0).mean(),
            "train/embedding_rank": torch.exp(-(p * p.clamp_min(1e-12).log()).sum()),
        }

    def validation_step(self, batch: dict[str, Tensor]) -> None:
        region_logits, nucleus_logits = self.probe_logits(batch)
        for name, logits, labels, metrics in (
            ("region", region_logits, batch["cell_labels"], self.region_metrics),
            ("nucleus", nucleus_logits, batch["labels"], self.nucleus_metrics),
        ):
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            metrics.update(logits.sigmoid(), labels.long())
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
            *self.pooling.named_parameters(),
            *self.projector.named_parameters(),
        ]:
            # Decay would shrink the RoVE frequencies, norms and special tokens.
            skip = parameter.ndim <= 1 or name.endswith("rope.freqs")
            (no_decay if skip else decay).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "lr": self.lr, "weight_decay": self.weight_decay},
                {"params": no_decay, "lr": self.lr, "weight_decay": 0.0},
                {
                    "params": [
                        *self.region_probe.parameters(),
                        *self.nucleus_probe.parameters(),
                    ],
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
