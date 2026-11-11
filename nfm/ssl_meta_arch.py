"""LeJEPA on local class tokens of nuclei, with detached region and nucleus probes."""

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
        projector_dim: int = 16,
        lamb: float = 0.02,
        lr: float = 5e-4,
        probe_lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lamb = lamb
        self.lr = lr
        self.probe_lr = probe_lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.backbone = NucleiTransformer(**backbone)
        dim = self.backbone.norm.normalized_shape[0]
        self.projector = MLP(
            dim, [2048, 2048, projector_dim], norm_layer=nn.BatchNorm1d
        )
        self.sigreg = SlicingUnivariateTest(EppsPulley(n_points=17), num_slices=256)
        self.region_probe = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.nucleus_probe = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.region_metrics = _metrics("validation/region/")
        self.nucleus_metrics = _metrics("validation/nucleus/")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Returns the class tokens, [class tokens, dim], and nucleus tokens."""
        nuclei, cls = self.backbone(
            batch["features"],
            batch["positions"],
            batch["lengths"],
            batch["neighbors"],
            batch["cls_positions"],
            batch["cls_lengths"],
            batch["cls_neighbors"],
        )
        return cls, nuclei

    def probe_logits(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        cls, nuclei = self(batch)
        return (
            self.region_probe(cls).squeeze(1),
            self.nucleus_probe(nuclei).squeeze(1),
        )

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        views = batch["unlabeled"]
        cls, _ = self(views)
        # Class tokens of different views that cover the same nuclei are paired,
        # and each predicts the center of its pair.
        first, second = views["pairs"]
        embeddings = torch.cat([cls[first], cls[second]])
        proj = self.projector(embeddings).unflatten(0, (2, -1))  # [2, pairs, dim]
        invariance = (proj.mean(0) - proj).square().mean()
        with torch.autocast(self.device.type, enabled=False):
            sigreg = self.sigreg(proj.float())
        lejepa = self.lamb * sigreg + (1 - self.lamb) * invariance

        # Labels must not reach the encoder, nor its batch norm statistics.
        labeled = batch["labeled"]
        self.backbone.eval()
        with torch.no_grad():
            region_features, nucleus_features = self(labeled)
        self.backbone.train()
        region_probe = F.binary_cross_entropy_with_logits(
            self.region_probe(region_features).squeeze(1), labeled["cls_labels"]
        )
        nucleus_probe = F.binary_cross_entropy_with_logits(
            self.nucleus_probe(nucleus_features).squeeze(1), labeled["labels"]
        )

        self.log_dict(
            {
                "train/lejepa_loss": lejepa,
                "train/invariance_loss": invariance,
                "train/sigreg_loss": sigreg,
                "train/tokens": float(len(views["features"]) + len(cls)),
                "train/class_tokens": float(len(cls)),
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
        """Spread of the paired class tokens, comparable across encoders.

        The std is of the L2-normalized embeddings, scaled so that isotropic ones
        give 1. The effective rank is the exponential of the entropy of the
        normalized singular values of the centered embeddings.
        """
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
            ("region", region_logits, batch["cls_labels"], self.region_metrics),
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
