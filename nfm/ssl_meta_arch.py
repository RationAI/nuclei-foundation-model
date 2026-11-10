"""LeJEPA on nuclei tokens with detached tile and nucleus probes."""

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
        self.tile_probe = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.nucleus_probe = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.tile_metrics = _metrics("validation/tile/")
        self.nucleus_metrics = _metrics("validation/nucleus/")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Returns the class tokens, [views, dim], and nucleus tokens, [nuclei, dim]."""
        tokens = self.backbone(batch["features"], batch["positions"], batch["lengths"])
        cls = cls_indices(batch["lengths"])
        nuclei = torch.ones(len(tokens), dtype=torch.bool, device=tokens.device)
        nuclei[cls] = False
        return tokens[cls], tokens[nuclei]

    def probe_logits(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        cls, nuclei = self(batch)
        return self.tile_probe(cls).squeeze(1), self.nucleus_probe(nuclei).squeeze(1)

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        views = batch["unlabeled"]
        n, v = views["batch_size"], views["views"]
        embeddings, _ = self(views)
        proj = self.projector(embeddings).reshape(v, n, -1)  # [views, batch, dim]
        invariance = (proj.mean(0) - proj).square().mean()
        with torch.autocast(self.device.type, enabled=False):
            sigreg = self.sigreg(proj.float())
        lejepa = self.lamb * sigreg + (1 - self.lamb) * invariance

        # Labels must not reach the encoder, nor its batch norm statistics.
        labeled = batch["labeled"]
        self.backbone.eval()
        with torch.no_grad():
            tile_features, nucleus_features = self(labeled)
        self.backbone.train()
        tile_probe = F.binary_cross_entropy_with_logits(
            self.tile_probe(tile_features).squeeze(1), labeled["labels"]
        )
        nucleus_probe = F.binary_cross_entropy_with_logits(
            self.nucleus_probe(nucleus_features).squeeze(1), labeled["nucleus_labels"]
        )

        self.log_dict(
            {
                "train/lejepa_loss": lejepa,
                "train/invariance_loss": invariance,
                "train/sigreg_loss": sigreg,
                "train/tokens": float(len(views["features"])),
                "train/tile_probe_loss": tile_probe,
                "train/nucleus_probe_loss": nucleus_probe,
                **self.embedding_statistics(embeddings.detach().float()),
            },
            batch_size=n,
            prog_bar=True,
        )
        return lejepa + tile_probe + nucleus_probe

    @staticmethod
    def embedding_statistics(embeddings: Tensor) -> dict[str, Tensor]:
        """Spread of the class tokens, comparable across encoders.

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
        tile_logits, nucleus_logits = self.probe_logits(batch)
        for name, logits, labels, metrics in (
            ("tile", tile_logits, batch["labels"], self.tile_metrics),
            ("nucleus", nucleus_logits, batch["nucleus_labels"], self.nucleus_metrics),
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
                        *self.tile_probe.parameters(),
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
