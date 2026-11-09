"""LeJEPA on nuclei tokens with a masked nucleus objective and detached probes."""

import math
from typing import Any

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchvision.ops import MLP

from nfm.modeling.sigreg import SIGReg
from nfm.modeling.transformer import NucleiTransformer, cls_indices


def _metrics(prefix: str) -> MetricCollection:
    return MetricCollection(
        {"AUROC": BinaryAUROC(), "AUPRC": BinaryAveragePrecision()}, prefix=prefix
    )


class SSLMetaArch(LightningModule):
    def __init__(
        self,
        backbone: dict[str, Any],
        projector_dim: int = 16,
        lamb: float = 0.02,
        ibot_weight: float = 1.0,
        sigreg_tokens: int = 4096,
        lr: float = 5e-4,
        probe_lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lamb = lamb
        self.ibot_weight = ibot_weight
        self.sigreg_tokens = sigreg_tokens
        self.lr = lr
        self.probe_lr = probe_lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.backbone = NucleiTransformer(**backbone)
        dim = self.backbone.norm.normalized_shape[0]
        self.projector = MLP(
            dim, [2048, 2048, projector_dim], norm_layer=nn.BatchNorm1d
        )
        self.token_projector = MLP(
            dim, [2048, 2048, projector_dim], norm_layer=nn.BatchNorm1d
        )
        self.sigreg = SIGReg()
        self.tile_probe = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.nucleus_probe = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.tile_metrics = _metrics("validation/tile/")
        self.nucleus_metrics = _metrics("validation/nucleus/")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Returns the class tokens, [views, dim], and nucleus tokens, [nuclei, dim]."""
        tokens = self.backbone(
            batch["features"], batch["positions"], batch["lengths"], batch.get("masked")
        )
        cls = cls_indices(batch["lengths"])
        nuclei = torch.ones(len(tokens), dtype=torch.bool, device=tokens.device)
        nuclei[cls] = False
        return tokens[cls], tokens[nuclei]

    def probe_logits(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        cls, nuclei = self(batch)
        return self.tile_probe(cls).squeeze(1), self.nucleus_probe(nuclei).squeeze(1)

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        views = batch["unlabeled"]
        n, g = views["batch_size"], views["global_views"]
        v = g + views["local_views"]
        tokens = self.backbone(
            views["features"], views["positions"], views["lengths"], views["masked"]
        )
        cls = tokens[cls_indices(views["lengths"])]
        proj = self.projector(cls[: v * n]).reshape(v, n, -1)  # [views, batch, dim]
        # All views predict the center of the global views.
        centers = proj[:g].mean(0)
        invariance = (centers - proj).square().mean()
        sigreg = self.sigreg(proj)
        lejepa = self.lamb * sigreg + (1 - self.lamb) * invariance

        # iBOT-style: masked nuclei predict the same nucleus in the other, unmasked
        # global view. The views are rotated and flipped differently, so the
        # position cannot be the answer. SIGReg on the token projections prevents
        # collapse without a teacher.
        predictions, targets = views["pairs"]
        pairs = torch.cat([tokens[targets], tokens[predictions]])
        target, prediction = self.token_projector(pairs).chunk(2)
        token_prediction = (target - prediction).square().mean()
        subset = torch.randperm(len(pairs), device=self.device)[: self.sigreg_tokens]
        token_sigreg = self.sigreg(torch.cat([target, prediction])[subset][None])
        ibot = self.lamb * token_sigreg + (1 - self.lamb) * token_prediction

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
                "train/ibot_loss": ibot,
                "train/token_prediction_loss": token_prediction,
                "train/token_sigreg_loss": token_sigreg,
                "train/predicted_nuclei": float(predictions.numel()) / n,
                "train/tokens": float(len(tokens)),
                "train/tile_probe_loss": tile_probe,
                "train/nucleus_probe_loss": nucleus_probe,
                "train/embedding_std": cls[: g * n].detach().float().std(0).mean(),
            },
            batch_size=n,
            prog_bar=True,
        )
        return lejepa + self.ibot_weight * ibot + tile_probe + nucleus_probe

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
            *self.token_projector.named_parameters(),
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
