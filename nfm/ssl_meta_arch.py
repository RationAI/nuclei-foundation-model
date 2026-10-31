"""LeJEPA on rendered nuclei tiles with a detached online Radboud probe."""

import math
from typing import Any

import timm
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchvision.ops import MLP

from nfm.modeling.sigreg import SIGReg


class SSLMetaArch(LightningModule):
    def __init__(
        self,
        backbone: str = "vit_small_patch8_224",
        image_size: int = 128,
        embed_dim: int = 512,
        projector_dim: int = 16,
        drop_path_rate: float = 0.1,
        lamb: float = 0.02,
        lr: float = 2e-3,
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
        self.backbone = timm.create_model(
            backbone,
            pretrained=False,
            in_chans=1,
            img_size=image_size,
            num_classes=embed_dim,
            drop_path_rate=drop_path_rate,
        )
        self.projector = MLP(
            embed_dim, [2048, 2048, projector_dim], norm_layer=nn.BatchNorm1d
        )
        self.sigreg = SIGReg()
        self.probe = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.val_metrics = MetricCollection(
            {"AUROC": BinaryAUROC(), "AUPRC": BinaryAveragePrecision()},
            prefix="validation/",
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.backbone((images.float() / 255 - 0.5) / 0.5)

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        views = batch["unlabeled"]  # [batch, views, height, width], uint8
        n, v = views.shape[:2]
        embeddings = self(views.flatten(0, 1).unsqueeze(1))
        proj = self.projector(embeddings).reshape(n, v, -1).transpose(0, 1)
        invariance = (proj.mean(0) - proj).square().mean()
        sigreg = self.sigreg(proj)
        lejepa = self.lamb * sigreg + (1 - self.lamb) * invariance

        # Labels must not reach the encoder, and the probe sees no drop path.
        self.backbone.eval()
        with torch.no_grad():
            features = self(batch["labeled"]["images"])
        self.backbone.train()
        logits = self.probe(features).squeeze(1)
        probe = F.binary_cross_entropy_with_logits(logits, batch["labeled"]["labels"])

        self.log_dict(
            {
                "train/lejepa_loss": lejepa,
                "train/invariance_loss": invariance,
                "train/sigreg_loss": sigreg,
                "train/probe_loss": probe,
                "train/embedding_std": embeddings.detach().std(0).mean(),
            },
            batch_size=n,
            prog_bar=True,
        )
        return lejepa + probe

    def validation_step(self, batch: dict[str, Tensor]) -> None:
        logits = self.probe(self(batch["images"])).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
        self.val_metrics.update(logits.sigmoid(), batch["labels"].long())
        self.log("validation/probe_loss", loss, batch_size=len(logits), sync_dist=True)
        self.log_dict(self.val_metrics, batch_size=len(logits))

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        *self.backbone.parameters(),
                        *self.projector.parameters(),
                    ],
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": self.probe.parameters(),
                    "lr": self.probe_lr,
                    "weight_decay": 1e-7,
                },
            ],
            fused=True,
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
