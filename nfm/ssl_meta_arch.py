"""Nucleus correspondence SSL and a continuously trained, detached linear probe."""

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

from nfm.configuration import Config
from nfm.modeling.objectives import NucleusObjective
from nfm.modeling.transformer import NFM


class SSLMetaArch(LightningModule):
    def __init__(
        self,
        warmup_steps: int = 1000,
        lr: float = 0.0003,
        regularizer: str = "sigreg",
        regularizer_weight: float | None = None,
        slices: int = 256,
        projector_dim: int = 128,
        weight_decay: float = 0.05,
        **config: Any,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.config = Config(**config)
        self.model = NFM(self.config)
        self.proj = nn.Sequential(
            nn.Linear(self.config.dim, 512), nn.GELU(), nn.Linear(512, projector_dim)
        )
        self.objective = NucleusObjective(regularizer, regularizer_weight, slices)
        self.probe = nn.Linear(self.config.dim, 1)
        self.val_metrics = MetricCollection(
            {
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "AUROC": BinaryAUROC(),
                "AUPRC": BinaryAveragePrecision(),
            },
            prefix="validation/",
        )

    def forward(self, batch) -> Tensor:
        return self.model(batch["efds"], batch["pos"], batch["block_mask"])

    def forward_unlabeled(self, batch: dict[str, Any]) -> Tensor:
        embeddings = self(batch)
        matched = embeddings[batch["pair_indices"]]  # [view, total matches, dim]
        counts = batch["pair_counts"]
        region_ids = torch.repeat_interleave(
            torch.arange(len(counts), device=counts.device), counts
        )
        means = matched.float().new_zeros(2, len(counts), matched.shape[-1])
        means.index_add_(1, region_ids, matched.float())
        means = means / counts.clamp_min(1)[None, :, None]
        nonempty = counts > 0

        projected = self.proj(matched)
        loss, components = self.objective(projected)
        size = len(batch["slide_ids"])
        for name, value in components.items():
            self.log(f"train/{name}", value, batch_size=size)
        self.log("train/ssl_loss", loss, batch_size=size, prog_bar=True)
        with torch.no_grad():
            self.log(
                "train/projector_std",
                projected.float().std(1, unbiased=False).mean()
                if projected.shape[1]
                else projected.new_zeros(()),
                batch_size=size,
            )
            residual = (
                embeddings[batch["pair_indices"]].float() - means[:, region_ids].float()
            )
            self.log(
                "train/within_region_std",
                residual.square().mean().sqrt()
                if residual.numel()
                else residual.new_zeros(()),
                batch_size=size,
            )
            occupied = means[:, nonempty].float()
            self.log(
                "train/between_region_std",
                occupied.var(1, unbiased=False).mean().sqrt()
                if occupied.shape[1]
                else occupied.new_zeros(()),
                batch_size=size,
            )
            self.log("train/matched_nuclei", counts.sum().float(), batch_size=size)
            self.log("train/matches_per_region", counts.float().mean(), batch_size=size)
            self.log(
                "train/empty_intersections",
                (counts == 0).sum().float(),
                batch_size=size,
            )
            self.log(
                "train/unique_slides",
                float(len(set(batch["slide_ids"]))),
                batch_size=size,
            )
            self.log("train/tokens", float(len(batch["efds"])), batch_size=size)
        return loss

    def forward_labeled(self, batch: dict[str, Any]) -> Tensor:
        # Probe labels must not influence encoder weights OR BN running stats.
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            embeddings = self(batch)
        self.model.train(was_training)

        score = batch["score"]
        logits = self.probe(embeddings[score])
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"][score])
        self.log(
            "train/probe_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=int(score.sum()),
        )
        return loss

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        ssl = self.forward_unlabeled(batch["unlabeled"])
        probe = self.forward_labeled(batch["labeled"])
        loss = ssl + probe
        self.log(
            "train/total_loss", loss, batch_size=len(batch["unlabeled"]["slide_ids"])
        )
        return loss

    def validation_step(self, batch: dict[str, Any]) -> None:
        embeddings = self(batch)
        logits = self.probe(embeddings)
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
        self.val_metrics.update(logits.sigmoid(), batch["labels"].long())
        self.log(
            "validation/probe_loss", loss, on_epoch=True, sync_dist=True, batch_size=1
        )
        self.log_dict(self.val_metrics, on_epoch=True, batch_size=1)

    def configure_optimizers(self):
        # Keep module registration order so checkpoint state maps to the same
        # parameters when a new model instance restores the optimizer.
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            group = no_decay if parameter.ndim <= 1 or ".rope." in name else decay
            group.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay},
                {"params": no_decay, "weight_decay": 0},
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # Native Python floats keep optimizer checkpoints weights_only-loadable.
        total = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup = min(self.warmup_steps, max(total - 1, 0))

        def schedule(step):
            if step < warmup:
                return 0.01 + 0.99 * step / max(warmup, 1)
            progress = min((step - warmup) / max(total - warmup, 1), 1.0)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
