"""Morphology retention and masked context prediction with detached probes."""

import math

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
from nfm.modeling.morphology import FrozenStandardizer, MorphologyEncoder


class MorphologyMetaArch(LightningModule):
    def __init__(
        self,
        dim: int = 256,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 4,
        efd_order: int = 16,
        intrinsic_dim: int = 128,
        rope_theta: float = 10000,
        lr: float = 3e-4,
        probe_lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_steps: int = 100,
        normalization_batches: int = 16,
        mask_ratio: float = 0.3,
        noise_std: float = 0.05,
        context_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if normalization_batches < 1 or not 0 < mask_ratio < 1:
            raise ValueError("Require calibration batches and a mask ratio in (0, 1)")
        if noise_std < 0 or context_weight < 0 or efd_order < 1:
            raise ValueError("Invalid corruption, loss weight, or EFD order")
        self.save_hyperparameters()
        architecture = Config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            efd_order=efd_order,
            rope_theta=rope_theta,
        )
        target_dim = 4 + min(8, efd_order)
        self.model = MorphologyEncoder(architecture, intrinsic_dim)
        # A linear decoder makes retained EFD information linearly recoverable.
        self.intrinsic_decoder = nn.Linear(intrinsic_dim, 4 * efd_order)
        self.context_decoder = nn.Linear(dim, target_dim)
        self.target_standardizer = FrozenStandardizer(target_dim, 0.02)
        self.register_buffer("calibration_batches", torch.zeros((), dtype=torch.long))
        self.probes = nn.ModuleDict(
            {
                "raw": nn.Linear(4 * efd_order, 1),
                "intrinsic": nn.Linear(intrinsic_dim, 1),
                "contextual": nn.Linear(intrinsic_dim + dim, 1),
            }
        )
        self.val_metrics = nn.ModuleDict(
            {
                name: MetricCollection(
                    {
                        "AUROC": BinaryAUROC(),
                        "AUPRC": BinaryAveragePrecision(),
                        "precision": BinaryPrecision(),
                        "recall": BinaryRecall(),
                    },
                    prefix=f"validation/{name}_",
                )
                for name in self.probes
            }
        )

    def forward(self, batch) -> dict[str, Tensor]:
        """Clean raw, intrinsic, and fused intrinsic/contextual nucleus features."""
        return self.model(batch)

    def on_fit_start(self) -> None:
        if self.trainer.world_size != 1:
            raise RuntimeError("This calibration pilot supports one device per run")

    @torch.no_grad()
    def calibrate(self, batch) -> None:
        self.model.standardizer.observe(batch["efds"])
        self.target_standardizer.observe(batch["morphology"])
        self.calibration_batches.add_(1)
        if self.calibration_batches >= self.hparams.normalization_batches:
            self.model.standardizer.freeze()
            self.target_standardizer.freeze()

    def sample_mask(self, region_ids: Tensor) -> Tensor:
        mask = torch.zeros_like(region_ids, dtype=torch.bool)
        for region in region_ids.unique():
            indices = (region_ids == region).nonzero().flatten()
            if len(indices) < 2:
                continue
            count = min(
                len(indices) - 1, max(1, round(len(indices) * self.hparams.mask_ratio))
            )
            selected = torch.randperm(len(indices), device=indices.device)[:count]
            mask[indices[selected]] = True
        return mask

    def reconstruction(
        self, batch, mask: Tensor | None = None, add_noise: bool = True
    ) -> tuple[Tensor, dict[str, Tensor]]:
        clean = self.model.standardizer(batch["efds"])
        targets = self.target_standardizer(batch["morphology"]).detach()
        noisy = (
            clean + torch.randn_like(clean) * self.hparams.noise_std
            if add_noise
            else clean
        )
        intrinsic = self.model.shape_encoder(noisy)
        prediction = self.intrinsic_decoder(intrinsic).float()
        intrinsic_loss = F.smooth_l1_loss(prediction, clean.detach())
        if mask is None:
            mask = self.sample_mask(batch["region_ids"])
        context = self.model.contextualize(
            intrinsic, batch["pos"], batch["block_mask"], mask
        )
        context_prediction = self.context_decoder(context).float()
        if mask.any():
            context_loss = F.smooth_l1_loss(context_prediction[mask], targets[mask])
            # An auditable context baseline: use only visible cells in this crop.
            regions = batch["region_ids"]
            count = len(batch["slide_ids"])
            sums = targets.new_zeros(count, targets.shape[1])
            counts = targets.new_zeros(count)
            sums.index_add_(0, regions[~mask], targets[~mask])
            counts.index_add_(
                0, regions[~mask], torch.ones_like(regions[~mask], dtype=targets.dtype)
            )
            baseline = (sums / counts.clamp_min(1)[:, None])[regions[mask]]
            context_baseline = F.smooth_l1_loss(baseline, targets[mask])
            context_mse = F.mse_loss(context_prediction[mask], targets[mask])
            baseline_mse = F.mse_loss(baseline, targets[mask])
        else:
            context_loss = context_prediction.sum() * 0
            context_baseline = context_mse = baseline_mse = context_loss.detach()
        loss = intrinsic_loss + self.hparams.context_weight * context_loss
        return loss, {
            "intrinsic_reconstruction": intrinsic_loss,
            "intrinsic_mean_baseline": F.smooth_l1_loss(
                torch.zeros_like(clean), clean.detach()
            ),
            "context_reconstruction": context_loss,
            "context_visible_mean_baseline": context_baseline,
            "context_mse": context_mse,
            "context_baseline_mse": baseline_mse,
            "masked_fraction": mask.float().mean(),
        }

    def train_probes(self, batch) -> Tensor:
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            features = self(batch)
        self.model.train(was_training)
        score = batch["score"]
        if not score.any():
            return sum(probe.weight.sum() * 0 for probe in self.probes.values())
        losses = []
        for name, probe in self.probes.items():
            logits = probe(features[name][score])
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"][score])
            self.log(f"train/{name}_probe_loss", loss, batch_size=int(score.sum()))
            losses.append(loss)
        self.log(
            "train/probe_prevalence",
            batch["labels"][score].mean(),
            batch_size=int(score.sum()),
        )
        return torch.stack(losses).mean()

    def training_step(self, batch):
        unlabeled = batch["unlabeled"]
        if not self.model.standardizer.ready:
            self.calibrate(unlabeled)
            self.log(
                "train/calibration_batches",
                self.calibration_batches.float(),
                batch_size=1,
            )
            # No gradients or parameter updates during input-only calibration.
            # Lightning still counts these batches in its global step/schedule.
            return None
        ssl_loss, components = self.reconstruction(unlabeled)
        size = len(unlabeled["slide_ids"])
        for name, value in components.items():
            self.log(f"train/{name}", value, batch_size=size)
        self.log("train/ssl_loss", ssl_loss, batch_size=size, prog_bar=True)
        self.log("train/tokens", float(len(unlabeled["efds"])), batch_size=size)
        return ssl_loss + self.train_probes(batch["labeled"])

    def validation_step(self, batch):
        features = self(batch)
        score = batch["score"]
        if not score.any():
            return
        targets = batch["labels"][score]
        self.log(
            "validation/prevalence",
            targets.mean(),
            batch_size=int(score.sum()),
            sync_dist=True,
        )
        for name, probe in self.probes.items():
            logits = probe(features[name][score])
            metrics = self.val_metrics[name]
            metrics.update(logits.sigmoid(), targets.long())
            self.log_dict(metrics, on_epoch=True, batch_size=int(score.sum()))
            self.log(
                f"validation/{name}_probe_loss",
                F.binary_cross_entropy_with_logits(logits, targets),
                batch_size=int(score.sum()),
                sync_dist=True,
            )
            self.log(
                f"validation/{name}_positive_fraction",
                (logits > 0).float().mean(),
                batch_size=int(score.sum()),
                sync_dist=True,
            )

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            if name.startswith("probes."):
                continue
            group = no_decay if parameter.ndim <= 1 or ".rope." in name else decay
            group.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0},
                {
                    "params": list(self.probes.parameters()),
                    "weight_decay": 0,
                    "lr": self.hparams.probe_lr,
                },
            ],
            lr=self.hparams.lr,
        )
        total = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup = min(self.hparams.warmup_steps, max(total - 1, 0))

        def encoder_schedule(step):
            if step < warmup:
                return 0.01 + 0.99 * step / max(warmup, 1)
            progress = min((step - warmup) / max(total - warmup, 1), 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, [encoder_schedule, encoder_schedule, lambda _: 1.0]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        if not gradient_clip_val:
            return
        # Probe labels cannot alter the encoder's clipping multiplier.
        clip = (
            torch.nn.utils.clip_grad_value_
            if gradient_clip_algorithm == "value"
            else torch.nn.utils.clip_grad_norm_
        )
        representation = [
            p for name, p in self.named_parameters() if not name.startswith("probes.")
        ]
        clip(representation, gradient_clip_val)
        clip(list(self.probes.parameters()), gradient_clip_val)
