"""Spatial-geometry pretraining with held-out spatial losses and random controls."""

import copy
import math

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)

from nfm.data.datasets.regional_dataset import HISTOGRAM_SIZES
from nfm.data.regional import regional_labeled_collate
from nfm.modeling.regional import RegionalEncoder


def poisson_deviance(rate, target):
    rate = rate.float().clamp_min(1e-6)
    target = target.float()
    return (
        rate - target + target * (target.clamp_min(1e-6).log() - rate.log())
    ).clamp_min(0)


def multiscale_deviance(rate, target, mask):
    terms = []
    while True:
        terms.append(poisson_deviance(rate[mask], target[mask]).mean())
        if rate.shape[-1] == 1:
            return torch.stack(terms).mean()
        shape = rate.shape
        rate = F.avg_pool2d(rate.reshape(-1, 1, shape[-2], shape[-1]), 2) * 4
        target = F.avg_pool2d(target.reshape(-1, 1, shape[-2], shape[-1]), 2) * 4
        rate = rate.reshape(*shape[:2], shape[-2] // 2, shape[-1] // 2)
        target = target.reshape_as(rate)


def histogram_kl(logits, counts, selected):
    terms, offset = [], 0
    for size in HISTOGRAM_SIZES:
        actual = counts[..., offset : offset + size][selected].float()
        population = actual.sum(-1)
        occupied = population > 0
        probabilities = actual / population.clamp_min(1)[:, None]
        log_prediction = F.log_softmax(
            logits[..., offset : offset + size][selected].float(), -1
        )
        values = (
            probabilities * (probabilities.clamp_min(1e-8).log() - log_prediction)
        ).sum(-1)
        terms.append(
            values[occupied].mean() if occupied.any() else log_prediction.sum() * 0
        )
        offset += size
    return torch.stack(terms).mean()


class RegionalMetaArch(LightningModule):
    labeled_collate_fn = staticmethod(regional_labeled_collate)

    def __init__(
        self,
        dim=256,
        hidden_dim=1024,
        num_heads=8,
        num_layers=8,
        efd_order=16,
        intrinsic_dim=128,
        grid_size=8,
        tile_size_um=64,
        fine_bins=8,
        inference_window_batch=32,
        lr=3e-4,
        probe_lr=1e-3,
        weight_decay=0.05,
        warmup_steps=100,
        normalization_batches=16,
        noise_std=0.05,
        occupancy_weight=1.0,
        morphology_weight=0.25,
        intrinsic_weight=1.0,
    ):
        super().__init__()
        if normalization_batches < 1 or fine_bins < 1 or fine_bins & (fine_bins - 1):
            raise ValueError(
                "Calibration is required and fine_bins must be a power of two"
            )
        if min(noise_std, occupancy_weight, morphology_weight, intrinsic_weight) < 0:
            raise ValueError("Loss and corruption weights must be nonnegative")
        self.save_hyperparameters()
        self.model = RegionalEncoder(
            efd_order,
            intrinsic_dim,
            dim,
            hidden_dim,
            num_heads,
            num_layers,
            grid_size,
            tile_size_um,
            fine_bins,
            inference_window_batch,
        )
        self.reference = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.intrinsic_decoder = nn.Linear(intrinsic_dim, 4 * efd_order)
        self.density_decoder = nn.Linear(dim, fine_bins**2)
        self.occupancy_decoder = nn.Linear(dim, fine_bins**2)
        self.morphology_decoder = nn.Linear(dim, sum(HISTOGRAM_SIZES))
        self.register_buffer("calibration_batches", torch.zeros((), dtype=torch.long))
        self.register_buffer("calibration_counts", torch.zeros((), dtype=torch.float64))
        self.register_buffer("calibration_bins", torch.zeros((), dtype=torch.float64))
        self.register_buffer(
            "calibration_occupied", torch.zeros((), dtype=torch.float64)
        )
        self.register_buffer(
            "calibration_histogram",
            torch.zeros(sum(HISTOGRAM_SIZES), dtype=torch.float64),
        )
        centers = self.model.centers
        weights = torch.exp(
            -torch.cdist(centers, centers).square() / (2 * (2 * tile_size_um) ** 2)
        )
        self.register_buffer("interpolation_weights", weights)
        dimensions = {
            "raw": 4 * efd_order,
            "intrinsic": intrinsic_dim,
            "contextual": intrinsic_dim + dim,
            "local_summary": self.model.summary_dim,
            "random_intrinsic": intrinsic_dim,
            "random_contextual": intrinsic_dim + dim,
        }
        self.probes = nn.ModuleDict(
            {name: nn.Linear(size, 1) for name, size in dimensions.items()}
        )
        self.val_metrics = nn.ModuleDict(
            {
                name: MetricCollection(
                    {
                        "AUPRC": BinaryAveragePrecision(),
                        "AUROC": BinaryAUROC(),
                        "precision": BinaryPrecision(),
                        "recall": BinaryRecall(),
                    },
                    prefix=f"validation/{name}_",
                )
                for name in dimensions
            }
        )

    def forward(self, batch):
        return self.model(batch)

    def on_fit_start(self):
        if self.trainer.world_size != 1:
            raise RuntimeError("The regional pilot currently supports one device")

    @torch.no_grad()
    def calibrate(self, batch):
        self.model.standardizer.observe(batch["inputs"]["efds"])
        visible = ~batch["inputs"]["mask"]
        counts = batch["targets"]["counts"][visible]
        self.calibration_counts.add_(counts.double().sum())
        self.calibration_bins.add_(counts.numel())
        self.calibration_occupied.add_((counts > 0).sum())
        self.calibration_histogram.add_(
            batch["targets"]["morphology"][visible].double().sum(0)
        )
        self.calibration_batches.add_(1)
        if self.calibration_batches >= self.hparams.normalization_batches:
            self.model.standardizer.freeze()
            self.reference.standardizer.load_state_dict(
                self.model.standardizer.state_dict()
            )
            rate = (self.calibration_counts / self.calibration_bins).clamp_min(1e-6)
            self.density_decoder.weight.zero_()
            self.density_decoder.bias.fill_(
                float(rate + torch.log(-torch.expm1(-rate)))
            )
            probability = (self.calibration_occupied / self.calibration_bins).clamp(
                1e-5, 1 - 1e-5
            )
            self.occupancy_decoder.weight.zero_()
            self.occupancy_decoder.bias.fill_(float(torch.logit(probability)))
            self.morphology_decoder.weight.zero_()
            self.morphology_decoder.bias.copy_(
                (self.calibration_histogram + 0.5).log().float()
            )

    def spatial_loss(self, batch, add_noise=True):
        inputs, target = batch["inputs"], batch["targets"]
        if not inputs["mask"].any():
            raise ValueError("Spatial reconstruction requires hidden tiles")
        if add_noise and self.hparams.noise_std:
            # Perturb visible shapes only, in the calibrated feature units.
            inputs = {
                **inputs,
                "efds": inputs["efds"]
                + torch.randn_like(inputs["efds"])
                * self.model.standardizer.scale
                * self.hparams.noise_std,
            }
        encoded = self.model.encode_windows(inputs)
        regions, mask = encoded["regions"], inputs["mask"]
        shape = (*mask.shape, self.hparams.fine_bins, self.hparams.fine_bins)
        rate = F.softplus(self.density_decoder(regions).float()).reshape(shape) + 1e-6
        occupancy_logits = self.occupancy_decoder(regions).float().reshape(shape)
        morphology_logits = self.morphology_decoder(regions).float()
        clean = self.model.standardizer(batch["inputs"]["efds"]).detach()
        retention = F.smooth_l1_loss(
            self.intrinsic_decoder(encoded["intrinsic"]).float(), clean
        )
        geometry = multiscale_deviance(rate, target["counts"], mask)
        occupancy = F.binary_cross_entropy_with_logits(
            occupancy_logits[mask], (target["counts"][mask] > 0).float()
        )
        morphology = histogram_kl(morphology_logits, target["morphology"], mask)
        # Both baselines are computed exclusively from visible cells/tiles.
        visible = (~mask).float()
        populations = encoded["visible_counts"].sum((-2, -1))
        weights = self.interpolation_weights[None] * visible[:, None, :]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        local_population = weights @ populations[..., None]
        local_rate = (
            local_population[..., None] / self.hparams.fine_bins**2
        ).expand_as(rate)
        average = (
            populations.sum(1) / visible.sum(1).clamp_min(1) / self.hparams.fine_bins**2
        )
        average = average[:, None, None, None].expand_as(rate)
        baseline = multiscale_deviance(local_rate, target["counts"], mask)
        global_baseline = multiscale_deviance(average, target["counts"], mask)
        visible_morphology = (
            target["morphology"].float().masked_fill(mask[..., None], 0)
        )
        interpolation = weights @ visible_morphology
        # Histograms use separate softmaxes; log(count + prior) yields smoothed proportions.
        morph_baseline = histogram_kl(
            (interpolation + 0.5).log(), target["morphology"], mask
        )
        loss = (
            geometry
            + self.hparams.occupancy_weight * occupancy
            + self.hparams.morphology_weight * morphology
            + self.hparams.intrinsic_weight * retention
        )
        return loss, {
            "geometry_deviance": geometry,
            "geometry_interpolation_baseline": baseline,
            "geometry_visible_mean_baseline": global_baseline,
            "spatial_skill": 1 - geometry / baseline.clamp_min(1e-6),
            "occupancy_BCE": occupancy,
            "morphology_KL": morphology,
            "morphology_interpolation_baseline": morph_baseline,
            "intrinsic_reconstruction": retention,
            "masked_fraction": mask.float().mean(),
            "hidden_nuclei": target["counts"][mask].sum(),
        }

    def probe_features(self, batch):
        with torch.no_grad():
            learned = self.model(batch)
            initial = self.reference(batch)
        return {
            **learned,
            "random_intrinsic": initial["intrinsic"],
            "random_contextual": initial["contextual"],
        }

    def train_probes(self, batch):
        features = self.probe_features(batch)
        score = batch["score"]
        if not score.any():
            return sum(head.weight.sum() * 0 for head in self.probes.values())
        losses = []
        with torch.autocast(device_type=batch["efds"].device.type, enabled=False):
            for name, head in self.probes.items():
                loss = F.binary_cross_entropy_with_logits(
                    head(features[name][score].float()), batch["labels"][score].float()
                )
                self.log(f"train/{name}_probe_loss", loss, batch_size=int(score.sum()))
                losses.append(loss)
        return torch.stack(losses).mean()

    def training_step(self, batch):
        if not self.model.standardizer.ready:
            self.calibrate(batch["unlabeled"])
            self.log(
                "train/calibration_batches",
                self.calibration_batches.float(),
                batch_size=1,
            )
            return None
        loss, metrics = self.spatial_loss(batch["unlabeled"])
        for name, value in metrics.items():
            self.log(
                f"train/{name}", value, batch_size=len(batch["unlabeled"]["slide_ids"])
            )
        self.log(
            "train/ssl_loss",
            loss,
            batch_size=len(batch["unlabeled"]["slide_ids"]),
            prog_bar=True,
        )
        return loss + self.train_probes(batch["labeled"])

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if "targets" in batch:
            _, metrics = self.spatial_loss(batch, add_noise=False)
            for name, value in metrics.items():
                self.log(
                    f"ssl_validation/{name}",
                    value,
                    batch_size=len(batch["slide_ids"]),
                    add_dataloader_idx=False,
                )
            return
        features = self.probe_features(batch)
        score = batch["score"]
        if not score.any():
            return
        target = batch["labels"][score].float()
        self.log(
            "validation/prevalence",
            target.mean(),
            batch_size=int(score.sum()),
            add_dataloader_idx=False,
        )
        with torch.autocast(device_type=batch["efds"].device.type, enabled=False):
            for name, head in self.probes.items():
                logits = head(features[name][score].float())
                self.val_metrics[name].update(logits.sigmoid(), target.long())
                self.log_dict(
                    self.val_metrics[name],
                    on_epoch=True,
                    batch_size=int(score.sum()),
                    add_dataloader_idx=False,
                )
                self.log(
                    f"validation/{name}_positive_fraction",
                    (logits > 0).float().mean(),
                    batch_size=int(score.sum()),
                    add_dataloader_idx=False,
                )

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad or name.startswith("probes."):
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay, "weight_decay": 0},
                {
                    "params": list(self.probes.parameters()),
                    "lr": self.hparams.probe_lr,
                    "weight_decay": 0,
                },
            ],
            lr=self.hparams.lr,
        )
        total = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup = min(self.hparams.warmup_steps, max(total - 1, 0))

        def schedule(step):
            if step < warmup:
                return 0.01 + 0.99 * step / max(warmup, 1)
            return 0.5 * (
                1 + math.cos(math.pi * min((step - warmup) / max(total - warmup, 1), 1))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, [schedule, schedule, lambda _: 1.0]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        if gradient_clip_val:
            representation = [
                p
                for name, p in self.named_parameters()
                if p.requires_grad and not name.startswith("probes.")
            ]
            clip = (
                torch.nn.utils.clip_grad_value_
                if gradient_clip_algorithm == "value"
                else torch.nn.utils.clip_grad_norm_
            )
            clip(representation, gradient_clip_val)
            clip(list(self.probes.parameters()), gradient_clip_val)
