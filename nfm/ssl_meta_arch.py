"""LeJEPA on cells clustered from the nucleus tokens, with detached probes."""

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

from nfm.modeling.clustering import Clustering, match_tracks, segment_mean
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
        clustering: dict[str, Any] | None = None,
        cluster_mix: float = 0.5,
        cluster_schedule: tuple[float, float] = (0.1, 0.3),
        min_overlap: float = 0.3,
        min_share: float = 0.5,
    ) -> None:
        """Initializes the model.

        Args:
            backbone: Arguments of the nucleus transformer.
            pooling: Arguments of the pooling of the cells.
            projector_dim: Output dimension of the LeJEPA projector.
            lamb: Weight of SIGReg against the invariance.
            lr: Peak learning rate of the encoder.
            probe_lr: Learning rate of the probes.
            weight_decay: Weight decay of the encoder.
            warmup_steps: Linear warmup of the learning rate.
            clustering: Arguments of the clustering of the cells.
            cluster_mix: Share of the tokens against the positions in the
                distances of the clustering of the cells, from 0 to 1.
            cluster_schedule: Shares of the training at which the mix starts
                to rise linearly from 0, while the tokens are still near random,
                and reaches `cluster_mix`.
            min_overlap: Least Jaccard index of the nuclei of the global cells
                of a track.
            min_share: Least share of the nuclei of a local cell that lie in the
                global cells of its track.
        """
        super().__init__()
        self.lamb = lamb
        self.lr = lr
        self.probe_lr = probe_lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.cluster_mix = cluster_mix
        self.cluster_schedule = tuple(cluster_schedule)
        self.min_overlap = min_overlap
        self.min_share = min_share
        self.backbone = NucleiTransformer(**backbone)
        dim = self.backbone.norm.normalized_shape[0]
        self.clustering = Clustering(dim, **(clustering or {}))
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

    def current_cluster_mix(self) -> float:
        """The share of the tokens in the clustering at this step.

        Without a trainer, as when a checkpoint is evaluated, it is the final one.
        """
        if self._trainer is None:
            return self.cluster_mix
        start, end = self.cluster_schedule
        total = max(int(self.trainer.estimated_stepping_batches), 1)
        progress = self.global_step / total
        if end <= start:
            return self.cluster_mix if progress >= start else 0.0
        return self.cluster_mix * min(max((progress - start) / (end - start), 0), 1)

    def encode(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Nucleus tokens, their cells and the cell embeddings.

        Returns:
            nuclei: The nucleus tokens, [nuclei, dim].
            cells: The cell of each nucleus, [nuclei], and `count` cells.
            regions: The cell embeddings, [count, dim].
            moved: The share of nuclei that left their initial cell.
        """
        nuclei = self.backbone(
            batch["features"], batch["positions"], batch["lengths"], batch["neighbors"]
        )
        cells, count, moved = self.clustering(nuclei, batch, self.current_cluster_mix())
        regions = self.pooling(nuclei, cells, count)
        return {
            "nuclei": nuclei,
            "cells": cells,
            "count": count,
            "regions": regions,
            "moved": moved,
        }

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        """Returns the cell embeddings, [cells, dim], and nucleus tokens."""
        encoded = self.encode(batch)
        return encoded["regions"], encoded["nuclei"]

    @staticmethod
    def cell_labels(encoded: dict[str, Any], labels: Tensor) -> Tensor:
        """A cell is positive if at least half of its nuclei are."""
        share = segment_mean(labels[:, None], encoded["cells"], encoded["count"])
        return (share[:, 0] >= 0.5).float()

    def probe_logits(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        """Region and nucleus logits, and the labels of the regions."""
        encoded = self.encode(batch)
        return (
            self.region_probe(encoded["regions"]).squeeze(1),
            self.nucleus_probe(encoded["nuclei"]).squeeze(1),
            self.cell_labels(encoded, batch["labels"]),
        )

    def encoder_eval(self, mode: bool = True) -> None:
        """Sets the encoder and the pooling, which belongs to it, to eval mode."""
        self.backbone.train(not mode)
        self.pooling.train(not mode)

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        views = batch["unlabeled"]
        encoded = self.encode(views)
        regions, count = encoded["regions"], encoded["count"]
        proj = self.projector(regions)
        # Cells that hold the same nuclei in different views form a track, and
        # each cell of a track predicts the mean of its cells in the global views.
        if views["global_views"] != 2:
            raise ValueError("the tracks start from exactly 2 global views")
        tracks, track_count = match_tracks(
            encoded["cells"],
            count,
            views["keys"],
            views["slots"],
            self.min_overlap,
            self.min_share,
        )
        slots = torch.zeros(count, dtype=torch.long, device=self.device)
        slots[encoded["cells"]] = views["slots"]
        tracked = tracks >= 0
        track, z = tracks[tracked], proj[tracked]
        is_global = slots[tracked] < views["global_views"]
        centers = segment_mean(z[is_global], track[is_global], track_count)
        invariance = (centers[track] - z).square().mean()
        # Each view slot is a sample of the embedding distribution.
        with torch.autocast(self.device.type, enabled=False):
            sigreg = torch.stack(
                [
                    self.sigreg(proj[slots == slot].float())
                    for slot in range(views["views_per_sample"])
                ]
            ).mean()
        lejepa = self.lamb * sigreg + (1 - self.lamb) * invariance

        # Labels must not reach the encoder, nor its batch norm statistics.
        labeled = batch["labeled"]
        self.encoder_eval()
        with torch.no_grad():
            probed = self.encode(labeled)
        self.encoder_eval(False)
        region_probe = F.binary_cross_entropy_with_logits(
            self.region_probe(probed["regions"]).squeeze(1),
            self.cell_labels(probed, labeled["labels"]),
        )
        nucleus_probe = F.binary_cross_entropy_with_logits(
            self.nucleus_probe(probed["nuclei"]).squeeze(1), labeled["labels"]
        )

        self.log_dict(
            {
                "train/lejepa_loss": lejepa,
                "train/invariance_loss": invariance,
                "train/sigreg_loss": sigreg,
                "train/tokens": float(len(views["features"])),
                "train/cells": float(count),
                "train/tracks": float(track_count),
                "train/tracked_cells": float(len(track)),
                "train/cluster_mix": self.current_cluster_mix(),
                "train/cluster_moved": encoded["moved"],
                "train/region_probe_loss": region_probe,
                "train/nucleus_probe_loss": nucleus_probe,
                **self.embedding_statistics(regions[tracked].detach().float()),
            },
            batch_size=views["batch_size"],
            prog_bar=True,
        )
        return lejepa + region_probe + nucleus_probe

    @staticmethod
    def embedding_statistics(embeddings: Tensor) -> dict[str, Tensor]:
        """Spread of the tracked cell embeddings, comparable across encoders.

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
        region_logits, nucleus_logits, region_labels = self.probe_logits(batch)
        for name, logits, labels, metrics in (
            ("region", region_logits, region_labels, self.region_metrics),
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
