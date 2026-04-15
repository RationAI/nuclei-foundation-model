from typing import Any

import lejepa
import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from nfm.configuration import Config
from nfm.modeling.concepts_loss import SpatialConceptLoss
from nfm.modeling.transformer import NFM


class SSLMetaArch(LightningModule):
    def __init__(
        self, warmup_steps: int, lamb: float, n_concepts: int, **config: Any
    ) -> None:
        super().__init__()
        self.warmup_steps = warmup_steps
        self.lamb = lamb

        self.config = Config(**config)
        self.model = NFM(self.config)
        self.scl = SpatialConceptLoss(self.config.dim, n_concepts)
        self.probe = nn.Linear(self.config.dim, 1)

        univariate_test = lejepa.univariate.EppsPulley(n_points=17)
        self.sigreg_loss = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test, num_slices=1024
        )

    def forward(
        self, x: Tensor, pos: Tensor, block_mask: BlockMask
    ) -> tuple[Tensor, Tensor]:
        return self.model(x, pos, block_mask)

    def forward_unlabeled(self, batch: dict[str, Any]) -> Tensor:
        batch_size = len(batch["g_seq_lens"])
        g_embed = self(batch["efds"], batch["pos"], batch["global_block_mask"])
        l_embed = self(batch["efds"], batch["pos"], batch["local_block_mask"])

        concepts = self.scl(g_embed, batch["knn_indices"])

        inv_loss = F.mse_loss(l_embed, g_embed)
        sigreg_loss = (self.sigreg_loss(g_embed) + self.sigreg_loss(l_embed)) / 2
        lejepa_loss = sigreg_loss * self.lamb + inv_loss * (1 - self.lamb)

        avg_norm = torch.linalg.norm(g_embed, dim=-1).mean()
        self.log("train/avg_norm", avg_norm, rank_zero_only=True, batch_size=batch_size)
        self.log(
            "train/sigreg_loss", sigreg_loss, rank_zero_only=True, batch_size=batch_size
        )
        self.log("train/inv_loss", inv_loss, rank_zero_only=True, batch_size=batch_size)
        self.log(
            "train/lejepa_loss",
            lejepa_loss,
            rank_zero_only=True,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log_dict(concepts, rank_zero_only=True, prog_bar=True)

        return lejepa_loss + concepts["spatial_loss"] + concepts["sae_loss"]

    def forward_labeled(self, batch: dict[str, Any]) -> Tensor:
        with torch.no_grad():
            embed = self(batch["efds"], batch["pos"], batch["block_mask"])

        probe_labels = self.probe(embed)

        probe_loss = F.binary_cross_entropy_with_logits(
            probe_labels, batch["labeled"]["labels"]
        )
        self.log(
            "train/probe_loss",
            probe_loss,
            rank_zero_only=True,
            prog_bar=True,
            on_epoch=True,
            batch_size=len(batch["labeled"]["seq_lens"]),
        )
        return probe_loss

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        unsupervised_loss = self.forward_unlabeled(batch["unlabeled"])
        supervised_loss = self.forward_labeled(batch["labeled"])

        return unsupervised_loss + supervised_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        no_decay_params = [
            w for n, w in self.named_parameters() if w.ndim == 1 or ".rope." in n
        ]
        decay_params = [p for p in self.parameters() if p not in set(no_decay_params)]
        params = [
            {"params": decay_params},
            {"params": no_decay_params, "weight_decay": 0},
        ]

        optimizer = torch.optim.AdamW(params, lr=0.0005, weight_decay=0.05)

        s1 = LinearLR(optimizer, start_factor=0.01, total_iters=self.warmup_steps)
        s2 = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches - self.warmup_steps,
            eta_min=1e-6,
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[s1, s2], milestones=[self.warmup_steps]
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]
