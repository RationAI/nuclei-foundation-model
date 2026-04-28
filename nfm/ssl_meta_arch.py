from typing import Any

import lejepa
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.ops import MLP

from nfm.configuration import Config
from nfm.modeling.transformer import NFM


# from nfm.modeling.concepts_loss import SpatialConceptLoss


class SSLMetaArch(LightningModule):
    def __init__(
        self, warmup_steps: int, lamb: float, n_concepts: int, **config: Any
    ) -> None:
        super().__init__()
        self.warmup_steps = warmup_steps
        self.lamb = lamb

        self.config = Config(**config)
        self.model = NFM(self.config)
        # self.scl = SpatialConceptLoss(self.config.dim, n_concepts)
        self.probe = nn.Linear(self.config.dim, 1)

        # self.proj = MLP(
        #     self.config.dim,
        #     hidden_channels=[2048, 2048, 256],
        #     norm_layer=nn.BatchNorm1d,
        # )
        # self.batch_norm = nn.BatchNorm1d(256, affine=False)

        # univariate_test = lejepa.univariate.EppsPulley(n_points=17)
        # self.sigreg_loss = lejepa.multivariate.SlicingUnivariateTest(
        #     univariate_test=univariate_test, num_slices=1024
        # )

    def forward(
        self, x: Tensor, pos: Tensor, block_mask: BlockMask
    ) -> tuple[Tensor, Tensor]:
        return self.model(x, pos, block_mask)

    def forward_unlabeled(self, batch: dict[str, Any]) -> Tensor:
        all_embed = self(batch["efds"], batch["pos"], batch["block_mask"])

        crops = torch.split(all_embed, batch["seq_lens"], dim=0)
        all_proj = self.batch_norm(
            self.proj(torch.stack([c.mean(dim=0) for c in crops]))
        )
        all_proj = rearrange(all_proj, "(b n) d -> b n d", n=8)
        batch_size = all_proj.shape[0]

        centers = all_proj[:, :2].mean(dim=1, keepdim=True)
        inv_loss = (all_proj - centers).square().mean()
        sigreg_loss = self.sigreg_loss(all_proj)
        lejepa_loss = sigreg_loss * self.lamb + inv_loss * (1 - self.lamb)

        self.log(
            "train/sigreg_loss",
            sigreg_loss,
            rank_zero_only=True,
            batch_size=batch_size,
            prog_bar=True,
        )
        self.log("train/inv_loss", inv_loss, rank_zero_only=True, batch_size=batch_size)
        self.log(
            "train/total_loss",
            lejepa_loss,
            rank_zero_only=True,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        return lejepa_loss

    def forward_labeled(self, batch: dict[str, Any]) -> Tensor:
        # with torch.no_grad():
        embed = self(batch["efds"], batch["pos"], batch["block_mask"])

        probe_labels = self.probe(embed)

        probe_loss = F.binary_cross_entropy_with_logits(probe_labels, batch["labels"])
        self.log(
            "train/probe_loss",
            probe_loss,
            rank_zero_only=True,
            prog_bar=True,
            on_epoch=True,
            batch_size=len(batch["seq_lens"]),
        )
        return probe_loss

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        # unsupervised_loss = self.forward_unlabeled(batch["unlabeled"])
        supervised_loss = self.forward_labeled(batch)
        return supervised_loss

    def validation_step(self, batch: dict[str, Any]) -> None:
        embed = self(batch["efds"], batch["pos"], batch["block_mask"])

        probe_labels = self.probe(embed)

        probe_loss = F.binary_cross_entropy_with_logits(probe_labels, batch["labels"])
        self.log(
            "validation/probe_loss",
            probe_loss,
            rank_zero_only=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=len(batch["seq_lens"]),
        )

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
