from typing import Any

import lejepa
import torch
from einops import rearrange
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from nfm.configuration import Config
from nfm.modeling.transformer import NucleiGraphEncoder


class SSLMetaArch(LightningModule):
    def __init__(self, warmup_steps: int, lamb: float, **config: Any) -> None:
        super().__init__()
        self.warmup_steps = warmup_steps
        self.lamb = lamb
        self.config = Config(**config)

        self.model = NucleiGraphEncoder(self.config)

        univariate_test = lejepa.univariate.EppsPulley(n_points=17)
        self.loss_fn = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test, num_slices=1024
        )

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        pos, embed = batch["local_crops"]
        _, local_proj = self.model(
            src=embed.flatten(0, 1),
            src_pos=pos.flatten(0, 1),
            tgt_pos=batch["local_spatial_registers"].flatten(0, 1),
        )
        local_proj = rearrange(local_proj, "(b n) d -> b n d", b=len(pos))

        pos, embed = batch["global_crops"]
        _, global_proj = self.model(
            src=embed.flatten(0, 1),
            src_pos=pos.flatten(0, 1),
            tgt_pos=batch["global_spatial_registers"].flatten(0, 1),
        )
        global_proj = rearrange(global_proj, "(b n) d -> b n d", b=len(pos))

        return global_proj, torch.cat([global_proj, local_proj], dim=1)

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        g_emb, a_emb = self(batch)

        centers = g_emb.mean(dim=1, keepdim=True)
        inv_loss = (a_emb - centers).square().mean()
        sigreg_loss = self.loss_fn(a_emb)
        lejepa_loss = sigreg_loss * self.lamb + inv_loss * (1 - self.lamb)

        self.log("train/sigreg_loss", sigreg_loss, rank_zero_only=True)
        self.log("train/inv_loss", inv_loss, rank_zero_only=True)
        self.log("train/lejepa_loss", lejepa_loss, rank_zero_only=True, prog_bar=True)

        return lejepa_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        no_decay_params = [
            w for n, w in self.model.named_parameters() if w.ndim == 1 or ".rope." in n
        ]
        decay_params = list(set(self.model.parameters()).difference(no_decay_params))
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

        return [optimizer], [scheduler]
