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
from nfm.modeling.transformer import NFM


class SSLMetaArch(LightningModule):
    def __init__(self, warmup_steps: int, lamb: float, **config: Any) -> None:
        super().__init__()
        self.warmup_steps = warmup_steps
        self.lamb = lamb

        self.config = Config(**config)
        self.model = NFM(self.config)
        self.linear_proj = nn.Linear(self.config.dim, 1)

        univariate_test = lejepa.univariate.EppsPulley(n_points=17)
        self.sigreg_loss = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test, num_slices=1024
        )

    def forward(
        self, x: Tensor, pos: Tensor, block_mask: BlockMask
    ) -> tuple[Tensor, Tensor]:
        block_mask.mask_mod = block_mask.mask_mod.to(x.device)
        return self.model(x, pos, block_mask)

    def forward_unlabeled(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        _, global_proj = self(batch["efds"], batch["pos"], batch["global_block_mask"])
        _, local_proj = self(batch["efds"], batch["pos"], batch["local_block_mask"])
        return global_proj, torch.cat([global_proj, local_proj], dim=1)

    def forward_labeled(self, batch: dict[str, Any]) -> Tensor:
        with torch.no_grad():
            embed, _ = self(batch["efds"], batch["pos"], batch["block_mask"])

        return self.linear_proj(embed)

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        g_emb, a_emb = self.forward_unlabeled(batch["unlabeled"])
        pred_labels = self.forward_labeled(batch["labeled"])

        centers = g_emb.mean(dim=1, keepdim=True)
        inv_loss = (a_emb - centers).square().mean()
        sigreg_loss = self.sigreg_loss(a_emb)
        lejepa_loss = sigreg_loss * self.lamb + inv_loss * (1 - self.lamb)

        probe_loss = F.cross_entropy(
            pred_labels[batch["labeled"]["seq_lens"]], batch["labeled"]["labels"]
        )

        avg_norm = torch.linalg.norm(a_emb, dim=-1).mean()
        self.log("train/avg_norm", avg_norm, rank_zero_only=True)

        self.log("train/sigreg_loss", sigreg_loss, rank_zero_only=True)
        self.log("train/inv_loss", inv_loss, rank_zero_only=True)
        self.log(
            "train/lejepa_loss",
            lejepa_loss,
            rank_zero_only=True,
            prog_bar=True,
            on_epoch=True,
        )
        self.log("train/probe_loss", probe_loss, rank_zero_only=True, prog_bar=True)

        return lejepa_loss + probe_loss

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
