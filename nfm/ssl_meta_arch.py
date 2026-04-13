from typing import Any

import lejepa
import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.ops import MLP

from nfm.configuration import Config
from nfm.modeling.transformer import NFM


class SSLMetaArch(LightningModule):
    def __init__(self, warmup_steps: int, lamb: float, **config: Any) -> None:
        super().__init__()
        self.warmup_steps = warmup_steps
        self.lamb = lamb

        self.config = Config(**config)
        self.model = NFM(self.config)
        self.probe = nn.Linear(self.config.dim, 1)

        self.proj = MLP(
            self.config.dim,
            hidden_channels=[
                self.config.proj_hidden_dim,
                self.config.proj_hidden_dim,
                self.config.proj_dim,
            ],
            norm_layer=nn.BatchNorm1d,
        )

        univariate_test = lejepa.univariate.EppsPulley(n_points=17)
        self.sigreg_loss = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test, num_slices=1024
        )

    def forward(
        self, x: Tensor, pos: Tensor, block_mask: BlockMask
    ) -> tuple[Tensor, Tensor]:
        return self.model(x, pos, block_mask)

    def forward_unlabeled(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        g_embed = self(batch["efds"], batch["pos"], batch["global_block_mask"])
        l_embed = self(batch["efds"], batch["pos"], batch["local_block_mask"])

        g_proj = self.proj(g_embed)
        l_proj = self.proj(l_embed)

        return g_proj, l_proj

    def forward_labeled(self, batch: dict[str, Any]) -> Tensor:
        with torch.no_grad():
            embed = self(batch["efds"], batch["pos"], batch["block_mask"])

        return self.probe(embed)

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        batch_size = len(batch["unlabeled"]["g_seq_lens"])
        g_emb, l_emb = self.forward_unlabeled(batch["unlabeled"])

        inv_loss = F.mse_loss(l_emb, g_emb)
        sigreg_loss = (self.sigreg_loss(g_emb) + self.sigreg_loss(l_emb)) / 2
        lejepa_loss = sigreg_loss * self.lamb + inv_loss * (1 - self.lamb)

        probe_labels = self.forward_labeled(batch["labeled"])
        probe_loss = F.binary_cross_entropy_with_logits(
            probe_labels, batch["labeled"]["labels"]
        )

        avg_norm = torch.linalg.norm(g_emb, dim=-1).mean()
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
        self.log(
            "train/probe_loss",
            probe_loss,
            rank_zero_only=True,
            prog_bar=True,
            on_epoch=True,
            batch_size=len(batch["labeled"]["seq_lens"]),
        )

        return lejepa_loss + probe_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        no_decay_params = [
            w for n, w in self.model.named_parameters() if w.ndim == 1 or ".rope." in n
        ]
        decay_params = (
            list(set(self.model.parameters()).difference(no_decay_params))
            + list(self.probe.parameters())
            + list(self.proj.parameters())
        )
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
