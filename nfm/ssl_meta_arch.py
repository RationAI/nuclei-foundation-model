from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from einops import rearrange
from lightning import LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim import Optimizer

from nfm.configuration import Config
from nfm.modeling.dino_head import DINOHead
from nfm.modeling.loss import DINOLoss, KoLeoLoss, iBOTPatchLoss
from nfm.modeling.transformer import Transformer
from nfm.utils import CosineScheduler


class SSLMetaArch(LightningModule):
    def __init__(
        self,
        ema_momentum: float,
        dino: dict[str, Any],
        ibot: dict[str, Any],
        **config: Any,
    ) -> None:
        super().__init__()
        self.config = Config(**config)
        self.ema_momentum = ema_momentum
        self.ibot_loss_weight = 1
        self.koleo_loss_weight = 1
        self.dino_loss_weight = 1

        self.student = nn.ModuleDict(
            {
                "backbone": Transformer(self.config),
                "dino_head": DINOHead(input_dim=self.config.dim, **dino),
                "ibot_head": DINOHead(input_dim=self.config.dim, **ibot),
            }
        )
        self.teacher = nn.ModuleDict(
            {
                "backbone": Transformer(self.config),
                "dino_head": DINOHead(input_dim=self.config.dim, **dino),
                "ibot_head": DINOHead(input_dim=self.config.dim, **ibot),
            }
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.dino_loss = DINOLoss()
        self.koleo_loss = KoLeoLoss()
        self.ibot_patch_loss = iBOTPatchLoss()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.teacher_temp = CosineScheduler(
                base_value=0.07,
                final_value=0.07,
                total_iters=self.trainer.estimated_stepping_batches,
                warmup_iters=1000,
                start_warmup_value=0.04,
            )

    def student_forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        pos, embed = batch["local_crops"]
        local_outputs = self.student.backbone(
            src=embed.flatten(0, 1),
            tgt_pos=batch["local_spatial_registers"].flatten(0, 1),
            src_pos=pos.flatten(0, 1),
            local_crops=True,
        )

        pos, embed = batch["global_crops"]
        pos_flat = pos.flatten(0, 1)
        embed_flat = embed.flatten(0, 1)

        # Per-crop masking: sample a drop probability p ~ U[0.2, 0.5] for each crop
        # and drop tokens independently with that probability within the crop.
        p_drop = torch.empty(
            embed_flat.shape[0], 1, 1, device=embed_flat.device, dtype=embed_flat.dtype
        ).uniform_(0.1, 0.5)
        mask = (
            torch.rand(
                embed_flat.shape[0],
                embed_flat.shape[1],
                1,
                device=embed_flat.device,
                dtype=embed_flat.dtype,
            )
            > p_drop
        )

        global_outputs = self.student.backbone(
            src=embed_flat * mask,
            tgt_pos=batch["global_spatial_registers"].flatten(0, 1),
            src_pos=pos_flat * mask,
        )

        local_cls_logits = self.student.dino_head(local_outputs["cls_token"])
        global_cls_logits = self.student.dino_head(global_outputs["cls_token"])
        global_patch_logits = self.student.ibot_head(global_outputs["patch_tokens"])

        revert_shape = partial(
            rearrange, pattern="(b n) ... -> b n ...", b=pos.shape[0]
        )
        return {
            "local_cls_logits": revert_shape(local_cls_logits),
            "global_cls_tokens": revert_shape(global_outputs["cls_token"]),
            "global_cls_logits": revert_shape(global_cls_logits),
            "global_patch_logits": revert_shape(global_patch_logits),
        }

    @torch.no_grad()
    def teacher_forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        pos, embed = batch["global_crops"]
        outputs = self.teacher.backbone(
            src=embed.flatten(0, 1),
            tgt_pos=batch["global_spatial_registers"].flatten(0, 1),
            src_pos=pos.flatten(0, 1),
        )

        # iBOT
        patch_tokens = rearrange(outputs["patch_tokens"], "bn k d -> (bn k) d")
        ibot_patch = self.teacher.ibot_head(patch_tokens)
        ibot_patch_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
            ibot_patch, self.teacher_temp[self.global_step]
        )

        # DINO
        cls = self.teacher.dino_head(outputs["cls_token"])
        cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
            cls, self.teacher_temp[self.global_step]
        )
        cls_centered = rearrange(cls_centered, "(b n) d -> b n d", b=pos.shape[0])

        return {
            "global_cls_logits": cls_centered,
            "global_patch_logits": rearrange(
                ibot_patch_centered,
                "(b n k) c -> b n k c",
                b=pos.shape[0],
                n=embed.shape[1],
            ),
        }

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        n_global_crops = batch["global_spatial_registers"].shape[1]
        n_local_crops = batch["local_spatial_registers"].shape[1]

        dino_global_terms = n_global_crops * (n_global_crops - 1)
        dino_local_terms = n_global_crops * n_local_crops
        dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
        dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)

        outputs = self.student_forward(batch)
        targets = self.teacher_forward(batch)

        ibot_loss = self.ibot_patch_loss(
            outputs["global_patch_logits"],
            targets["global_patch_logits"],
        )
        dino_local_loss = self.dino_loss(
            outputs["local_cls_logits"],
            targets["global_cls_logits"],
        )
        dino_global_loss = self.dino_loss(
            outputs["global_cls_logits"],
            targets["global_cls_logits"],
            ignore_diagonal=True,
        )
        koleo_loss = self.koleo_loss(outputs["global_cls_tokens"])

        total_loss = (
            ibot_loss * self.ibot_loss_weight
            + dino_local_loss * dino_local_scale * self.dino_loss_weight
            + dino_global_loss * dino_global_scale * self.dino_loss_weight
            + koleo_loss * self.koleo_loss_weight
        )

        self.log("train/dino_loss", dino_local_loss, rank_zero_only=True)
        self.log("train/dino_loss", dino_global_loss, rank_zero_only=True)
        self.log("train/ibot_loss", ibot_loss, rank_zero_only=True)
        self.log("train/koleo_loss", koleo_loss, rank_zero_only=True)
        self.log(
            "train/total_loss",
            total_loss,
            rank_zero_only=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        no_decay_params = [
            w
            for n, w in self.student.named_parameters()
            if w.ndim == 1 or ".rope." in n
        ]
        decay_params = list(set(self.student.parameters()).difference(no_decay_params))
        params = [
            {"params": decay_params},
            {"params": no_decay_params, "weight_decay": 0},
        ]

        optimizer = torch.optim.AdamW(params, lr=0.0004, weight_decay=0.04)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1.0e-06,
        )
        return [optimizer], [scheduler]

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer | LightningOptimizer,
        optimizer_closure: Callable[[], Any] | None = None,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        # update teacher with EMA
        with torch.no_grad():
            student_params = list(self.student.parameters())
            teacher_params = list(self.teacher.parameters())
            torch._foreach_mul_(teacher_params, self.ema_momentum)
            torch._foreach_add_(
                teacher_params, student_params, alpha=1 - self.ema_momentum
            )
