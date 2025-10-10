from collections.abc import Callable
from typing import Any

import torch
from einops import rearrange
from lightning import LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.optim import Optimizer

from nfm.configuration import Config
from nfm.modeling.dino_head import DINOHead
from nfm.modeling.loss import DINOLoss, KoLeoLoss, iBOTPatchLoss
from nfm.modeling.transformer import Transformer
from nfm.utils import CosineScheduler


class SSLMetaArch(LightningModule):
    def __init__(
        self, ema_momentum: float, dino: DictConfig, ibot: DictConfig, **config: Any
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

        self.dino_loss = DINOLoss()
        self.koleo_loss = KoLeoLoss()
        self.ibot_patch_loss = iBOTPatchLoss()

    def setup(self, stage: str) -> None:
        # self.trainer.num_training_batches is inf for some reason
        if stage == "fit":
            self.teacher_temp = CosineScheduler(
                base_value=0.07,
                final_value=0.07,
                total_iters=30 * 10,  # self.trainer.num_training_batches,
                warmup_iters=30 * 10,  # self.trainer.num_training_batches,
                start_warmup_value=0.04,
            )

    def student_forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        pattern = "b n ... -> (b n) ..."

        pos, embed = batch["local_crops"]
        local_outputs = self.student.backbone(
            src=rearrange(embed, pattern),
            tgt_pos=rearrange(batch["local_spatial_registers"], pattern),
            src_pos=rearrange(pos, pattern),
            local_crops=True,
        )

        pos, embed = batch["global_crops"]
        global_outputs = self.student.backbone(
            src=rearrange(embed, pattern),
            tgt_pos=rearrange(batch["global_spatial_registers"], pattern),
            src_pos=rearrange(pos, pattern),
        )

        local_cls_logits = self.student.dino_head(local_outputs["cls_token"])
        global_cls_logits = self.student.dino_head(global_outputs["cls_token"])
        global_patch_logits = self.student.ibot_head(global_outputs["patch_tokens"])

        return {
            "local_cls_logits": rearrange(
                local_cls_logits, "(b n) d -> b n d", b=pos.shape[0]
            ),
            "global_cls_tokens": rearrange(
                global_outputs["cls_token"], "(b n) d -> b n d", b=pos.shape[0]
            ),
            "global_cls_logits": rearrange(
                global_cls_logits, "(b n) d -> b n d", b=pos.shape[0]
            ),
            "global_patch_logits": rearrange(
                global_patch_logits, "(b n) d -> b n d", b=pos.shape[0]
            ),
        }

    @torch.inference_mode()
    def teacher_forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        pattern = "b n ... -> (b n) ..."
        pos, embed = batch["global_crops"]
        outputs = self.teacher.backbone(
            src=rearrange(embed, pattern),
            tgt_pos=rearrange(batch["global_spatial_registers"], pattern),
            src_pos=rearrange(pos, pattern),
        )

        # iBOT
        ibot_patch = self.teacher.ibot_head(outputs["patch_tokens"])
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
            "global_patch_logits": ibot_patch_centered,
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

        self.log(
            "train/dino_loss", dino_local_loss, sync_dist=True, rank_zero_only=True
        )
        self.log(
            "train/dino_loss", dino_global_loss, sync_dist=True, rank_zero_only=True
        )
        self.log("train/ibot_loss", ibot_loss, sync_dist=True, rank_zero_only=True)
        self.log("train/koleo_loss", koleo_loss, sync_dist=True, rank_zero_only=True)
        self.log(
            "train/total_loss",
            total_loss,
            sync_dist=True,
            rank_zero_only=True,
            prog_bar=True,
        )

        return total_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=0.0004, weight_decay=0.04
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs * self.trainer.num_training_batches,
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
