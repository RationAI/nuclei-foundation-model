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
        self.ibot_patch_loss = iBOTPatchLoss(patch_out_dim=self.config.dim)

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

    def batch_forward_backbone(
        self,
        backbone: nn.Module,
        crops: tuple[Tensor, Tensor],
        tokens: tuple[Tensor, Tensor],
        local_crops: bool = False,
    ) -> dict[str, Tensor]:
        args = {"b": crops[0].shape[0]}
        pattern = "b n ... -> (b n) ..."
        reverse = "(b n) ... -> b n ..."
        if local_crops:
            args["n"] = crops[0].shape[1]
            pattern = "b n l ... -> (b n l) ..."
            reverse = "(b n l) ... -> b n l ..."

        pos, embed = crops
        token_pos, token_embed = tokens

        embed = rearrange(embed, pattern)
        pos = rearrange(pos, pattern)
        token_embed = rearrange(token_embed, pattern)
        token_pos = rearrange(token_pos, pattern)

        outputs = backbone(token_embed, embed, token_pos, pos, local_crops)

        outputs["patch_tokens"] = rearrange(outputs["patch_tokens"], reverse, **args)
        outputs["cls_token"] = rearrange(outputs["cls_token"], reverse, **args)

        return outputs

    def student_forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        local_outputs = self.batch_forward_backbone(
            self.student.backbone,
            batch["local_crops"],
            batch["local_crop_tokens"],
            local_crops=True,
        )
        local_cls_logits = self.student.dino_head(local_outputs["cls_token"])

        global_outputs = self.batch_forward_backbone(
            self.student.backbone,
            batch["global_crops"],
            batch["global_crop_tokens"],
        )
        global_patch_logits = self.student.ibot_head(global_outputs["patch_tokens"])
        global_cls_logits = self.student.dino_head(global_outputs["cls_token"])

        return {
            "local_cls_logits": local_cls_logits,
            "global_cls_tokens": global_outputs["cls_token"],
            "global_cls_logits": global_cls_logits,
            "global_patch_logits": global_patch_logits,
        }

    @torch.inference_mode()
    def teacher_forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        outputs = self.batch_forward_backbone(
            self.teacher.backbone,
            batch["global_crops"],
            batch["global_crop_tokens"],
        )

        # iBOT
        ibot_patch_tokens = self.teacher.ibot_head(outputs["patch_tokens"])

        n_masked_patches_tensor = (
            ibot_patch_tokens.shape[0]
            * ibot_patch_tokens.shape[1]
            * ibot_patch_tokens.shape[2]
        )
        ibot_patch_tokens = self.ibot_patch_loss.sinkhorn_knopp_teacher(
            ibot_patch_tokens,
            teacher_temp=self.teacher_temp[self.global_step],
            n_masked_patches_tensor=n_masked_patches_tensor,
        )

        # DINO
        cls_tokens = outputs["cls_token"].flip(
            0
        )  # reverse so A is matched to B in the global crops dino loss
        cls_tokens = self.teacher.dino_head(cls_tokens)
        cls_tokens = self.dino_loss.sinkhorn_knopp_teacher(
            cls_tokens,
            teacher_temp=self.teacher_temp[self.global_step],
        ).permute(1, 0, 2)
        # .view(n_global_crops, -1, *cls_tokens.shape[1:])

        # cls_tokens = rearrange(cls_tokens, "b n c -> (b n) c")

        return {
            "global_cls_logits": cls_tokens,
            "global_patch_logits": ibot_patch_tokens,
        }

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        outputs = self.student_forward(batch)
        targets = self.teacher_forward(batch)

        ibot_loss = self.ibot_patch_loss.forward_masked(
            outputs["global_patch_logits"],
            targets["global_patch_logits"],
        )
        dino_loss = self.dino_loss(
            outputs["global_cls_logits"],
            targets["global_cls_logits"],
        )
        koleo_loss = self.koleo_loss(outputs["global_cls_tokens"])

        self.log(
            "train/dino_loss",
            dino_loss,
            sync_dist=True,
            prog_bar=True,
            batch_size=1,
            rank_zero_only=True,
        )
        self.log("train/ibot_loss", ibot_loss, sync_dist=True)
        self.log("train/koleo_loss", koleo_loss, sync_dist=True)

        return (
            ibot_loss * self.ibot_loss_weight
            # + dino_loss * self.dino_loss_weight
            # + koleo_loss * self.koleo_loss_weight
        )

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
