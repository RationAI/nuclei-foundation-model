from collections.abc import Callable
from typing import Any

import torch
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
    def __init__(self, dino: DictConfig, ibot: DictConfig, **config: Any) -> None:
        super().__init__()
        self.config = Config(**config)

        self.student = nn.ModuleDict(
            {
                "backbone": Transformer(self.config),
                "dino_head": DINOHead(
                    input_dim=self.config.dim,
                    hidden_dim=dino.hidden_dim,
                    bottleneck_dim=dino.bottleneck_dim,
                    output_dim=dino.num_prototypes,
                    num_layers=3,
                ),
                "ibot_head": DINOHead(
                    input_dim=self.config.dim,
                    hidden_dim=ibot.hidden_dim,
                    bottleneck_dim=ibot.bottleneck_dim,
                    output_dim=ibot.num_prototypes,
                    num_layers=3,
                ),
            }
        )
        self.teacher = nn.ModuleDict(
            {
                "backbone": Transformer(self.config),
                "dino_head": DINOHead(
                    input_dim=self.config.dim,
                    hidden_dim=dino.hidden_dim,
                    bottleneck_dim=dino.bottleneck_dim,
                    output_dim=dino.num_prototypes,
                    num_layers=3,
                ),
                "ibot_head": DINOHead(
                    input_dim=self.config.dim,
                    hidden_dim=ibot.hidden_dim,
                    bottleneck_dim=ibot.bottleneck_dim,
                    output_dim=ibot.num_prototypes,
                    num_layers=3,
                ),
            }
        )

        self.dino_loss = DINOLoss()
        self.koleo_loss = KoLeoLoss()
        self.ibot_patch_loss = iBOTPatchLoss(patch_out_dim=self.config.dim)

        self.momentum = CosineScheduler(
            base_value=0.994,
            final_value=1,
            total_iters=self.trainer.max_epochs * self.trainer.num_training_batches,
        )
        self.teacher_temp = CosineScheduler(
            base_value=0.07,
            final_value=0.07,
            total_iters=30 * self.trainer.num_training_batches,
            warmup_iters=30 * self.trainer.num_training_batches,
            start_warmup_value=0.04,
        )

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mask_indices_list = batch["mask_indices_list"]
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = batch["upperbound"]

        student_global_backbone_output_dict, student_local_backbone_output_dict = (
            self.student(
                [batch["collated_global_crops"], batch["collated_local_crops"]],
                masks=[batch["collated_masks"], None],
            )
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict[
            "x_norm_clstoken"
        ]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict[
                "x_norm_patchtokens"
            ]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(
                upperbound, _dim
            )
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(
                    ibot_student_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                )
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(
                    buffer_tensor_patch_tokens.unsqueeze(0)
                )
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(
                    buffer_tensor_patch_tokens
                )[:n_masked_patches]

        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(
            inputs_for_student_head_list
        )
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        return (
            student_global_cls_tokens,
            student_local_cls_tokens_after_head,
            student_global_cls_tokens_after_head,
            student_global_masked_patch_tokens_after_head,
        )

    @torch.no_grad()
    def teacher_forward(self, global_crops: Tensor) -> tuple[Tensor, Tensor]:
        n_global_crops = len(global_crops)
        embbed = self.teacher.backbone(global_crops)

        # dino
        cls_tokens = outputs["cls_token"].chunk(n_global_crops)
        cls_tokens = torch.cat((cls_tokens[1], cls_tokens[0]))
        # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss

        # iBOT
        ibot_teacher_patch_tokens = outputs["x_norm_patchtokens"]
        _dim = ibot_teacher_patch_tokens.shape[-1]
        buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
        torch.index_select(
            ibot_teacher_patch_tokens.flatten(0, 1),
            dim=0,
            index=mask_indices_list,
            out=buffer_tensor_teacher[:n_masked_patches],
        )

        masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(
            buffer_tensor_teacher
        )[:n_masked_patches]

        # sinkhorn_knopp centering
        cls_tokens_softmaxed_centered = self.dino_loss.sinkhorn_knopp_teacher(
            cls_tokens, teacher_temp=self.teacher_temp[self.global_step]
        ).view(n_global_crops, -1, *cls_tokens.shape[1:])

        masked_teacher_ibot_softmaxed_centered = (
            self.ibot_patch_loss.sinkhorn_knopp_teacher(
                masked_teacher_patch_tokens_after_head,
                teacher_temp=self.teacher_temp[self.global_step],
                n_masked_patches_tensor=n_masked_patches_tensor,
            )
        )

        return (
            cls_tokens_softmaxed_centered,
            masked_teacher_ibot_softmaxed_centered,
        )

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        n_global_crops = 2
        n_local_crops = 8

        global_crops = batch["collated_global_crops"]

        masks = batch["collated_masks"]
        mask_indices_list = batch["mask_indices_list"]
        n_masked_patches = mask_indices_list.shape[0]
        masks_weight = batch["masks_weight"].cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = (
            self.teacher_forward(global_crops)
        )

        (
            student_global_cls_tokens,
            student_local_cls_tokens_after_head,
            student_global_cls_tokens_after_head,
            student_global_masked_patch_tokens_after_head,
        ) = self(batch)

        loss_dict = {}

        loss_accumulator = 0  # for backprop

        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(
                    n_local_crops
                ),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        # compute dino loss
        dino_global_crops_loss = (
            self.dino_loss(
                student_output_list=[student_global_cls_tokens_after_head],
                teacher_out_softmaxed_centered_list=[
                    teacher_dino_softmaxed_centered_list.flatten(0, 1)
                ],  # these were chunked and stacked in reverse so A is matched to B
            )
            * loss_scales
            / (n_global_crops_loss_terms + n_local_crops_loss_terms)
        )

        loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

        # accumulate loss
        loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

        student_cls_tokens = student_global_cls_tokens

        # koleo loss
        koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
            self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
        )  # we don't apply koleo loss between cls tokens of a same image
        loss_accumulator += koleo_loss
        loss_dict["koleo_loss"] = (
            koleo_loss / loss_scales
        )  # this is to display the same losses as before but we can remove eventually

        # compute loss
        ibot_patch_loss = (
            self.ibot_patch_loss.forward_masked(
                student_global_masked_patch_tokens_after_head,
                masked_teacher_ibot_softmaxed_centered,
                student_masks_flat=masks,
                n_masked_patches=n_masked_patches,
                masks_weight=masks_weight,
            )
            * loss_scales
            * ibot_loss_scale
        )

        # store for display
        loss_dict["ibot_loss"] = ibot_patch_loss / 2

        # accumulate loss
        loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        return loss_accumulator

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=2.0e-04, weight_decay=0.04
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

        m = self.momentum[self.global_step]
        with torch.no_grad():
            student_params = list(self.student.parameters())
            teacher_params = list(self.teacher.parameters())
            torch._foreach_mul_(teacher_params, m)
            torch._foreach_add_(teacher_params, student_params, alpha=1 - m)
