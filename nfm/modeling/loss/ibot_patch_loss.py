# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Matěj Pekár from https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/ibot_patch_loss.py

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


# ruff: noqa: N801
class iBOTPatchLoss(nn.Module):
    def __init__(
        self,
        patch_out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self,
        teacher_output: Tensor,
        teacher_temp: float,
        n_masked_patches_tensor: Tensor,
        n_iterations: int = 3,
    ) -> Tensor:
        teacher_output = teacher_output.float()

        q = torch.exp(
            teacher_output / teacher_temp
        ).T  # Q is K-by-B for consistency with notations from our paper
        # B = Q.shape[1] * world_size # number of samples to assign
        b = torch.tensor(
            n_masked_patches_tensor,
            dtype=torch.float32,
            device=teacher_output.device,
        )

        if dist.is_initialized():
            dist.all_reduce(b)
        k = q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_q = torch.sum(q)
        if dist.is_initialized():
            dist.all_reduce(sum_q)
        q /= sum_q

        for _ in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            q /= sum_of_rows
            q /= k

            # normalize each column: total weight per sample must be 1/B
            q /= torch.sum(q, dim=0, keepdim=True)
            q /= b

        q *= b  # the columns must sum to 1 so that Q is an assignment
        return q.T

    def forward_masked(
        self,
        student_patch_tokens_masked: Tensor,
        teacher_patch_tokens_masked: Tensor,
        student_masks_flat: Tensor = None,
        n_masked_patches=None,
        masks_weight=None,
    ) -> Tensor:
        s = student_patch_tokens_masked
        t = teacher_patch_tokens_masked.clone()
        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)

        if student_masks_flat is None:
            return -loss.sum() / s.shape[0]

        # if masks_weight is None:
        #     masks_weight = (
        #         (1 / student_masks_flat.sum(-1).clamp(min=1.0))
        #         .unsqueeze(-1)
        #         .expand_as(student_masks_flat)[student_masks_flat]
        #     )
        # if n_masked_patches is not None:
        #     loss = loss[:n_masked_patches]
        # loss = loss * masks_weight
        # return -loss.sum() / student_masks_flat.shape[0]
