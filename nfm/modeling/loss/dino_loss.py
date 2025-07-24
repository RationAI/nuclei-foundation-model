# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Matěj Pekár from https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/dino_clstoken_loss.py

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


class DINOLoss(nn.Module):
    def __init__(self, student_temp: float = 0.1) -> None:
        super().__init__()
        self.student_temp = student_temp

    @torch.no_grad()
    def sinkhorn_knopp_teacher(
        self, teacher_output: Tensor, teacher_temp: float, n_iterations: int = 3
    ) -> Tensor:
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        q = torch.exp(
            teacher_output / teacher_temp
        ).T  # Q is K-by-B for consistency with notations from our paper
        b = q.shape[1] * world_size  # number of samples to assign
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

    def forward(
        self,
        student_output_list: list[Tensor],
        teacher_out_softmaxed_centered_list: list[Tensor],
    ) -> Tensor:
        """Cross-entropy between softmax outputs of the teacher and student networks."""
        total_loss = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                total_loss -= torch.sum(t * lsm, dim=-1).mean()
        return total_loss

    def forward(
        self,
        student_output: Tensor,
        teacher_out_softmaxed_centered: Tensor,
    ) -> Tensor:
        """Cross-entropy between softmax outputs of the teacher and student networks."""
        lsm = F.log_softmax(student_output / self.student_temp, dim=-1)
        lsm = rearrange(lsm, "ns b d -> ns 1 b d")
        teacher_out_softmaxed_centered = rearrange(
            teacher_out_softmaxed_centered, "nt b d -> 1 nt b d"
        )

        loss = -torch.sum(teacher_out_softmaxed_centered * lsm, dim=-1)
        return loss.mean()
