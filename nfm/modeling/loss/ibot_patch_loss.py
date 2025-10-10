# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Matěj Pekár from https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/ibot_patch_loss.py

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


class iBOTPatchLoss(nn.Module):
    def __init__(self, student_temp: float = 0.1) -> None:
        super().__init__()
        self.student_temp = student_temp

    def sinkhorn_knopp_teacher(
        self, teacher_output: Tensor, teacher_temp: float, n_iterations: int = 3
    ) -> Tensor:
        """Performs the Sinkhorn-Knopp algorithm to obtain a distribution over prototypes that is uniform over the batch.

        Args:
            teacher_output: Teacher outputs of shape (batch_size, num_prototypes).
            teacher_temp: Temperature parameter for the teacher outputs.
            n_iterations: Number of iterations for the Sinkhorn-Knopp algorithm.
        """
        teacher_output = teacher_output.float()
        q = torch.exp(teacher_output / teacher_temp)

        b, k = q.shape
        if dist.is_initialized():
            b *= dist.get_world_size()

        # make the matrix sums to 1
        sum_q = torch.sum(q)
        if dist.is_initialized():
            dist.all_reduce(sum_q)
        q /= sum_q

        for _ in range(n_iterations):
            # normalize each column: total weight per prototype must be 1/K
            sum_of_cols = torch.sum(q, dim=0, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_cols)
            q /= sum_of_cols * k

            # normalize each row: total weight per sample must be 1/B
            q /= torch.sum(q, dim=1, keepdim=True) * b

        q *= b  # the columns must sum to 1 so that Q is an assignment
        return q

    def forward(
        self, student_patch_tokens: Tensor, teacher_patch_tokens: Tensor
    ) -> Tensor:
        t = teacher_patch_tokens.float()
        s = student_patch_tokens.float()
        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        return -loss.mean()
