# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Matěj Pekár from https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer.

    From Sablayrolles et al. - 2018 - Spreading vectors for similarity search
    """

    def __init__(self) -> None:
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_nns_inner(self, x: Tensor) -> Tensor:
        """Pairwise nearest neighbors for L2-normalized vectors."""
        dots = x @ x.T  # parwise dot products (= inverse distance)
        dots.fill_diagonal_(-1)
        return torch.max(dots, dim=1)[1]  # max inner prod -> min distance

    def forward(self, student_output: Tensor, eps: float = 1e-8) -> Tensor:
        """Compute the KoLeo loss.

        Args:
            student_output ([N, D]): Backbone output of student.
            eps: Small value to avoid numerical instability.
        """
        student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
        indices = self.pairwise_nns_inner(student_output)
        distances = self.pdist(student_output, student_output[indices])  # NxD, NxD -> N
        return -torch.log(distances + eps).mean()
