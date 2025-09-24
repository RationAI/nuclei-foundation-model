# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by Matěj Pekár from https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.distributed.nn import all_gather


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer.

    From Sablayrolles et al. - 2018 - Spreading vectors for similarity search
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        """Compute the KoLeo loss.

        Args:
            x ([B, N, D]): Backbone output of student.
        """
        x = F.normalize(x.float(), eps=self.eps, p=2, dim=-1)
        x = rearrange(x, "b n d -> n b d")

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            all_x = torch.cat(all_gather(x), dim=1)  # N, B*world, D
            rank = dist.get_rank()
        else:
            all_x = x
            rank = 0

        # nearest neighbor search
        with torch.no_grad():
            similarity = torch.bmm(x, all_x.transpose(1, 2))
            n, local_b, global_b = similarity.shape

            # Trick to fill diagonal with -1
            similarity.view(n, -1)[:, rank * local_b :: (global_b + 1)].fill_(-1)

            # gather nearest neighbors
            indices = torch.argmax(similarity, dim=-1)
            nearest_neighbors = torch.gather(all_x, 1, indices[..., None].expand_as(x))

        # Calculate pairwise distance between local vectors and their found NNs
        distances = torch.norm(x - nearest_neighbors, p=2, dim=-1)
        return -torch.log(distances + self.eps).mean()
