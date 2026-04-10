import math
from typing import Self

import numpy as np
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.utils.rnn import pad_sequence


class _PaddingMaskMod:
    """Pickle-safe callable used by BlockMask for sequence-length padding.

    Pre-computed per-batch scalar to avoid dynamic indexing in pointwise subgraph.
    """

    def __init__(self, seq_lens: Tensor) -> None:
        # Store as expanded view to avoid dynamic indexing in the mask function
        # When using spawn mp_context, CUDA can be initialized in workers
        self.seq_lens = seq_lens

    def to(self, device: torch.device) -> Self:
        self.seq_lens = self.seq_lens.to(device)
        return self

    def __call__(self, b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
        # seq_lens is already on CUDA from __init__, same device as q/kv
        return (q < self.seq_lens[b]) & (kv < self.seq_lens[b])


def create_batched_block_quantized_knn_mask(
    neighbor_indices_list: list[Tensor],
    seq_lens: Tensor,
    block_size: int,
    symmetric: bool = False,
) -> BlockMask:
    """Creates a batched BlockMask directly from variable-length neighbor indices.

    Natively supports sequences not divisible by block_size without artificial padding.

    Args:
        neighbor_indices_list: List of Tensors shape (N_i, K) containing neighbor indices.
        seq_lens: List of unpadded sequence lengths for each batch item.
        block_size: Size of the attention block.
        symmetric: Whether to symmetrize the connections.

    Returns:
        Batched BlockMask configured with full/partial kernel metadata.
    """
    neighbor_indices = pad_sequence(
        neighbor_indices_list, batch_first=True, padding_value=-1
    )
    B, N, K = neighbor_indices.shape
    num_blocks = math.ceil(N / block_size)

    q_idx = torch.arange(N).view(1, N, 1).expand(B, N, K)
    kv_idx = neighbor_indices
    seq_lens_view = seq_lens.view(B, 1, 1)
    valid_mask = (q_idx < seq_lens_view) & (kv_idx < seq_lens_view) & (kv_idx >= 0)

    q_block_ids = q_idx // block_size
    kv_block_ids = kv_idx // block_size

    adj_matrix = torch.zeros((B, num_blocks, num_blocks), dtype=torch.bool)
    b_idx = torch.arange(B).view(B, 1, 1).expand(B, N, K)
    adj_matrix[b_idx[valid_mask], q_block_ids[valid_mask], kv_block_ids[valid_mask]] = (
        True
    )

    if symmetric:
        adj_matrix = adj_matrix | adj_matrix.mT

    kv_num_blocks = adj_matrix.sum(dim=-1).to(torch.int32)

    col_indices = (
        torch.arange(num_blocks)
        .view(1, 1, num_blocks)
        .expand(B, num_blocks, num_blocks)
    )
    masked_col_indices = torch.where(
        adj_matrix, col_indices, torch.tensor(num_blocks + 1)
    )
    sorted_indices, _ = masked_col_indices.sort(dim=-1)

    kv_indices = torch.where(
        sorted_indices > num_blocks,
        torch.tensor(-1, dtype=torch.int32),
        sorted_indices.to(torch.int32),
    )

    kv_num_blocks = kv_num_blocks.unsqueeze(1)
    kv_indices = kv_indices.unsqueeze(1)

    full_kv_indices = kv_indices.clone()
    num_fully_valid_blocks = seq_lens // block_size

    q_blk_idx = torch.arange(num_blocks).view(1, 1, num_blocks)
    mixed_q_mask = q_blk_idx >= num_fully_valid_blocks.view(B, 1, 1)
    mixed_kv_mask = full_kv_indices >= num_fully_valid_blocks.view(B, 1, 1, 1)

    full_kv_indices.masked_fill_(mixed_q_mask.unsqueeze(-1), -1)
    full_kv_indices.masked_fill_(mixed_kv_mask, -1)
    full_kv_num_blocks = (full_kv_indices != -1).sum(dim=-1).to(torch.int32)

    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        BLOCK_SIZE=(block_size, block_size),
        mask_mod=_PaddingMaskMod(seq_lens),
    )


def block_spatial_sort(points: np.ndarray, block_size: int) -> np.ndarray:
    n = len(points)
    out = np.arange(n)

    # Stack holds (start, end, depth) — operate on out[start:end] in-place
    stack = [(0, n, 0)]

    while stack:
        start, end, depth = stack.pop()
        size = end - start

        if size <= block_size:
            continue

        segment = out[start:end]

        # Partial sort: only need to put left_blocks*block_size elements on the left
        num_blocks = math.ceil(size / block_size)
        left_blocks = num_blocks // 2
        split = left_blocks * block_size

        # np.argpartition avoids full sort — O(n) instead of O(n log n) per level
        axis = depth % 2
        local_pts = points[segment, axis]
        pivot_idx = np.argpartition(local_pts, split - 1)
        segment[:] = segment[pivot_idx]

        stack.append((start, start + split, depth + 1))
        stack.append((start + split, end, depth + 1))

    return out
