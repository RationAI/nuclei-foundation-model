import math
from copy import copy
from typing import Self

import numpy as np
import torch
import torch.nn.attention.flex_attention
import torch.nn.functional as F
from einops import repeat
from torch import Tensor, nn
from torch.utils._pytree import tree_map_only


class _MaskMod(nn.Module):
    def __init__(self, doc_ids: Tensor, max_sequence_length: int) -> None:
        super().__init__()
        self.register_buffer(
            "doc_ids", F.pad(doc_ids, (0, max_sequence_length - len(doc_ids)), value=-1)
        )

    def __call__(self, b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
        # If the tokens don't belong to the same document, zero out the attention.
        return (self.doc_ids[q] >= 0) & (self.doc_ids[q] == self.doc_ids[kv])


class BlockMask(torch.nn.attention.flex_attention.BlockMask):
    def to(self, device: torch.device | str) -> Self:
        moved = copy(self)
        moved.__dict__ = tree_map_only(
            torch.Tensor | _MaskMod,
            lambda x: x.to(device),
            self.__dict__,
        )
        return moved


def create_ragged_block_quantized_knn_mask(
    neighbor_indices_list: list[Tensor],
    block_size: int,
    *,
    max_sequence_length: int,
) -> BlockMask:
    """Use the packed length for attention and a fixed capacity for doc_ids."""
    device = neighbor_indices_list[0].device

    # === 1. Tightly Pack & Shift KNN Indices ===
    packed_neighbors_list = []
    doc_ids_list = []
    current_offset = 0

    for doc_id, neighbors in enumerate(neighbor_indices_list):
        N_i = len(neighbors)

        # Vectorized offset (faster and more memory efficient than .clone() + valid_mask)
        packed_neighbors_list.append(
            torch.where(neighbors >= 0, neighbors + current_offset, neighbors)
        )

        doc_ids_list.append(
            torch.where(
                (neighbors >= 0).any(dim=1),
                torch.full((N_i,), doc_id, dtype=torch.int32, device=device),
                torch.full((N_i,), -1, dtype=torch.int32, device=device),
            )
        )
        current_offset += N_i

    packed_neighbors = torch.cat(packed_neighbors_list, dim=0)  # (N_total, K)
    doc_ids = torch.cat(doc_ids_list, dim=0)  # (N_total,)

    seq_len, K = packed_neighbors.shape
    if seq_len > max_sequence_length:
        raise ValueError(
            f"Packed length {seq_len} exceeds doc_ids capacity {max_sequence_length}"
        )
    num_blocks = math.ceil(seq_len / block_size)

    # === 2. Build Global Adjacency Matrix (2D) ===
    # Map token-level connections to block-level connections
    q_idx = torch.arange(seq_len, device=device)
    q_block_ids = repeat(q_idx // block_size, "n -> n k", k=K)
    kv_block_ids = packed_neighbors // block_size

    valid_conn = packed_neighbors >= 0

    # We keep this 2D until the end to save memory and avoid broad casting overhead
    adj_matrix = torch.zeros((num_blocks, num_blocks), dtype=torch.bool, device=device)
    adj_matrix[q_block_ids[valid_conn], kv_block_ids[valid_conn]] = True

    kv_num_blocks = adj_matrix.sum(dim=-1, dtype=torch.int32)

    # === 3. Compress to Dense KV Indices ===
    col_indices = repeat(
        torch.arange(num_blocks, device=device, dtype=torch.int32),
        "k -> n k",
        n=num_blocks,
    )

    # Push invalid connections to the back (value: num_blocks + 1) and sort them out
    masked_col_indices = torch.where(adj_matrix, col_indices, num_blocks + 1)
    kv_indices, _ = masked_col_indices.sort(dim=-1)
    kv_indices[kv_indices > num_blocks] = -1  # Demote invalid connections to -1

    # === 4. Optimize Fast Path (Pure vs Mixed Blocks) ===
    # This temporary block view only classifies full vs partial blocks. The
    # attention sequence remains at its actual length.
    block_doc_ids = F.pad(
        doc_ids, (0, num_blocks * block_size - seq_len), value=-2
    ).view(num_blocks, block_size)
    is_pure_block = (block_doc_ids == block_doc_ids[:, :1]).all(dim=1)
    is_pure_block &= block_doc_ids[:, 0] >= 0

    valid_kv_mask = kv_indices >= 0
    safe_kv_indices = torch.where(valid_kv_mask, kv_indices, 0)

    is_kv_pure = is_pure_block[safe_kv_indices]

    mixed_q_mask = ~is_pure_block
    mixed_kv_mask = valid_kv_mask & (~is_kv_pure)

    full_kv_indices = kv_indices.clone()

    # Demote boundary-crossing blocks to the slow path (-1)
    full_kv_indices.masked_fill_(mixed_q_mask.unsqueeze(-1), -1)
    full_kv_indices.masked_fill_(mixed_kv_mask, -1)

    sort_keys = torch.where(full_kv_indices == -1, num_blocks + 1, full_kv_indices)
    sorted_full_indices, _ = sort_keys.sort(dim=-1)

    # Push the -1 "holes" to the back so valid indices are contiguous
    full_kv_indices = torch.where(
        sorted_full_indices > num_blocks,
        torch.tensor(-1, dtype=torch.int32, device=device),
        sorted_full_indices,
    )

    full_kv_num_blocks = (full_kv_indices != -1).sum(dim=-1, dtype=torch.int32)

    # FlexAttention processes full and partial lists separately: they must be
    # disjoint, otherwise pure blocks are attended to twice.
    partial_indices = kv_indices.masked_fill(
        (~mixed_q_mask[:, None]) & is_kv_pure & valid_kv_mask, -1
    )
    partial_keys = torch.where(partial_indices >= 0, partial_indices, num_blocks + 1)
    partial_keys = partial_keys.sort(dim=-1).values
    kv_indices = torch.where(partial_keys < num_blocks, partial_keys, -1)
    kv_num_blocks = (kv_indices >= 0).sum(dim=-1, dtype=torch.int32)

    # === 5. Reshape for BlockMask (Batch=1, Heads=1) ===
    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks.view(1, 1, num_blocks),
        kv_indices=kv_indices.view(1, 1, num_blocks, num_blocks),
        full_kv_num_blocks=full_kv_num_blocks.view(1, 1, num_blocks),
        full_kv_indices=full_kv_indices.view(1, 1, num_blocks, num_blocks),
        BLOCK_SIZE=(block_size, block_size),
        mask_mod=_MaskMod(doc_ids, max_sequence_length),
        seq_lengths=(seq_len, seq_len),
    )


def block_spatial_sort(
    points: np.ndarray, block_size: int, global_offset: int = 0
) -> np.ndarray:
    n = len(points)
    out = np.arange(n)
    stack = [(0, n, 0)]

    while stack:
        start, end, depth = stack.pop()

        # Translate to global sequence indices to align with hardware blocks
        global_start = global_offset + start
        global_end = global_offset + end - 1

        start_block = global_start // block_size
        end_block = global_end // block_size

        # If the entire segment fits within a single global block, stop splitting
        if start_block == end_block:
            continue

        # Pick a split boundary at a global block transition
        split_block = (start_block + end_block + 1) // 2

        # Translate the chosen global boundary back to a local split size
        split_local_idx = split_block * block_size - global_offset
        split_size = split_local_idx - start

        segment = out[start:end]
        axis = depth % 2
        local_pts = points[segment, axis]

        # Partition array based on the globally-aligned split size
        pivot_idx = np.argpartition(local_pts, split_size - 1)
        segment[:] = segment[pivot_idx]

        stack.append((start, start + split_size, depth + 1))
        stack.append((start + split_size, end, depth + 1))

    return out
