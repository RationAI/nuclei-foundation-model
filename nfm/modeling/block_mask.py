import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial import KDTree
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


def attend_all_mask_mod(
    batch: Tensor, head: Tensor, q_idx: Tensor, kv_idx: Tensor
) -> Tensor:
    return torch.ones_like(batch, dtype=torch.bool)


def create_block_quantized_knn_mask(
    kdtree: KDTree,
    points: NDArray[np.floating],
    n_points_unpadded: int,
    k: int,
    block_size: int,
    symmetric: bool = True,
) -> BlockMask:
    """Generates a single-item BlockMask from a KDTree and a corresponding point array.

    Padded points (at the end of the array) are excluded so that they neither attend to nor are attended by any key/value blocks.

    Args:
        kdtree: KDTree built over the points.
        points: A sorted (n, d) numpy array of positions, n must be divisible by the `block_size`.
        k: Number of neighbors to query, at least 1.
        n_points_unpadded: Number of points without the padding.
        block_size: Number of points per block.

    Returns:
        A BlockMask object with layouts (Batch, Head, ...):
            - kv_num_blocks: (1, 1, num_blocks), number of key/value blocks per query block
            - kv_indices: (1, 1, num_blocks, num_blocks), indices of key/value blocks
            - q_num_blocks: (1, 1, num_blocks), count of query blocks per key/value block (derived).
            - q_indices: (1, 1, num_blocks, num_blocks), indices of query blocks (derived).
            - BLOCK_SIZE: (block_size, block_size)
            - shape: (1, 1, seq_length, seq_length)
        where num_blocks = seq_length // block_size, seq_length % block_size = 0
    """
    n_points = points.shape[0]
    assert k >= 1 and n_points % block_size == 0
    num_blocks = n_points // block_size

    # 1. Build Block Adjacency Mask from a kNN Query (Q -> K mapping)
    # ----------------------------------------------------------------
    _, neighbor_indices = kdtree.query(points[:n_points_unpadded], k=k)
    neighbor_indices = neighbor_indices[:, None] if k == 1 else neighbor_indices

    q_block_ids = np.arange(n_points_unpadded) // block_size
    kv_block_ids = neighbor_indices // block_size

    adj_matrix = np.zeros((num_blocks, num_blocks), dtype=bool)
    valid_mask = neighbor_indices < n_points_unpadded
    q_block_ids_expanded = np.broadcast_to(q_block_ids[:, None], valid_mask.shape)

    # mark blocks as connected if any point in Q attends to any point in K
    adj_matrix[q_block_ids_expanded[valid_mask], kv_block_ids[valid_mask]] = True

    # Enforce symmetric block graph
    if symmetric:
        adj_matrix = adj_matrix | adj_matrix.T

    # 2. Convert adjacency to BlockMask format:
    # ----------------------------------------------------------------
    kv_counts = adj_matrix.sum(axis=1)
    kv_num_blocks = torch.from_numpy(kv_counts).int().unsqueeze(0)
    kv_indices = torch.full((1, num_blocks, num_blocks), -1, dtype=torch.int32)

    rows, cols = np.nonzero(adj_matrix)
    # sort connections by Q-block and then KV-block to ensure that slot indices are contiguous
    # within each Q-block (required by BlockMask)
    order = np.lexsort((cols, rows))  # sort keys are (secondary, primary)
    rows = rows[order]
    cols = cols[order]

    # compute the slot indices for kv_indices[0, Q-block, slot]
    cum_counts = np.cumsum(kv_counts)
    q_block_offsets = np.zeros_like(cum_counts)
    q_block_offsets[1:] = cum_counts[:-1]
    global_idx = np.arange(len(rows))
    slot_idx = global_idx - q_block_offsets[rows]

    kv_indices[0, rows, slot_idx] = torch.from_numpy(cols).int()

    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks.unsqueeze(0),  # add head dim
        kv_indices=kv_indices.unsqueeze(0),  # add head dim
        full_kv_num_blocks=kv_num_blocks.unsqueeze(0),  # add head dim
        full_kv_indices=kv_indices.unsqueeze(0),  # add head dim
        BLOCK_SIZE=block_size,
        # mask_mod=attend_all_mask_mod,
    )


def batch_block_masks(masks: list[BlockMask]) -> BlockMask:
    """Batch a list of single-item BlockMask objects into one batched BlockMask.

    All masks must have the same sequence length and block size.
    Different neighbor counts (at the block level) are handled by padding.

    Args:
        masks: List of BlockMask objects.

    Returns:
        Batched BlockMask object with layouts (Batch, Head, ...):
            - kv_num_blocks: (b, 1, num_blocks)
            - kv_indices: (b, 1, num_blocks, max_kv_blocks)
            - q_num_blocks: (b, 1, num_blocks) (derived)
            - q_indices: (b, 1, num_blocks, num_blocks) (derived)
            - BLOCK_SIZE: (block_size, block_size)
            - shape: (b, 1, seq_length, seq_length)
        where
            b = batch size,
            num_blocks = seq_length // block_size,
            max_kv_blocks = maximum number of KV blocks per query block across the batch,
        The "mask_mod" is inherited from the first mask.
    """
    assert all(m.shape == masks[0].shape for m in masks)
    assert all(m.BLOCK_SIZE == masks[0].BLOCK_SIZE for m in masks)

    kv_num_blocks = torch.cat([m.kv_num_blocks for m in masks], dim=0)
    kv_indices_list = [m.kv_indices for m in masks]

    max_kv_len = max(kv.shape[-1] for kv in kv_indices_list)
    padded_kv_indices = [
        torch.nn.functional.pad(kv, (0, max_kv_len - kv.shape[-1]), "constant", -1)
        for kv in kv_indices_list
    ]
    kv_indices = torch.cat(padded_kv_indices, dim=0)

    batched_mask = BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=None,
        full_kv_indices=None,
        BLOCK_SIZE=masks[0].BLOCK_SIZE,
        mask_mod=masks[0].mask_mod,
    )
    return batched_mask
