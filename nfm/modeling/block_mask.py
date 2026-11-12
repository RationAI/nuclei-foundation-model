"""Neighborhood attention over packed views of nuclei.

The sequence holds the nuclei of all views, each view sorted so that a block of
`block_size` tokens is a compact patch of tissue. Within a view, a nucleus
attends to the nuclei in every block holding a nearest neighbor of a nucleus in
its own block. Neighborhoods are thus quantized to blocks: block pairs within
one view are full blocks, and the token-level mask is evaluated only for blocks
that hold nuclei of two views.
"""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


def neighborhood_mask(
    lengths: Tensor, neighbors: Tensor, block_size: int = 128
) -> BlockMask:
    """The block mask for packed views.

    Args:
        lengths: Nuclei per view, [views].
        neighbors: Indices of the nearest nuclei of each nucleus, [nuclei, k].
        block_size: Block size of the flex attention kernel.
    """
    device = neighbors.device
    tokens = len(neighbors)
    blocks = -(-tokens // block_size)
    documents = torch.full((blocks * block_size,), -1, device=device)
    views = torch.arange(len(lengths), device=device)
    documents[:tokens] = torch.repeat_interleave(views, lengths)

    # Blocks of nuclei reached by each block of nuclei.
    reached = torch.zeros(blocks, blocks, dtype=torch.bool, device=device)
    rows = (torch.arange(tokens, device=device) // block_size)[:, None]
    reached[rows.expand_as(neighbors), neighbors // block_size] = True

    def mask_mod(b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
        same_view = documents[q] == documents[kv]
        return same_view & reached[q // block_size, kv // block_size]

    # Full blocks hold nuclei of a single view, so the view check always passes.
    by_block = documents.view(blocks, block_size)
    valid = by_block >= 0
    lowest = torch.where(valid, by_block, len(lengths)).amin(1)
    highest = by_block.amax(1)
    pure = (lowest == highest) & valid.all(1)
    full = reached & pure[:, None] & pure[None] & (lowest[:, None] == lowest[None])
    partial = reached & ~full

    def ordered(selected: Tensor) -> tuple[Tensor, Tensor]:
        counts = selected.sum(-1, dtype=torch.int32)
        indices = torch.argsort(~selected, dim=-1, stable=True).to(torch.int32)
        return counts[None, None], indices[None, None]

    kv_num_blocks, kv_indices = ordered(partial)
    full_kv_num_blocks, full_kv_indices = ordered(full)
    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(tokens, tokens),
    )
