"""Neighborhood attention over packed views, with local class tokens.

The sequence holds the nuclei of all views, each view sorted so that a block of
`block_size` tokens is a compact patch of tissue, followed by the class tokens.
A query attends to a key only within the same view, and

- a nucleus to the nuclei in every block holding a nearest neighbor of a
  nucleus in its own block,
- a class token to the nuclei in the blocks holding the nuclei it covers,
- a nucleus to the class tokens that cover a nucleus in its block,
- a class token only to itself, not to other class tokens.

Neighborhoods are thus quantized to blocks: pairs of blocks of nuclei of one
view are full blocks, so the token-level mask is evaluated only for blocks with
class tokens or with nuclei of two views.
"""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


def neighborhood_mask(
    lengths: Tensor,
    neighbors: Tensor,
    cls_lengths: Tensor,
    cls_neighbors: Tensor,
    block_size: int = 128,
) -> BlockMask:
    """The block mask for nuclei followed by class tokens.

    Args:
        lengths: Nuclei per view, [views].
        neighbors: Indices of the nearest nuclei of each nucleus, [nuclei, k].
        cls_lengths: Class tokens per view, [views].
        cls_neighbors: Indices of the nuclei each class token covers, [cls, k].
        block_size: Block size of the flex attention kernel.
    """
    device = neighbors.device
    nuclei, classes = len(neighbors), len(cls_neighbors)
    tokens = nuclei + classes
    blocks = -(-tokens // block_size)
    padded = blocks * block_size
    views = torch.arange(len(lengths), device=device)
    documents = torch.full((padded,), -1, dtype=torch.long, device=device)
    documents[:nuclei] = torch.repeat_interleave(views, lengths)
    documents[nuclei:tokens] = torch.repeat_interleave(views, cls_lengths)
    # Class tokens come after the nuclei, so a tensor of the count tells them
    # apart without a lookup, and keeps the mask the same for every batch.
    first_cls = torch.tensor(nuclei, device=device)

    # Blocks of nuclei reached by each nucleus block and by each class token.
    nuclei_blocks = torch.zeros(blocks, blocks, dtype=torch.bool, device=device)
    rows = (torch.arange(nuclei, device=device) // block_size)[:, None]
    nuclei_blocks[rows.expand_as(neighbors), neighbors // block_size] = True
    # A row for every position after the nuclei, padding included.
    rows = max(padded - nuclei, 1)
    cls_blocks = torch.zeros(rows, blocks, dtype=torch.bool, device=device)
    rows = torch.arange(classes, device=device)[:, None]
    cls_blocks[rows.expand_as(cls_neighbors), cls_neighbors // block_size] = True

    # Evaluated for every pair of a partial block, so it uses few lookups: one
    # for the coverage of a class token, in either direction, one for nuclei.
    def mask_mod(b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
        q_block, kv_block = q // block_size, kv // block_size
        q_cls, kv_cls = q >= first_cls, kv >= first_cls
        token = (torch.where(q_cls, q, kv) - first_cls).clamp(min=0)
        covered = cls_blocks[token, torch.where(q_cls, kv_block, q_block)]
        allowed = torch.where(
            q_cls == kv_cls,
            torch.where(q_cls, q == kv, nuclei_blocks[q_block, kv_block]),
            covered,
        )
        return (documents[q] == documents[kv]) & allowed

    # Block pairs with any allowed token pair, a superset is enough.
    reached = nuclei_blocks.clone()
    cls_block = (nuclei + torch.arange(classes, device=device)) // block_size
    covered, block = cls_blocks[:classes].nonzero(as_tuple=True)
    reached[cls_block[covered], block] = True
    reached[block, cls_block[covered]] = True
    reached[cls_block, cls_block] = True
    by_block = documents.view(blocks, block_size)
    valid = by_block >= 0
    lowest = torch.where(valid, by_block, len(lengths)).amin(1)
    highest = by_block.amax(1)
    reached &= (lowest[None] <= highest[:, None]) & (highest[None] >= lowest[:, None])

    # Full blocks: nuclei of a single view on both sides, all pairs allowed.
    has_cls = torch.arange(blocks, device=device) >= nuclei // block_size
    pure = (lowest == highest) & valid.all(1) & ~has_cls
    full = (
        nuclei_blocks & pure[:, None] & pure[None] & (lowest[:, None] == lowest[None])
    )
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
