"""Transformer over packed sequences of nuclei tokens."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from nfm.modeling.rove import RoVE


flex_attention = torch.compile(flex_attention, dynamic=True)


def document_mask(lengths: Tensor, block_size: int = 128) -> BlockMask:
    """Full attention within each packed sequence, none across them.

    Args:
        lengths: Tokens per sequence.
        block_size: Block size of the flex attention kernel.
    """
    device = lengths.device
    documents = torch.repeat_interleave(
        torch.arange(len(lengths), device=device), lengths
    )
    tokens = len(documents)
    # The kernel works on whole blocks, the tail of the last one is out of range.
    # A document id of its own keeps the mask in bounds there.
    documents = F.pad(documents, (0, -tokens % block_size), value=len(lengths))
    blocks = documents.view(-1, block_size)
    first, last = blocks[:, 0], blocks[:, -1]
    # Sequences are contiguous, so blocks interact when their sequence ranges do.
    partial = (first[None] <= last[:, None]) & (last[None] >= first[:, None])
    pure = first == last
    full = pure[:, None] & pure[None] & (first[:, None] == first[None])
    partial &= ~full

    def ordered(blocks: Tensor) -> tuple[Tensor, Tensor]:
        counts = blocks.sum(-1, dtype=torch.int32)
        indices = torch.argsort(~blocks, dim=-1, stable=True).to(torch.int32)
        return counts[None, None], indices[None, None]

    def mask_mod(b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
        return documents[q] == documents[kv]

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


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope_theta: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.v_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.rope = RoVE(self.head_dim, theta=rope_theta)

    def forward(self, x: Tensor, positions: Tensor, block_mask: BlockMask) -> Tensor:
        # [tokens, 3 * dim] -> 3 x [1, heads, tokens, head_dim]
        q, k, v = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(1, 2, 0, 3)
        q = self.rope(self.q_norm(q), positions)
        k = self.rope(self.k_norm(k), positions)
        v = self.rope(self.v_norm(v), positions)
        x = flex_attention(q[None], k[None], v[None], block_mask=block_mask)[0]
        x = self.rope(x, positions, inverse=True)
        return self.out(x.transpose(0, 1).flatten(1))


class FeedForward(nn.Module):
    """GeGLU."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.gelu(gate, approximate="tanh") * up)


class Block(nn.Module):
    """Pre- and post-norm residual block, as in Gemma."""

    def __init__(
        self, dim: int, num_heads: int, hidden_dim: int, rope_theta: float
    ) -> None:
        super().__init__()
        self.attn = SelfAttention(dim, num_heads, rope_theta)
        self.ffn = FeedForward(dim, hidden_dim)
        self.pre_attn_norm = nn.RMSNorm(dim)
        self.post_attn_norm = nn.RMSNorm(dim)
        self.pre_ffn_norm = nn.RMSNorm(dim)
        self.post_ffn_norm = nn.RMSNorm(dim)

    def forward(self, x: Tensor, positions: Tensor, block_mask: BlockMask) -> Tensor:
        x = x + self.post_attn_norm(
            self.attn(self.pre_attn_norm(x), positions, block_mask)
        )
        return x + self.post_ffn_norm(self.ffn(self.pre_ffn_norm(x)))


class NucleiTransformer(nn.Module):
    """Encodes views of nuclei, each a class token followed by nucleus tokens.

    Nucleus tokens embed the shape descriptors, positions enter only through RoVE.
    """

    def __init__(
        self,
        in_features: int = 64,
        dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        hidden_dim: int = 1024,
        rope_theta: float = 100.0,
        block_size: int = 128,
        compile: bool = True,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        # Descriptors scale with the nucleus size, which has to stay visible.
        self.input_norm = nn.BatchNorm1d(in_features, affine=False)
        self.embed = nn.Linear(in_features, dim)
        self.cls_token = nn.Parameter(torch.randn(dim) * 0.02)
        self.blocks = nn.ModuleList(
            Block(dim, num_heads, hidden_dim, rope_theta) for _ in range(depth)
        )
        self.norm = nn.RMSNorm(dim)
        if compile:
            # Fusing the norms, rotations and activations halves the memory.
            for block in self.blocks:
                block.compile(dynamic=True)

    def forward(
        self,
        features: Tensor,
        positions: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """Encodes packed views.

        Args:
            features: Shape descriptors, [tokens, in_features]. Rows of the class
                tokens are ignored.
            positions: Positions in micrometers, [tokens, 2].
            lengths: Tokens per view, the first token of each is the class token.

        Returns:
            Output tokens, [tokens, dim].
        """
        cls = cls_indices(lengths)
        nuclei = torch.ones(len(features), dtype=torch.bool, device=features.device)
        nuclei[cls] = False
        x = self.embed(self.input_norm(features[nuclei]))
        tokens = x.new_empty(len(features), x.size(1))
        tokens[nuclei] = x
        tokens[cls] = self.cls_token.to(x.dtype)
        tokens = tokens.float()  # fp32 residual stream
        block_mask = document_mask(lengths, self.block_size)
        for block in self.blocks:
            tokens = block(tokens, positions, block_mask)
        return self.norm(tokens)


def cls_indices(lengths: Tensor) -> Tensor:
    """Index of the class token of each packed view."""
    return torch.cumsum(lengths, 0) - lengths
