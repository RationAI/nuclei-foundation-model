"""Transformer over packed views of nuclei."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.utils.checkpoint import checkpoint

from nfm.modeling.block_mask import neighborhood_mask
from nfm.modeling.rove import RoVE


flex_attention = torch.compile(flex_attention, dynamic=True)


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
        # Norms return float32 under autocast, attention runs in the dtype of the
        # projection, bfloat16 under autocast.
        dtype = q.dtype
        q = self.rope(self.q_norm(q), positions).to(dtype)
        k = self.rope(self.k_norm(k), positions).to(dtype)
        v = self.rope(self.v_norm(v), positions).to(dtype)
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
    """Pre- and post-norm residual block, as in Gemma.

    With `checkpointed`, only the input is kept for the backward pass, and the
    block runs again to recompute the rest.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        rope_theta: float,
        checkpointed: bool = False,
    ) -> None:
        super().__init__()
        self.checkpointed = checkpointed
        self.attn = SelfAttention(dim, num_heads, rope_theta)
        self.ffn = FeedForward(dim, hidden_dim)
        self.pre_attn_norm = nn.RMSNorm(dim)
        self.post_attn_norm = nn.RMSNorm(dim)
        self.pre_ffn_norm = nn.RMSNorm(dim)
        self.post_ffn_norm = nn.RMSNorm(dim)

    def forward(self, x: Tensor, positions: Tensor, block_mask: BlockMask) -> Tensor:
        if self.checkpointed and torch.is_grad_enabled():
            return checkpoint(
                self.residual, x, positions, block_mask, use_reentrant=False
            )
        return self.residual(x, positions, block_mask)

    def residual(self, x: Tensor, positions: Tensor, block_mask: BlockMask) -> Tensor:
        x = x + self.post_attn_norm(
            self.attn(self.pre_attn_norm(x), positions, block_mask)
        )
        return x + self.post_ffn_norm(self.ffn(self.pre_ffn_norm(x)))


class NucleiTransformer(nn.Module):
    """Encodes views of nuclei with neighborhood attention, see `neighborhood_mask`.

    Nucleus tokens embed the shape features, and positions enter only through RoVE.
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
        mlp_embed: bool = True,
        checkpoint_every: int = 0,
        compile: bool = True,
    ) -> None:
        """Initializes the transformer.

        Args:
            in_features: Shape features per nucleus.
            dim: Token dimension.
            depth: Number of blocks.
            num_heads: Attention heads per block.
            hidden_dim: Hidden dimension of the GeGLU feed-forward.
            rope_theta: Base of the RoVE frequencies, positions are in micrometers.
            block_size: Block size of the flex attention kernel.
            mlp_embed: Embeds the shape features with an MLP, else linearly.
            checkpoint_every: Activation checkpointing of every n-th block, which
                then keeps only its input for the backward pass. 1 checkpoints all
                blocks, 0 none.
            compile: Compiles the blocks.
        """
        super().__init__()
        self.block_size = block_size
        # Shape features scale with the nucleus size, which has to stay visible.
        self.input_norm = nn.BatchNorm1d(in_features, affine=False)
        self.embed = (
            nn.Sequential(nn.Linear(in_features, dim), nn.GELU(), nn.Linear(dim, dim))
            if mlp_embed
            else nn.Linear(in_features, dim)
        )
        self.blocks = nn.ModuleList(
            Block(
                dim,
                num_heads,
                hidden_dim,
                rope_theta,
                checkpointed=checkpoint_every > 0 and i % checkpoint_every == 0,
            )
            for i in range(depth)
        )
        self.norm = nn.RMSNorm(dim)
        if compile:
            # Fusing the norms, rotations and activations halves the memory.
            for block in self.blocks:
                block.compile(dynamic=True)

    def forward(
        self, features: Tensor, positions: Tensor, lengths: Tensor, neighbors: Tensor
    ) -> Tensor:
        """Encodes packed views.

        Args:
            features: Shape features of the nuclei, [nuclei, in_features].
            positions: Positions of the nuclei in micrometers, [nuclei, 2].
            lengths: Nuclei per view, [views].
            neighbors: Nearest nuclei of each nucleus, [nuclei, k].

        Returns:
            Output nucleus tokens, [nuclei, dim].
        """
        tokens = self.embed(self.input_norm(features)).float()  # fp32 residual stream
        block_mask = neighborhood_mask(lengths, neighbors, self.block_size)
        for block in self.blocks:
            tokens = block(tokens, positions, block_mask)
        return self.norm(tokens)
