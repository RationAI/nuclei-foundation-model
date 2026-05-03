import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torchvision.ops import MLP

from nfm.configuration import Config
from nfm.modeling.layers import FeedForward, RoPE


flex_attention = torch.compile(flex_attention)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope_theta: float) -> None:
        super().__init__()
        self.head_dim = dim // num_heads

        self.rope = RoPE(self.head_dim, theta=rope_theta)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.v_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)

    def forward(self, x: Tensor, pos: Tensor, block_mask: BlockMask) -> Tensor:
        q, k, v = rearrange(
            self.qkv(x), "b n (three h d) -> three b h n d", three=3, d=self.head_dim
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        x = flex_attention(
            query=self.rope(q, pos),
            key=self.rope(k, pos),
            value=v,
            block_mask=block_mask,
        )
        x = rearrange(x, "b h n d -> b n (h d)")

        return self.wo(x)


class Layer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.attn = SelfAttention(
            dim=config.dim, num_heads=config.num_heads, rope_theta=config.rope_theta
        )
        self.ffn = FeedForward(config.dim, config.hidden_dim)

        self.pre_attn_norm = nn.RMSNorm(config.dim)
        self.pre_ffn_norm = nn.RMSNorm(config.dim)
        self.post_attn_norm = nn.RMSNorm(config.dim)
        self.post_ffn_norm = nn.RMSNorm(config.dim)

    def forward(self, x: Tensor, pos: Tensor, block_mask: BlockMask) -> Tensor:
        y = self.pre_attn_norm(x)
        x = x + self.post_attn_norm(self.attn(y, pos, block_mask))

        y = self.pre_ffn_norm(x)
        return x + self.post_ffn_norm(self.ffn(y))


class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Layer(config) for _ in range(config.num_layers))
        self.norm = nn.RMSNorm(config.dim)

    def forward(self, x: Tensor, pos: Tensor, block_mask: BlockMask) -> Tensor:
        """Forward pass of the Transformer model.

        Args:
            x: Target sequence of shape (b, n, d)
            pos: Target positions of shape (b, n, 2)
            block_mask: Block mask for attention
        """
        for layer in self.layers:
            x = layer(x, pos, block_mask)

        return self.norm(x)


class NFM(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4 * config.efd_order, affine=False)
        self.backbone = Transformer(config)
        self.polygon_proj = nn.Linear(4 * config.efd_order, config.dim)
        self.position_encoder = MLP(
            2, [config.dim, config.dim], activation_layer=nn.SiLU
        )

    def forward(
        self, x: Tensor, pos: Tensor, block_mask: BlockMask, seq_len: int
    ) -> Tensor:
        """Forward pass of the Transformer model.

        Args:
            x: Target sequence of shape (n, d)
            pos: Target positions of shape (n, 2)
            block_mask: Block mask for attention
        """
        normalized = torch.zeros_like(x)
        normalized[:seq_len] = self.bn(x[:seq_len])

        # x = self.polygon_proj(normalized) + self.position_encoder(pos / 10000)
        x = self.position_encoder(pos / 1000)
        return self.backbone(x[None], pos[None], block_mask).squeeze(0)
