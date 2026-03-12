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

    def forward(self, x: Tensor, pos: Tensor, block_mask: BlockMask) -> Tensor:
        q, k, v = rearrange(
            self.qkv(x), "b n (three h d) -> three b h n d", three=3, d=self.head_dim
        )
        q = self.q_norm(q)
        k = self.k_norm(k)

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

        self.self_attn = SelfAttention(
            dim=config.dim, num_heads=config.num_heads, rope_theta=config.rope_theta
        )
        self.ffn = FeedForward(config.dim, config.hidden_dim)

        self.pre_self_attn_norm = nn.RMSNorm(config.dim)
        self.pre_ffn_norm = nn.RMSNorm(config.dim)

    def forward(self, x: Tensor, pos: Tensor, block_mask: BlockMask) -> Tensor:
        y = self.pre_self_attn_norm(x)
        x = x + self.self_attn(y, pos, block_mask)

        y = self.pre_ffn_norm(x)
        return x + self.ffn(y)


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

        self.proj = MLP(
            config.dim,
            hidden_channels=[
                config.proj_hidden_dim,
                config.proj_hidden_dim,
                config.proj_dim,
            ],
            norm_layer=nn.BatchNorm1d,
        )

    def forward(
        self, x: Tensor, pos: Tensor, block_mask: BlockMask
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the Transformer model.

        Args:
            x: Target sequence of shape (b, n, d)
            pos: Target positions of shape (b, n, 2)
            block_mask: Block mask for attention
        """
        x = self.bn(x.flatten(0, 1)).view_as(x)
        x = self.polygon_proj(x)

        embed = self.backbone(x, pos, block_mask)

        # 2. Global Average Pooling
        return embed, self.proj(embed.mean(dim=1))
