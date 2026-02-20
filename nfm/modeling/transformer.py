import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torchvision.ops import MLP

from nfm.configuration import Config
from nfm.modeling.layers import FeedForward, RoPE


class SelfAttention(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, rope_theta: float, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.head_dim = dim // num_heads

        self.rope = RoPE(self.head_dim, theta=rope_theta)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        q, k, v = rearrange(
            self.qkv(x), "b n (three h d) -> three b h n d", three=3, d=self.head_dim
        )
        q = self.q_norm(q)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(
            query=self.rope(q, pos),
            key=self.rope(k, pos),
            value=v,
            dropout_p=self.dropout if self.training else 0.0,
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

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        y = self.pre_self_attn_norm(x)
        x = x + self.self_attn(y, pos)

        y = self.pre_ffn_norm(x)
        return x + self.ffn(y)


class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.bn = nn.BatchNorm1d(4 * config.efd_order, affine=False)
        self.polygon_proj = nn.Linear(4 * config.efd_order, config.dim)

        self.self_layers = nn.ModuleList(
            Layer(config) for _ in range(config.num_layers)
        )
        self.norm = nn.RMSNorm(config.dim)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """Forward pass of the Transformer model.

        Args:
            x: Target sequence of shape (b, n, d)
            pos: Target positions of shape (b, n, 2)
        """
        # Ignore zero tokens as they are padded polygons
        x_flatten = x.flatten(0, 1)
        non_zero = x_flatten.abs().sum(dim=-1) != 0
        if non_zero.any():
            x_flatten[non_zero] = self.bn(x_flatten[non_zero])

        x = self.polygon_proj(x)

        for layer in self.self_layers:
            x = layer(x, pos)

        return self.norm(x)


class NucleiGraphEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.backbone = Transformer(config)

        self.proj = MLP(
            config.dim,
            hidden_channels=[
                config.proj_hidden_dim,
                config.proj_hidden_dim,
                config.proj_dim,
            ],
            norm_layer=nn.BatchNorm1d,
        )
        self.final_norm = nn.BatchNorm1d(config.proj_dim, affine=False)

    def forward(self, x: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the Transformer model.

        Args:
            x: Target sequence of shape (b, n, d)
            pos: Target positions of shape (b, n, 2)
        """
        embed = self.backbone(x, pos)
        # 2. Global Average Pooling
        return embed, self.final_norm(self.proj(embed.mean(dim=1)))
