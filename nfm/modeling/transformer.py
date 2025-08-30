import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from nfm.configuration import Config
from nfm.modeling.layers import FeedForward, RoPE


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = dropout
        self.head_dim = dim // num_heads

        self.rope = RoPE(self.head_dim, theta=10000)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(
        self, tgt: Tensor, src: Tensor, tgt_pos: Tensor, src_pos: Tensor
    ) -> Tensor:
        q = rearrange(self.q(tgt), "b n (h d) -> b h n d", d=self.head_dim)
        k, v = rearrange(
            self.kv(src), "b n (two h d) -> two b h n d", two=2, d=self.head_dim
        )
        q = self.q_norm(q)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(
            query=self.rope(q, tgt_pos),
            key=self.rope(k, src_pos),
            value=v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        x = rearrange(x, "b h n d -> b n (h d)")

        return self.wo(x)


class CrossLayer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.self_attn = Attention(dim=config.dim, num_heads=config.num_heads)
        self.cross_attn = Attention(dim=config.dim, num_heads=config.num_heads)
        self.ffn = FeedForward(config.dim, config.hidden_dim)

        self.pre_self_attn_norm = nn.RMSNorm(config.dim)
        self.pre_cross_attn_norm = nn.RMSNorm(config.dim)
        self.pre_ffn_norm = nn.RMSNorm(config.dim)

    def forward(
        self, tgt: Tensor, src: Tensor, tgt_pos: Tensor, src_pos: Tensor
    ) -> Tensor:
        y = self.pre_cross_attn_norm(tgt)
        tgt = tgt + self.cross_attn(y, src, tgt_pos, src_pos)

        y = self.pre_self_attn_norm(tgt)
        tgt = tgt + self.self_attn(y, y, tgt_pos, tgt_pos)

        y = self.pre_ffn_norm(tgt)
        return tgt + self.ffn(y)


class Layer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.self_attn = Attention(dim=config.dim, num_heads=config.num_heads)
        self.ffn = FeedForward(config.dim, config.hidden_dim)

        self.pre_self_attn_norm = nn.RMSNorm(config.dim)
        self.pre_ffn_norm = nn.RMSNorm(config.dim)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        y = self.pre_self_attn_norm(x)
        x = x + self.self_attn(y, y, pos, pos)

        y = self.pre_ffn_norm(x)
        return x + self.ffn(y)


class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.cross_layers = nn.ModuleList(
            CrossLayer(config) for _ in range(config.num_cross_layers)
        )
        self.self_layers = nn.ModuleList(
            Layer(config) for _ in range(config.num_self_layers)
        )
        self.cls_token = nn.Parameter(torch.randn(config.dim))
        self.cls_norm = nn.RMSNorm(config.dim)
        self.norm = nn.RMSNorm(config.dim)

    def forward(
        self,
        tgt: Tensor,
        src: Tensor,
        tgt_pos: Tensor,
        src_pos: Tensor,
        local_crops: bool = False,
    ) -> dict[str, Tensor]:
        """Forward pass of the Transformer model.

        Args:
            tgt: Target sequence of shape (b, n, d)
            src: Source sequence of shape (b, m, d)
            tgt_pos: Target positions of shape (b, n, 2)
            src_pos: Source positions of shape (b, m, 2)

        Returns:
            A dictionary containing:
                - "cls_token": The class token of shape (b, d)
                - "patch_tokens": The patch tokens of shape (b, n, d)
        """
        cls_tokens = repeat(self.cls_token, "d -> b 1 d", b=tgt.shape[0])
        tgt = torch.cat((cls_tokens, tgt), dim=1)
        tgt_pos = torch.cat((torch.zeros_like(tgt_pos[:, :1]), tgt_pos), dim=1)

        for layer in self.cross_layers:
            tgt = layer(tgt, src, tgt_pos, src_pos)

        for layer in self.self_layers:
            tgt = layer(tgt, tgt_pos)

        # Final normalization
        patch_tokens = self.norm(tgt[:, 1:])
        if self.training and local_crops:
            cls_token = self.cls_norm(tgt[:, 0])
        else:
            cls_token = self.norm(tgt[:, 0])

        return {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
        }
