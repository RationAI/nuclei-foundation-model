import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from nfm.configuration import Config
from nfm.modeling.layers import CayleySTRING, FeedForward


@torch.autocast("cuda", enabled=False)
def relative_to_absolute_pos(pos: Tensor, step_x: float, step_y: float) -> Tensor:
    pos = pos.sigmoid()
    h, w = pos.shape[1:3]

    anchor_x = torch.arange(w, dtype=torch.float32, device=pos.device) * step_x
    anchor_y = torch.arange(h, dtype=torch.float32, device=pos.device) * step_y

    absolute_x = pos[..., 0] * step_x + anchor_x
    absolute_y = pos[..., 1] * step_y + anchor_y.unsqueeze(1)
    return torch.stack((absolute_x, absolute_y), dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = dropout
        self.head_dim = dim // num_heads

        self.rope = CayleySTRING(self.head_dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(
        self, tgt: Tensor, src: Tensor, tgt_pos: Tensor, src_pos: Tensor
    ) -> Tensor:
        q = rearrange(self.q(tgt), "b n (h d) -> b h n d", d=self.head_dim)
        k, v = rearrange(
            self.kv(src), "b n (two h d) -> two b h n d", two=2, d=self.head_dim
        )

        x = F.scaled_dot_product_attention(
            query=self.rope(q, tgt_pos),
            key=self.rope(k, src_pos),
            value=v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        x = rearrange(x, "b h n d -> b n (h d)")

        return self.wo(x)


class Layer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.self_attn = Attention(dim=config.dim, num_heads=config.num_heads)
        self.cross_attn = Attention(dim=config.dim, num_heads=config.num_heads)
        self.ffn = FeedForward(config.dim, config.dim * 4)

        self.pre_self_attn_norm = nn.RMSNorm(config.dim)
        self.pre_cross_attn_norm = nn.RMSNorm(config.dim)
        self.pre_ffn_norm = nn.RMSNorm(config.dim)
        self.post_self_attn_norm = nn.RMSNorm(config.dim)
        self.post_cross_attn_norm = nn.RMSNorm(config.dim)
        self.post_ffn_norm = nn.RMSNorm(config.dim)

    def forward(
        self, tgt: Tensor, src: Tensor, tgt_pos: Tensor, src_pos: Tensor
    ) -> Tensor:
        y = self.pre_cross_attn_norm(tgt)
        tgt = tgt + self.post_cross_attn_norm(self.cross_attn(y, src, tgt_pos, src_pos))

        y = self.pre_self_attn_norm(tgt)
        tgt = tgt + self.post_self_attn_norm(self.self_attn(y, y, tgt_pos, tgt_pos))

        y = self.pre_ffn_norm(tgt)
        return tgt + self.post_ffn_norm(self.ffn(y))


class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.layers = nn.ModuleList(Layer(config) for _ in range(config.num_layers))
        self.class_head = nn.Linear(config.dim, config.num_classes)

    def forward(
        self, tgt: Tensor, src: Tensor, tgt_pos: Tensor, src_pos: Tensor
    ) -> dict[str, Tensor]:
        for layer in self.layers:
            tgt = layer(tgt, src, tgt_pos, src_pos)

        return {"logits": self.class_head(tgt)}
