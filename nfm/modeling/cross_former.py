import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

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


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout

        self.pe = CayleySTRING(dim, num_heads)
        self.query = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(
        self, tgt: Tensor, src: Tensor, tgt_coords: Tensor, src_coord: Tensor
    ) -> Tensor:
        q = rearrange(self.query(tgt), "b n (h d) -> b h n d", h=self.num_heads)
        k, v = rearrange(
            self.kv(src), "b n (two h d) -> two b h n d", two=2, h=self.num_heads
        )
        x = F.scaled_dot_product_attention(
            query=self.pe(q, tgt_coords),
            key=self.pe(k, src_coord),
            value=v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        tgt = self.wo(rearrange(x, "b h n d -> b n (h d)"))

        return tgt


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout

        self.pe = CayleySTRING(dim, num_heads)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:
        q, k, v = rearrange(
            self.qkv(x), "b n (three h d) -> three b h n d", three=3, h=self.num_heads
        )
        x = F.scaled_dot_product_attention(
            query=self.pe(q, coords),
            key=self.pe(k, coords),
            value=v,
            dropout_p=self.dropout if self.training else 0.0,
        )

        return self.wo(rearrange(x, "b h n d -> b n (h d)"))


class Layer(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()

        self.cross_attention = CrossAttention(dim, num_heads)
        self.cross_attention_norm = nn.LayerNorm(dim)

        self.self_attention = SelfAttention(dim, num_heads)
        self.self_attention_norm = nn.LayerNorm(dim)

        self.ffn = FeedForward(dim, dim * 4)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(
        self, tgt: Tensor, src: Tensor, tgt_coords: Tensor, src_coords: Tensor
    ) -> Tensor:
        x = self.cross_attention_norm(tgt)
        tgt = tgt + self.cross_attention(x, src, tgt_coords, src_coords)

        x = self.self_attention_norm(tgt)
        tgt = tgt + self.self_attention(x, tgt_coords)

        x = self.ffn_norm(tgt)
        tgt = tgt + self.ffn(x)

        return tgt


class Transformer(nn.Module):
    def __init__(
        self, dim: int, num_layers: int, num_heads: int, num_classes: int
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(Layer(dim, num_heads) for _ in range(num_layers))
        self.class_head = nn.Linear(dim, num_classes)

    def forward(
        self, tgt: Tensor, src: Tensor, tgt_coords: Tensor, src_coords: Tensor
    ) -> dict[str, Tensor]:
        for layer in self.layers:
            tgt = layer(tgt=tgt, src=src, tgt_coords=tgt_coords, src_coords=src_coords)

        return {"logits": self.class_head(tgt)}
