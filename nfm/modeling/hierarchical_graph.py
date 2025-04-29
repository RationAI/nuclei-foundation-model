import math

import torch
from timm.layers.drop import DropPath
from torch import Tensor, nn


class MLP(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act_layer: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        assert num_layers > 1

        layers = []
        h = [hidden_dim] * (num_layers - 1)
        for n, k in zip([input_dim, *h], h, strict=False):
            layers.append(nn.Linear(n, k))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        super().__init__(*layers)


class GraphMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim * 2
        self.proj = nn.Linear(dim, self.dim, bias=False)
        self.norm = nn.LayerNorm(self.dim)

    def forward(
        self, x: Tensor, coords: Tensor, indices: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Forward function for Graph Merging.

        Args:
            x ([b, subgraph_size, dim]): Hidden states.
            coords ([b, subgraph_size, 2]): Absolute reference coords.
            indices ([b]): Indices for merging. Each group must have the same size.
        """
        x = self.norm(self.proj(x).mean(dim=1))
        coords = coords.mean(dim=1)

        # group by indices
        masks = indices == torch.unique(indices).unsqueeze(1)
        x = x[None].expand(len(masks), -1, -1)[masks].view(len(masks), -1, self.dim)
        coords = coords[None].expand(len(masks), -1, -1)[masks].view(len(masks), -1, 2)

        return x, coords


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        self.mha = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(dim)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        self.ffn = MLP(dim, dim * mlp_ratio, dim, 2, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:
        """Forward function for Window Self-Attention.

        Args:
            x ([b, n, c]): Hidden states.
            coords ([b, n, 2]): Absolute reference coords.
            mask: Attention mask.
        """
        # Self-Attention
        rel_coords = coords[:, None] - coords[:, :, None]
        bias = self.cpb_mlp(rel_coords)  # (b, n, num_heads)
        bias = bias.permute(0, 3, 1, 2).flatten(0, 1)
        tgt = self.mha(x, x, x, attn_mask=bias)[0]
        x = x + self.drop_path(self.attention_norm(tgt))

        # FFN
        x = x + self.drop_path(self.ffn_norm(self.ffn(x)))
        return x


class Stage(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.grou_merging = GraphMerging(dim)
        self.blocks = nn.ModuleList(
            Block(dim, num_heads, dropout=dropout) for _ in range(depth)
        )

    def forward(self, x: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
        if self.downsample is not None:
            x, coords = self.downsample(x, coords)

        for block in self.blocks:
            x = block(x, coords)
        return x, coords


class HGraphTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        depths: list[int],
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.stages = nn.ModuleList(
            Stage(
                dim=dim * 2**i,
                num_heads=num_heads,
                depth=depth,
                downsample=None if i == 0 else GraphMerging(dim * 2 ** (i - 1)),
                dropout=dropout,
            )
            for i, depth in enumerate(depths)
        )

        self.class_head = nn.ModuleList(
            nn.Linear(dim * 2**i, num_classes) for i in range(len(depths))
        )
        self.init_weights()

    def init_weights(self) -> None:
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for head in self.class_head:
            nn.init.constant_(head.bias, bias_value)

    def forward(
        self, x: Tensor, coords: Tensor
    ) -> dict[str, Tensor | list[dict[str, Tensor]]]:
        logits_list: list[Tensor] = []

        for i, stage in enumerate(self.stages):
            x = stage(x, coords)
            logits_list.append(self.class_head[i](x))

        return {
            f"stage_{i}": [{"logits": logits}] for i, logits in enumerate(logits_list)
        }
