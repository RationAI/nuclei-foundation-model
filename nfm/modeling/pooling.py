"""Pooling of nucleus tokens into embeddings of their Voronoi cells.

Cells have different numbers of nuclei, so attention within a cell is computed
with segment operations: a softmax within each cell and sums into cells.
"""

import torch
from torch import Tensor, nn

from nfm.modeling.transformer import FeedForward


def segment_sum(values: Tensor, segments: Tensor, count: int) -> Tensor:
    """Sums of the rows of values in each segment, [count, ...]."""
    return values.new_zeros(count, *values.shape[1:]).index_add_(0, segments, values)


def segment_softmax(scores: Tensor, segments: Tensor, count: int) -> Tensor:
    """Softmax of scores [items, heads] over the items of each segment."""
    index = segments[:, None].expand_as(scores)
    maxima = scores.new_full((count, scores.size(1)), -torch.inf)
    maxima = maxima.scatter_reduce(0, index, scores, "amax")
    weights = (scores - maxima[segments]).exp()
    return weights / segment_sum(weights, segments, count)[segments]


class SimPool(nn.Module):
    """Attention pooling with the cell average as the query.

    The average of the normalized tokens of a cell is the query, the tokens are
    the keys, and the attention weights a generalized mean of the tokens, shifted
    to be non-negative, with the exponent `gamma`.

    Reference: Keep It SimPool, https://arxiv.org/abs/2309.06891
    """

    def __init__(self, dim: int, gamma: float = 1.25) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.gamma = gamma

    def forward(self, tokens: Tensor, cells: Tensor, count: int) -> Tensor:
        """Pools tokens [nuclei, dim] into their cells, [count, dim]."""
        with torch.autocast(tokens.device.type, enabled=False):
            x = self.norm(tokens.float())
            sizes = torch.bincount(cells, minlength=count).clamp_min(1)
            query = self.query(segment_sum(x, cells, count) / sizes[:, None])
            scores = (self.key(x) * query[cells]).sum(-1, keepdim=True)
            weights = segment_softmax(scores * x.size(1) ** -0.5, cells, count)
            lowest = torch.full((count,), torch.inf, device=x.device)
            lowest = lowest.scatter_reduce(0, cells, x.amin(1), "amin")
            values = (x - lowest[cells, None]).clamp_min(1e-6) ** self.gamma
            pooled = segment_sum(weights * values, cells, count)
            return pooled.clamp_min(1e-6) ** (1 / self.gamma)


class AttentionPool(nn.Module):
    """Cross-attention layers in which a query per cell attends to its nuclei.

    The query starts as the cell average plus a learned token, and the queries do
    not attend to each other.
    """

    def __init__(
        self, dim: int, num_heads: int = 12, depth: int = 2, hidden_dim: int = 2048
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.token = nn.Parameter(torch.randn(dim) * 0.02)
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "query_norm": nn.RMSNorm(dim),
                    "key_norm": nn.RMSNorm(dim),
                    "query": nn.Linear(dim, dim, bias=False),
                    "key_value": nn.Linear(dim, 2 * dim, bias=False),
                    "out": nn.Linear(dim, dim, bias=False),
                    "ffn_norm": nn.RMSNorm(dim),
                    "ffn": FeedForward(dim, hidden_dim),
                }
            )
            for _ in range(depth)
        )
        self.norm = nn.RMSNorm(dim)

    def forward(self, tokens: Tensor, cells: Tensor, count: int) -> Tensor:
        """Pools tokens [nuclei, dim] into their cells, [count, dim]."""
        sizes = torch.bincount(cells, minlength=count).clamp_min(1)
        x = segment_sum(tokens, cells, count) / sizes[:, None] + self.token
        for layer in self.layers:
            query = layer["query"](layer["query_norm"](x)).unflatten(
                -1, (self.num_heads, -1)
            )
            key, value = (
                layer["key_value"](layer["key_norm"](tokens))
                .unflatten(-1, (2, self.num_heads, -1))
                .unbind(1)
            )
            scores = (key * query[cells]).sum(-1).float() * key.size(-1) ** -0.5
            weights = segment_softmax(scores, cells, count)  # [nuclei, heads]
            attended = segment_sum(weights[..., None] * value.float(), cells, count)
            x = x + layer["out"](attended.flatten(1).to(x.dtype))
            x = x + layer["ffn"](layer["ffn_norm"](x))
        return self.norm(x)


POOLINGS = {"simpool": SimPool, "attention": AttentionPool}


def pooling(dim: int, kind: str = "simpool", **kwargs: object) -> nn.Module:
    """A pooling head of the given kind, `simpool` or `attention`."""
    return POOLINGS[kind](dim, **kwargs)
