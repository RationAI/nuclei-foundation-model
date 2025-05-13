import math

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn.utils import parametrize


def init_freqs(head_dim: int, num_heads: int, pos_dim: int, theta: float) -> Tensor:
    """Taken from https://github.com/naver-ai/rope-vit/blob/main/self-attn/rope_self_attn.py."""
    freqs_x = []
    freqs_y = []
    freqs = 1 / (theta ** (torch.arange(0, head_dim, 2 * pos_dim).float() / head_dim))
    for _ in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi
        fx = torch.cat(
            [freqs * torch.cos(angles), freqs * torch.cos(torch.pi / 2 + angles)],
            dim=-1,
        )
        fy = torch.cat(
            [freqs * torch.sin(angles), freqs * torch.sin(torch.pi / 2 + angles)],
            dim=-1,
        )
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    return torch.stack([freqs_x, freqs_y], dim=0)


class Skew(nn.Module):
    """Skew-symmetric matrix parameterization."""

    def forward(self, x: Tensor) -> Tensor:
        a = x.triu(1)
        return a - a.transpose(-1, -2)

    def right_inverse(self, x: Tensor) -> Tensor:
        return x.triu(1)


class CayleySTRING(nn.Module):
    """Implements the Cayley-STRING positional encoding.

    Based on "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING"
    (https://arxiv.org/abs/2502.02562).

    Applies RoPE followed by multiplication with a learnable orthogonal matrix P
    parameterized by the Cayley transform: P = (I - S)(I + S)^-1, where S is
    a learnable skew-symmetric matrix.

    Args:
        dim (int): The feature dimension of the input tensor. Must be even.
        max_seq_len (int): The maximum sequence length.
        base (int): The base value for the RoPE frequency calculation. Defaults to 10000.
        pos_dim (int): The dimensionality of the position vectors (e.g., 1 for 1D, 2 for 2D). Defaults to 1.
    """

    def __init__(
        self, dim: int, num_heads: int, pos_dim: int = 2, theta: float = 100.0
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "Dimension must be divisible by num_heads."

        head_dim = dim // num_heads

        self.freqs = nn.Parameter(init_freqs(head_dim, num_heads, pos_dim, theta))

        self.S = nn.Parameter(torch.zeros(head_dim, head_dim))
        parametrize.register_parametrization(self, "S", Skew())

        self.register_buffer("I", torch.eye(head_dim), persistent=False)

        self.init_weights()

    def init_weights(self) -> None:
        self.S = nn.init.kaiming_uniform_(self.S, a=math.sqrt(5))

    @parametrize.cached()
    @torch.autocast("cuda", enabled=False)
    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """Apply Cayley-STRING positional encoding.

        Args:
            x ([b, h, n, d]): Input tensor.
            positions ([b, n, pos_dim]): Positions tensor.
        """
        # Compute (I + S)^-1 @ x
        y = torch.linalg.solve(
            self.I + self.S, rearrange(x.float(), "b h n d -> h d (b n)")
        )

        # change of basis
        px = torch.matmul(self.I - self.S, y)
        px = rearrange(px, "h d (b n) -> b h n d", b=x.size(0)).contiguous()

        # apply RoPE-Mixed
        angles = torch.einsum("bnk,khc->bhnc", positions, self.freqs)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        px_ = torch.view_as_complex(rearrange(px, "... (d two) -> ... d two", two=2))
        out = rearrange(torch.view_as_real(px_ * freqs_cis), "... d two -> ... (d two)")

        return out.type_as(x)
