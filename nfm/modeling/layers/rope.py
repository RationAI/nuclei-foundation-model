import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn.utils.parametrizations import orthogonal


class RoPE(nn.Module):
    """Implements the N dimensional RoPE positional encoding.

    Applies RoPE-Mixed on input under learnable orthogonal transformation P.

    Reference:
        - "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING" (https://arxiv.org/abs/2502.02562)
        - "Rethinking RoPE: A Mathematical Blueprint for N-dimensional Rotary Positional Embedding" (https://arxiv.org/abs/2504.06308)
    """

    def __init__(self, dim: int, pos_dim: int = 2, theta: float = 100.0) -> None:
        """Initialize RoPE module.

        Args:
            dim: The feature dimension of the input tensor. Must be even.
            pos_dim: The dimensionality of the position vectors (e.g., 1 for 1D, 2 for 2D).
            theta: The base value for the RoPE frequency calculation.
        """
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.freqs = nn.Parameter(repeat(freqs, "d -> p d", p=pos_dim).clone())
        self.P = orthogonal(
            nn.Linear(dim, dim, bias=False), orthogonal_map="householder"
        )

    @torch.autocast("cuda", dtype=torch.float32)
    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """Apply RoPE positional encoding.

        Args:
            x ([b, h, n, d]): Input tensor.
            positions ([b, n, pos_dim]): Positions tensor.
        """
        px = self.P(x.float())

        # apply RoPE-Mixed
        freqs = positions.to(self.freqs) @ self.freqs
        freqs_cis = rearrange(
            torch.polar(torch.ones_like(freqs), freqs), "b n c -> b 1 n c"
        )
        px_ = torch.view_as_complex(rearrange(px, "... (d two) -> ... d two", two=2))
        out = rearrange(torch.view_as_real(px_ * freqs_cis), "... d two -> ... (d two)")

        return out.to(x)
