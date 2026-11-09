import torch
from torch import Tensor, nn


class RoVE(nn.Module):
    """2D Rotary Value Embeddings with learnable mixed frequencies.

    Rotates queries, keys and values by their positions, and the attention
    output back by the query position, so the output is relative as well.

    References:
        - RoVE: https://arxiv.org/abs/2606.11275
        - RoPE-Mixed: https://arxiv.org/abs/2403.13298
    """

    def __init__(self, dim: int, theta: float = 100.0) -> None:
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        angles = 2 * torch.pi * torch.rand_like(freqs)
        self.freqs = nn.Parameter(
            torch.stack((freqs * angles.cos(), freqs * angles.sin()))
        )  # [2, dim / 2], one frequency vector per rotated pair

    def forward(self, x: Tensor, positions: Tensor, inverse: bool = False) -> Tensor:
        """Rotates x [..., tokens, dim] by positions [tokens, 2]."""
        with torch.autocast(positions.device.type, enabled=False):
            angles = positions.float() @ self.freqs.float()  # [tokens, dim / 2]
            if inverse:
                angles = -angles
            cos, sin = angles.cos(), angles.sin()
            real, imag = x.float()[..., 0::2], x.float()[..., 1::2]
            out = torch.stack((real * cos - imag * sin, real * sin + imag * cos), -1)
        return out.flatten(-2).to(x.dtype)
