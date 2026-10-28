import torch
from torch import Tensor, nn


class RoVE(nn.Module):
    """N-dimensional Rotary Value Embeddings (RoVE).

    Uses learnable RoPE-Mixed frequencies and applies inverse RoPE
    to the output of attention.

    References:
        - RoVE: https://arxiv.org/abs/2606.11275
        - N-dimensional RoPE: https://arxiv.org/abs/2504.06308
    """

    def __init__(self, dim: int, pos_dim: int = 2, theta: float = 100.0) -> None:
        super().__init__()
        assert dim % 2 == 0, "Feature dimension must be even."

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        angles = 2 * torch.pi * torch.rand_like(freqs)
        self.freqs = nn.Parameter(
            torch.stack((freqs * angles.cos(), freqs * angles.sin()))
        )

    def forward(self, x: Tensor, positions: Tensor, inverse: bool = False) -> Tensor:
        """Apply forward or inverse RoPE.

        Args:
            x: Tensor [b, h, n, d].
            positions: Tensor [b, n, pos_dim].
            inverse: Whether to apply the inverse rotation.

        Returns:
            Rotated tensor [b, h, n, d].
        """
        with torch.autocast(device_type=positions.device.type, enabled=False):
            freqs = positions.float() @ self.freqs.float()

        if inverse:
            freqs = -freqs

        cos = freqs.cos()[:, None]
        sin = freqs.sin()[:, None]
        x_float = x.float()
        real = x_float[..., 0::2]
        imag = x_float[..., 1::2]
        out = torch.stack(
            (real * cos - imag * sin, real * sin + imag * cos), dim=-1
        ).flatten(-2)
        return out.to(x.dtype)
