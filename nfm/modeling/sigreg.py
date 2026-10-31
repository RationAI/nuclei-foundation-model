import torch
from torch import Tensor, nn


class SIGReg(nn.Module):
    """Sliced Epps-Pulley test pushing embeddings to an isotropic Gaussian.

    Follows the reference implementation in the LeJEPA repository (MINIMAL.md).
    """

    def __init__(self, knots: int = 17, slices: int = 256) -> None:
        super().__init__()
        self.slices = slices
        t = torch.linspace(0, 3, knots)
        weights = torch.full((knots,), 2 * 3 / (knots - 1))
        weights[[0, -1]] = 3 / (knots - 1)
        window = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: Tensor) -> Tensor:
        """Computes the statistic for projections of shape [views, batch, dim]."""
        with torch.autocast(proj.device.type, enabled=False):
            proj = proj.float()
            A = torch.randn(proj.size(-1), self.slices, device=proj.device)
            A = A / A.norm(dim=0)
            x_t = (proj @ A).unsqueeze(-1) * self.t
            err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
            return ((err @ self.weights) * proj.size(-2)).mean()
