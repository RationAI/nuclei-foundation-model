"""Nucleus agreement with correctly oriented distribution regularizers."""

import lejepa
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NucleusObjective(nn.Module):
    def __init__(
        self,
        regularizer: str = "sigreg",
        weight: float | None = None,
        slices: int = 256,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if regularizer not in {"sigreg", "visreg", "vicreg"}:
            raise ValueError(f"Unknown regularizer: {regularizer}")
        self.kind = regularizer
        self.weight = (
            (1.0 if regularizer == "vicreg" else 0.1) if weight is None else weight
        )
        if self.weight < 0 or slices < 1 or eps <= 0:
            raise ValueError("Invalid regularizer parameters")
        self.slices = slices
        self.eps = eps
        self.sigreg = lejepa.multivariate.SlicingUnivariateTest(
            lejepa.univariate.EppsPulley(n_points=17), num_slices=slices
        )

    def forward(self, paired: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Accept [2, matched nuclei, dimensions], or the equal-count 4D form."""
        if paired.ndim not in {3, 4} or paired.shape[0] != 2:
            raise ValueError(
                "Expected [2, samples, dimensions] (optionally with a region axis)"
            )
        with torch.autocast(device_type=paired.device.type, enabled=False):
            z = paired.float().reshape(2, -1, paired.shape[-1])
            if z.shape[1] < 2:
                # No meaningful population test is possible. Keep the graph
                # connected for backward/DDP and let the continuous probe train.
                zero = paired.sum() * 0
                return zero, {
                    "alignment": zero,
                    "regularizer": zero,
                    "ssl_skipped": zero.detach() + 1,
                }
            alignment = F.mse_loss(z[0], z[1])
            centered = z - z.mean(1, keepdim=True)
            variance = centered.square().mean(1)
            std = (variance + self.eps).sqrt()
            components = {}
            if self.kind == "sigreg":
                regularization = self.sigreg(z)
            elif self.kind == "visreg":
                directions = torch.randn(z.shape[-1], self.slices, device=z.device)
                directions = F.normalize(directions, dim=0)
                projected = (centered / std.detach()[:, None]) @ directions
                quantiles = torch.arange(1, z.shape[1] + 1, device=z.device) / (
                    z.shape[1] + 1
                )
                normal = torch.distributions.Normal(0.0, 1.0)
                target = normal.icdf(quantiles)[None, :, None]
                components = {
                    "scale": (std - 1).square().mean(),
                    "shape": (projected.sort(dim=1).values - target).square().mean(),
                    "center": z.mean(1).square().mean(),
                }
                regularization = sum(components.values())
            else:
                # Conventional VICReg ratios, scaled so alignment has weight one.
                covariance = centered.transpose(1, 2) @ centered / (z.shape[1] - 1)
                diagonal = torch.diag_embed(covariance.diagonal(dim1=-2, dim2=-1))
                components = {
                    "variance": F.relu(1 - std).mean(),
                    "covariance": (covariance - diagonal).square().sum((-2, -1)).mean()
                    / z.shape[-1],
                }
                regularization = components["variance"] + components["covariance"] / 25
            loss = alignment + self.weight * regularization
        return loss, {
            "alignment": alignment,
            "regularizer": regularization,
            "ssl_skipped": loss.detach().new_zeros(()),
            **components,
        }
