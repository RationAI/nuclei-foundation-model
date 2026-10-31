"""Fixed normalization and independent intrinsic/contextual feature paths."""

import torch
from torch import Tensor, nn

from nfm.configuration import Config
from nfm.modeling.transformer import Transformer


class FrozenStandardizer(nn.Module):
    """Streaming population moments, frozen after an explicit calibration phase."""

    def __init__(self, dimensions: int, minimum_scale: float) -> None:
        super().__init__()
        self.minimum_scale = minimum_scale
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("mean", torch.zeros(dimensions, dtype=torch.float64))
        self.register_buffer("m2", torch.zeros(dimensions, dtype=torch.float64))
        self.register_buffer("scale", torch.ones(dimensions))
        self.register_buffer("ready", torch.tensor(False))

    @torch.no_grad()
    def observe(self, values: Tensor) -> None:
        if self.ready:
            raise RuntimeError("Normalization is frozen")
        values = values.detach().double()
        if not len(values) or not torch.isfinite(values).all():
            raise ValueError("Calibration requires nonempty finite features")
        count = len(values)
        mean = values.mean(0)
        m2 = (values - mean).square().sum(0)
        combined = self.count + count
        delta = mean - self.mean
        self.m2.add_(m2 + delta.square() * self.count * count / combined)
        self.mean.add_(delta * count / combined)
        self.count.copy_(combined)

    @torch.no_grad()
    def freeze(self) -> None:
        if self.count < 2:
            raise RuntimeError("At least two observations are required")
        self.scale.copy_((self.m2 / self.count).sqrt().clamp_min(self.minimum_scale))
        self.ready.fill_(True)

    def forward(self, values: Tensor) -> Tensor:
        if not self.ready:
            raise RuntimeError("Run SSL calibration before encoding or validation")
        with torch.autocast(device_type=values.device.type, enabled=False):
            return (values.float() - self.mean.float()) / self.scale


class MorphologyEncoder(nn.Module):
    def __init__(self, config: Config, intrinsic_dim: int) -> None:
        super().__init__()
        self.standardizer = FrozenStandardizer(4 * config.efd_order, 1e-3)
        self.shape_encoder = nn.Sequential(
            nn.Linear(4 * config.efd_order, 256),
            nn.GELU(),
            nn.Linear(256, intrinsic_dim),
        )
        self.context_projection = nn.Linear(intrinsic_dim, config.dim)
        self.mask_token = nn.Parameter(torch.zeros(config.dim))
        self.context_encoder = Transformer(config)

    def contextualize(
        self,
        intrinsic: Tensor,
        positions: Tensor,
        block_mask,
        mask: Tensor | None = None,
    ) -> Tensor:
        tokens = self.context_projection(intrinsic)
        if mask is not None:
            # Shape information is removed before any cross-nucleus operation.
            tokens = torch.where(
                mask[:, None], self.mask_token.to(tokens.dtype), tokens
            )
        return self.context_encoder(tokens[None], positions[None], block_mask).squeeze(
            0
        )

    def forward(self, batch) -> dict[str, Tensor]:
        raw = self.standardizer(batch["efds"])
        intrinsic = self.shape_encoder(raw)
        context = self.contextualize(intrinsic, batch["pos"], batch["block_mask"])
        return {
            "raw": raw,
            "intrinsic": intrinsic,
            "contextual": torch.cat([intrinsic, context], dim=-1),
        }
