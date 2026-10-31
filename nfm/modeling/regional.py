"""A visible-cell pooler and a small spatial transformer over physical tiles."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from nfm.modeling.morphology import FrozenStandardizer


class RegionalLayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.output = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        batch, length, dim = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .view(batch, length, 3, self.num_heads, dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        attended = (
            F.scaled_dot_product_attention(q, k, v)
            .transpose(1, 2)
            .reshape(batch, length, dim)
        )
        x = x + self.output(attended)
        return x + self.ffn(self.norm2(x))


class RegionalEncoder(nn.Module):
    def __init__(
        self,
        efd_order=16,
        intrinsic_dim=128,
        dim=256,
        hidden_dim=1024,
        num_heads=8,
        num_layers=4,
        grid_size=8,
        tile_size_um=64,
        fine_bins=8,
        inference_window_batch=32,
    ):
        super().__init__()
        if dim % 4 or dim % num_heads or inference_window_batch < 1:
            raise ValueError("Invalid attention dimensions or inference batch size")
        self.grid_size, self.tile_size_um = grid_size, tile_size_um
        self.fine_bins, self.dim = fine_bins, dim
        self.inference_window_batch = inference_window_batch
        self.standardizer = FrozenStandardizer(4 * efd_order, 1e-3)
        self.shape_encoder = nn.Sequential(
            nn.Linear(4 * efd_order, 256), nn.GELU(), nn.Linear(256, intrinsic_dim)
        )
        # Include the nucleus's own raw shape as well as its tile summary.
        self.summary_dim = 12 * efd_order + fine_bins**2 + 1
        self.patch_projection = nn.Sequential(
            nn.Linear(2 * intrinsic_dim + fine_bins**2 + 1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.layers = nn.ModuleList(
            [RegionalLayer(dim, hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        axis = (torch.arange(grid_size).float() + 0.5) * tile_size_um
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        centers = torch.stack([xx, yy], -1).reshape(-1, 2)
        periods = torch.logspace(
            math.log10(2 * tile_size_um),
            math.log10(4 * grid_size * tile_size_um),
            dim // 4,
        )
        phases = centers[..., None] * (2 * math.pi / periods)
        self.register_buffer(
            "position_embedding", torch.cat([phases.sin(), phases.cos()], -1).flatten(1)
        )
        self.register_buffer("centers", centers)

    def pool(self, raw, intrinsic, positions, window_ids, windows):
        g, r = self.grid_size, self.fine_bins
        tile_xy = torch.floor(positions / self.tile_size_um).long().clamp(0, g - 1)
        patch_ids = window_ids * (g * g) + tile_xy[:, 1] * g + tile_xy[:, 0]
        fine_xy = (
            torch.floor((positions / self.tile_size_um - tile_xy) * r)
            .long()
            .clamp(0, r - 1)
        )
        bin_ids = patch_ids * (r * r) + fine_xy[:, 1] * r + fine_xy[:, 0]
        counts = torch.bincount(patch_ids, minlength=windows * g * g).float()
        maps = (
            torch.bincount(bin_ids, minlength=windows * g * g * r * r)
            .float()
            .view(-1, r * r)
        )

        def moments(values):
            values = values.float()
            total = values.new_zeros(windows * g * g, values.shape[-1])
            squares = torch.zeros_like(total)
            total.index_add_(0, patch_ids, values)
            squares.index_add_(0, patch_ids, values.square())
            mean = total / counts.clamp_min(1)[:, None]
            variance = (
                squares / counts.clamp_min(1)[:, None] - mean.square()
            ).clamp_min(0)
            return torch.cat([mean, (variance + 1e-6).sqrt()], -1)

        geometry = torch.cat([maps.log1p(), counts[:, None].log1p()], -1)
        tokens = torch.cat([moments(intrinsic), geometry], -1).view(windows, g * g, -1)
        summary = torch.cat([moments(raw), geometry], -1).view(windows, g * g, -1)
        return tokens, summary, patch_ids, maps.view(windows, g * g, r, r)

    def encode_windows(self, inputs):
        """Only the visible input dictionary is accepted; targets are separate."""
        raw = self.standardizer(inputs["efds"])
        intrinsic = self.shape_encoder(raw)
        mask = inputs["mask"]
        if mask.shape[1] != self.grid_size**2:
            raise ValueError("Dataset and encoder grids differ")
        tokens, summary, patch_ids, maps = self.pool(
            raw, intrinsic, inputs["pos"], inputs["window_ids"], len(mask)
        )
        if mask.flatten()[patch_ids].any():
            raise ValueError("Hidden cells must be removed from the encoder inputs")
        tokens = self.patch_projection(tokens)
        tokens = torch.where(mask[..., None], self.mask_token.to(tokens.dtype), tokens)
        tokens = tokens + self.position_embedding.to(tokens.dtype)
        for layer in self.layers:
            tokens = layer(tokens)
        return {
            "raw": raw,
            "intrinsic": intrinsic,
            "regions": self.norm(tokens),
            "summary": summary,
            "patch_ids": patch_ids,
            "visible_counts": maps,
        }

    def encode_slide(self, batch, return_regions=False):
        """Cover every nucleus once, batching independent physical context windows."""
        positions, efds = batch["pos"], batch["efds"]
        if not len(efds):
            raise ValueError("Slide inference requires at least one nucleus")
        docs = batch.get("document_ids")
        if docs is None:
            docs = batch["block_mask"].mask_mod.doc_ids[: len(efds)].long()
        side = self.grid_size * self.tile_size_um
        window_xy = torch.floor(positions / side).long()
        keys, inverse, lengths = torch.unique(
            torch.cat([docs[:, None], window_xy], -1),
            dim=0,
            return_inverse=True,
            return_counts=True,
        )
        order = torch.argsort(inverse, stable=True)
        bounds = F.pad(lengths.cumsum(0), (1, 0)).cpu().tolist()
        raw = self.standardizer(efds)
        intrinsic = self.shape_encoder(raw)
        # CUDA autocast can emit BF16 shapes but FP32 normalized context.
        # Preserve the latter precision when scattering back into nucleus order.
        contextual = raw.new_empty(len(raw), self.dim)
        summaries = raw.new_empty(len(raw), self.summary_dim - raw.shape[-1])
        regional, population = [], []
        for start in range(0, len(keys), self.inference_window_batch):
            end = min(len(keys), start + self.inference_window_batch)
            keep = order[bounds[start] : bounds[end]]
            local = positions[keep] - window_xy[keep] * side
            tokens, summary, patch_ids, maps = self.pool(
                raw[keep], intrinsic[keep], local, inverse[keep] - start, end - start
            )
            tokens = self.patch_projection(tokens) + self.position_embedding.to(
                intrinsic.dtype
            )
            for layer in self.layers:
                tokens = layer(tokens)
            tokens = self.norm(tokens).float()
            contextual[keep] = tokens.flatten(0, 1)[patch_ids]
            summaries[keep] = summary.flatten(0, 1)[patch_ids]
            if return_regions:
                regional.append(tokens)
                population.append(maps.sum((-2, -1)))
        features = {
            "raw": raw,
            "intrinsic": intrinsic,
            "contextual": torch.cat([intrinsic, contextual], -1),
            "local_summary": torch.cat([raw, summaries], -1),
        }
        if not return_regions:
            return features
        return {
            "nuclei": features,
            "regions": {
                "features": torch.cat(regional),
                "counts": torch.cat(population),
                "centers_um": keys[:, None, 1:].to(positions.dtype) * side
                + self.centers,
                "document_ids": keys[:, 0],
            },
        }

    def forward(self, batch):
        return self.encode_slide(batch)
