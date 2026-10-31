"""Single clean crops and fixed geometric targets for morphology pretraining."""

import numpy as np
from degraph import build_spatial_graph

from nfm.data.datasets.nuclei_dataset import NucleiDataset, Sample


def morphology_targets(polygons: np.ndarray, efds: np.ndarray) -> np.ndarray:
    """Physical size, rotation-invariant shape, and relative harmonic amplitudes."""
    centered = polygons.astype(np.float64) - polygons.mean(1, keepdims=True)
    following = np.roll(centered, -1, axis=1)
    area = (
        np.abs(
            (
                centered[..., 0] * following[..., 1]
                - centered[..., 1] * following[..., 0]
            ).sum(1)
        )
        / 2
    )
    perimeter = np.linalg.norm(following - centered, axis=-1).sum(1)
    if not np.isfinite(centered).all() or np.any(area <= 0) or np.any(perimeter <= 0):
        raise ValueError("Morphology targets require finite, nondegenerate polygons")
    covariance = centered.transpose(0, 2, 1) @ centered / centered.shape[1]
    axes = np.linalg.eigvalsh(covariance).clip(1e-12)
    radii = np.linalg.norm(centered, axis=-1)
    scalars = np.stack(
        [
            np.log(area),
            np.log(perimeter),
            0.5 * np.log(axes[:, 1] / axes[:, 0]),
            np.clip(4 * np.pi * area / perimeter**2, 0, 1),
            radii.std(1) / radii.mean(1).clip(1e-12),
        ],
        axis=1,
    )
    coefficients = efds.reshape(len(efds), -1, 4).astype(np.float64)
    amplitudes = np.linalg.norm(coefficients, axis=-1)
    ratios = amplitudes[:, 1:8] / amplitudes[:, :1].clip(1e-12)
    result = np.concatenate([scalars, np.log(ratios.clip(1e-4))], axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite morphology targets")
    return result.astype(np.float32)


class MorphologyNucleiDataset(NucleiDataset):
    """Reuse physical retrieval and graph traversal, without paired augmentations."""

    def __getitem__(self, idx: int) -> Sample:
        slide = self.slides.iloc[idx]
        points, polygons, _, seed = self._candidates(slide)
        graph = build_spatial_graph(points)
        indices = self.find_component(seed, self.crop_size, graph, points)
        polygons = polygons[indices]
        positions, efds = self.polygon_to_efd(polygons)
        return {
            "pos": positions,
            "efds": efds,
            "morphology": morphology_targets(polygons, efds),
            "slide_id": str(slide.id),
        }
