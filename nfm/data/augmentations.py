"""Polygon augmentations for nuclei shape and spatial data.

All augmentations operate on reconstructed polygon vertices with shape
``(n_nuclei, n_boundary_points, 2)`` and are applied *before* EFD computation
so that the resulting descriptors and centroids reflect the transformed geometry.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class PolygonAugmentation(Protocol):
    """Protocol for an augmentation callable."""

    def __call__(
        self, polygons: NDArray[np.float32], **kwargs
    ) -> dict[str, NDArray[np.float32]]: ...


class Flip:
    """Flip the crop horizontally and/or vertically.

    Each axis is flipped independently with 50% probability, simulating
    variations in tissue orientation.
    """

    def __call__(
        self, polygons: NDArray[np.float32], **kwargs
    ) -> dict[str, NDArray[np.float32]]:
        centre = polygons.mean(axis=(0, 1))
        polygons = polygons - centre
        if np.random.rand() < 0.5:
            polygons[..., 0] = -polygons[..., 0]
        if np.random.rand() < 0.5:
            polygons[..., 1] = -polygons[..., 1]
        polygons = polygons + centre
        return {"polygons": polygons, **kwargs}


class PositionJitter:
    """Translate each nucleus by an independent random 2-D vector.

    Because polygons are already absolute coordinates, adding per-nucleus shifts
    moves the centroids while preserving the boundary shape.
    """

    def __init__(self, max_shift: float = 5.0) -> None:
        self.max_shift = max_shift

    def __call__(
        self, polygons: NDArray[np.float32], **kwargs
    ) -> dict[str, NDArray[np.float32]]:
        n = polygons.shape[0]
        shifts = np.random.uniform(
            -self.max_shift, self.max_shift, size=(n, 1, 2)
        ).astype(np.float32)
        return {"polygons": polygons + shifts, **kwargs}


class FieldRotation:
    """Rotate the entire crop around its geometric centre.

    The rotation is applied to all polygon vertices as a single rigid body
    transform, thereby changing the relative spatial layout of the nuclei.
    """

    def __init__(self, max_angle: float = np.pi) -> None:
        self.max_angle = max_angle

    def __call__(
        self, polygons: NDArray[np.float32], **kwargs
    ) -> dict[str, NDArray[np.float32]]:
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        centre = polygons.mean(axis=(0, 1))
        local = polygons.reshape(-1, 2) - centre
        rotated = local @ R.T
        return {
            "polygons": (rotated + centre).reshape(polygons.shape),
            **kwargs,
        }


class NucleiRotation:
    """Rotate each nucleus independently around its own centroid.

    Each polygon is rotated by a different random angle, simulating variations
    in nucleus orientation while preserving the overall spatial arrangement.
    """

    def __init__(self, max_angle: float = np.pi) -> None:
        self.max_angle = max_angle

    def __call__(
        self, polygons: NDArray[np.float32], **kwargs
    ) -> dict[str, NDArray[np.float32]]:
        n = polygons.shape[0]
        angles = np.random.uniform(-self.max_angle, self.max_angle, size=n).astype(
            np.float32
        )
        sin_a = np.sin(angles)[:, None]
        cos_a = np.cos(angles)[:, None]

        centroids = polygons.mean(axis=1, keepdims=True)
        local = polygons - centroids

        rotated = np.empty_like(local)
        rotated[..., 0] = cos_a * local[..., 0] - sin_a * local[..., 1]
        rotated[..., 1] = sin_a * local[..., 0] + cos_a * local[..., 1]

        return {"polygons": (rotated + centroids).astype(np.float32), **kwargs}


class FieldScale:
    """Uniform scaling of the whole crop around its geometric centre.

    A single scale factor is drawn and applied to every vertex, changing the
    apparent field-of-view magnification.
    """

    def __init__(self, scale_range: tuple[float, float] = (0.9, 1.1)) -> None:
        self.scale_range = scale_range

    def __call__(
        self, polygons: NDArray[np.float32], **kwargs
    ) -> dict[str, NDArray[np.float32]]:
        scale = float(np.random.uniform(*self.scale_range))
        centre = polygons.reshape(-1, 2).mean(axis=0)
        return {
            "polygons": (centre + (polygons - centre) * scale).astype(np.float32),
            **kwargs,
        }


class PolygonScale:
    """Independent scaling of each nucleus around its own centroid.

    Each polygon is scaled independently, simulating variations in nucleus size
    while leaving the neighbour positions unchanged.
    """

    def __init__(self, scale_range: tuple[float, float] = (0.9, 1.1)) -> None:
        self.scale_range = scale_range

    def __call__(
        self, polygons: NDArray[np.float32], **kwargs
    ) -> dict[str, NDArray[np.float32]]:
        centroids = polygons.mean(axis=1, keepdims=True)
        directions = polygons - centroids
        n = polygons.shape[0]
        scales = np.random.uniform(*self.scale_range, size=(n, 1, 1)).astype(np.float32)
        return {
            "polygons": (centroids + directions * scales).astype(np.float32),
            **kwargs,
        }


class ShapeDistortion:
    """Radial shape perturbation.

    For each boundary vertex the direction from the polygon centroid is
    computed, then a small scalar offset is added along that direction.
    This deforms the outline while keeping the polygon roughly star-shaped.
    """

    def __init__(self, noise_std: float = 1.0) -> None:
        self.noise_std = noise_std

    def __call__(
        self, polygons: NDArray[np.float32], **kwargs
    ) -> dict[str, NDArray[np.float32]]:
        centroids = polygons.mean(axis=1, keepdims=True)
        directions = polygons - centroids
        norms = np.linalg.norm(directions, axis=2, keepdims=True)
        safe_norms = np.where(norms < 1e-8, 1.0, norms)
        unit_directions = directions / safe_norms

        noise = np.random.normal(
            0.0, self.noise_std, size=(polygons.shape[0], polygons.shape[1], 1)
        ).astype(np.float32)
        return {
            "polygons": (polygons + unit_directions * noise).astype(np.float32),
            **kwargs,
        }


class AffineSkew:
    """Random shear/skew applied to the whole crop around its centre.

    A 2x2 matrix ``[[1, sx], [sy, 1]]`` is sampled and applied uniformly to
    all vertices, introducing parallelogram-like spatial distortion.
    """

    def __init__(self, skew_range: tuple[float, float] = (-0.1, 0.1)) -> None:
        self.skew_range = skew_range

    def __call__(
        self, polygons: NDArray[np.float32], **kwargs
    ) -> dict[str, NDArray[np.float32]]:
        skew_x = float(np.random.uniform(*self.skew_range))
        skew_y = float(np.random.uniform(*self.skew_range))
        M = np.array([[1.0, skew_x], [skew_y, 1.0]], dtype=np.float32)

        centre = polygons.reshape(-1, 2).mean(axis=0)
        local = polygons.reshape(-1, 2) - centre
        skewed = local @ M.T
        return {
            "polygons": (skewed + centre).reshape(polygons.shape).astype(np.float32),
            **kwargs,
        }


class RandomDrop:
    """Randomly drop a fraction of nuclei from the crop.

    A random subset of polygons is removed.  The output length is therefore
    variable, which means downstream ``sample_crops`` indices become stale if
    this transform is used inside :meth:`NucleiDataset.__getitem__`.  It is
    most useful in isolated testing or in a pipeline that re-computes crops
    after dropping.
    """

    def __init__(self, p: float = 0.05) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p = p

    def __call__(
        self, polygons: NDArray[np.float32], labels: NDArray[np.float32]
    ) -> dict[str, NDArray[np.float32]]:
        n = polygons.shape[0]
        n_keep = max(1, int(n * (1 - self.p)))
        if n_keep >= n or np.random.rand() < 0.5:
            return {"polygons": polygons, "labels": labels}

        keep = np.random.choice(n, n_keep, replace=False)
        return {"polygons": polygons[keep], "labels": labels[keep]}


class ClusterDrop:
    """Drop a spatially coherent cluster of nuclei.

    A random seed nucleus is chosen and the nearest ``p*n`` nuclei (measured
    by centroid distance) are removed.  Like :class:`RandomDrop`, the output
    length is variable.
    """

    def __init__(self, p: float = 0.1) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p = p

    def __call__(
        self, polygons: NDArray[np.float32], labels: NDArray[np.float32]
    ) -> dict[str, NDArray[np.float32]]:
        n = polygons.shape[0]
        n_drop = min(n - 1, int(n * self.p))
        if n_drop <= 0 or np.random.rand() < 0.5:
            return {"polygons": polygons, "labels": labels}

        centroids = polygons.mean(axis=1)
        start = int(np.random.randint(0, n))
        distances = np.linalg.norm(centroids - centroids[start], axis=1)
        drop = np.argpartition(distances, n_drop - 1)[:n_drop]
        mask = np.ones(n, dtype=bool)
        mask[drop] = False
        return {"polygons": polygons[mask], "labels": labels[mask]}


class Compose:
    """Compose a sequence of polygon augmentations.

    Augmentations are applied in order.  Each augmentation draws its own
    randomness, so the overall result is stochastic.
    """

    def __init__(self, augmentations: Sequence[PolygonAugmentation]) -> None:
        self.augmentations = augmentations

    def __call__(self, **kwargs) -> dict[str, NDArray[np.float32]]:
        for aug in self.augmentations:
            kwargs = aug(**kwargs)
        return kwargs
