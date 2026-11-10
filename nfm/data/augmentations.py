"""Augmentations of nuclei contours in a view frame centered on the view center.

Each one takes polygons with shape [nuclei, vertices, 2] in micrometers and other
per-nucleus arrays as keyword arguments, and returns them in a dict.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


type Nuclei = dict[str, NDArray[Any]]


class Compose:
    def __init__(self, transforms: Sequence[Callable[..., Nuclei]]) -> None:
        self.transforms = transforms

    def __call__(self, **nuclei: NDArray[Any]) -> Nuclei:
        for transform in self.transforms:
            nuclei = transform(**nuclei)
        return nuclei


class Affine:
    """Random rotation, flips and scaling of the whole field around the center."""

    def __init__(
        self,
        rotation: bool = True,
        flip: bool = True,
        scale: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self.rotation = rotation
        self.flip = flip
        self.scale = scale

    def __call__(self, polygons: NDArray[np.float32], **other: NDArray[Any]) -> Nuclei:
        angle = np.random.uniform(0, 2 * np.pi) if self.rotation else 0.0
        matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        if self.flip and np.random.rand() < 0.5:
            matrix = matrix @ np.diag([-1.0, 1.0])
        matrix *= np.random.uniform(*self.scale)
        return {"polygons": (polygons @ matrix.T.astype(np.float32)), **other}


class Holes:
    """Removes the nuclei in random square holes, like coarse dropout.

    Holes are centered on random nuclei, with side lengths in micrometers.
    """

    def __init__(
        self,
        count: tuple[int, int] = (1, 8),
        size: tuple[float, float] = (6.0, 20.0),
        p: float = 0.5,
    ) -> None:
        self.count = count
        self.size = size
        self.p = p

    def __call__(self, **nuclei: NDArray[Any]) -> Nuclei:
        centroids = nuclei["polygons"].mean(1)
        if len(centroids) == 0 or np.random.rand() >= self.p:
            return nuclei
        count = np.random.randint(self.count[0], self.count[1] + 1)
        centers = centroids[np.random.randint(len(centroids), size=count)]
        sizes = np.random.uniform(*self.size, size=(count, 1))
        inside = np.abs(centroids[:, None] - centers[None]).max(2) <= sizes.T / 2
        keep = ~inside.any(1)
        return {key: value[keep] for key, value in nuclei.items()}


class ShapeNoise:
    """Smooth radial deformation and rescaling of each contour independently.

    The radial offsets are a random Fourier series over the vertices, with the
    harmonics up to `harmonics` and the offsets relative to the nucleus radius.
    Each nucleus is also rescaled by a factor drawn from `scale`.
    """

    def __init__(
        self,
        std: float = 0.05,
        harmonics: int = 4,
        scale: tuple[float, float] = (1.0, 1.0),
        p: float = 1.0,
    ) -> None:
        self.std = std
        self.harmonics = harmonics
        self.scale = scale
        self.p = p

    def __call__(self, polygons: NDArray[np.float32], **other: NDArray[Any]) -> Nuclei:
        if np.random.rand() >= self.p:
            return {"polygons": polygons, **other}
        n, vertices = polygons.shape[:2]
        centroids = polygons.mean(1, keepdims=True)
        angles = 2 * np.pi * np.arange(vertices) / vertices
        k = np.arange(1, self.harmonics + 1)[:, None]  # [harmonics, 1]
        amplitudes = np.random.normal(0, self.std, (n, self.harmonics, 1))
        phases = np.random.uniform(0, 2 * np.pi, (n, self.harmonics, 1))
        noise = (amplitudes * np.cos(k * angles + phases)).sum(1)  # [n, vertices]
        factor = (1 + noise) * np.random.uniform(*self.scale, (n, 1))
        polygons = centroids + (polygons - centroids) * factor[..., None]
        return {"polygons": polygons.astype(np.float32), **other}


class Drop:
    """Removes each nucleus with probability `p`, along with its other arrays."""

    def __init__(self, p: float = 0.1) -> None:
        self.p = p

    def __call__(self, **nuclei: NDArray[Any]) -> Nuclei:
        keep = np.random.rand(len(nuclei["polygons"])) >= self.p
        return {key: value[keep] for key, value in nuclei.items()}
