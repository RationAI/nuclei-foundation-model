"""Nuclei contours to tokens: shape descriptors and positions in a view frame."""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray


type View = dict[str, NDArray[Any]]
type Augmentation = Callable[..., dict[str, NDArray[Any]]]


def radial_polygons(
    points: NDArray[np.float32], radii: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Contours from the store's radial format, a [sin, cos] convention in pixels."""
    angles = np.arange(radii.shape[1]) * 2 * np.pi / radii.shape[1]
    directions = np.stack([np.sin(angles), np.cos(angles)], axis=-1)
    return (points[:, None] + radii[..., None] * directions).astype(np.float32)


def elliptic_fourier_descriptors(
    polygons: NDArray[np.floating], order: int
) -> NDArray[np.float32]:
    """Translation and start-point invariant contour descriptors.

    Contours are first oriented counter-clockwise, so flipped views and other
    sources with the opposite vertex order are encoded alike.

    Args:
        polygons: Vertices with shape [nuclei, vertices, 2].
        order: Number of harmonics.

    Returns:
        Descriptors with shape [nuclei, 4 * order].
    """
    contour = polygons.astype(np.float64)
    x, y = contour[..., 0], contour[..., 1]
    area = (x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y).sum(1)
    contour[area < 0] = contour[area < 0, ::-1]
    contour = np.concatenate([contour, contour[:, :1]], axis=1)

    dxy = np.diff(contour, axis=1)
    dt = np.linalg.norm(dxy, axis=2)
    t = np.concatenate([np.zeros((len(contour), 1)), np.cumsum(dt, axis=1)], axis=1)
    period = np.maximum(t[:, -1:, None], 1e-8)
    orders = np.arange(1, order + 1)[None, :, None]
    phi = 2 * np.pi * orders * t[:, None] / period  # [nuclei, order, vertices + 1]
    d_cos = np.diff(np.cos(phi), axis=2)
    d_sin = np.diff(np.sin(phi), axis=2)
    direction = dxy / np.maximum(dt, 1e-8)[..., None]
    dx, dy = direction[:, None, :, 0], direction[:, None, :, 1]
    scale = period / (2 * orders**2 * np.pi**2)
    a, b = scale[..., 0] * (dx * d_cos).sum(2), scale[..., 0] * (dx * d_sin).sum(2)
    c, d = scale[..., 0] * (dy * d_cos).sum(2), scale[..., 0] * (dy * d_sin).sum(2)

    # Shift the start point to the major axis of the first harmonic ellipse.
    a1, b1, c1, d1 = a[:, :1], b[:, :1], c[:, :1], d[:, :1]
    theta = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1), a1**2 - b1**2 + c1**2 - d1**2)
    cos, sin = np.cos(orders[..., 0] * theta), np.sin(orders[..., 0] * theta)
    coeffs = np.stack(
        [a * cos + b * sin, -a * sin + b * cos, c * cos + d * sin, -c * sin + d * cos],
        axis=2,
    )
    # The major axis has two ends. Moving to the other one, a shift by pi,
    # flips the sign of the odd harmonics. Pick the end in the positive x direction.
    coeffs[coeffs[:, 0, 0] < 0] *= (-1.0) ** orders[0]
    return coeffs.reshape(len(coeffs), 4 * order).astype(np.float32)


def crop_view(
    polygons: NDArray[np.float32],
    center: NDArray[np.float32],
    size: float,
    max_nuclei: int | None = None,
    augmentation: Augmentation | None = None,
    order: int = 16,
    **extras: NDArray[Any],
) -> View:
    """A square view of the nuclei whose centroids lie inside it.

    The augmentation acts in the view frame, centered on the view center, before
    cropping, so rotations, flips and scaling turn and resize the field itself.
    It may shrink the field by less than sqrt(2).

    Args:
        polygons: Vertices in micrometers with shape [nuclei, vertices, 2].
        center: View center in micrometers.
        size: View side length in micrometers.
        max_nuclei: Keeps only this many nuclei, the closest to the center.
        augmentation: Called with polygons and the extras, returns them changed.
        order: Number of harmonics of the shape descriptors.
        extras: Per-nucleus arrays, such as ids or labels, selected alongside.

    Returns:
        Shape descriptors, positions relative to the view center and the extras.
    """
    # Only nuclei within `size` of the center can end up in the view, as long as
    # the augmentation shrinks the field by less than sqrt(2). Shapes are not
    # augmented for the others.
    near = np.linalg.norm(polygons[:, 0] - center, axis=1) <= size
    view = {
        "polygons": polygons[near] - center,
        **{key: value[near] for key, value in extras.items()},
    }
    if augmentation is not None:
        view = augmentation(**view)
    positions = view["polygons"].mean(1)
    keep = np.flatnonzero(np.abs(positions).max(1) <= size / 2)
    if max_nuclei is not None and len(keep) > max_nuclei:
        distances = np.linalg.norm(positions[keep], axis=1)
        keep = keep[np.argpartition(distances, max_nuclei - 1)[:max_nuclei]]
    polygons = view.pop("polygons")[keep]
    return {
        "efds": elliptic_fourier_descriptors(polygons, order),
        "positions": positions[keep].astype(np.float32),
        **{key: value[keep] for key, value in view.items()},
    }
