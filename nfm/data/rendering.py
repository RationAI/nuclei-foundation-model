"""Rasterize nuclei polygons into grayscale tiles."""

import cv2
import numpy as np
from numpy.typing import NDArray


cv2.setNumThreads(0)  # DataLoader workers are already the unit of parallelism.


def render(
    polygons: NDArray[np.float32],
    center: NDArray[np.float32],
    field: float,
    size: int,
    angle: float = 0.0,
    flip: bool = False,
) -> NDArray[np.uint8]:
    """Render a square tile with white nuclei on a black background.

    Args:
        polygons: Vertices in micrometers with shape [nuclei, vertices, 2].
        center: Tile center in micrometers.
        field: Tile side length in micrometers.
        size: Tile side length in pixels.
        angle: Rotation around the tile center in radians.
        flip: Whether to mirror the tile.
    """
    # The first vertex is enough to discard nuclei far outside of the tile.
    near = np.abs(polygons[:, 0] - center).max(axis=1) < field
    points = polygons[near] - center
    cos, sin = np.cos(angle), np.sin(angle)
    points = points @ np.array([[cos, sin], [-sin, cos]], dtype=np.float32)
    if flip:
        points = points * np.array([-1, 1], dtype=np.float32)
    # A small margin avoids cv2 anti-aliasing artifacts where nuclei cross the border.
    margin = 2
    points = (points * size / field + size / 2 + margin) * 16  # 4 fractional bits
    image = np.zeros((size + 2 * margin, size + 2 * margin), dtype=np.uint8)
    # One call per nucleus, a single call would leave holes where nuclei overlap.
    for polygon in points.astype(np.int32):
        cv2.fillPoly(image, [polygon], 255, cv2.LINE_AA, shift=4)
    return image[margin:-margin, margin:-margin]


def random_view(
    polygons: NDArray[np.float32],
    center: NDArray[np.float32],
    field: float,
    size: int,
    scale: tuple[float, float],
    drop: float,
    blur: float,
) -> NDArray[np.uint8]:
    """Render a random resized crop of a tile directly from its polygons.

    Cropping in vector space keeps zoomed views sharp, and any rotation is valid
    because tissue has no canonical orientation. All vertices stay within
    field / sqrt(2) of the center.
    """
    crop = field * np.sqrt(np.random.uniform(*scale))
    crop_center = center + np.random.uniform(-1, 1, 2) * (field - crop) / 2
    kept = polygons[np.random.rand(len(polygons)) >= drop]
    image = render(
        kept,
        crop_center.astype(np.float32),
        crop,
        size,
        np.random.uniform(0, 2 * np.pi),
        np.random.rand() < 0.5,
    )
    if np.random.rand() < blur:
        image = cv2.GaussianBlur(image, (0, 0), np.random.uniform(0.1, 1.5))
    return image
