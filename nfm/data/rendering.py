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
) -> NDArray[np.uint8]:
    """Render a square tile with white nuclei on a black background.

    Args:
        polygons: Vertices in micrometers with shape [nuclei, vertices, 2].
        center: Tile center in micrometers.
        field: Tile side length in micrometers.
        size: Tile side length in pixels.
    """
    # The first vertex is enough to discard nuclei far outside of the tile.
    near = np.abs(polygons[:, 0] - center).max(axis=1) < field
    points = polygons[near] - center
    # A small margin avoids cv2 anti-aliasing artifacts where nuclei cross the border.
    margin = 2
    points = (points * size / field + size / 2 + margin) * 16  # 4 fractional bits
    image = np.zeros((size + 2 * margin, size + 2 * margin), dtype=np.uint8)
    # One call per nucleus, a single call would leave holes where nuclei overlap.
    for polygon in points.astype(np.int32):
        cv2.fillPoly(image, [polygon], 255, cv2.LINE_AA, shift=4)
    return image[margin:-margin, margin:-margin]
