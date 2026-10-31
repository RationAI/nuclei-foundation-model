"""Reconstruct nuclear contours from the radial detection format."""

import numpy as np


def radial_polygons(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Reconstruct the store's [sin, cos] radial convention in pixel units."""
    angle = np.arange(radii.shape[1], dtype=np.float64) * (2 * np.pi / radii.shape[1])
    directions = np.stack([np.sin(angle), np.cos(angle)], axis=-1)
    return points[:, None].astype(np.float64) + radii[..., None] * directions
