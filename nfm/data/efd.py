import numpy as np
from numpy.typing import NDArray


def _phase_shift_efd(
    coeffs: NDArray[np.float64],
    theta: NDArray[np.float64],
    harmonic_indices: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply a start-point phase shift θ to each harmonic."""
    cos_terms = np.cos(harmonic_indices * theta[:, None])
    sin_terms = np.sin(harmonic_indices * theta[:, None])

    rotated = np.empty_like(coeffs)
    a, b, c, d = coeffs.transpose(2, 0, 1)

    rotated[..., 0] = a * cos_terms + b * sin_terms
    rotated[..., 1] = -a * sin_terms + b * cos_terms
    rotated[..., 2] = c * cos_terms + d * sin_terms
    rotated[..., 3] = -c * sin_terms + d * cos_terms
    return rotated


def normalize_efd_for_starting_point(
    coeffs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate EFD coefficients to eliminate contour start-point dependence."""
    mask = np.linalg.norm(coeffs[:, 0], axis=1) > np.finfo(np.float64).eps
    if not np.any(mask):
        return coeffs

    a1, b1, c1, d1 = coeffs[mask, 0].T
    harmonic_indices = np.arange(1, coeffs.shape[1] + 1)
    theta = 0.5 * np.arctan2(2.0 * (a1 * b1 + c1 * d1), a1**2 - b1**2 + c1**2 - d1**2)

    rotated = _phase_shift_efd(coeffs[mask], theta, harmonic_indices)

    # Condition for pi rotation
    pi_mask = (rotated[:, 0, 0] < 0.0) | (
        np.isclose(rotated[:, 0, 0], 0.0) & (rotated[:, 0, 2] < 0.0)
    )
    if np.any(pi_mask):
        rotated[pi_mask] = _phase_shift_efd(
            coeffs[mask & pi_mask], theta[pi_mask] + np.pi, harmonic_indices
        )

    coeffs[mask] = rotated
    return coeffs


def elliptic_fourier_descriptors(
    contour: NDArray[np.float64], order: int
) -> NDArray[np.float64]:
    """Computes the Elliptic Fourier Descriptors for a set of contours.

    Args:
        contour: Array of shape (N, M, 2) representing N contours,
            each with M points in 2D.
        order: The order of Fourier coefficients to calculate.
    """
    contour = np.concatenate((contour, contour[:, :1]), axis=1)

    dxy = np.diff(contour, axis=1)
    dt = np.linalg.norm(dxy, axis=2)
    t = np.concatenate([np.zeros((contour.shape[0], 1)), np.cumsum(dt, axis=1)], axis=1)
    T = t[:, -1:, None]

    orders = np.arange(1, order + 1)[None, :, None]
    consts = T / (2 * orders**2 * np.pi**2)
    phi = 2 * np.pi * t[:, None] * orders / T

    d_cos_phi = np.cos(phi[..., 1:]) - np.cos(phi[..., :-1])
    d_sin_phi = np.sin(phi[..., 1:]) - np.sin(phi[..., :-1])

    dxy_norm = dxy / dt[..., None]

    a = np.sum(dxy_norm[:, None, :, 0] * d_cos_phi, axis=2)
    b = np.sum(dxy_norm[:, None, :, 0] * d_sin_phi, axis=2)
    c = np.sum(dxy_norm[:, None, :, 1] * d_cos_phi, axis=2)
    d = np.sum(dxy_norm[:, None, :, 1] * d_sin_phi, axis=2)

    coeffs = consts * np.stack([a, b, c, d], axis=2)
    return normalize_efd_for_starting_point(coeffs)
