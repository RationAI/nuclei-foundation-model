"""Augmentations of nuclei contours in a view frame centered on the view center.

Each one takes polygons with shape [nuclei, vertices, 2] in micrometers and other
per-nucleus arrays as keyword arguments, and returns them in a dict.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


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


class NucleusAffine:
    """Random affine map of each nucleus around its own centroid.

    Each nucleus is stretched along a random axis, by independent factors from
    `scale` along it and across it, and then rotated by up to `rotation` degrees.
    The positions of the nuclei do not change.
    """

    def __init__(
        self, rotation: float = 15.0, scale: tuple[float, float] = (0.9, 1.1)
    ) -> None:
        self.rotation = rotation
        self.scale = scale

    def __call__(self, polygons: NDArray[np.float32], **other: NDArray[Any]) -> Nuclei:
        n = len(polygons)
        angle = np.deg2rad(np.random.uniform(-self.rotation, self.rotation, n))
        axis = np.random.uniform(0, np.pi, n)
        stretch = np.random.uniform(*self.scale, (n, 2))
        # R(angle) R(axis) diag(stretch) R(-axis), one 2 x 2 matrix per nucleus.
        matrix = _rotations(angle + axis) @ (stretch[:, :, None] * _rotations(-axis))
        centroids = polygons.mean(1, keepdims=True)
        offsets = (polygons - centroids) @ matrix.transpose(0, 2, 1)
        return {"polygons": (centroids + offsets).astype(np.float32), **other}


def _rotations(angles: NDArray[np.floating]) -> NDArray[np.floating]:
    cos, sin = np.cos(angles), np.sin(angles)
    return np.stack([np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], -2)


class ShapeNoise:
    """Smooth radial deformation of each contour independently.

    The radial offsets are a random Fourier series over the vertices, with the
    harmonics up to `harmonics` and the offsets relative to the nucleus radius.
    """

    def __init__(self, std: float = 0.05, harmonics: int = 4, p: float = 1.0) -> None:
        self.std = std
        self.harmonics = harmonics
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
        polygons = centroids + (polygons - centroids) * (1 + noise[..., None])
        return {"polygons": polygons.astype(np.float32), **other}


class Jitter:
    """Moves each nucleus by an independent Gaussian offset of `std` micrometers.

    Shapes are kept, but the distances between nuclei change, so views sharing
    nuclei cannot be matched by their exact layout.
    """

    def __init__(self, std: float = 1.0) -> None:
        self.std = std

    def __call__(self, polygons: NDArray[np.float32], **other: NDArray[Any]) -> Nuclei:
        offsets = np.random.normal(0, self.std, (len(polygons), 1, 2))
        return {"polygons": (polygons + offsets).astype(np.float32), **other}


class Drop:
    """Removes each nucleus with probability `p`, along with its other arrays."""

    def __init__(self, p: float = 0.1) -> None:
        self.p = p

    def __call__(self, **nuclei: NDArray[Any]) -> Nuclei:
        keep = np.random.rand(len(nuclei["polygons"])) >= self.p
        return {key: value[keep] for key, value in nuclei.items()}


def _random_field(
    points: NDArray[np.floating], scale: float, features: int = 64
) -> NDArray[np.float64]:
    """A smooth random function of unit variance at the points.

    Random Fourier features of a Gaussian kernel: bumps about `scale`
    micrometers wide.
    """
    frequencies = np.random.normal(0, 1 / scale, (features, 2))
    phases = np.random.uniform(0, 2 * np.pi, features)
    return np.sqrt(2 / features) * np.cos(points @ frequencies.T + phases).sum(-1)


def _log_uniform(low: float, high: float) -> float:
    return float(np.exp(np.random.uniform(np.log(low), np.log(high))))


class FieldDrop:
    """Removes the nuclei where a smooth random field is highest.

    The field has bumps of a random length scale, so the dropped nuclei form
    irregular regions, from scattered clusters to whole structures. The cut is
    the quantile of the field over the nuclei, so a view loses the same share
    of nuclei whether it is dense or sparse, large or small. Near the cut a
    nucleus is dropped with a probability that rises smoothly, which leaves the
    region borders ragged.
    """

    def __init__(
        self,
        fraction: tuple[float, float] = (0.1, 0.3),
        scale: tuple[float, float] = (15.0, 90.0),
        softness: float = 0.15,
        features: int = 64,
    ) -> None:
        """Initializes the drop.

        Args:
            fraction: Range of the share of nuclei to drop.
            scale: Range of the length scale of the field in micrometers,
                drawn log-uniformly.
            softness: Width of the smooth cut, in field standard deviations.
            features: Random Fourier features of the field.
        """
        self.fraction = fraction
        self.scale = scale
        self.softness = softness
        self.features = features

    def __call__(self, **nuclei: NDArray[Any]) -> Nuclei:
        centroids = nuclei["polygons"].mean(1)
        if len(centroids) == 0:
            return nuclei
        field = _random_field(centroids, _log_uniform(*self.scale), self.features)
        cut = np.quantile(field, 1 - np.random.uniform(*self.fraction))
        z = np.clip((field - cut) / (self.softness * max(field.std(), 1e-8)), -50, 50)
        keep = np.random.rand(len(field)) >= 1 / (1 + np.exp(-z))
        return {key: value[keep] for key, value in nuclei.items()}


class Warp:
    """Smooth deformation of the whole view, as sectioning stretches tissue.

    Each nucleus moves by a smooth random displacement field at its centroid, of
    a random length scale and amplitude, so nearby nuclei move together. The
    field barely changes across a nucleus, so the shapes are kept.
    """

    def __init__(
        self,
        amplitude: tuple[float, float] = (1.0, 4.0),
        scale: tuple[float, float] = (50.0, 200.0),
        features: int = 64,
    ) -> None:
        """Initializes the warp.

        Args:
            amplitude: Range of the standard deviation of the displacement of
                each axis in micrometers.
            scale: Range of the length scale of the field in micrometers,
                drawn log-uniformly.
            features: Random Fourier features of the field.
        """
        self.amplitude = amplitude
        self.scale = scale
        self.features = features

    def __call__(self, polygons: NDArray[np.float32], **other: NDArray[Any]) -> Nuclei:
        centroids = polygons.mean(1)
        scale = _log_uniform(*self.scale)
        amplitude = np.random.uniform(*self.amplitude)
        shift = amplitude * np.stack(
            [_random_field(centroids, scale, self.features) for _ in range(2)], -1
        )
        return {"polygons": (polygons + shift[:, None]).astype(np.float32), **other}


class ShapeScale:
    """Scales every nucleus of the view by one factor around its own centroid.

    Scanners, stains and segmenters bias nucleus sizes consistently across a
    slide, while the positions of the nuclei do not change.
    """

    def __init__(self, scale: tuple[float, float] = (0.92, 1.08)) -> None:
        self.scale = scale

    def __call__(self, polygons: NDArray[np.float32], **other: NDArray[Any]) -> Nuclei:
        centroids = polygons.mean(1, keepdims=True)
        factor = np.random.uniform(*self.scale)
        polygons = centroids + (polygons - centroids) * factor
        return {"polygons": polygons.astype(np.float32), **other}


def _areas(polygons: NDArray[np.floating]) -> NDArray[np.float64]:
    x, y = polygons[..., 0].astype(np.float64), polygons[..., 1].astype(np.float64)
    return 0.5 * np.abs((x * np.roll(y, -1, -1) - np.roll(x, -1, -1) * y).sum(-1))


def _hull_contours(
    points: NDArray[np.floating], centers: NDArray[np.floating], vertices: int
) -> NDArray[np.float32]:
    """Convex hulls of point sets as radial contours around inner centers.

    The hull is the intersection of the half-planes bounding the points in
    `vertices` directions, from its support function, so it is exact up to a
    polygon with that many sides drawn around it.

    Args:
        points: Point sets with shape [hulls, points, 2].
        centers: A point inside each hull, [hulls, 2].
        vertices: Vertices of each contour, and directions of the half-planes.
    """
    angles = 2 * np.pi * np.arange(vertices) / vertices
    directions = np.stack([np.cos(angles), np.sin(angles)], -1)  # [vertices, 2]
    support = ((points - centers[:, None]) @ directions.T).max(1)  # [hulls, vertices]
    # Distance along each ray to each half-plane, kept where the ray meets it.
    cos = directions @ directions.T  # [ray, half-plane]
    reach = support[:, None] / np.where(cos > 1e-6, cos, np.nan)[None]
    radii = np.nanmin(np.where(reach > 0, reach, np.nan), -1)
    return (centers[:, None] + radii[..., None] * directions).astype(np.float32)


def _gaps(
    first: NDArray[np.floating],
    second: NDArray[np.floating],
    first_centers: NDArray[np.floating],
    second_centers: NDArray[np.floating],
    facing: int = 16,
) -> NDArray[Any]:
    """Smallest distance between the vertices of paired contours, [pairs].

    Only the `facing` vertices of each contour closest to the other's center
    are compared, where two neighboring contours come closest.
    """

    def toward(contours: NDArray[Any], centers: NDArray[Any]) -> NDArray[Any]:
        distance = ((contours - centers[:, None]) ** 2).sum(-1)
        index = np.argpartition(distance, facing - 1, axis=1)[:, :facing]
        return np.take_along_axis(contours, index[..., None], 1)

    offsets = (
        toward(first, second_centers)[:, :, None]
        - toward(second, first_centers)[:, None]
    )
    return np.sqrt((offsets**2).sum(-1).min((1, 2)))


class Merge:
    """Merges touching pairs of nuclei into one, as segmenters often do.

    Two nuclei touch when their contours come within `max_gap` micrometers,
    by the closest pair of vertices. Each touching pair merges with probability
    `p`, and a nucleus merges at most once. The merged contour is the convex hull
    of the pair, as a detector of star-convex contours outlines two nuclei it
    takes for one. The merged nucleus keeps the id and other arrays of one of
    the two.
    """

    def __init__(
        self, p: float = 0.2, max_gap: float = 0.5, neighbors: int = 6
    ) -> None:
        """Initializes the merge.

        Args:
            p: Probability that a touching pair merges.
            max_gap: Largest gap between the contours of a touching pair, in
                micrometers.
            neighbors: Nearest nuclei, by centroid, that each nucleus may touch.
        """
        self.p = p
        self.max_gap = max_gap
        self.neighbors = neighbors

    def __call__(self, **nuclei: NDArray[Any]) -> Nuclei:
        polygons = nuclei["polygons"]
        if len(polygons) < 2:
            return nuclei
        centroids = polygons.mean(1)
        reach = np.sqrt(((polygons - centroids[:, None]) ** 2).sum(-1).max(1))
        k = min(self.neighbors, len(polygons) - 1) + 1
        distance, nearest = cKDTree(centroids).query(centroids, k)
        a = np.repeat(np.arange(len(polygons)), k - 1)
        b, distance = nearest[:, 1:].ravel(), distance[:, 1:].ravel()
        # Each pair once, close enough that the contours may touch, and drawn
        # with probability p before the costlier gap is measured.
        candidate = (
            (a < b)
            & (distance <= reach[a] + reach[b] + self.max_gap)
            & (np.random.rand(len(a)) < self.p)
        )
        a, b = a[candidate], b[candidate]
        if len(a) == 0:
            return nuclei
        gaps = _gaps(polygons[a], polygons[b], centroids[a], centroids[b])
        touching = np.flatnonzero(gaps <= self.max_gap)
        # A nucleus merges at most once, the pairs taken in random order.
        used = np.zeros(len(polygons), dtype=bool)
        pairs = []
        for t in np.random.permutation(touching):
            if not used[a[t]] and not used[b[t]]:
                used[a[t]] = used[b[t]] = True
                pairs.append(t)
        if not pairs:
            return nuclei
        a, b = a[pairs], b[pairs]
        # The pair's centroids weighted by area lie inside the hull.
        weights = np.stack([_areas(polygons[a]), _areas(polygons[b])], -1)
        weights /= weights.sum(-1, keepdims=True)
        centers = weights[:, :1] * centroids[a] + weights[:, 1:] * centroids[b]
        points = np.concatenate([polygons[a], polygons[b]], axis=1)
        merged = _hull_contours(points, centers, polygons.shape[1])
        # Each merged nucleus is the first or the second of its pair at random.
        swap = np.random.rand(len(a)) < 0.5
        kept, gone = np.where(swap, b, a), np.where(swap, a, b)
        polygons = polygons.copy()
        polygons[kept] = merged
        keep = np.ones(len(polygons), dtype=bool)
        keep[gone] = False
        nuclei = {**nuclei, "polygons": polygons}
        return {key: value[keep] for key, value in nuclei.items()}


def _covered(
    inner: NDArray[np.floating], outer: NDArray[np.floating], rings: int = 4
) -> NDArray[np.float64]:
    """Share of the area of each inner contour that lies inside its outer one.

    Points spread evenly over the area of the inner contour, on rings through
    every second vertex, are tested against the outer contour, which is
    star-shaped around its vertex mean like the contours of the store.

    Args:
        inner: Contours with shape [pairs, vertices, 2].
        outer: Contours with shape [pairs, vertices, 2].
        rings: Rings of points, at equal area steps.

    Returns:
        The covered share of each inner contour, [pairs].
    """
    center = inner.mean(1, keepdims=True)
    fractions = np.sqrt((np.arange(rings) + 0.5) / rings)[None, :, None, None]
    points = center[:, None] + fractions * (inner[:, None, ::2] - center[:, None])
    points = points.reshape(len(inner), -1, 2)
    # The radius of the outer contour along the direction of each point.
    middle = outer.mean(1, keepdims=True)
    offsets = outer - middle
    angles = np.arctan2(offsets[..., 1], offsets[..., 0])
    radii = np.linalg.norm(offsets, axis=-1)
    order = np.argsort(angles, axis=1)
    angles = np.take_along_axis(angles, order, 1)
    radii = np.take_along_axis(radii, order, 1)
    angles = np.concatenate(
        [angles[:, -1:] - 2 * np.pi, angles, angles[:, :1] + 2 * np.pi], 1
    )
    radii = np.concatenate([radii[:, -1:], radii, radii[:, :1]], 1)
    toward = points - middle
    theta = np.arctan2(toward[..., 1], toward[..., 0])
    right = (angles[:, None, :] <= theta[..., None]).sum(-1)
    right = np.clip(right, 1, angles.shape[1] - 1)
    a0 = np.take_along_axis(angles, right - 1, 1)
    a1 = np.take_along_axis(angles, right, 1)
    r0 = np.take_along_axis(radii, right - 1, 1)
    r1 = np.take_along_axis(radii, right, 1)
    weight = (theta - a0) / np.maximum(a1 - a0, 1e-8)
    inside = np.linalg.norm(toward, axis=-1) < r0 + weight * (r1 - r0)
    return inside.mean(1)


class FalseDetections:
    """Adds spurious nuclei near the tissue, as detectors find in debris and blur.

    A random share of the view's count is added, each a rotated copy of a random
    nucleus of the view scaled by a factor from `scale`, placed up to `spread`
    micrometers from another random nucleus. Like real detections, a share
    `overlapping` of the copies may cover, and be covered by, up to
    `max_overlap` of the area of any of their `neighbors` nearest real nuclei,
    and the rest must stay clear of them, up to `tolerance`. A copy that
    overlaps more is placed again, up to `tries` times, and is left out if it
    never fits, so packed tissue gets fewer. They have id -1, which is
    never tracked, and the other arrays of the nucleus they copy.
    """

    def __init__(
        self,
        fraction: tuple[float, float] = (0.0, 0.03),
        scale: tuple[float, float] = (0.5, 1.1),
        spread: float = 20.0,
        overlapping: float = 0.35,
        max_overlap: float = 0.3,
        tolerance: float = 0.02,
        tries: int = 8,
        neighbors: int = 4,
    ) -> None:
        self.fraction = fraction
        self.scale = scale
        self.spread = spread
        self.overlapping = overlapping
        self.max_overlap = max_overlap
        self.tolerance = tolerance
        self.tries = tries
        self.neighbors = neighbors

    def __call__(self, **nuclei: NDArray[Any]) -> Nuclei:
        polygons = nuclei["polygons"]
        count = round(np.random.uniform(*self.fraction) * len(polygons))
        if count == 0 or len(polygons) == 0:
            return nuclei
        source = np.random.randint(len(polygons), size=count)
        shapes = polygons[source] - polygons[source].mean(1, keepdims=True)
        rotations = _rotations(np.random.uniform(0, 2 * np.pi, count))
        shapes = shapes @ rotations.transpose(0, 2, 1)
        shapes *= np.random.uniform(*self.scale, (count, 1, 1))
        allowed = np.where(
            np.random.rand(count) < self.overlapping, self.max_overlap, self.tolerance
        )
        centroids = polygons.mean(1)
        tree = cKDTree(centroids)
        reach = np.sqrt(((polygons - centroids[:, None]) ** 2).sum(-1).max(1))
        shape_reach = np.sqrt((shapes**2).sum(-1).max(1))
        k = min(self.neighbors, len(polygons))
        added_polygons = np.zeros_like(shapes)
        placed = np.zeros(count, dtype=bool)
        for _ in range(self.tries):
            pending = np.flatnonzero(~placed)
            if len(pending) == 0:
                break
            anchors = polygons[np.random.randint(len(polygons), size=len(pending))]
            radius = self.spread * np.sqrt(np.random.rand(len(pending)))
            direction = np.random.uniform(0, 2 * np.pi, len(pending))
            places = anchors.mean(1) + radius[:, None] * np.stack(
                [np.cos(direction), np.sin(direction)], -1
            )
            candidates = places[:, None] + shapes[pending]
            distance, near = tree.query(places, k)
            distance, near = distance.reshape(-1, k), near.reshape(-1, k)
            # Only the real nuclei close enough to touch are tested.
            touch = distance < shape_reach[pending, None] + reach[near]
            copy, slot = np.nonzero(touch)
            real = polygons[near[copy, slot]]
            overlap = np.zeros((len(pending), k))
            if len(copy):
                overlap[copy, slot] = np.maximum(
                    _covered(candidates[copy], real), _covered(real, candidates[copy])
                )
            fits = overlap.max(1) <= allowed[pending]
            added_polygons[pending[fits]] = candidates[fits]
            placed[pending[fits]] = True
        if not placed.any():
            return nuclei
        source = source[placed]
        added = {
            key: value[source].copy()
            for key, value in nuclei.items()
            if key != "polygons"
        }
        if "ids" in added:
            added["ids"][:] = -1
        added["polygons"] = added_polygons[placed].astype(np.float32)
        return {
            key: np.concatenate([value, added[key]]) for key, value in nuclei.items()
        }
