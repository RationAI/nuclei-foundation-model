"""Graph-grown overlapping crops with identity-preserving independent views."""

import heapq
import random
from typing import Any

import lance
import numpy as np
import pandas as pd
from degraph import build_spatial_graph
from numpy.typing import NDArray
from torch.utils.data import Dataset

from nfm.data.augmentations import Compose, FieldRotation, Flip, ShapeDistortion
from nfm.data.efd import elliptic_fourier_descriptors
from nfm.data.geometry import radial_polygons


type Sample = dict[str, Any]


class NucleiDataset(Dataset[Sample]):
    def __init__(
        self,
        slides: pd.DataFrame,
        nuclei_path: str,
        crop_size: int = 4096,
        alpha: float = 0.85,
        efd_order: int = 16,
        augmentation: Compose | None = None,
    ) -> None:
        self.slides = slides.reset_index(drop=True)
        self.nuclei_path = nuclei_path
        self.crop_size = crop_size
        self.alpha, self.efd_order = alpha, efd_order
        self.augmentation = augmentation or Compose(
            [Flip(), FieldRotation(), ShapeDistortion(noise_std=0.01)]
        )

    def __len__(self) -> int:
        return len(self.slides)

    def find_component(
        self,
        idx: int,
        k: int,
        graph: list[list[tuple[int, float]]],
        centroids: NDArray[np.float32],
        indices: set[int] | None = None,
    ) -> list[int]:
        component_indices = []
        visited = np.zeros(len(centroids), dtype=np.bool_)
        pq = [(0.0, idx)]
        start_dist = np.linalg.norm(centroids - centroids[idx], axis=1)
        while pq and len(component_indices) < k:
            _, current_idx = heapq.heappop(pq)
            if visited[current_idx]:
                continue
            visited[current_idx] = True
            component_indices.append(current_idx)
            for n_idx, edge_dist in graph[current_idx]:
                if not visited[n_idx] and (indices is None or n_idx in indices):
                    cost = self.alpha * edge_dist + (1 - self.alpha) * start_dist[n_idx]
                    heapq.heappush(pq, (cost, n_idx))
        return component_indices

    def radial_to_polygon(self, points: NDArray, radial_distances: NDArray) -> NDArray:
        t = np.linspace(0, 1, radial_distances.shape[-1] + 1, dtype=np.float32)[:-1]
        cos = np.cos(2 * np.pi * t)
        sin = np.sin(2 * np.pi * t)

        polar = radial_distances[..., None] * np.stack([sin, cos], axis=-1)
        return points[:, None] + polar

    def polygon_to_efd(self, polygons: NDArray) -> tuple[NDArray, NDArray]:
        """Encode SSL/probe contours with the original EFD-only input."""
        centroids = polygons.mean(axis=1)
        efds = elliptic_fourier_descriptors(polygons.astype(np.float64), self.efd_order)
        return centroids, efds.reshape(-1, self.efd_order * 4).astype(np.float32)

    def sample_crops(self, points: NDArray, seed: int):
        """Grow A, then bias B's seed toward later nuclei in A's traversal."""
        graph = build_spatial_graph(points)
        first = np.asarray(self.find_component(seed, self.crop_size, graph, points))

        seed = first[int(random.triangular(0, len(first) - 1, len(first) * 0.9))]
        second = np.asarray(self.find_component(seed, self.crop_size, graph, points))

        return first, second

    def _candidates(self, slide: pd.Series) -> tuple[NDArray, NDArray, NDArray, int]:
        ds = lance.dataset(
            f"{self.nuclei_path}/organ={slide.organ}/dataset={slide.dataset}/slide_id={slide.id}.lance"
        )

        seed = np.asarray(
            ds.sample(1, columns=["points"])["points"][0].as_py(), dtype=np.float64
        )
        table = ds.to_table(
            nearest={
                "column": "points",
                "q": seed.astype(np.float32),
                "k": min(ds.count_rows(), int(2 * self.crop_size / (1 - self.alpha))),
                "metric": "euclidean",
            },
            with_row_id=True,
            disable_scoring_autoprojection=True,
        )

        mpp = np.asarray([slide.mpp_x, slide.mpp_y], dtype=np.float64)
        points = np.asarray(table["points"].to_pylist(), dtype=np.float64)
        radial_distances = np.asarray(
            table["radial_distances"].to_pylist(), dtype=np.float64
        )
        ids = np.asarray(table["_rowid"].to_pylist(), dtype=np.uint64)
        polygons = radial_polygons(points, radial_distances) * mpp
        points *= mpp

        # The Delaunay implementation requires distinct positions. Retain the
        # first persistent ID at each physical location, including the seed.
        _, unique = np.unique(
            np.round(points / 1e-4).astype(np.int64), axis=0, return_index=True
        )
        points, polygons, ids = (
            points[unique],
            polygons[unique],
            ids[unique],
        )

        seed = int(np.argmin(np.linalg.norm(points - seed * mpp, axis=1)))
        return points.astype(np.float32), polygons, ids, seed

    def __getitem__(self, idx: int) -> Sample:
        slide = self.slides.iloc[idx]
        points, polygons, ids, seed = self._candidates(slide)
        first, second = self.sample_crops(points, seed)

        views = []
        for crop in [first, second]:
            result = self.augmentation(
                polygons=polygons[crop].copy(), indices=ids[crop].copy()
            )
            view_ids = np.asarray(result["indices"])
            pos, efds = self.polygon_to_efd(result["polygons"])
            views.append((pos, efds, view_ids))

        return {
            "pos": [v[0] for v in views],
            "efds": [v[1] for v in views],
            "indices": [v[2] for v in views],
            "matched_ids": np.intersect1d(views[0][2], views[1][2]),
            "slide_id": str(slide.id),
        }
