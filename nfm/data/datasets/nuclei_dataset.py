import heapq
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
from degraph import build_spatial_graph
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from torch.utils.data import Dataset

from nfm.data.efd import elliptic_fourier_descriptors

type Sample = dict[
    str, tuple[NDArray[np.float32], NDArray[np.float32]] | NDArray[np.float32]
]


class NucleiDataset(Dataset[Sample]):
    def __init__(
        self,
        slides_path: str | Path,
        nuclei_path: str | Path,
        global_crop_k: int = 4096,
        local_crop_k: int = 768,
        n_global_crops: int = 2,
        n_local_crops: int = 8,
        n_local_spatial_registers: int = 48,
        n_global_spatial_registers: int = 256,
        alpha: float = 0.85,
        efd_order: int = 16,
    ) -> None:
        self.slides = pd.read_parquet(
            slides_path, columns=["id", "mpp_x", "mpp_y", "dataset", "organ"]
        )
        self.nuclei_path = Path(nuclei_path)
        self.global_crop_k = global_crop_k
        self.local_crop_k = local_crop_k
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
        self.n_local_spatial_registers = n_local_spatial_registers
        self.n_global_spatial_registers = n_global_spatial_registers
        self.alpha = alpha
        self.efd_order = efd_order

    def __len__(self) -> int:
        return len(self.slides)

    def sample_spatial_registers(
        self, points: np.ndarray, n_samples: int
    ) -> NDArray[np.float32]:
        kmeans = KMeans(n_clusters=n_samples)
        kmeans.fit(points)
        return kmeans.cluster_centers_

    def find_component(
        self,
        idx: int,
        k: int,
        graph: list[list[tuple[int, float]]],
        centroids: NDArray[np.float32],
        indices: set[int] | None = None,
    ) -> list[int]:
        component_indices = []
        visited = np.zeros(len(centroids), dtype=np.bool)

        pq = []
        heapq.heappush(pq, (0, idx))
        start_point_coords = centroids[idx]

        while pq and len(component_indices) < k:
            _, current_idx = heapq.heappop(pq)
            if visited[current_idx]:
                continue

            visited[current_idx] = True
            component_indices.append(current_idx)

            for n_idx, edge_dist in graph[current_idx]:
                if not visited[n_idx] and (indices is None or n_idx in indices):
                    start_dist = np.linalg.norm(centroids[n_idx] - start_point_coords)
                    cost = self.alpha * edge_dist + (1 - self.alpha) * start_dist
                    heapq.heappush(pq, (cost, n_idx))

        return component_indices

    def radial_to_efd(
        self,
        points: NDArray[np.float32],
        radial_distances: NDArray[np.float32],
        mpp_x: float,
        mpp_y: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        t = np.linspace(0, 1, radial_distances.shape[-1] + 1, dtype=np.float32)[:-1]
        cos = np.cos(2 * np.pi * t)
        sin = np.sin(2 * np.pi * t)

        polar = radial_distances[..., None] * np.stack([sin, cos], axis=-1)
        polygons = points[:, None] + polar

        polygons[..., 0] *= mpp_x
        polygons[..., 1] *= mpp_y

        centroids = polygons.mean(axis=1)
        efd = elliptic_fourier_descriptors(polygons.astype(np.float64), self.efd_order)

        return centroids, efd.astype(np.float32)

    def pad_crops(
        self, crops: NDArray[np.float32], target_k: int
    ) -> NDArray[np.float32]:
        pad_len = target_k - crops.shape[1]
        return np.pad(
            crops,
            ((0, 0), (0, pad_len), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    def __getitem__(self, idx: int) -> Sample:
        slide = self.slides.iloc[idx]
        df = pd.read_parquet(
            self.nuclei_path
            / f"organ={slide.organ}/dataset={slide.dataset}/slide_id={slide.id}"
        )

        points = np.stack(df.points.values, dtype=np.float32)
        # delaunay triangulation fails with duplicate points - remove them
        _, unique_idx = np.unique(points.round(decimals=1), axis=0, return_index=True)
        points = points[unique_idx]
        df = df.iloc[unique_idx].reset_index(drop=True)
        graph = build_spatial_graph(points)

        # Global crops generation
        global_crops_indices: list[list[int]] = []
        seed = random.randint(0, len(points) - 1)
        for _ in range(self.n_global_crops):
            indices = self.find_component(seed, self.global_crop_k, graph, points)
            global_crops_indices.append(indices)
            seed_idx = int(random.triangular(0, len(indices) - 1, len(indices) * 0.9))
            seed = indices[seed_idx]

        # Local crop generation
        all_indices_set = set(itertools.chain.from_iterable(global_crops_indices))
        all_indices = list(all_indices_set)
        index_map = {k: v for v, k in enumerate(all_indices)}
        global_crops_indices = [
            [index_map[i] for i in crop] for crop in global_crops_indices
        ]

        local_crops_indices = [
            [
                index_map[i]
                for i in self.find_component(
                    seed, self.local_crop_k, graph, points, indices=all_indices_set
                )
            ]
            for seed in np.random.choice(all_indices, self.n_local_crops, replace=False)
        ]

        centroids, efds = self.radial_to_efd(
            points[all_indices],
            np.stack(df.radial_distances.values[all_indices]),
            mpp_x=slide.mpp_x,
            mpp_y=slide.mpp_y,
        )
        efds = efds.reshape(-1, self.efd_order * 4)

        return {
            "global_crops": (
                self.pad_crops(centroids[global_crops_indices], self.global_crop_k),
                self.pad_crops(efds[global_crops_indices], self.global_crop_k),
            ),
            "local_crops": (
                self.pad_crops(centroids[local_crops_indices], self.local_crop_k),
                self.pad_crops(efds[local_crops_indices], self.local_crop_k),
            ),
            "global_spatial_registers": np.stack(
                [
                    self.sample_spatial_registers(c, self.n_global_spatial_registers)
                    for c in centroids[global_crops_indices]
                ]
            ),
            "local_spatial_registers": np.stack(
                [
                    self.sample_spatial_registers(c, self.n_local_spatial_registers)
                    for c in centroids[local_crops_indices]
                ]
            ),
        }
