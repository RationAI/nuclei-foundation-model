import heapq
import itertools
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from degraph import build_spatial_graph
from numpy.typing import NDArray
from sklearn.neighbors import KernelDensity
from torch.utils.data import Dataset


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
    ) -> None:
        self.slides = pd.read_parquet(slides_path)
        self.nuclei_path = Path(nuclei_path)
        self.global_crop_k = global_crop_k
        self.local_crop_k = local_crop_k
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
        self.n_local_spatial_registers = n_local_spatial_registers
        self.n_global_spatial_registers = n_global_spatial_registers
        self.alpha = alpha

    def __len__(self) -> int:
        return len(self.slides)

    def _sample_spatial_registers(
        self, points: np.ndarray, n_samples: int
    ) -> NDArray[np.float32]:
        kde = KernelDensity(bandwidth=0.5, kernel="gaussian")
        kde.fit(points)
        return kde.sample(n_samples)

    def _find_component(
        self,
        idx: int,
        k: int,
        graph: list[list[tuple[int, float]]],
        centroids: Sequence[float],
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

    def _get_polygons(self, df: pd.DataFrame) -> NDArray[np.float32]:
        radial_distances = np.stack(df.radial_distances.values)
        points = np.stack(df.points.values)

        t = np.linspace(0, 1, radial_distances.shape[-1] + 1, dtype=np.float32)[:-1]
        cos = np.cos(2 * np.pi * t)
        sin = np.sin(2 * np.pi * t)

        polar = radial_distances[..., None] * np.stack([sin, cos], axis=-1)
        return points[:, None] + polar

    def __getitem__(self, idx: int) -> Sample:
        slide = self.slides.iloc[idx]
        df = pd.read_parquet(self.nuclei_path / f"slide_id={slide['slide_id']}.parquet")

        polygons = self._get_polygons(df)
        polygons[..., 0] *= slide["mpp_x"]
        polygons[..., 1] *= slide["mpp_y"]

        centroids = polygons.mean(axis=1)
        graph = build_spatial_graph(centroids)

        # Global crops generation
        global_crops: list[list[int]] = []
        seed = random.randint(0, len(centroids) - 1)
        for _ in range(self.n_global_crops):
            crop = self._find_component(seed, self.global_crop_k, graph, centroids)
            global_crops.append(crop)
            seed_idx = int(random.triangular(0, len(crop) - 1, len(crop) * 0.9))
            seed = crop[seed_idx]

        global_spatial_registers = np.stack(
            [
                self._sample_spatial_registers(
                    centroids[crop], self.n_global_spatial_registers
                )
                for crop in global_crops
            ]
        )

        # Local crop generation
        all_global_crops = list(set(itertools.chain.from_iterable(global_crops)))
        crops = np.random.choice(
            all_global_crops, size=self.n_local_crops, replace=False
        )
        local_crops = [
            self._find_component(
                crop, self.local_crop_k, graph, centroids, set(all_global_crops)
            )
            for crop in crops
        ]

        local_spatial_registers = np.stack(
            [
                self._sample_spatial_registers(
                    centroids[crop], self.n_local_spatial_registers
                )
                for crop in local_crops
            ]
        )

        return {
            "global_crops": (centroids[global_crops], polygons[global_crops]),
            "local_crops": (centroids[local_crops], polygons[local_crops]),
            "global_spatial_registers": global_spatial_registers,
            "local_spatial_registers": local_spatial_registers,
        }
