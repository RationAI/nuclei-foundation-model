import heapq
import itertools
import random
from collections.abc import Container, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from graph_pytorch_ext import build_adjacency_graph
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from sklearn.neighbors import KernelDensity
from torch.utils.data import Dataset


type AdjacencyGraph = list[list[tuple[int, float]]]
type Sample = dict[str, tuple[NDArray, NDArray] | NDArray[np.float32]]


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
        alpha: float = 0.8,
        target_mpp: float = 0.25,
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
        self.target_mpp = target_mpp

    def __len__(self) -> int:
        return len(self.slides)

    def _sample_spatial_registers(
        self, points: np.ndarray, n_samples: int
    ) -> NDArray[np.float32]:
        kde = KernelDensity(bandwidth=0.5, kernel="gaussian")
        kde.fit(points)
        return kde.sample(n_samples)

    def _build_graph(self, points: NDArray[np.uint32]) -> AdjacencyGraph:
        tri = Delaunay(points)
        distances = np.linalg.norm(
            points[tri.simplices[:, [0, 1, 2]]]
            - points[np.roll(tri.simplices[:, [0, 1, 2]], shift=-1, axis=1)],
            axis=2,
        )
        adj_graph = build_adjacency_graph(
            tri.simplices.astype(np.int64),
            distances.astype(np.float32),
            len(points),
        )
        return adj_graph

    def _find_component(
        self,
        idx: int,
        k: int,
        graph: AdjacencyGraph,
        centroids: Sequence[float],
        indices: Container[int] | None = None,
    ) -> list[int]:
        component_indices = []
        in_component = np.zeros(len(centroids), dtype=np.bool)

        pq = []
        heapq.heappush(pq, (0, idx))
        start_point_coords = centroids[idx]

        while len(pq) != 0 and len(component_indices) < k:
            _, current_idx = heapq.heappop(pq)
            if in_component[current_idx]:
                continue

            in_component[current_idx] = True
            component_indices.append(current_idx)

            for n_idx, edge_dist in graph[current_idx]:
                if (indices is None or n_idx in indices) and not in_component[n_idx]:
                    start_dist = np.linalg.norm(centroids[n_idx] - start_point_coords)
                    hybrid_cost = self.alpha * edge_dist + (1 - self.alpha) * start_dist
                    heapq.heappush(pq, (hybrid_cost, n_idx))

        return component_indices

    def _normalize(self, centroids, slide):
        scale_x = slide["mpp_x"] / self.target_mpp
        scale_y = slide["mpp_y"] / self.target_mpp

        return centroids * (scale_x, scale_y)

    def __getitem__(self, idx: int) -> Sample:
        slide = self.slides.iloc[idx]
        df = pd.read_parquet(self.nuclei_path / f"slide_id={slide['slide_id']}")

        centroids = self._normalize(np.stack(df.centroid.values), slide)
        graph = self._build_graph(centroids)

        # assert len(centroids) >= self.global_crop_k
        # if len(centroids) < self.global_crop_k:
        #     pass

        # Global crops generation
        global_crops: list[list[int]] = []
        seed = random.randint(0, len(centroids) - 1)
        for _ in range(self.n_global_crops):
            crop = self._find_component(seed, self.global_crop_k, graph, centroids)
            global_crops.append(crop)
            seed_idx = int(random.triangular(0, len(crop) - 1, len(crop) * 0.8))
            seed = crop[seed_idx]

        global_spatial_registers = [
            self._sample_spatial_registers(
                centroids[crop], self.n_global_spatial_registers
            )
            for crop in global_crops
        ]

        # Local crop generation
        all_global_crops = list(set(itertools.chain.from_iterable(global_crops)))
        crops = np.random.choice(
            all_global_crops, size=self.n_local_crops, replace=False
        )
        local_crops = [
            self._find_component(
                crop, self.local_crop_k, graph, centroids, all_global_crops
            )
            for crop in crops
        ]

        local_spatial_registers = [
            self._sample_spatial_registers(
                centroids[crop], self.n_local_spatial_registers
            )
            for crop in local_crops
        ]

        return {
            "global_crops": (centroids[global_crops], embeddings[global_crops]),
            "local_crops": (centroids[local_crops], embeddings[local_crops]),
            "global_spatial_registers": global_spatial_registers,
            "local_spatial_registers": local_spatial_registers,
        }
