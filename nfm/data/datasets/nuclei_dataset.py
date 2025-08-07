import heapq
import random
from pathlib import Path

import numpy as np
import pandas as pd
from graph_pytorch_ext import build_adjacency_graph
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from sklearn.neighbors import KernelDensity
from torch.utils.data import Dataset


type AdjacencyGraph = list[list[tuple[int, float]]]


class NucleiDataset(Dataset):
    def __init__(
        self,
        base_path: Path,
        global_crop_k: int = 4096,
        local_crop_k: int = 768,
        local_crop_tokens: int = 48,
        global_crop_tokens: int = 256,
        n_local_crops: int = 8,
        alpha: float = 0.8,
    ) -> None:
        self.paths = list(base_path.rglob("nuclei/slide_id=*"))
        self.global_crop_k = global_crop_k
        self.local_crop_k = local_crop_k
        self.global_crop_tokens = global_crop_tokens
        self.local_crop_tokens = local_crop_tokens
        self.n_local_crops = n_local_crops
        self.alpha = alpha

    def __len__(self) -> int:
        return len(self.paths)

    def _get_tokens(
        self, points: NDArray, centroids: NDArray, token_count: int
    ) -> NDArray:
        kde = KernelDensity(bandwidth=0.5, kernel="gaussian")
        kde.fit(points)
        tokens = kde.sample(token_count)
        distances = np.linalg.norm(centroids[:, None, :] - tokens[None, :, :], axis=2)
        indices = np.argmin(distances, axis=0)
        return indices

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
        centroids: NDArray[np.float32],
        indices: NDArray[np.uint32] | None = None,
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

    def __getitem__(self, idx: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        df = pd.read_parquet(self.paths[idx])
        centroids = np.stack(df.centroid.values)
        embeddings = np.stack(df.embedding.values)

        assert len(centroids) >= self.global_crop_k

        graph = self._build_graph(centroids)

        # Global crop token generation
        global_crops = []
        seed = random.randint(0, len(centroids) - 1)
        global_crops.append(
            self._find_component(seed, self.global_crop_k, graph, centroids)
        )

        seed = int(
            random.triangular(0, self.global_crop_k - 1, self.global_crop_k * 0.8)
        )
        global_crops.append(
            self._find_component(
                global_crops[0][seed], self.global_crop_k, graph, centroids
            )
        )

        # Global crop token generation
        global_tokens = [
            self._get_tokens(centroids[idx], centroids, self.global_crop_tokens)
            for idx in global_crops
        ]

        # Local crop generation
        local_crops = []
        for i, g_tokens in enumerate(global_tokens):
            crops = []
            tokens = np.random.choice(g_tokens, size=self.n_local_crops, replace=False)
            crops = [
                self._find_component(
                    token, self.local_crop_k, graph, centroids, global_crops[i]
                )
                for token in tokens
            ]
            local_crops.append(crops)

        # Local crop token generation
        local_tokens = []
        for crops in local_crops:
            tokens = [
                self._get_tokens(centroids[crop], centroids, self.local_crop_tokens)
                for crop in crops
            ]
            local_tokens.append(tokens)

        return {
            "global_crops": (centroids[global_crops], embeddings[global_crops]),
            "local_crops": (centroids[local_crops], embeddings[local_crops]),
            "global_crop_tokens": (centroids[global_tokens], embeddings[global_tokens]),
            "local_crop_tokens": (centroids[local_tokens], embeddings[local_tokens]),
        }
