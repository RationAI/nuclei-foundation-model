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


SLIDES_ROOT = Path("/workspace/nuclei/subsets")
# The number of global and local crops is fixed to global_crop_count=2 and local_crop_count=8
LOCAL_CROP_COUNT = 8


class NucleiDataset(Dataset):
    def __init__(
        self,
        data_root: Path,  # Path("/flash/project_465002057/nuclei/subsets_results/")
        global_crop_k: int = 4096,
        local_crop_k: int = 768,
        local_crop_tokens: int = 48,
        global_crop_tokens: int = 256,
        alpha: float = 0.8,
    ) -> None:
        self.data_root = data_root
        self._load_dataframe()
        self.alpha = alpha
        self.global_crop_k = global_crop_k
        self.local_crop_k = local_crop_k
        self.global_crop_tokens = global_crop_tokens
        self.local_crop_tokens = local_crop_tokens

    def _get_slide_data_path(self, row):
        path = Path(row["path"])
        return (
            self.data_root
            / path.relative_to(SLIDES_ROOT).parent
            / f"results/nuclei/slide_id={row['id']}"
        )

    def _load_dataframe(self) -> None:
        df = pd.read_parquet(list(self.data_root.glob("*/*/results/slides/*.parquet")))
        self.paths = df.apply(self._get_slide_data_path, axis=1).to_list()

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

    def _build_graph(self, points) -> list[list[tuple[int, float]]]:
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

    def _find_hybrid_component(
        self,
        idx: int,
        k: int,
        graph,
        centroids: NDArray,
        indices: NDArray | None = None,
    ) -> list[int]:
        n_points = len(centroids)
        start_point_coords = centroids[idx]

        pq = []
        in_component = np.zeros(n_points, dtype=bool)
        component_indices = []

        heapq.heappush(pq, (0, idx))

        while len(pq) != 0 and len(component_indices) < k:
            _, current_idx = heapq.heappop(pq)

            if in_component[current_idx]:
                continue

            in_component[current_idx] = True
            component_indices.append(current_idx)

            neighbor_idxs = [idx for idx, _ in graph[current_idx]]
            start_dists = np.linalg.norm(
                centroids[neighbor_idxs] - start_point_coords, axis=1
            )
            for i, (neighbor_idx, edge_dist) in enumerate(graph[current_idx]):
                if (indices is None or neighbor_idx in indices) and not in_component[
                    neighbor_idx
                ]:
                    start_dist = start_dists[i]
                    hybrid_cost = self.alpha * edge_dist + (1 - self.alpha) * start_dist
                    heapq.heappush(pq, (hybrid_cost, neighbor_idx))
        return component_indices

    def __getitem__(self, idx: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        df = pd.read_parquet(self.paths[idx])
        centroids = np.stack(df.centroid.values)
        embeddings = np.stack(df.embedding.values)

        graph = self._build_graph(centroids)

        # Global crop token generation
        global_crops = []
        seed = random.randint(0, len(centroids))
        global_crops.append(
            self._find_hybrid_component(seed, self.global_crop_k, graph, centroids)
        )

        # Randomly select a seed for the second global crop
        seed = random.randint(int(self.global_crop_k * 0.8), self.global_crop_k)
        global_crops.append(
            self._find_hybrid_component(
                global_crops[0][seed], self.global_crop_k, graph, centroids
            )
        )

        # Global crop token generation
        global_crop_tokenss = []
        for global_idx in global_crops:
            global_crop_tokens = self._get_tokens(
                centroids[global_idx], centroids, self.global_crop_tokens
            )
            global_crop_tokenss.append(global_crop_tokens)

        # Local crop generation
        local_crops = []
        for i, global_crop_tokens in enumerate(global_crop_tokenss):
            crops = []
            tokens = np.random.choice(
                global_crop_tokens, size=LOCAL_CROP_COUNT, replace=False
            )
            for token in tokens:
                crop = self._find_hybrid_component(
                    token, self.local_crop_k, graph, centroids, global_crops[i]
                )
                crops.append(crop)
            local_crops.append(crops)

        # Local crop token generation
        local_crop_tokenss = [[] for _ in range(len(global_crops))]
        for i, crops in enumerate(local_crops):
            for crop in crops:
                local_crop_tokenss[i].append(
                    self._get_tokens(centroids[crop], centroids, self.local_crop_tokens)
                )

        return {
            "global_crops": (centroids[global_crops], embeddings[global_crops]),
            "local_crops": (centroids[local_crops], embeddings[local_crops]),
            "global_crop_tokens": (
                centroids[global_crop_tokenss],
                embeddings[global_crop_tokenss],
            ),
            "local_crop_tokens": (
                centroids[local_crop_tokenss],
                embeddings[local_crop_tokenss],
            ),
        }
