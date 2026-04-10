import heapq
import random

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from degraph import build_spatial_graph
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.utils.data import Dataset

from nfm.data.efd import elliptic_fourier_descriptors
from nfm.modeling.block_mask import block_spatial_sort


type Sample = dict[str, Tensor]


class NucleiDataset(Dataset[Sample]):
    def __init__(
        self,
        slides_path: str,
        nuclei_path: str,
        global_crop_k: int = 4096,
        alpha: float = 0.85,
        efd_order: int = 16,
        knn: int = 16,
        block_size: int = 128,
    ) -> None:
        self.slides = pd.read_parquet(
            slides_path, columns=["id", "mpp_x", "mpp_y", "dataset", "organ"]
        )
        self.block_size = block_size
        self.nuclei_path = nuclei_path
        self.global_crop_k = global_crop_k
        self.alpha = alpha
        self.efd_order = efd_order
        self.nbrs = NearestNeighbors(n_neighbors=knn, metric="euclidean")

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

    def radial_to_polygon(
        self, points: NDArray[np.float32], radial_distances: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        t = np.linspace(0, 1, radial_distances.shape[-1] + 1, dtype=np.float32)[:-1]
        cos = np.cos(2 * np.pi * t)
        sin = np.sin(2 * np.pi * t)

        polar = radial_distances[..., None] * np.stack([sin, cos], axis=-1)
        return points[:, None] + polar

    def polygon_to_efd(
        self, polygons: NDArray[np.float32], mpp_x: float, mpp_y: float
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        polygons[..., 0] *= mpp_x
        polygons[..., 1] *= mpp_y

        centroids = polygons.mean(axis=1)
        efd = elliptic_fourier_descriptors(polygons.astype(np.float64), self.efd_order)

        return centroids, efd.reshape(-1, self.efd_order * 4).astype(np.float32)

    def read_radial_distances(
        self, pf: pq.ParquetFile, indices: NDArray[np.intp]
    ) -> NDArray[np.float32]:
        rg_lengths = [
            pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)
        ]
        rg_starts = np.concatenate(([0], np.cumsum(rg_lengths)))
        rg_matches = np.searchsorted(rg_starts, indices, side="right") - 1
        unique_rgs = np.unique(rg_matches)

        # Read only the necessary row groups
        radial_table = pf.read_row_groups(unique_rgs, columns=["radial_distances"])

        # Map sorted_file_indices to indices in the new concatenated table
        # We need to adjust indices based on the row groups we actually read
        read_rg_lengths = [rg_lengths[i] for i in unique_rgs]
        new_rg_starts = np.concatenate(([0], np.cumsum(read_rg_lengths)[:-1]))

        # Create lookup: RG_index -> start_in_new_table
        rg_lookup = np.zeros(pf.num_row_groups, dtype=np.int64)
        rg_lookup[unique_rgs] = new_rg_starts

        # Calculate indices in the new table
        indices_in_new = indices - rg_starts[rg_matches] + rg_lookup[rg_matches]

        # Extract values and restore original order
        return (
            radial_table["radial_distances"]
            .take(indices_in_new)
            .combine_chunks()
            .values.to_numpy()
            .reshape(-1, 64)
        )

    def downsample_points(
        self, points: NDArray[np.float32], limit: int
    ) -> tuple[NDArray[np.float32], NDArray[np.intp]]:
        """Downsample points to a specified limit using a distance-based approach."""
        if len(points) <= limit:
            return points, np.arange(len(points))

        center_idx = random.randint(0, len(points) - 1)
        dists = np.linalg.norm(points - points[center_idx], axis=1)
        keep_indices = np.argpartition(dists, limit)[:limit]

        return points[keep_indices], keep_indices

    def __getitem__(self, idx: int) -> Sample:
        slide = self.slides.iloc[idx]
        pf = pq.ParquetFile(
            f"{self.nuclei_path}/organ={slide.organ}/dataset={slide.dataset}/slide_id={slide.id}/nuclei.parquet"
        )
        points_col = pf.read(columns=["points"])["points"].combine_chunks()
        points = points_col.values.to_numpy().reshape(-1, 2).astype(np.float32)

        points, keep_indices = self.downsample_points(
            points, limit=int(self.global_crop_k / (1 - self.alpha))
        )
        graph = build_spatial_graph(points)

        # crop generation
        seed = random.randint(0, len(points) - 1)
        indices = self.find_component(seed, self.global_crop_k, graph, points)

        polygons = self.radial_to_polygon(
            points[indices], self.read_radial_distances(pf, keep_indices[indices])
        )
        centroids, efds = self.polygon_to_efd(
            polygons, mpp_x=slide.mpp_x, mpp_y=slide.mpp_y
        )

        sort_indices = block_spatial_sort(centroids, self.block_size)
        centroids = centroids[sort_indices]
        efds = efds[sort_indices]

        _, knn = self.nbrs.fit(centroids).kneighbors(centroids)
        pad_len = self.global_crop_k - len(centroids)

        return {
            "pos": torch.from_numpy(np.pad(centroids, ((0, pad_len), (0, 0)))),
            "efds": torch.from_numpy(np.pad(efds, ((0, pad_len), (0, 0)))),
            "global_knn": torch.from_numpy(
                np.pad(knn, ((0, pad_len), (0, 0)), constant_values=-1)
            ),
            "local_knn": torch.from_numpy(
                np.pad(knn[:, :1], ((0, pad_len), (0, 0)), constant_values=-1)
            ),
            "seq_len": len(centroids),
        }
