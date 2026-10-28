import random

import numpy as np
import pandas as pd
import torch
from degraph import build_spatial_graph

from nfm.data.datasets.nuclei_dataset import NucleiDataset, Sample


class ClassificationNucleiDataset(NucleiDataset):
    def __init__(
        self,
        slides: pd.DataFrame,
        annotation_path: str,
        crop_size: int = 4096,
        alpha: float = 0.8,
        efd_order: int = 16,
    ) -> None:
        self.slides = slides.reset_index(drop=True)

        annotations = pd.read_parquet(
            annotation_path, columns=["slide_id", "id", "annot_label"]
        )
        annotations = annotations[annotations["slide_id"].isin(self.slides.slide_id)]
        self.annotations = {
            slide_id: group.set_index("id")["annot_label"]
            for slide_id, group in annotations.groupby("slide_id")
        }
        self.positivity = {
            slide.slide_id: float(self.annotations[slide.slide_id].mean())
            if slide.is_carcinoma
            else 0.0
            for slide in self.slides.itertuples()
        }
        self.crop_size = crop_size
        self.alpha = alpha
        self.efd_order = efd_order

    def __getitem__(self, idx: int) -> Sample:
        slide = self.slides.iloc[idx]
        nuclei = pd.read_parquet(slide.slide_nuclei_path).sort_values("id")
        mpps = np.array([slide.mpp_x, slide.mpp_y], dtype=np.float32)
        points = np.stack(nuclei["centroid"]).astype(np.float32) * mpps
        labels = (
            self.annotations[slide.slide_id].loc[nuclei["id"]].to_numpy(np.float32)
            if slide.is_carcinoma
            else np.zeros(len(nuclei), dtype=np.float32)
        )
        indices = np.arange(len(nuclei))

        # With 75% positive slides, seed 2/3 of their crops in positive nuclei.
        positive_seed = slide.is_carcinoma and torch.rand(()).item() < 2 / 3
        valid_seeds = np.flatnonzero(labels == int(positive_seed))
        seed = (
            int(random.choice(valid_seeds))
            if len(valid_seeds)
            else random.randrange(len(nuclei))
        )
        center = points[seed]

        limit = int(self.crop_size / max(1 - self.alpha, 1e-4))
        if len(points) > limit:
            distances = np.linalg.norm(points - center, axis=1)
            indices = np.argpartition(distances, limit - 1)[:limit]

        # Delaunay graph construction requires distinct points.
        quantized = np.round(points[indices] / 1e-4).astype(np.int64)
        _, unique = np.unique(quantized, axis=0, return_index=True)
        indices = indices[np.sort(unique)]
        crop_points = points[indices]
        graph = build_spatial_graph(crop_points)
        seed = int(np.argmin(np.linalg.norm(crop_points - center, axis=1)))
        indices = indices[self.find_component(seed, self.crop_size, graph, crop_points)]

        polygons = (
            np.stack(nuclei["polygon"].iloc[indices])
            .reshape(len(indices), -1, 2)
            .astype(np.float32)
        )
        centroids, efds = self.polygon_to_efd(polygons * mpps)

        return {
            "pos": centroids,
            "efds": torch.from_numpy(efds),
            "labels": torch.from_numpy(labels[indices])[:, None],
            "seq_len": len(centroids),
        }
