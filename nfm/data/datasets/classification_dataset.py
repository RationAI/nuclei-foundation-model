import random

import numpy as np
import pandas as pd
import torch
from degraph import build_spatial_graph

from nfm.data.augmentations import PolygonAugmentation
from nfm.data.datasets.nuclei_dataset import NucleiDataset, Sample


class ClassificationNucleiDataset(NucleiDataset):
    def __init__(
        self,
        slides_path: str,
        nuclei_path: str,
        global_crop_k: int = 2048,
        alpha: float = 0.85,
        efd_order: int = 16,
        augmentation: PolygonAugmentation | list[PolygonAugmentation] | None = None,
    ) -> None:
        super().__init__(
            slides_path=slides_path,
            nuclei_path=nuclei_path,
            global_crop_k=global_crop_k,
            alpha=alpha,
            efd_order=efd_order,
            augmentation=augmentation,
        )

    def __getitem__(self, idx: int) -> Sample:
        slide = self.slides.iloc[idx]
        df = pd.read_parquet(
            f"{self.nuclei_path}/{slide.organ}/{slide.dataset}/annotation_labels/slide_id={slide.id}",
            columns=["polygon", "annot_coverage"],
        )
        polygons = np.stack(df["polygon"]).reshape(-1, 64, 2).astype(np.float32)
        labels = np.stack(df["annot_coverage"])

        points, keep_indices = self.downsample_points(
            polygons.mean(axis=1), limit=int(self.global_crop_k / (1 - self.alpha))
        )
        graph = build_spatial_graph(points)

        seed = random.randint(0, len(points) - 1)
        comp_indices = self.find_component(seed, self.global_crop_k, graph, points)
        indices = keep_indices[comp_indices]

        crop_polygons = polygons[indices]
        crop_polygons = self.augmentation(crop_polygons)

        centroids, efds = self.polygon_to_efd(
            crop_polygons, mpp_x=slide.mpp_x, mpp_y=slide.mpp_y
        )
        labels = labels[indices]

        return {
            "pos": centroids,
            "efds": torch.from_numpy(efds),
            "labels": torch.from_numpy(labels > 0.5)[:, None],
            "seq_len": len(centroids),
        }
