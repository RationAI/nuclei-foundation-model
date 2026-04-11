import random

import numpy as np
import pandas as pd
import torch
from degraph import build_spatial_graph

from nfm.data.datasets.nuclei_dataset import NucleiDataset, Sample
from nfm.modeling.block_mask import block_spatial_sort


class ClassificationNucleiDataset(NucleiDataset):
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

        # crop generation
        seed = random.randint(0, len(points) - 1)
        comp_indices = self.find_component(seed, self.global_crop_k, graph, points)
        indices = keep_indices[comp_indices]

        centroids, efds = self.polygon_to_efd(
            polygons[indices], mpp_x=slide.mpp_x, mpp_y=slide.mpp_y
        )

        sort_indices = block_spatial_sort(centroids, self.block_size)

        centroids = centroids[sort_indices]
        efds = efds[sort_indices]
        labels = labels[indices][sort_indices]

        _, knn = self.nbrs.fit(centroids).kneighbors(centroids)

        return {
            "pos": torch.from_numpy(centroids),
            "efds": torch.from_numpy(efds),
            "knn": torch.from_numpy(knn),
            "labels": torch.from_numpy(labels > 0.5)[:, None],
            "seq_len": len(centroids),
        }
