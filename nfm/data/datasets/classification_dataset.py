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
        polygons = np.stack(df["polygons"]).reshape(-1, 64, 2).astype(np.float32)
        labels = np.stack(df["annot_coverage"])

        points, keep_indices = self.downsample_points(
            polygons.mean(axis=1), limit=int(self.global_crop_k / (1 - self.alpha))
        )
        graph = build_spatial_graph(points)

        # crop generation
        seed = random.randint(0, len(points) - 1)
        indices = self.find_component(seed, self.global_crop_k, graph, points)

        centroids, efds = self.polygon_to_efd(
            polygons[keep_indices], mpp_x=slide.mpp_x, mpp_y=slide.mpp_y
        )

        indices = block_spatial_sort(centroids, self.block_size)

        centroids = centroids[indices]
        efds = efds[indices]
        labels = labels[keep_indices][indices]

        _, knn = self.nbrs.fit(centroids).kneighbors(centroids)
        pad_len = self.global_crop_k - len(centroids)

        return {
            "pos": torch.from_numpy(np.pad(centroids, ((0, pad_len), (0, 0)))),
            "efds": torch.from_numpy(np.pad(efds, ((0, pad_len), (0, 0)))),
            "knn": torch.from_numpy(
                np.pad(knn, ((0, pad_len), (0, 0)), constant_values=-1)
            ),
            "labels": torch.from_numpy(labels > 0.5),
            "seq_len": len(centroids),
        }
