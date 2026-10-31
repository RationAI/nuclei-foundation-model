from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from nfm.data.rendering import render


class ClassificationNucleiDataset(Dataset[dict[str, Any]]):
    """Rendered tiles of one Radboud slide, positive if >= 50 % nuclei are positive.

    Training draws tiles centered on random nuclei. Evaluation uses the occupied
    cells of a fixed non-overlapping grid, subsampled reproducibly per slide.
    """

    def __init__(
        self,
        slides: pd.DataFrame,
        annotation_path: str,
        tiles_per_slide: int,
        grid: bool = False,
        field: float = 128.0,
        size: int = 128,
    ) -> None:
        self.slides = slides.reset_index(drop=True)
        self.annotation_path = annotation_path
        self.tiles_per_slide = tiles_per_slide
        self.grid = grid
        self.field = field
        self.size = size

    def __len__(self) -> int:
        return len(self.slides)

    def tile_centers(self, centroids: np.ndarray, idx: int) -> np.ndarray:
        if not self.grid:
            picks = np.random.randint(len(centroids), size=self.tiles_per_slide)
            return centroids[picks]
        cells = np.unique(np.floor(centroids / self.field), axis=0)
        count = min(len(cells), self.tiles_per_slide)
        cells = np.random.default_rng(idx).permutation(cells)[:count]
        return ((cells + 0.5) * self.field).astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        slide = self.slides.iloc[idx]
        nuclei = pd.read_parquet(
            slide.slide_nuclei_path, columns=["id", "centroid", "polygon"]
        )
        labels = (
            pd.read_parquet(
                f"{self.annotation_path}/{slide.slide_id}.parquet",
                columns=["id", "annot_label"],
            )
            .set_index("id")["annot_label"]
            .loc[nuclei["id"]]
            .to_numpy(np.float32)
            if slide.is_carcinoma
            else np.zeros(len(nuclei), dtype=np.float32)
        )
        mpp = np.array([slide.mpp_x, slide.mpp_y], dtype=np.float32)
        centroids = np.stack(nuclei["centroid"]).astype(np.float32) * mpp
        polygons = (
            np.stack(nuclei["polygon"]).astype(np.float32).reshape(len(nuclei), -1, 2)
            * mpp
        )

        centers = self.tile_centers(centroids, idx)
        images = [render(polygons, c, self.field, self.size) for c in centers]
        tile_labels = [
            labels[np.abs(centroids - c).max(axis=1) <= self.field / 2].mean() >= 0.5
            for c in centers
        ]
        return {
            "images": torch.from_numpy(np.stack(images)).unsqueeze(1),
            "labels": torch.tensor(tile_labels, dtype=torch.float32),
        }
