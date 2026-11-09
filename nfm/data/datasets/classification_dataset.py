import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from nfm.data.nuclei import View, crop_view


class ClassificationNucleiDataset(Dataset[list[View]]):
    """Tiles of one Radboud slide with nucleus labels.

    A tile is positive if at least 50 % of its nuclei are. Training draws tiles
    centered on random nuclei. Evaluation uses the occupied cells of a fixed
    non-overlapping grid, subsampled reproducibly per slide.
    """

    def __init__(
        self,
        slides: pd.DataFrame,
        annotation_path: str,
        tiles_per_slide: int,
        grid: bool = False,
        field: float = 112.0,
        efd_order: int = 16,
    ) -> None:
        self.slides = slides.reset_index(drop=True)
        self.annotation_path = annotation_path
        self.tiles_per_slide = tiles_per_slide
        self.grid = grid
        self.field = field
        self.efd_order = efd_order

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

    def __getitem__(self, idx: int) -> list[View]:
        slide = self.slides.iloc[idx]
        nuclei = pd.read_parquet(slide.slide_nuclei_path, columns=["id", "polygon"])
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
        polygons = (
            np.stack(nuclei["polygon"]).astype(np.float32).reshape(len(nuclei), -1, 2)
            * mpp
        )
        tiles = []
        for center in self.tile_centers(polygons.mean(1), idx):
            tile = crop_view(
                polygons, center, self.field, order=self.efd_order, labels=labels
            )
            tile["label"] = np.float32(tile["labels"].mean() >= 0.5)
            tiles.append(tile)
        return tiles
