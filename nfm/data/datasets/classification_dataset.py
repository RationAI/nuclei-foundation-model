import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from nfm.data.nuclei import View, crop_view


class ClassificationNucleiDataset(Dataset[list[View]]):
    """Tiles of one Radboud slide with nucleus labels.

    A tile is positive if at least 50 % of its nuclei are. With `tiles_per_slide`,
    tiles are centered on random nuclei, for training. Without it, the whole
    slide is divided into the cells of a fixed grid, so every nucleus is in
    exactly one tile, for evaluation.
    """

    def __init__(
        self,
        slides: pd.DataFrame,
        annotation_path: str,
        tiles_per_slide: int | None = None,
        field: float = 112.0,
        efd_order: int = 16,
    ) -> None:
        self.slides = slides.reset_index(drop=True)
        self.annotation_path = annotation_path
        self.tiles_per_slide = tiles_per_slide
        self.field = field
        self.efd_order = efd_order

    def __len__(self) -> int:
        return len(self.slides)

    def tile_nuclei(
        self, centroids: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray | slice]]:
        """Center of each tile and the nuclei it is cropped from."""
        if self.tiles_per_slide is not None:
            picks = np.random.randint(len(centroids), size=self.tiles_per_slide)
            return [(centroids[i], slice(None)) for i in picks]
        cells = np.floor(centroids / self.field).astype(np.int64)
        cells, members = np.unique(cells, axis=0, return_inverse=True)
        order = np.argsort(members, kind="stable")
        groups = np.split(order, np.cumsum(np.bincount(members))[:-1])
        centers = ((cells + 0.5) * self.field).astype(np.float32)
        return list(zip(centers, groups, strict=True))

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
        for center, members in self.tile_nuclei(polygons.mean(1)):
            tile = crop_view(
                polygons[members],
                center,
                self.field,
                order=self.efd_order,
                labels=labels[members],
            )
            tile["label"] = np.float32(tile["labels"].mean() >= 0.5)
            tiles.append(tile)
        return tiles
