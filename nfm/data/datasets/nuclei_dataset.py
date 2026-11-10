import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyarrow import FixedSizeListArray
from torch.utils.data import Dataset

from nfm.data.nuclei import (
    Augmentation,
    Descriptor,
    View,
    crop_view,
    radial_distances,
    radial_polygons,
)


def _to_numpy(column: FixedSizeListArray) -> NDArray[np.float32]:
    flat = column.combine_chunks().flatten().to_numpy().astype(np.float32)
    return flat.reshape(len(column), -1)


class NucleiDataset(Dataset[list[View]]):
    """Views of the nuclei of a random field, each a random square crop of it.

    A crop covers a random share of the field area, like a random resized crop
    of a tile, but keeps the physical scale. View centers are random nuclei, so
    no view is empty.
    """

    def __init__(
        self,
        slides: pd.DataFrame,
        nuclei_path: str,
        augmentation: Augmentation,
        views: int = 4,
        field: float = 128.0,
        scale: tuple[float, float] = (0.3, 1.0),
        region_nuclei: int = 512,
        descriptor: Descriptor = radial_distances,
    ) -> None:
        """Initializes the dataset.

        Args:
            slides: Slides to sample fields from.
            nuclei_path: Root of the lance nuclei store.
            augmentation: Applied to each view.
            views: Number of views.
            field: Side length of the field in micrometers.
            scale: Range of the share of the field area covered by a view.
            region_nuclei: Nuclei loaded around the field center. Must cover a
                circle of radius field / sqrt(2), so rotated views stay inside.
            descriptor: Shape features of a contour in the view frame.
        """
        self.slides = slides.reset_index(drop=True)
        self.nuclei_path = nuclei_path
        self.augmentation = augmentation
        self.views = views
        self.field = field
        self.scale = scale
        self.region_nuclei = region_nuclei
        self.descriptor = descriptor

    def __len__(self) -> int:
        return len(self.slides)

    def load_region(self, slide: pd.Series) -> tuple[NDArray[np.float32], ...]:
        """Contours in micrometers of the nuclei around a random nucleus."""
        # Imported here so the main process never loads lance, which is not
        # fork-safe and would make forked DataLoader workers crash.
        import lance

        ds = lance.dataset(
            f"{self.nuclei_path}/organ={slide.organ}/dataset={slide.dataset}/slide_id={slide.id}/nuclei.lance"
        )
        count = ds.count_rows()
        seed = _to_numpy(
            ds.take([np.random.randint(count)], columns=["points"])["points"]
        )[0]
        table = ds.to_table(
            columns=["points", "radial_distances"],
            nearest={
                "column": "points",
                "q": seed,
                "k": min(count, self.region_nuclei),
                "metric": "euclidean",
            },
            disable_scoring_autoprojection=True,
        )
        mpp = np.array([slide.mpp_x, slide.mpp_y], dtype=np.float32)
        points = _to_numpy(table["points"])
        polygons = radial_polygons(points, _to_numpy(table["radial_distances"]))
        return polygons * mpp, seed * mpp

    def __getitem__(self, idx: int) -> list[View]:
        polygons, seed = self.load_region(self.slides.iloc[idx])
        centroids = polygons.mean(1)
        offsets = np.abs(centroids - seed).max(1)
        views = []
        for _ in range(self.views):
            size = self.field * np.sqrt(np.random.uniform(*self.scale))
            # Centers keep the whole view inside the field, the seed always can.
            inside = offsets <= max((self.field - size) / 2, offsets.min())
            center = centroids[np.random.choice(np.flatnonzero(inside))]
            views.append(
                crop_view(
                    polygons,
                    center,
                    size,
                    augmentation=self.augmentation,
                    descriptor=self.descriptor,
                )
            )
        return views
