from collections.abc import Callable

import lance
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyarrow import FixedSizeListArray
from torch.utils.data import Dataset

from nfm.data.rendering import render


def _to_numpy(column: FixedSizeListArray) -> NDArray[np.float32]:
    flat = column.combine_chunks().flatten().to_numpy().astype(np.float32)
    return flat.reshape(len(column), -1)


class NucleiDataset(Dataset[NDArray[np.uint8]]):
    """Multiple rendered views of a random tile from the whole cohort."""

    def __init__(
        self,
        slides: pd.DataFrame,
        nuclei_path: str,
        augmentation: Callable[..., dict[str, NDArray[np.uint8]]],
        views: int = 4,
        field: float = 128.0,
        size: int = 128,
        region_nuclei: int = 512,
    ) -> None:
        self.slides = slides.reset_index(drop=True)
        self.nuclei_path = nuclei_path
        self.views = views
        self.field = field
        self.size = size
        self.augmentation = augmentation
        # Must cover a circle of radius field / sqrt(2) around the tile center.
        self.region_nuclei = region_nuclei

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> NDArray[np.uint8]:
        slide = self.slides.iloc[idx]
        ds = lance.dataset(
            f"{self.nuclei_path}/organ={slide.organ}/dataset={slide.dataset}/slide_id={slide.id}/nuclei.lance"
        )
        count = ds.count_rows()
        seed = _to_numpy(
            ds.take([np.random.randint(count)], columns=["points"])["points"]
        )
        table = ds.to_table(
            columns=["points", "radial_distances"],
            nearest={
                "column": "points",
                "q": seed[0],
                "k": min(count, self.region_nuclei),
                "metric": "euclidean",
            },
            disable_scoring_autoprojection=True,
        )
        points = _to_numpy(table["points"])
        radii = _to_numpy(table["radial_distances"])

        # The store keeps radial contours in a [sin, cos] convention, in pixels.
        angles = np.arange(radii.shape[1]) * 2 * np.pi / radii.shape[1]
        directions = np.stack([np.sin(angles), np.cos(angles)], axis=-1)
        mpp = np.array([slide.mpp_x, slide.mpp_y], dtype=np.float32)
        polygons = (points[:, None] + radii[..., None] * directions) * mpp
        polygons = polygons.astype(np.float32)

        # Oversized so that any rotation still fills the final tile.
        image = render(
            polygons,
            seed[0] * mpp,
            self.field * np.sqrt(2),
            int(np.ceil(self.size * np.sqrt(2))),
        )
        return np.stack(
            [self.augmentation(image=image)["image"] for _ in range(self.views)]
        )
