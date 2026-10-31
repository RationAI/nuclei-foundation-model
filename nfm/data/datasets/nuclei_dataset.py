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


class NucleiDataset(Dataset[dict[str, NDArray[np.uint8]]]):
    """Global and local views of a random region, rendered at a random resolution.

    Follows the Virchow2 recipe: mixed magnification and extended-context
    translation, where all views are crops of one larger context with minimal resizing.
    """

    def __init__(
        self,
        slides: pd.DataFrame,
        nuclei_path: str,
        global_augmentation: Callable[..., dict[str, NDArray[np.uint8]]],
        local_augmentation: Callable[..., dict[str, NDArray[np.uint8]]],
        global_views: int = 2,
        local_views: int = 8,
        context_size: int = 392,
        mpps: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
        mpp_weights: tuple[float, ...] = (0.2, 0.4, 0.2, 0.2),
        max_density: float = 0.0125,
    ) -> None:
        self.slides = slides.reset_index(drop=True)
        self.nuclei_path = nuclei_path
        self.global_augmentation = global_augmentation
        self.local_augmentation = local_augmentation
        self.global_views = global_views
        self.local_views = local_views
        self.context_size = context_size
        self.mpps = mpps
        self.mpp_weights = mpp_weights
        self.max_density = max_density  # nuclei per um^2, sizes the region query

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> dict[str, NDArray[np.uint8]]:
        slide = self.slides.iloc[idx]
        ds = lance.dataset(
            f"{self.nuclei_path}/organ={slide.organ}/dataset={slide.dataset}/slide_id={slide.id}/nuclei.lance"
        )
        count = ds.count_rows()
        seed = _to_numpy(
            ds.take([np.random.randint(count)], columns=["points"])["points"]
        )
        field = self.context_size * np.random.choice(self.mpps, p=self.mpp_weights)
        # The query is a circle that has to enclose the square context.
        nuclei = int(self.max_density * np.pi * field**2 / 2)
        table = ds.to_table(
            columns=["points", "radial_distances"],
            nearest={
                "column": "points",
                "q": seed[0],
                "k": min(count, nuclei),
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

        context = render(polygons, seed[0] * mpp, field, self.context_size)
        return {
            "global": np.stack(
                [
                    self.global_augmentation(image=context)["image"]
                    for _ in range(self.global_views)
                ]
            ),
            "local": np.stack(
                [
                    self.local_augmentation(image=context)["image"]
                    for _ in range(self.local_views)
                ]
            ),
        }
