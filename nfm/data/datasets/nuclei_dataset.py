from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyarrow import FixedSizeListArray
from torch.utils.data import Dataset

from nfm.data.nuclei import Augmentation, View, crop_view, radial_polygons


def _to_numpy(column: FixedSizeListArray) -> NDArray[np.float32]:
    flat = column.combine_chunks().flatten().to_numpy().astype(np.float32)
    return flat.reshape(len(column), -1)


class NucleiDataset(Dataset[dict[str, list[View]]]):
    """Global and local views of the nuclei of a random region.

    Follows the Virchow2 recipe in nuclei instead of pixels: a random field of
    view (mixed magnification) and extended-context translation, where all views
    are crops of one larger context. View centers are random nuclei, so no view
    is empty. Every nucleus keeps its id so views can be matched nucleus by nucleus.
    """

    def __init__(
        self,
        slides: pd.DataFrame,
        nuclei_path: str,
        global_augmentation: Augmentation,
        local_augmentation: Augmentation,
        global_views: int = 2,
        local_views: int = 8,
        fields: tuple[float, ...] = (56.0, 112.0, 224.0, 448.0),
        field_weights: tuple[float, ...] = (0.2, 0.4, 0.2, 0.2),
        context_scale: float = 1.75,
        local_scale: float = 0.4375,
        max_global_nuclei: int = 512,
        max_local_nuclei: int = 128,
        max_density: float = 0.0125,
        efd_order: int = 16,
    ) -> None:
        """Initializes the dataset.

        Args:
            slides: Slides to sample regions from.
            nuclei_path: Root of the lance nuclei store.
            global_augmentation: Applied to each global view.
            local_augmentation: Applied to each local view.
            global_views: Number of global views.
            local_views: Number of local views.
            fields: Side lengths of the global views in micrometers.
            field_weights: Probabilities of the fields.
            context_scale: Context side length relative to the global views.
            local_scale: Local view side length relative to the global views.
            max_global_nuclei: Token budget of a global view.
            max_local_nuclei: Token budget of a local view.
            max_density: Nuclei per um^2, sizes the region query.
            efd_order: Number of harmonics of the shape descriptors.
        """
        self.slides = slides.reset_index(drop=True)
        self.nuclei_path = nuclei_path
        self.global_augmentation = global_augmentation
        self.local_augmentation = local_augmentation
        self.global_views = global_views
        self.local_views = local_views
        self.fields = fields
        self.field_weights = field_weights
        self.context_scale = context_scale
        self.local_scale = local_scale
        self.max_global_nuclei = max_global_nuclei
        self.max_local_nuclei = max_local_nuclei
        self.max_density = max_density
        self.efd_order = efd_order

    def __len__(self) -> int:
        return len(self.slides)

    def load_context(
        self, slide: Any, field: float
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Contours in micrometers within a context around a random nucleus."""
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
        # A view rotated around a center anywhere in the context stays within the
        # circle enclosing the square context.
        radius = field / np.sqrt(2)
        nuclei = int(self.max_density * np.pi * radius**2)
        table = ds.to_table(
            columns=["points", "radial_distances"],
            nearest={
                "column": "points",
                "q": seed,
                "k": min(count, nuclei),
                "metric": "euclidean",
            },
            disable_scoring_autoprojection=True,
        )
        mpp = np.array([slide.mpp_x, slide.mpp_y], dtype=np.float32)
        points = _to_numpy(table["points"])
        polygons = radial_polygons(points, _to_numpy(table["radial_distances"]))
        return polygons * mpp, seed * mpp

    def __getitem__(self, idx: int) -> dict[str, list[View]]:
        slide = self.slides.iloc[idx]
        size = np.random.choice(self.fields, p=self.field_weights)
        context = size * self.context_scale
        polygons, seed = self.load_context(slide, context)
        centroids = polygons.mean(1)
        ids = np.arange(len(polygons))

        def view(size: float, max_nuclei: int, augmentation: Augmentation) -> View:
            # Centers keep the whole view inside the context.
            inside = np.abs(centroids - seed).max(1) <= (context - size) / 2
            center = centroids[np.random.choice(np.flatnonzero(inside))]
            return crop_view(
                polygons,
                center,
                size,
                max_nuclei,
                augmentation,
                self.efd_order,
                ids=ids,
            )

        local = size * self.local_scale
        return {
            "global": [
                view(size, self.max_global_nuclei, self.global_augmentation)
                for _ in range(self.global_views)
            ],
            "local": [
                view(local, self.max_local_nuclei, self.local_augmentation)
                for _ in range(self.local_views)
            ],
        }
