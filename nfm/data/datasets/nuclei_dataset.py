from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyarrow import FixedSizeListArray
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from nfm.data.neighborhoods import Neighborhoods, match_cells
from nfm.data.nuclei import (
    Augmentation,
    Descriptor,
    make_view,
    radial_distances,
    radial_polygons,
)


def _to_numpy(column: FixedSizeListArray) -> NDArray[np.float32]:
    flat = column.combine_chunks().flatten().to_numpy().astype(np.float32)
    return flat.reshape(len(column), -1)


class NucleiDataset(Dataset[dict[str, Any]]):
    """Overlapping views of the nuclei around a random nucleus.

    Each view is the nuclei closest to its own center, a random number of them,
    like a random crop of a variable share of the region. Centers are drawn among
    the nuclei closest to the region seed, so the views overlap. Voronoi cells of
    different views that hold the same nuclei are paired for the objective.
    """

    def __init__(
        self,
        slides: pd.DataFrame,
        nuclei_path: str,
        augmentation: Augmentation,
        neighborhoods: Neighborhoods,
        views: int = 4,
        view_nuclei: tuple[int, int] = (1024, 4096),
        center_nuclei: int = 1024,
        region_nuclei: int = 8192,
        min_overlap: float = 0.3,
        descriptor: Descriptor = radial_distances,
    ) -> None:
        """Initializes the dataset.

        Args:
            slides: Slides to sample regions from.
            nuclei_path: Root of the lance nuclei store.
            augmentation: Applied to each view.
            neighborhoods: Adds the nucleus neighbors and Voronoi cells to a view.
            views: Number of views.
            view_nuclei: Range of the number of nuclei in a view.
            center_nuclei: View centers are among this many nuclei closest to the
                seed, which sets how far apart the views are.
            region_nuclei: Nuclei loaded around the seed. A view is cut short
                where it reaches beyond them.
            min_overlap: Least Jaccard index of the nuclei of two paired cells.
            descriptor: Shape features of a contour in the view frame.
        """
        self.slides = slides.reset_index(drop=True)
        self.nuclei_path = nuclei_path
        self.augmentation = augmentation
        self.neighborhoods = neighborhoods
        self.views = views
        self.view_nuclei = view_nuclei
        self.center_nuclei = center_nuclei
        self.region_nuclei = region_nuclei
        self.min_overlap = min_overlap
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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        polygons, seed = self.load_region(self.slides.iloc[idx])
        centroids = polygons.mean(1)
        tree = cKDTree(centroids)
        _, near_seed = tree.query(seed, min(self.center_nuclei, len(centroids)))
        ids = np.arange(len(polygons))
        views = []
        for _ in range(self.views):
            center = centroids[np.random.choice(np.atleast_1d(near_seed))]
            count = np.random.randint(self.view_nuclei[0], self.view_nuclei[1] + 1)
            _, nuclei = tree.query(center, min(count, len(centroids)))
            nuclei = np.atleast_1d(nuclei)
            view = make_view(
                polygons[nuclei],
                center,
                self.augmentation,
                self.descriptor,
                ids=ids[nuclei],
            )
            views.append(self.neighborhoods(view))
        return {"views": views, "pairs": match_cells(views, self.min_overlap)}
