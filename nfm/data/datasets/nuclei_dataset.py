from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyarrow import FixedSizeListArray
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from nfm.data.neighborhoods import Neighborhoods
from nfm.data.nuclei import (
    Augmentation,
    Descriptor,
    View,
    make_view,
    radial_distances,
    radial_polygons,
)


def _to_numpy(column: FixedSizeListArray) -> NDArray[np.float32]:
    flat = column.combine_chunks().flatten().to_numpy().astype(np.float32)
    return flat.reshape(len(column), -1)


class NucleiDataset(Dataset[dict[str, Any]]):
    """Global and local views of the nuclei around a random nucleus, as in LeJEPA.

    Each view is the nuclei closest to its own center, a random number of them,
    like a random crop. The centers of the global views are among the nuclei
    closest to the region seed, so the global views overlap, and the centers of
    the local views are among the nuclei of the global views. The model
    clusters each view into cells and tracks the cells across the views by the
    ids of their nuclei.
    """

    def __init__(
        self,
        slides: pd.DataFrame,
        nuclei_path: str,
        augmentation: Augmentation,
        neighborhoods: Neighborhoods,
        global_views: int = 2,
        global_nuclei: tuple[int, int] = (3072, 4096),
        local_views: int = 6,
        local_nuclei: tuple[int, int] = (512, 1024),
        center_nuclei: int = 1024,
        region_nuclei: int = 8192,
        descriptor: Descriptor = radial_distances,
    ) -> None:
        """Initializes the dataset.

        Args:
            slides: Slides to sample regions from.
            nuclei_path: Root of the lance nuclei store.
            augmentation: Applied to each view.
            neighborhoods: Adds the nucleus neighbors and initial cells to a view.
            global_views: Number of global views, whose cells start the tracks.
            global_nuclei: Range of the number of nuclei in a global view.
            local_views: Number of local views, whose cells join the tracks.
            local_nuclei: Range of the number of nuclei in a local view.
            center_nuclei: Global view centers are among this many nuclei
                closest to the seed, which sets how far apart they are.
            region_nuclei: Nuclei loaded around the seed. A view is cut short
                where it reaches beyond them.
            descriptor: Shape features of a contour in the view frame.
        """
        self.slides = slides.reset_index(drop=True)
        self.nuclei_path = nuclei_path
        self.augmentation = augmentation
        self.neighborhoods = neighborhoods
        self.global_views = global_views
        self.global_nuclei = global_nuclei
        self.local_views = local_views
        self.local_nuclei = local_nuclei
        self.center_nuclei = center_nuclei
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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        polygons, seed = self.load_region(self.slides.iloc[idx])
        centroids = polygons.mean(1)
        tree = cKDTree(centroids)
        ids = np.arange(len(polygons))

        def view(center: NDArray[np.float32], sizes: tuple[int, int]) -> View:
            count = np.random.randint(sizes[0], sizes[1] + 1)
            _, nuclei = tree.query(center, min(count, len(centroids)))
            nuclei = np.atleast_1d(nuclei)
            return self.neighborhoods(
                make_view(
                    polygons[nuclei],
                    center,
                    self.augmentation,
                    self.descriptor,
                    ids=ids[nuclei],
                )
            )

        _, near_seed = tree.query(seed, min(self.center_nuclei, len(centroids)))
        near_seed = np.atleast_1d(near_seed)
        views = [
            view(centroids[np.random.choice(near_seed)], self.global_nuclei)
            for _ in range(self.global_views)
        ]
        covered = np.unique(np.concatenate([v["ids"] for v in views]))
        if len(covered) == 0:  # everything dropped
            covered = near_seed
        views += [
            view(centroids[np.random.choice(covered)], self.local_nuclei)
            for _ in range(self.local_views)
        ]
        return {"views": views, "global_views": self.global_views}
