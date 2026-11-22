from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyarrow import FixedSizeListArray
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from nfm.data.neighborhoods import Neighborhoods, match_tracks
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
    the local views are among the nuclei of the global views. Voronoi cells that
    hold the same nuclei in different views form tracks for the objective.
    """

    def __init__(
        self,
        slides: pd.DataFrame,
        nuclei_path: str,
        augmentation: Augmentation,
        neighborhoods: Neighborhoods,
        local_augmentation: Augmentation | None = None,
        global_views: int = 2,
        global_nuclei: tuple[int, int] = (3072, 4096),
        local_views: int = 6,
        local_nuclei: tuple[int, int] = (512, 1024),
        center_nuclei: int = 1024,
        region_nuclei: int = 8192,
        min_overlap: float = 0.3,
        min_share: float = 0.5,
        descriptor: Descriptor = radial_distances,
    ) -> None:
        """Initializes the dataset.

        Args:
            slides: Slides to sample regions from.
            nuclei_path: Root of the lance nuclei store.
            augmentation: Applied to each global view, and to each local view
                unless `local_augmentation` is given.
            neighborhoods: Adds the nucleus neighbors and Voronoi cells to a view.
            local_augmentation: Applied to each local view, which is small and
                usually dropped less, so the tracks still reach it.
            global_views: Number of global views, whose cells start the tracks.
            global_nuclei: Range of the number of nuclei in a global view.
            local_views: Number of local views, whose cells join the tracks.
            local_nuclei: Range of the number of nuclei in a local view.
            center_nuclei: Global view centers are among this many nuclei
                closest to the seed, which sets how far apart they are.
            region_nuclei: Nuclei loaded around the seed. A view is cut short
                where it reaches beyond them.
            min_overlap: Least Jaccard index of the nuclei of the cells of a
                track in the global views.
            min_share: Least share of the nuclei of a local cell that lie in the
                global cells of its track.
            descriptor: Shape features of a contour in the view frame.
        """
        self.slides = slides.reset_index(drop=True)
        self.nuclei_path = nuclei_path
        self.augmentation = augmentation
        self.local_augmentation = local_augmentation or augmentation
        self.neighborhoods = neighborhoods
        self.global_views = global_views
        self.global_nuclei = global_nuclei
        self.local_views = local_views
        self.local_nuclei = local_nuclei
        self.center_nuclei = center_nuclei
        self.region_nuclei = region_nuclei
        self.min_overlap = min_overlap
        self.min_share = min_share
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

        def crop(
            center: NDArray[np.float32], sizes: tuple[int, int]
        ) -> tuple[NDArray[np.float32], NDArray[np.intp]]:
            count = np.random.randint(sizes[0], sizes[1] + 1)
            _, nuclei = tree.query(center, min(count, len(centroids)))
            return center, np.atleast_1d(nuclei)

        def view(
            center: NDArray[np.float32],
            nuclei: NDArray[np.intp],
            augmentation: Augmentation,
        ) -> View:
            return self.neighborhoods(
                make_view(
                    polygons[nuclei],
                    center,
                    augmentation,
                    self.descriptor,
                    ids=ids[nuclei],
                )
            )

        # All crops are chosen before any augmentation, so the local centers
        # are among the nuclei of the global crops, whatever the views drop.
        _, near_seed = tree.query(seed, min(self.center_nuclei, len(centroids)))
        near_seed = np.atleast_1d(near_seed)
        crops = [
            crop(centroids[np.random.choice(near_seed)], self.global_nuclei)
            for _ in range(self.global_views)
        ]
        covered = np.unique(np.concatenate([nuclei for _, nuclei in crops]))
        crops += [
            crop(centroids[np.random.choice(covered)], self.local_nuclei)
            for _ in range(self.local_views)
        ]
        views = [
            view(
                center,
                nuclei,
                self.augmentation if i < self.global_views else self.local_augmentation,
            )
            for i, (center, nuclei) in enumerate(crops)
        ]
        tracks = match_tracks(
            views, self.global_views, self.min_overlap, self.min_share
        )
        return {"views": views, "tracks": tracks, "global_views": self.global_views}
