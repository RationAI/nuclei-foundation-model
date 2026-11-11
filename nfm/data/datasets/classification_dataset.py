import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from nfm.data.neighborhoods import Neighborhoods
from nfm.data.nuclei import Descriptor, View, make_view, radial_distances


class ClassificationNucleiDataset(Dataset[View]):
    """Nuclei of one Radboud slide with nucleus and region labels.

    A region is the neighborhood of a local class token, and it is positive if at
    least 50 % of its nuclei are. With `crop_nuclei`, a view is that many nuclei
    closest to a random nucleus, for training. Without it, a view is the whole
    slide, so every nucleus is evaluated exactly once.
    """

    def __init__(
        self,
        slides: pd.DataFrame,
        annotation_path: str,
        neighborhoods: Neighborhoods,
        crop_nuclei: int | None = None,
        descriptor: Descriptor = radial_distances,
    ) -> None:
        self.slides = slides.reset_index(drop=True)
        self.annotation_path = annotation_path
        self.neighborhoods = neighborhoods
        self.crop_nuclei = crop_nuclei
        self.descriptor = descriptor

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> View:
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
        centroids = polygons.mean(1)
        if self.crop_nuclei is None:
            center = centroids.mean(0)
        else:
            center = centroids[np.random.randint(len(centroids))]
            count = min(self.crop_nuclei, len(centroids))
            _, members = cKDTree(centroids).query(center, count)
            polygons, labels = polygons[members], labels[members]
        view = self.neighborhoods(
            make_view(polygons, center, descriptor=self.descriptor, labels=labels)
        )
        region = view["labels"][view["cls_neighbors"]].mean(1) >= 0.5
        return {**view, "cls_labels": region.astype(np.float32)}
