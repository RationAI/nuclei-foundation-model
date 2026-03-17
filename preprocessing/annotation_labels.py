from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import numpy as np
import pandas as pd
import ray
import tifffile
from einops import rearrange
from numpy.typing import NDArray
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.ray import read_slides


def label_nuclei(
    slide_record: dict[str, Any],
    nuclei_dir: Path,
    annot_masks_dir: Path,
) -> Iterator[dict[str, Any]]:
    """Labels nuclei for a single slide based on an annotation mask.

    For each nucleus, computes the fraction of polygon vertices that fall
    inside the annotation mask. Nuclei with coverage >= overlap_thr are
    labeled as annotated (1), otherwise 0.
    """
    slide_path = slide_record["path"]
    slide_id = slide_path.stem
    dataset_name = slide_path.parents[0].name

    nuclei_path = nuclei_dir / dataset_name / f"slide_id={slide_id}"
    nuclei = pd.read_parquet(nuclei_path, columns=["id", "polygon"])

    annot_mask_path = annot_masks_dir / f"{slide_id}.tiff"
    annot_mask: NDArray[np.uint8] = tifffile.imread(annot_mask_path).squeeze()

    mask_extent_y, mask_extent_x = annot_mask.shape
    with OpenSlide(slide_path) as slide:
        wsi_extent_x, wsi_extent_y = slide.dimensions
    scale_x = mask_extent_x / wsi_extent_x
    scale_y = mask_extent_y / wsi_extent_y

    polygons = rearrange(nuclei["polygon"].tolist(), "b (v d) -> b v d", d=2)
    coords = np.round(polygons * np.array([scale_x, scale_y])).astype(int)
    x_coords = np.clip(coords[..., 0], 0, mask_extent_x - 1)
    y_coords = np.clip(coords[..., 1], 0, mask_extent_y - 1)
    coverage = np.mean(annot_mask[y_coords, x_coords] != 0, axis=1)

    for nuc_id, cov in zip(nuclei["id"], coverage, strict=True):
        yield {
            "slide_id": slide_id,
            "id": nuc_id,
            "annot_coverage": cov,
        }


@with_cli_args(["+preprocessing=annotation_labels"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = ray.data.from_items(
        [{"path": p} for p in Path(config.slides_path).glob("*.tif")]
    )

    labeled_nuclei = slides.flat_map(
        label_nuclei,
        fn_kwargs={
            "nuclei_dir": Path(config.nuclei_path),
            "annot_masks_dir": Path(config.annot_masks_path),
        },
        num_cpus=1,
        memory=3 * 1024**3,
    )

    with TemporaryDirectory() as tmp_dir:
        labeled_nuclei.write_parquet(tmp_dir, partition_cols=["slide_id"])
        logger.log_artifacts(
            local_dir=tmp_dir, artifact_path=config.mlflow_artifact_path
        )
    ray.shutdown()


if __name__ == "__main__":
    main()
