import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict

import hydra
import numpy as np
import ray
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles, read_slide_tiles
from ray.data.expressions import col
from transformers import AutoImageProcessor, AutoModelForObjectDetection


class SlideRecord(TypedDict):
    path: str
    extent_x: int
    extent_y: int
    tile_extent_x: int
    tile_extent_y: int
    stride_x: int
    stride_y: int
    mpp_x: float
    mpp_y: float
    level: int
    downsample: float
    slide_id: str
    scale_factor: float


class TileRecord(TypedDict):
    tile_x: int
    tile_y: int
    path: str
    slide_id: str
    mpp_x: float
    mpp_y: float
    extent_x: int
    extent_y: int
    tile_extent_x: int
    tile_extent_y: int
    level: int


class NucleusRecord(TypedDict):
    id: str
    slide_id: str
    polygon: NDArray[np.float32]
    centroid: NDArray[np.float32]


class TilePolygonRecord(TypedDict):
    slide_id: str
    tile_x: int
    tile_y: int
    polygons: list[NDArray[np.float32]]


class Model:
    device = "cuda"

    def __init__(self) -> None:
        self.model = AutoModelForObjectDetection.from_pretrained(
            "RationAI/LSP-DETR",
            trust_remote_code=True,
        ).to(self.device)
        self.model = self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(
            "RationAI/LSP-DETR",
            trust_remote_code=True,
        )

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def __call__(self, batch: dict[str, Any]) -> TilePolygonRecord:
        """Segments nuclei in a tile and extracts polygons.

        Args:
            batch (dict): Tile metadata and image.
                - "path" (str): Slide path.
                - "tile_x" (int): X-coordinate of the tile.
                - "tile_y" (int): Y-coordinate of the tile.
                - "tile" (PIL.Image or np.ndarray): Tile image.
        """
        inputs = self.processor(
            batch["tile"].copy(), device=self.device, return_tensors="pt"
        )
        outputs = self.model(**inputs)
        results = self.processor.post_process(outputs)

        return {
            "slide_id": batch["slide_id"],
            "tile_x": batch["tile_x"],
            "tile_y": batch["tile_y"],
            "polygons": [result["polygons"].cpu().numpy() for result in results],
        }


def tiling(slide_record: SlideRecord) -> Iterator[TileRecord]:
    """Yields metadata for unprocessed tiles of a slide.

    Note: The tiling step is not separated as per usual, as tiles are only used within
          this pipeline and persisting them would not have an additional benefit.
    """
    for x, y in grid_tiles(
        slide_extent=(slide_record["extent_x"], slide_record["extent_y"]),
        tile_extent=(slide_record["tile_extent_x"], slide_record["tile_extent_y"]),
        stride=(slide_record["stride_x"], slide_record["stride_y"]),
        last="keep",
    ):
        yield {
            "tile_x": x,
            "tile_y": y,
            "path": slide_record["path"],
            "slide_id": Path(slide_record["path"]).stem,
            "mpp_x": slide_record["mpp_x"],
            "mpp_y": slide_record["mpp_y"],
            "extent_x": slide_record["extent_x"],
            "extent_y": slide_record["extent_y"],
            "tile_extent_x": slide_record["tile_extent_x"],
            "tile_extent_y": slide_record["tile_extent_y"],
            "level": slide_record["level"],
        }


def drop_duplicates(
    tile_record: TilePolygonRecord, tile_extent: int, overlap: int
) -> Iterator[NucleusRecord]:
    """Filters out nuclei near tile borders to avoid duplicates.

    For each nucleus, its centroid is computed and checked to ensure it lies within
    the non-overlapping region. Remaining polygons and centroids are adjusted to
    absolute slide coordinates.
    """
    if len(tile_record["polygons"]) == 0:
        return

    polygons_arr = np.stack(tile_record["polygons"], axis=0)
    centroids = polygons_arr.mean(axis=1)
    keep = np.all(centroids >= overlap / 2, axis=-1) & np.all(
        centroids < tile_extent - overlap / 2, axis=-1
    )

    offset = np.array((tile_record["tile_x"], tile_record["tile_y"]), dtype=np.float32)
    polygons = polygons_arr[keep] + offset
    centroids = centroids[keep] + offset

    for i, (polygon, centroid) in enumerate(zip(polygons, centroids, strict=True)):
        nucleus_key = f"{tile_record['slide_id']}{tile_record['tile_x']}{tile_record['tile_y']}{i}"

        yield {
            "id": hashlib.sha256(nucleus_key.encode()).hexdigest(),
            "slide_id": tile_record["slide_id"],
            "polygon": polygon,
            "centroid": centroid,
        }


def filter_tissue_tiles(row: dict[str, Any]) -> bool:
    return row["tile"].std() > 8


@with_cli_args(["+preprocessing=nuclei_segmentation"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, _: MLFlowLogger) -> None:
    slides = read_slides(
        config.slides_path,
        tile_extent=config.tile_extent,
        stride=config.tile_extent - config.overlap,
        level=0,
    )

    tiles = slides.flat_map(tiling, num_cpus=0.1, memory=128 * 1024**2).repartition(
        target_num_rows_per_block=128
    )
    tissue_tiles = (
        tiles.with_column(
            "tile",
            read_slide_tiles(
                col("path"),
                col("tile_x"),
                col("tile_y"),
                col("tile_extent_x"),
                col("tile_extent_y"),
                col("level"),
            ),
            num_cpus=1,
            memory=5 * 1024**3,
        )
        .filter(filter_tissue_tiles, memory=3 * 1024**3)
        .repartition(target_num_rows_per_block=config.batch_size * 16)
    )

    nuclei = tissue_tiles.map_batches(
        Model,
        num_gpus=1,
        num_cpus=0,
        batch_size=config.batch_size,
        memory=5 * 1024**3,
        zero_copy_batch=True,
    )
    nuclei = nuclei.flat_map(
        drop_duplicates,
        fn_kwargs={"tile_extent": config.tile_extent, "overlap": config.overlap},
        num_cpus=0.1,
        memory=1.5 * 1024**3,
    )
    nuclei.write_parquet(config.nuclei_path, partition_cols=["slide_id"])
    ray.shutdown()


if __name__ == "__main__":
    main()
