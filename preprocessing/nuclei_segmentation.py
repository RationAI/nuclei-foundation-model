from typing import Any, TypedDict

import numpy as np
import ray
import torch
from histopath.ray.datasource import OpenSlideMetaDatasource
from histopath.tiling import grid_tiles
from histopath.tiling.openslide_tile_reader import openslide_tile_reader
from histopath.tiling.utils import row_hash
from numpy.typing import NDArray
from transformers import AutoImageProcessor, AutoModelForObjectDetection


PATH = ""
TILE_EXTENT = 2048
OVERLAP = 64
STRIDE = TILE_EXTENT - OVERLAP


class Nuclei(TypedDict):
    slide_id: str
    polygon: NDArray[np.float32]
    centroid: NDArray[np.float32]
    embedding: NDArray[np.float32]
    is_edge: bool


class Model:
    device = "cuda"

    def __init__(self) -> None:
        self.model = AutoModelForObjectDetection.from_pretrained(
            "RationAI/LSP-DETR",
            trust_remote_code=True,
            token="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",
        ).to(self.device)
        self.model = self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(
            "RationAI/LSP-DETR",
            trust_remote_code=True,
            token="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",
        )

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        inputs = self.processor(batch["tile"], device=self.device, return_tensors="pt")
        outputs = self.model(**inputs)
        results = self.processor.post_process(outputs)

        batch["polygons"] = [result["polygons"].cpu().numpy() for result in results]
        batch["embeddings"] = [result["embeddings"].cpu().numpy() for result in results]
        return batch


def tiling(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"tile_x": x, "tile_y": y, **row}
        for x, y in grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
        )
    ]


def filter_tissue(row: dict[str, Any]) -> bool:
    return row["tile"].std() > 8


def drop_duplicates(row: dict[str, Any]) -> list[dict[str, Any]]:
    centroids = row["polygons"].min(axis=1)

    keep = np.all(centroids >= OVERLAP / 2, axis=-1) & np.all(
        centroids < TILE_EXTENT - OVERLAP / 2, axis=-1
    )

    offset = np.array((row["tile_x"], row["tile_y"]), dtype=np.float32)
    polygons = row["polygons"][keep] + offset
    embeddings = row["embeddings"][keep]
    centroids = centroids[keep] + offset

    return [
        {
            "slide_id": row["id"],
            "polygon": polygon,
            "embedding": embedding,
            "centroid": centroid,
        }
        for polygon, embedding, centroid in zip(
            polygons, embeddings, centroids, strict=True
        )
    ]


if __name__ == "__main__":
    slides = ray.data.read_datasource(
        OpenSlideMetaDatasource(PATH, mpp=0.25, tile_extent=TILE_EXTENT, stride=STRIDE)
    ).map(row_hash, num_cpus=0.1, memory=300 * 1024 * 1024)
    slides.write_parquet("slides")

    tiles = slides.flat_map(tiling, num_cpus=0.2, memory=300 * 1024 * 1024).repartition(
        target_num_rows_per_block=200
    )
    tissue_tiles = tiles.map(
        openslide_tile_reader, num_cpus=0.25, memory=300 * 1024 * 1024
    ).filter(filter_tissue)
    nuclei = tissue_tiles.map_batches(
        Model, num_gpus=1, num_cpus=0, batch_size=20, concurrency=1
    )
    nuclei.flat_map(
        drop_duplicates, num_cpus=0.1, memory=300 * 1024 * 1024
    ).write_parquet("nuclei", partition_cols=["slide_id"])
