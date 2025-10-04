from typing import Any

import numpy as np
import torch
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles, read_slide_tiles
from ratiopath.tiling.utils import row_hash
from transformers import AutoImageProcessor, AutoModelForObjectDetection


PATH = ""
TILE_EXTENT = 2048
OVERLAP = 64
STRIDE = TILE_EXTENT - OVERLAP


def tiling(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "tile_x": x,
            "tile_y": y,
            "path": row["path"],
            "slide_id": row["id"],
            "level": row["level"],
            "tile_extent_x": row["tile_extent_x"],
            "tile_extent_y": row["tile_extent_y"],
        }
        for x, y in grid_tiles(
            slide_extent=(row["extent_x"], row["extent_y"]),
            tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
            stride=(row["stride_x"], row["stride_y"]),
            last="keep",
        )
    ]


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

        return {
            "slide_id": batch["slide_id"],
            "tile_x": batch["tile_x"],
            "tile_y": batch["tile_y"],
            "polygons": [result["polygons"].cpu().numpy() for result in results],
            "embeddings": [result["embeddings"].cpu().numpy() for result in results],
        }


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
            "slide_id": row["slide_id"],
            "polygon": polygon,
            "embedding": embedding,
            "centroid": centroid,
        }
        for polygon, embedding, centroid in zip(
            polygons, embeddings, centroids, strict=True
        )
    ]


if __name__ == "__main__":
    slides = read_slides(PATH, mpp=0.25, tile_extent=TILE_EXTENT, stride=STRIDE)
    slides = slides.map(row_hash, num_cpus=0.1, memory=128 * 1024 * 1024)
    slides.write_parquet("slides")

    tiles = slides.flat_map(tiling, num_cpus=0.2, memory=128 * 1024 * 1024).repartition(
        target_num_rows_per_block=128
    )

    tissue_tiles = tiles.map_batches(
        read_slide_tiles, num_cpus=1, memory=4 * 1024 * 1024 * 1024
    ).filter(lambda row: row["tile"].std() > 8, memory=1.5 * 1024 * 1024 * 1024)
    nuclei = tissue_tiles.map_batches(
        Model,
        num_gpus=1,
        num_cpus=0,
        batch_size=16,
        memory=3 * 1024 * 1024 * 1024,
        concurrency=8,
        zero_copy_batch=True,
    )
    nuclei.flat_map(
        drop_duplicates, num_cpus=0.1, memory=1.5 * 1024 * 1024 * 1024
    ).write_parquet("nuclei", partition_cols=["slide_id"])
