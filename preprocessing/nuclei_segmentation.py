import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow
import ray
import torch
from ratiopath.ray import read_slides
from ratiopath.tiling import grid_tiles, read_slide_tiles
from ratiopath.tiling.utils import row_hash
from ray.data._internal.datasource.parquet_datasink import ParquetDatasink
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# INPUT_PATHS = sys.argv[1]  # $BATCH_FILE from Slurm
OUTPUT_FOLDER = "/flash/project_465002057/segmentation/test_results"
LOG_DIR = Path("/flash/project_465002057/segmentation/seg_logs")

TILE_EXTENT = 2048
OVERLAP = 64
STRIDE = TILE_EXTENT - OVERLAP


class ParquetDatasinkWithLogs(ParquetDatasink):
    def _write_parquet_files(
        self,
        tables: list["pyarrow.Table"],
        filename: str,
        output_schema: "pyarrow.Schema",
        write_uuid: str,
        write_kwargs: dict[str, Any],
    ) -> None:
        for col in ["tile_x", "tile_y"]:
            idx = output_schema.get_field_index(col)
            if idx != -1:
                output_schema = output_schema.remove(idx)

        super()._write_parquet_files(
            [t.drop_columns(["tile_x", "tile_y"]) for t in tables],
            filename,
            output_schema,
            write_uuid,
            write_kwargs,
        )

        for table in tables:
            df = table.to_pandas()
            for slide_id, group in df.groupby("slide_id"):
                log_entries = group[["tile_x", "tile_y"]].drop_duplicates(
                    ignore_index=True
                )
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                with open(LOG_DIR / f"{slide_id}.log", "a") as f:
                    f.write(log_entries.to_string(header=False, index=False))
                    f.write("\n")


def init_ray_with_slurm_limits():
    print("SLURM memory limits (if available):")
    print("SLURM_MEM_PER_NODE:", os.environ.get("SLURM_MEM_PER_NODE"))
    print("SLURM_CPUS_ON_NODE:", os.environ.get("SLURM_CPUS_ON_NODE"))

    slurm_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE"))
    slurm_mem_gb = os.environ.get("SLURM_MEM_PER_NODE")
    slurm_mem_bytes = int(slurm_mem_gb) * 1024 * 1024
    object_store_mem = int(slurm_mem_bytes * 0.3)

    print("🛠️ Starting Ray with SLURM memory settings...")
    print("memory", slurm_mem_bytes)
    print("object_store_mem", object_store_mem)

    ray.init(
        _memory=slurm_mem_bytes,
        object_store_memory=object_store_mem,
        enable_resource_isolation=True,
        num_cpus=slurm_cpus,
    )

    print("🧠 Ray available resources:")
    print(ray.available_resources())


def tiling(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    processed_tiles = set()
    log_file = LOG_DIR / f"{row['id']}.log"
    if log_file.exists():
        with log_file.open("r") as f:
            for line in f:
                x_str, y_str = line.split()
                processed_tiles.add((int(x_str), int(y_str)))

    for x, y in grid_tiles(
        slide_extent=(row["extent_x"], row["extent_y"]),
        tile_extent=(row["tile_extent_x"], row["tile_extent_y"]),
        stride=(row["stride_x"], row["stride_y"]),
        last="keep",
    ):
        if (x, y) not in processed_tiles:
            yield {
                "tile_x": x,
                "tile_y": y,
                "path": row["path"],
                "organ": row["organ"],
                "dataset": row["dataset"],
                "slide_id": row["id"],
                "level": row["level"],
                "tile_extent_x": row["tile_extent_x"],
                "tile_extent_y": row["tile_extent_y"],
            }


class Model:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            "organ": batch["organ"],
            "dataset": batch["dataset"],
            "slide_id": batch["slide_id"],
            "tile_x": batch["tile_x"],
            "tile_y": batch["tile_y"],
            "polygons": [result["polygons"].cpu().numpy() for result in results],
            "radial_distances": outputs["radial_distances"].cpu().numpy(),
            "points": outputs["points"].cpu().numpy(),
        }


def filter_tissue_tiles(row: dict[str, Any]) -> bool:
    if row["tile"].std() > 8:
        return True

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / f"{row['slide_id']}.log", "a") as f:
        f.write(f"{row['tile_x']} {row['tile_y']}\n")

    return False


def drop_duplicates(row: dict[str, Any]) -> list[dict[str, Any]]:
    centroids = row["polygons"].min(axis=1)

    keep = np.all(centroids >= OVERLAP / 2, axis=-1) & np.all(
        centroids < TILE_EXTENT - OVERLAP / 2, axis=-1
    )

    offset = np.array((row["tile_x"], row["tile_y"]), dtype=np.float32)
    radial_distances = row["radial_distances"][keep]
    points = row["points"][keep] + offset

    return [
        {
            "tile_x": row["tile_x"],
            "tile_y": row["tile_y"],
            "organ": row["organ"],
            "dataset": row["dataset"],
            "slide_id": row["slide_id"],
            "radial_distances": radial_distances,
            "points": points,
        }
        for radial_distances, points in zip(radial_distances, points, strict=True)
    ]


if __name__ == "__main__":
    init_ray_with_slurm_limits()
    print("STARTING SEGMENTATION")

    # with open(INPUT_PATHS, "r") as f:
    #     paths = [line.strip() for line in f if line.strip()]

    paths = "/flash/project_465002057/data/adrenal_gland/TCGA/TCGA-XG-A823-01A-01-TS1.85232749-ECFB-4F2B-9CDC-4E1012B035E1.svs"
    slides = read_slides(paths, mpp=0.25, tile_extent=TILE_EXTENT, stride=STRIDE)
    slides = slides.map(row_hash, num_cpus=0.1, memory=128 * 1024 * 1024)
    slides.write_parquet(
        f"{OUTPUT_FOLDER}/slides",
        partition_cols=["organ", "dataset"],
    )

    tiles = slides.flat_map(tiling, num_cpus=0.2, memory=128 * 1024 * 1024).repartition(
        target_num_rows_per_block=128
    )

    tissue_tiles = (
        tiles.map_batches(read_slide_tiles, num_cpus=1, memory=5 * 1024 * 1024 * 1024)
        .filter(lambda row: row["tile"].std() > 8, memory=3 * 1024 * 1024 * 1024)
        .repartition(target_num_rows_per_block=180)
    )

    nuclei = tissue_tiles.map_batches(
        Model,
        num_gpus=1,
        num_cpus=0,
        batch_size=18,
        memory=3 * 1024 * 1024 * 1024,
        concurrency=4,
        zero_copy_batch=True,
    )

    nuclei = nuclei.flat_map(
        drop_duplicates, num_cpus=0.1, memory=1.5 * 1024 * 1024 * 1024
    )

    nuclei.write_datasink(
        ParquetDatasinkWithLogs(
            f"{OUTPUT_FOLDER}/cells",
            partition_cols=["organ", "dataset", "slide_id"],
        )
    )
