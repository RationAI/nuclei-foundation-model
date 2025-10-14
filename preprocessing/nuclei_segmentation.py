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


INPUT_SLIDES = ""
OUTPUT_SLIDES = "slides"
OUTPUT_NUCLEI = "nuclei"
TILE_EXTENT = 2048
OVERLAP = 64
STRIDE = TILE_EXTENT - OVERLAP


def get_log_file(organ: str, dataset: str, slide_id: str) -> Path:
    log_file = (
        Path(OUTPUT_NUCLEI)
        / f"organ={organ}"
        / f"dataset={dataset}"
        / f"{slide_id}.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return log_file


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

                log_file = get_log_file(
                    organ=group["organ"].iloc[0],
                    dataset=group["dataset"].iloc[0],
                    slide_id=slide_id,
                )
                with open(log_file, "a") as f:
                    f.write(log_entries.to_string(header=False, index=False))
                    f.write("\n")


def tiling(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    processed_tiles = set()
    log_file = get_log_file(
        organ=row["organ"], dataset=row["dataset"], slide_id=row["id"]
    )
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
                "slide_id": row["id"],
                "organ": row["organ"],
                "dataset": row["dataset"],
                "level": row["level"],
                "tile_extent_x": row["tile_extent_x"],
                "tile_extent_y": row["tile_extent_y"],
            }


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
        inputs = self.processor(
            batch["tile"].copy(), device=self.device, return_tensors="pt"
        )
        outputs = self.model(**inputs)

        return {
            "slide_id": batch["slide_id"],
            "organ": batch["organ"],
            "dataset": batch["dataset"],
            "tile_x": batch["tile_x"],
            "tile_y": batch["tile_y"],
            "radial_distances": outputs["radial_distances"].cpu().numpy(),
            "points": outputs["points"].cpu().numpy(),
        }


def filter_tissue_tiles(row: dict[str, Any]) -> bool:
    if row["tile"].std() > 8:
        return True

    log_file = get_log_file(
        organ=row["organ"], dataset=row["dataset"], slide_id=row["id"]
    )
    with open(log_file / f"{row['slide_id']}.log", "a") as f:
        f.write(f"{row['tile_x']} {row['tile_y']}\n")

    return False


def drop_duplicates(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    keep = np.all(row["points"] >= OVERLAP / 2, axis=-1) & np.all(
        row["points"] < TILE_EXTENT - OVERLAP / 2, axis=-1
    )

    offset = np.array((row["tile_x"], row["tile_y"]), dtype=np.float32)
    radial_distances = row["radial_distances"][keep]
    points = row["points"][keep] + offset

    for radial_distances, points in zip(radial_distances, points, strict=True):
        yield {
            "slide_id": row["slide_id"],
            "organ": row["organ"],
            "dataset": row["dataset"],
            "tile_x": row["tile_x"],
            "tile_y": row["tile_y"],
            "radial_distances": radial_distances,
            "points": points,
        }


def add_slide_metadata(row: dict[str, Any]) -> dict[str, Any]:
    row["organ"] = Path(row["path"]).parent.name
    row["dataset"] = Path(row["path"]).parent.parent.name
    return row_hash(row)


def main() -> None:
    slides = read_slides(INPUT_SLIDES, mpp=0.25, tile_extent=TILE_EXTENT, stride=STRIDE)
    slides = slides.map(add_slide_metadata, num_cpus=0.1, memory=128 * 1024 * 1024)
    slides.write_parquet(
        OUTPUT_SLIDES,
        partition_cols=["organ", "dataset"],
    )

    tiles = slides.flat_map(tiling, num_cpus=0.2, memory=128 * 1024**2).repartition(
        target_num_rows_per_block=128
    )

    tissue_tiles = tiles.map_batches(
        read_slide_tiles, num_cpus=1, memory=5 * 1024**3
    ).filter(filter_tissue_tiles, memory=3 * 1024**3)
    tissue_tiles = tissue_tiles.repartition(target_num_rows_per_block=180)
    nuclei = tissue_tiles.map_batches(
        Model,
        num_gpus=1,
        num_cpus=0,
        batch_size=18,
        memory=3 * 1024**3,
        concurrency=4,
        zero_copy_batch=True,
    )
    nuclei = nuclei.flat_map(drop_duplicates, num_cpus=0.1, memory=1.5 * 1024**3)

    nuclei.write_datasink(
        ParquetDatasinkWithLogs(
            OUTPUT_NUCLEI,
            partition_cols=["organ", "dataset", "slide_id"],
        )
    )


if __name__ == "__main__":
    ray.init(
        num_cpus=int(os.environ["SLURM_CPUS_ON_NODE"]),
        _memory=(memory := int(os.environ["SLURM_MEM_PER_NODE"]) * 1024**2),
        object_store_memory=int(memory * 0.3),
        enable_resource_isolation=True,
    )
    main()
    ray.shutdown()
