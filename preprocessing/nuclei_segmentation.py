import posixpath
from typing import Any, TypedDict

import numpy as np
import pyarrow
import ray
import torch
from histopath.ray.datasource import OpenSlideMetaDatasource
from histopath.tiling import grid_tiles
from histopath.tiling.openslide_tile_reader import openslide_tile_reader
from histopath.tiling.utils import row_hash
from numpy.typing import NDArray
from ray.data._internal.datasource.parquet_datasink import ParquetDatasink
from ray.data.block import Block, BlockAccessor
from ray.data.datasource import FilenameProvider
from shapely import Polygon, STRtree
from transformers import AutoImageProcessor, AutoModelForObjectDetection


PATH = ""
TILE_EXTENT = 1024
OVERLAP = 64
STRIDE = TILE_EXTENT - OVERLAP


class Nuclei(TypedDict):
    slide_id: str
    polygon: NDArray[np.float32]
    centroid: NDArray[np.float32]
    embedding: NDArray[np.float32]
    is_edge: bool


class SlideFilenameProvider(FilenameProvider):
    def get_filename_for_block(
        self, block: Block, task_index: int, block_index: int
    ) -> str:
        block_acc = BlockAccessor.for_block(block)
        row = next(block_acc.iter_rows(public_row_format=False))
        return row["slide_id"] + ".parquet"


class Datasink(ParquetDatasink):
    def __init__(
        self, path: str, ignore_cols: str | list[str] | None = None, **kwargs: Any
    ):
        self.ignore_cols = ignore_cols
        super().__init__(path, **kwargs)

    def _write_single_file(
        self,
        path: str,
        tables: list[pyarrow.Table],
        filename: str,
        output_schema: pyarrow.Schema,
        write_kwargs: dict[str, Any],
    ) -> None:
        import pyarrow.parquet as pq

        row_group_size = write_kwargs.pop("row_group_size", None)
        output_schema = pyarrow.schema(
            [field for field in output_schema if field.name not in self.ignore_cols]
        )

        write_path = posixpath.join(path, filename)
        with (
            self.open_output_stream(write_path) as file,
            pq.ParquetWriter(file, output_schema, **write_kwargs) as writer,
        ):
            for table in tables:
                if self.ignore_cols is not None:
                    table = table.drop_columns(self.ignore_cols)
                table = table.cast(output_schema)
                writer.write_table(table, row_group_size=row_group_size)


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


def aggregate_nuclei(block: Block) -> dict[str, Any]:
    def clasify_nucleus_location(polygon: NDArray[np.float32]) -> bool | None:
        """Classifies a nucleus based on its location within a tile.

        Returns:
            - True: If the nucleus is fully within the tile's core (non-overlap) area.
            - False: If the nucleus is within the tile's edge (overlap) area.
            - None: If the nucleus should be discarded (e.g., outside tile bounds).
        """
        min = polygon.min(axis=0)
        max = polygon.max(axis=0)

        if np.any(min < 0) or np.any(max >= TILE_EXTENT):
            if np.any(max >= OVERLAP) or np.any(min < TILE_EXTENT - OVERLAP):
                return False
            return None

        return np.all(max >= OVERLAP) and np.all(min < TILE_EXTENT - OVERLAP)

    unique = {
        "slide_id": [],
        "polygons": [],
        "embeddings": [],
        "centroids": [],
    }
    candidate = []

    block_acc = BlockAccessor.for_block(block)
    for row in block_acc.iter_rows(public_row_format=False):
        offset = np.array((row["tile_x"], row["tile_y"]), dtype=np.float32)
        for i, polygon in enumerate(row["polygons"]):
            if not Polygon(polygon).is_valid:
                continue

            if (is_center := clasify_nucleus_location(polygon)) is None:
                continue

            polygon = polygon + offset
            if is_center:
                unique["slide_id"].append(row["id"])
                unique["polygons"].append(polygon)
                unique["embeddings"].append(row["embeddings"][i])
                unique["centroids"].append(polygon.mean(axis=0))
            else:
                candidate.append(
                    {
                        "slide_id": row["id"],
                        "polygon": polygon,
                        "embedding": row["embeddings"][i],
                    }
                )

    polygons = [Polygon(nucleus["polygon"]) for nucleus in candidate]
    tree = STRtree(polygons)
    for i, query in enumerate(polygons):
        is_duplicate = False

        for key_idx in tree.query(query, predicate="intersects"):
            if i <= key_idx:
                continue

            key = tree.geometries.take(key_idx)
            iou = query.intersection(key).area / query.union(key).area
            if iou > 0.8:
                is_duplicate = True
                break

        if not is_duplicate:
            unique["slide_id"].append(candidate[i]["slide_id"])
            unique["polygons"].append(candidate[i]["polygon"])
            unique["embeddings"].append(candidate[i]["embedding"])
            unique["centroids"].append(candidate[i]["polygon"].mean(axis=0))

    return unique


if __name__ == "__main__":
    slides = ray.data.read_datasource(
        OpenSlideMetaDatasource(PATH, mpp=0.25, tile_extent=TILE_EXTENT, stride=STRIDE)
    ).map(row_hash, num_cpus=0.1, memory=300 * 1024 * 1024)
    slides.write_parquet("slides")

    tiles = slides.flat_map(tiling, num_cpus=0.2, memory=300 * 1024 * 1024).repartition(
        target_num_rows_per_block=200
    )
    tissue_tiles = tiles.map(openslide_tile_reader, memory=300 * 1024 * 1024).filter(
        filter_tissue
    )
    nuclei = tissue_tiles.map_batches(
        Model, num_gpus=1, num_cpus=0, batch_size=20, concurrency=1
    )
    aggregated = nuclei.groupby("id").map_groups(
        aggregate_nuclei, batch_format=None, memory=1024 * 1024 * 1024
    )
    aggregated.write_datasink(
        Datasink(
            "nuclei", ignore_cols="slide_id", filename_provider=SlideFilenameProvider()
        )
    )
