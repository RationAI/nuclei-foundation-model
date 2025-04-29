from typing import Any

import numpy as np
import ray
import torch
from histopath.ray.datasource import OpenSlideMetaDatasource
from histopath.tiling import grid_tiles
from histopath.tiling.openslide_tile_reader import openslide_tile_reader
from histopath.tiling.utils import row_hash
from numpy.typing import NDArray
from ray.data.aggregate import AggregateFnV2
from ray.data.block import Block, BlockAccessor
from torch import Tensor


type TileMap = dict[tuple[int, int], Any]


class SlideNucleiAggregator(AggregateFnV2):
    def __init__(self) -> None:
        super().__init__(
            name="SlideNucleiAggregator", on="id", zero_factory=dict, ignore_nulls=True
        )

    def combine(self, current_accumulator: TileMap, new: TileMap) -> TileMap:
        """Combines new partially aggregated value (previously returned
        from `aggregate_block` partial aggregations into a singular partial
        aggregation) with the previously stored accumulator"""
        ...

    def aggregate_block(self, block: Block) -> dict[tuple[int, int], Any]:
        tile_map = {}
        block_acc = BlockAccessor.for_block(block)
        for row in block_acc.iter_rows(public_row_format=False):
            print(row)

        return tile_map


class LSPSwin:
    def __init__(self) -> None:
        self.model = torch.jit.load("model.pt")
        self.model = self.model.to("cuda").eval()

    @torch.inference_mode()
    def __call__(self, batch: dict[str, Any]) -> dict[str, NDArray[np.uint8]]:
        outputs = self.model(self.preprocess(batch["image"]))
        batch["nuclei"] = self.get_star_polygons(
            outputs,
            batch["image"].shape[1],
            batch["image"].shape[2],
        )
        return batch

    def preprocess(self, images: NDArray[np.uint8]) -> Tensor:
        # Apply preprocessing here
        batch = torch.from_numpy(images).float()
        batch = batch.permute(0, 3, 1, 2)
        return batch

    def get_star_polygons(
        self, outputs: dict[str, Tensor], height: int, width: int
    ) -> list[NDArray[np.float32]]:
        labels = outputs["logits"].argmax(dim=-1)
        non_no_object_indices = labels != outputs["logits"].shape[1] - 1
        size = torch.tensor([width, height], device=labels.device)
        t = torch.linspace(
            0, 1, outputs["radial_distances"].shape[-1] + 1, device=labels.device
        )[:-1]
        cos = torch.cos(2 * torch.pi * t)
        sin = torch.sin(2 * torch.pi * t)
        output = []
        for b, indices in enumerate(non_no_object_indices):
            points = outputs["points"][b, indices] * size
            radial_distances = outputs["radial_distances"][b, indices]
            contours = points[:, None] + radial_distances[..., None] * torch.stack(
                [sin, cos], dim=-1
            )
            output.append(contours.cpu().numpy())
        return output


def tiling(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"tile_x": tile[0], "tile_y": tile[1], "slide_id": row["id"]}
        for tile in grid_tiles(
            (row["extent_x"], row["extent_y"]),
            (row["tile_extent_x"], row["tile_extent_y"]),
            (row["stride_x"], row["stride_y"]),
        )
    ]


def flat_nuclei(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "centroid_x": nuclei["centroid_x"],
            "centroid_y": nuclei["centroid_y"],
            "polygon": nuclei["polygon"],
            "slide_id": row["id"],
        }
        for nuclei in row["nuclei"]
    ]


slides = ray.data.read_datasource(
    OpenSlideMetaDatasource("", tile_extent=2048, stride=64, mpp=0.25)
).map(row_hash)

slides.flat_map(tiling).map(openslide_tile_reader).map_batches(
    LSPSwin, num_gpus=1, batch_size=32, concurrency=8
).aggregate(SlideNucleiAggregator()).flat_map(flat_nuclei).write_parquet("")
