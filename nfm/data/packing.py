"""Packs variable-length views into one sequence of nucleus tokens.

Each view is sorted so that the attention blocks are spatially compact. The
Voronoi cells of all views are numbered consecutively.
"""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from nfm.data.nuclei import View


type Batch = dict[str, Any]


def spatial_order(
    points: NDArray[np.float32], offset: int, block_size: int
) -> NDArray[Any]:
    """Orders points so that each attention block holds a compact patch of them.

    The points are split recursively along their wider extent, always at a block
    boundary of the packed sequence, where the first point lands at `offset`.
    """
    order = np.arange(len(points))
    stack = [(0, len(points))]
    while stack:
        start, end = stack.pop()
        first, last = (offset + start) // block_size, (offset + end - 1) // block_size
        if first >= last:
            continue
        split = (first + last + 1) // 2 * block_size - offset
        segment = order[start:end]
        selected = points[segment]
        axis = np.ptp(selected, axis=0).argmax()
        segment[:] = segment[np.argpartition(selected[:, axis], split - start - 1)]
        stack += [(start, split), (split, end)]
    return order


def pack(views: list[View], block_size: int) -> Batch:
    """Concatenates the views into one sequence, with indices into it.

    Returns:
        features: Shape features of the nuclei, [nuclei, features].
        positions: Positions of the nuclei in their view frames, [nuclei, 2].
        lengths: Nuclei per view, [views].
        neighbors: Nearest nuclei of each nucleus, [nuclei, neighbors].
        cells: Cell of each nucleus, numbered across the views, [nuclei].
        cell_lengths: Cells per view, [views].
        cell_count: Number of cells.
        labels, cell_labels: Nucleus and region labels, when the views have them.
    """
    features, positions, neighbors, cells, labels = [], [], [], [], []
    offset = cell_offset = 0
    for view in views:
        order = spatial_order(view["positions"], offset, block_size)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        features.append(view["features"][order])
        positions.append(view["positions"][order])
        neighbors.append(inverse[view["neighbors"][order]] + offset)
        cells.append(view["cells"][order] + cell_offset)
        if "labels" in view:
            labels.append(view["labels"][order])
        offset += len(order)
        cell_offset += len(view["cell_seeds"])
    batch = {
        "features": torch.from_numpy(np.concatenate(features)),
        "positions": torch.from_numpy(np.concatenate(positions)),
        "lengths": torch.tensor([len(v["positions"]) for v in views]),
        "neighbors": torch.from_numpy(np.concatenate(neighbors)),
        "cells": torch.from_numpy(np.concatenate(cells)),
        "cell_lengths": torch.tensor([len(v["cell_seeds"]) for v in views]),
        "cell_count": cell_offset,
    }
    if labels:
        batch["labels"] = torch.from_numpy(np.concatenate(labels))
        batch["cell_labels"] = torch.from_numpy(
            np.concatenate([v["cell_labels"] for v in views])
        )
    return batch


def collate_views(batch: list[dict[str, Any]], block_size: int) -> Batch:
    """Packs the views of all samples, with the tracks of their cells.

    Returns:
        The packed views, sample by sample, and for each cell: `tracks`, its
        track numbered across the samples or -1, and `slots`, the index of its
        view within its sample, global views first. `global_views` is the number
        of global views per sample, and `views_per_sample` of all views.
    """
    views = [view for sample in batch for view in sample["views"]]
    packed = pack(views, block_size)
    tracks, slots, offset = [], [], 0
    for sample in batch:
        count = max((int(t.max(initial=-1)) for t in sample["tracks"]), default=-1)
        for slot, cell_tracks in enumerate(sample["tracks"]):
            tracks.append(np.where(cell_tracks >= 0, cell_tracks + offset, -1))
            slots.append(np.full(len(cell_tracks), slot))
        offset += count + 1
    return {
        **packed,
        "tracks": torch.from_numpy(np.concatenate(tracks)).long(),
        "slots": torch.from_numpy(np.concatenate(slots)).long(),
        "track_count": offset,
        "global_views": batch[0]["global_views"],
        "views_per_sample": len(batch[0]["views"]),
        "batch_size": len(batch),
    }


def collate_slides(batch: list[View], block_size: int) -> Batch:
    """Packs slide views with their nucleus and region labels."""
    return pack(batch, block_size)
