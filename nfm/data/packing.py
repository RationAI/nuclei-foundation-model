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
        ids: Nucleus ids, when the views have them.
    """
    features, positions, neighbors, cells, labels, ids = [], [], [], [], [], []
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
        if "ids" in view:
            ids.append(view["ids"][order])
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
    if ids:
        batch["ids"] = torch.from_numpy(np.concatenate(ids))
    return batch


def nucleus_tracks(
    ids: NDArray[Any], slots: NDArray[Any], global_views: int, count: int
) -> tuple[NDArray[Any], NDArray[Any], int]:
    """Tracks of the nuclei of one sample across its views.

    A nucleus in every global view starts a track, of which `count` are drawn
    at random, and its copies in the local views join it.

    Args:
        ids: Id of each nucleus of the sample, the same in every view, or -1
            for nuclei added by the augmentations, [nuclei].
        slots: View of each nucleus within the sample, global views first.
        global_views: Number of global views.
        count: Most tracks to draw.

    Returns:
        The indices of the tracked nuclei, their tracks and the number of tracks.
    """
    real = ids >= 0  # -1, shared by all added nuclei, is never tracked
    shared = np.unique(ids[(slots == 0) & real])
    for slot in range(1, global_views):
        shared = np.intersect1d(shared, ids[(slots == slot) & real], assume_unique=True)
    if len(shared) > count:
        shared = np.sort(np.random.choice(shared, count, replace=False))
    at = np.searchsorted(shared, ids).clip(max=max(len(shared) - 1, 0))
    tracked = np.flatnonzero(shared[at] == ids) if len(shared) else np.zeros(0, int)
    return tracked, at[tracked], len(shared)


def collate_views(
    batch: list[dict[str, Any]], block_size: int, nucleus_tracks_per_sample: int = 256
) -> Batch:
    """Packs the views of all samples, with the tracks of their cells and nuclei.

    Returns:
        The packed views, sample by sample, and for each cell: `tracks`, its
        track numbered across the samples or -1, and `slots`, the index of its
        view within its sample, global views first. Of the tracked nuclei, up to
        `nucleus_tracks_per_sample` tracks per sample: `nucleus_index`, their
        index in the packed views, `nucleus_tracks` and `nucleus_slots`.
        `global_views` is the number of global views per sample, and
        `views_per_sample` of all views.
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

    per_sample = len(batch[0]["views"])
    ids = packed.pop("ids").numpy()
    lengths = packed["lengths"].numpy()
    token_slots = np.repeat(np.tile(np.arange(per_sample), len(batch)), lengths)
    sample_lengths = lengths.reshape(len(batch), per_sample).sum(1)
    index, nucleus_tracks_, nucleus_offset, start = [], [], 0, 0
    for length in sample_lengths:
        part = slice(start, start + length)
        tracked, track, track_count = nucleus_tracks(
            ids[part],
            token_slots[part],
            batch[0]["global_views"],
            nucleus_tracks_per_sample,
        )
        index.append(tracked + start)
        nucleus_tracks_.append(track + nucleus_offset)
        nucleus_offset += track_count
        start += length
    index_ = np.concatenate(index)
    return {
        **packed,
        "tracks": torch.from_numpy(np.concatenate(tracks)).long(),
        "slots": torch.from_numpy(np.concatenate(slots)).long(),
        "track_count": offset,
        "nucleus_index": torch.from_numpy(index_).long(),
        "nucleus_tracks": torch.from_numpy(np.concatenate(nucleus_tracks_)).long(),
        "nucleus_slots": torch.from_numpy(token_slots[index_]).long(),
        "nucleus_track_count": nucleus_offset,
        "global_views": batch[0]["global_views"],
        "views_per_sample": per_sample,
        "batch_size": len(batch),
    }


def collate_slides(batch: list[View], block_size: int) -> Batch:
    """Packs slide views with their nucleus and region labels."""
    return pack(batch, block_size)
