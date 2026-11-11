"""Packs variable-length views into one sequence of tokens.

The nuclei of all views come first, each view sorted so that the attention
blocks are spatially compact, then the local class tokens of all views. Class
tokens have no features and sit at their positions in the view frame.
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
        cls_positions: Positions of the class tokens, [class tokens, 2].
        cls_lengths: Class tokens per view, [views].
        cls_neighbors: Nuclei covered by each class token, [class tokens, k].
        labels, cls_labels: Nucleus and region labels, when the views have them.
    """
    packed: dict[str, list[NDArray[Any]]] = {
        key: [] for key in ("features", "positions", "neighbors", "cls_neighbors")
    }
    labels = []
    offset = 0
    for view in views:
        order = spatial_order(view["positions"], offset, block_size)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        packed["features"].append(view["features"][order])
        packed["positions"].append(view["positions"][order])
        packed["neighbors"].append(inverse[view["neighbors"][order]] + offset)
        packed["cls_neighbors"].append(inverse[view["cls_neighbors"]] + offset)
        if "labels" in view:
            labels.append(view["labels"][order])
        offset += len(order)
    batch = {
        key: torch.from_numpy(np.concatenate(value)) for key, value in packed.items()
    }
    batch["lengths"] = torch.tensor([len(v["positions"]) for v in views])
    batch["cls_positions"] = torch.from_numpy(
        np.concatenate([v["cls_positions"] for v in views])
    )
    batch["cls_lengths"] = torch.tensor([len(v["cls_positions"]) for v in views])
    if labels:
        batch["labels"] = torch.from_numpy(np.concatenate(labels))
        batch["cls_labels"] = torch.from_numpy(
            np.concatenate([v["cls_labels"] for v in views])
        )
    return batch


def collate_views(batch: list[dict[str, Any]], block_size: int) -> Batch:
    """Packs the views of all samples, with the paired class tokens.

    Returns:
        The packed views, sample by sample, and `pairs`, the indices of the
        paired class tokens, [2, pairs].
    """
    views = [view for sample in batch for view in sample["views"]]
    packed = pack(views, block_size)
    lengths = packed["cls_lengths"].numpy()
    starts = np.cumsum(lengths) - lengths
    per_sample = len(batch[0]["views"])
    pairs = [
        np.stack(
            [
                starts[b * per_sample + rows[:, 0]] + rows[:, 1],
                starts[b * per_sample + rows[:, 2]] + rows[:, 3],
            ]
        )
        for b, rows in enumerate(sample["pairs"] for sample in batch)
    ]
    return {
        **packed,
        "pairs": torch.from_numpy(np.concatenate(pairs, axis=1)).long(),
        "batch_size": len(batch),
    }


def collate_slides(batch: list[View], block_size: int) -> Batch:
    """Packs slide views with their nucleus and region labels."""
    return pack(batch, block_size)
