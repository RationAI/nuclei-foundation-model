"""Packs variable-length views into one sequence of tokens.

Each view becomes a class token followed by one token per nucleus. The class
token has no features and sits at the view center, the origin of the positions.
"""

from typing import Any

import numpy as np
import torch
from torch import Tensor

from nfm.data.nuclei import View


type Batch = dict[str, Any]


def pack(views: list[View]) -> dict[str, Tensor]:
    """Concatenates the views into one sequence.

    Returns:
        features: Shape descriptors, zeros for the class tokens, [tokens, features].
        positions: Positions relative to the view center, [tokens, 2].
        lengths: Tokens per view, including the class token, [views].
    """
    width = views[0]["efds"].shape[1]
    cls = np.zeros((1, width), dtype=np.float32)
    origin = np.zeros((1, 2), dtype=np.float32)
    return {
        "features": torch.from_numpy(
            np.concatenate([x for v in views for x in (cls, v["efds"])])
        ),
        "positions": torch.from_numpy(
            np.concatenate([x for v in views for x in (origin, v["positions"])])
        ),
        "lengths": torch.tensor([len(v["positions"]) + 1 for v in views]),
    }


def blob_mask(positions: np.ndarray, ratio: float) -> np.ndarray:
    """Masks the share `ratio` of nuclei closest to a random nucleus."""
    count = round(ratio * len(positions))
    if count == 0:
        return np.zeros(len(positions), dtype=bool)
    anchor = positions[np.random.randint(len(positions))]
    distances = np.linalg.norm(positions - anchor, axis=1)
    mask = np.zeros(len(positions), dtype=bool)
    mask[np.argpartition(distances, count - 1)[:count]] = True
    return mask


def collate_views(
    batch: list[dict[str, list[View]]],
    mask_probability: float = 0.5,
    mask_ratio: tuple[float, float] = (0.1, 0.5),
) -> Batch:
    """Packs global views, local views and masked copies of some global views.

    Views are ordered view-major, so view `v` of sample `b` is sequence
    `v * batch_size + b`. As in DINOv2, only a part of the global views is masked.
    A masked nucleus is paired with the same nucleus in the next global view of
    the sample, if it is there.

    Returns:
        The packed tokens with `masked` for the masked inputs, `pairs` with the
        masked and target token indices, [2, pairs], and the view counts.
    """
    n = len(batch)
    global_views = [
        s["global"][v] for v in range(len(batch[0]["global"])) for s in batch
    ]
    local_views = [s["local"][v] for v in range(len(batch[0]["local"])) for s in batch]
    count = round(mask_probability * len(global_views))
    chosen = np.random.permutation(len(global_views))[:count]
    masks = [
        blob_mask(global_views[i]["positions"], np.random.uniform(*mask_ratio))
        for i in chosen
    ]

    views = global_views + local_views + [global_views[i] for i in chosen]
    packed = pack(views)
    starts = np.cumsum(packed["lengths"].numpy()) - packed["lengths"].numpy()
    first_masked = len(global_views) + len(local_views)
    masked = np.zeros(len(packed["features"]), dtype=bool)
    predictions, targets = [np.zeros(0, int)], [np.zeros(0, int)]
    for k, (i, mask) in enumerate(zip(chosen, masks, strict=True)):
        tokens = starts[first_masked + k] + 1 + np.flatnonzero(mask)
        masked[tokens] = True
        partner = (i + n) % len(global_views)
        _, ours, theirs = np.intersect1d(
            global_views[i]["ids"][mask],
            global_views[partner]["ids"],
            assume_unique=True,
            return_indices=True,
        )
        predictions.append(tokens[ours])
        targets.append(starts[partner] + 1 + theirs)

    return {
        **packed,
        "masked": torch.from_numpy(masked),
        "pairs": torch.from_numpy(
            np.stack([np.concatenate(predictions), np.concatenate(targets)])
        ).long(),
        "batch_size": n,
        "global_views": len(batch[0]["global"]),
        "local_views": len(batch[0]["local"]),
    }


def collate_tiles(batch: list[list[View]]) -> Batch:
    """Packs the tiles of all slides with tile and nucleus labels."""
    tiles = [tile for slide in batch for tile in slide]
    return {
        **pack(tiles),
        "labels": torch.tensor(np.array([t["label"] for t in tiles])),
        "nucleus_labels": torch.from_numpy(
            np.concatenate([t["labels"] for t in tiles])
        ),
    }
