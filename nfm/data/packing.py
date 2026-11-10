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
        features: Shape features, zeros for the class tokens, [tokens, features].
        positions: Positions relative to the view center, [tokens, 2].
        lengths: Tokens per view, including the class token, [views].
    """
    width = views[0]["features"].shape[1]
    cls = np.zeros((1, width), dtype=np.float32)
    origin = np.zeros((1, 2), dtype=np.float32)
    return {
        "features": torch.from_numpy(
            np.concatenate([x for v in views for x in (cls, v["features"])])
        ),
        "positions": torch.from_numpy(
            np.concatenate([x for v in views for x in (origin, v["positions"])])
        ),
        "lengths": torch.tensor([len(v["positions"]) + 1 for v in views]),
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
