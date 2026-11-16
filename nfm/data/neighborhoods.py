"""Spatial neighborhoods of nuclei and the initial cells of the clustering.

The cells themselves are clustered by the model from the tokens, starting from
these cells (`nfm.modeling.clustering`).
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from nfm.data.nuclei import View


def farthest_point_sampling(points: NDArray[np.float32], count: int) -> NDArray[Any]:
    """Indices of `count` points spread out evenly, starting from a random one."""
    count = min(count, len(points))
    chosen = np.empty(count, dtype=np.int64)
    if count == 0:
        return chosen
    chosen[0] = np.random.randint(len(points))
    distances = np.linalg.norm(points - points[chosen[0]], axis=1)
    for i in range(1, count):
        chosen[i] = distances.argmax()
        distances = np.minimum(
            distances, np.linalg.norm(points - points[chosen[i]], axis=1)
        )
    return chosen


def _nearest(tree: cKDTree, queries: NDArray[np.float32], k: int) -> NDArray[Any]:
    """The k nearest points, repeating the last one if there are fewer."""
    if tree.n == 0:
        return np.zeros((len(queries), k), dtype=np.int64)
    _, indices = tree.query(queries, min(k, tree.n))
    indices = indices.reshape(len(queries), -1)
    return np.pad(indices, ((0, 0), (0, k - indices.shape[1])), mode="edge")


class Neighborhoods:
    """Adds the nearest neighbors of each nucleus and initial cells to a view.

    Cell seeds are nuclei chosen by farthest point sampling, one per
    `nuclei_per_cell` nuclei, and each nucleus starts in the cell of the seed
    closest to it. Indices are local to the view.
    """

    def __init__(self, neighbors: int = 16, nuclei_per_cell: int = 64) -> None:
        self.neighbors = neighbors
        self.nuclei_per_cell = nuclei_per_cell

    def __call__(self, view: View) -> View:
        positions = view["positions"]
        count = -(-len(positions) // self.nuclei_per_cell)
        seeds = positions[farthest_point_sampling(positions, count)].reshape(-1, 2)
        cells = (
            cKDTree(seeds).query(positions)[1]
            if len(seeds)
            else np.zeros(0, dtype=np.int64)
        )
        return {
            **view,
            "neighbors": _nearest(cKDTree(positions), positions, self.neighbors),
            "cells": cells.astype(np.int64),
            "cell_seeds": seeds,
        }
