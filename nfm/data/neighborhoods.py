"""Spatial neighborhoods of nuclei and their Voronoi cells."""

from itertools import combinations
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
    """Adds the nearest neighbors of each nucleus and Voronoi cells to a view.

    Cell seeds are nuclei chosen by farthest point sampling, one per
    `nuclei_per_cell` nuclei, and each nucleus belongs to the cell of the seed
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


def match_cells(views: list[View], min_overlap: float) -> NDArray[Any]:
    """Pairs of cells in different views that hold the same nuclei.

    Two cells are paired when each is the other's best match by the Jaccard
    index of their nucleus ids, and that index is at least `min_overlap`.

    Returns:
        Rows of (view, cell, other view, other cell), [pairs, 4].
    """
    size = max(int(v["ids"].max(initial=0)) for v in views) + 1
    members = []
    for view in views:
        member = np.zeros((len(view["cell_seeds"]), size), dtype=np.float32)
        member[view["cells"], view["ids"]] = 1
        members.append(member)
    pairs = [np.zeros((0, 4), dtype=np.int64)]
    for a, b in combinations(range(len(views)), 2):
        if len(members[a]) == 0 or len(members[b]) == 0:
            continue
        shared = members[a] @ members[b].T
        union = members[a].sum(1)[:, None] + members[b].sum(1)[None] - shared
        jaccard = shared / np.maximum(union, 1)
        best_b, best_a = jaccard.argmax(1), jaccard.argmax(0)
        i = np.flatnonzero(
            (best_a[best_b] == np.arange(len(best_b)))
            & (jaccard[np.arange(len(best_b)), best_b] >= min_overlap)
        )
        pairs.append(
            np.stack([np.full_like(i, a), i, np.full_like(i, b), best_b[i]], axis=1)
        )
    return np.concatenate(pairs)
