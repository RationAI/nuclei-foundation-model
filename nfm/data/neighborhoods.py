"""Spatial neighborhoods of nuclei and their Voronoi cells."""

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


def _members(view: View, size: int) -> NDArray[np.float32]:
    """Membership of the nucleus ids in the cells of a view, [cells, size].

    Nuclei with id -1, added by the augmentations, belong to no track.
    """
    members = np.zeros((len(view["cell_seeds"]), size), dtype=np.float32)
    real = view["ids"] >= 0
    members[view["cells"][real], view["ids"][real]] = 1
    return members


def match_tracks(
    views: list[View], global_views: int, min_overlap: float, min_share: float
) -> list[NDArray[Any]]:
    """Tracks of cells that hold the same nuclei in different views.

    A track starts from a cell of the first global view that is the mutual best
    match, by the Jaccard index of their nucleus ids, of a cell in every other
    global view, with an index of at least `min_overlap`. A cell of a local view
    joins the track whose global cells hold the largest share of its nuclei, if
    that share is at least `min_share`. Local cells are small and often cut by
    the view border, so their Jaccard index with a whole track is low even when
    they lie inside it.

    Returns:
        For each view, the track of each of its cells, or -1, [cells].
    """
    size = max(int(v["ids"].max(initial=-1)) for v in views) + 1
    members = [_members(view, size) for view in views[:global_views]]
    anchor = members[0]
    keep = np.ones(len(anchor), dtype=bool)
    partners = []
    for other in members[1:]:
        if len(other) == 0:
            keep[:] = False
            partners.append(np.zeros(len(anchor), dtype=np.int64))
            continue
        shared = anchor @ other.T
        union = anchor.sum(1)[:, None] + other.sum(1)[None] - shared
        jaccard = shared / np.maximum(union, 1)
        best, best_back = jaccard.argmax(1), jaccard.argmax(0)
        keep &= best_back[best] == np.arange(len(anchor))
        keep &= jaccard[np.arange(len(anchor)), best] >= min_overlap
        partners.append(best)
    starts = np.flatnonzero(keep)
    tracks = [np.full(len(view["cell_seeds"]), -1, dtype=np.int64) for view in views]
    tracks[0][starts] = np.arange(len(starts))
    nuclei = anchor[starts]
    for v, best in enumerate(partners, start=1):
        tracks[v][best[starts]] = np.arange(len(starts))
        nuclei = np.maximum(nuclei, members[v][best[starts]])
    if len(starts) == 0:
        return tracks
    for v in range(global_views, len(views)):
        local = _members(views[v], size)
        share = (local @ nuclei.T) / np.maximum(local.sum(1)[:, None], 1)
        best = share.argmax(1)
        joined = share[np.arange(len(local)), best] >= min_share
        tracks[v][joined] = best[joined]
    return tracks
