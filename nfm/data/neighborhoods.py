"""Spatial neighborhoods of nuclei and the local class tokens."""

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
    """Adds the neighbors of each nucleus and the local class tokens to a view.

    Class tokens are placed at nuclei chosen by farthest point sampling, one per
    `nuclei_per_cls` nuclei, and each covers its `cls_neighbors` nearest nuclei.
    Indices are local to the view.
    """

    def __init__(
        self, neighbors: int = 16, nuclei_per_cls: int = 32, cls_neighbors: int = 128
    ) -> None:
        self.neighbors = neighbors
        self.nuclei_per_cls = nuclei_per_cls
        self.cls_neighbors = cls_neighbors

    def __call__(self, view: View) -> View:
        positions = view["positions"]
        tree = cKDTree(positions)
        count = -(-len(positions) // self.nuclei_per_cls)
        cls_positions = positions[farthest_point_sampling(positions, count)]
        return {
            **view,
            "neighbors": _nearest(tree, positions, self.neighbors),
            "cls_positions": cls_positions.reshape(-1, 2),
            "cls_neighbors": _nearest(tree, cls_positions, self.cls_neighbors),
        }


def match_class_tokens(views: list[View], min_overlap: float) -> NDArray[Any]:
    """Pairs of class tokens in different views that cover the same nuclei.

    Two class tokens are paired when each is the other's best match by the
    Jaccard index of their nucleus ids, and that index is at least `min_overlap`.

    Returns:
        Rows of (view, class token, other view, other class token), [pairs, 4].
    """
    size = max(int(v["ids"].max(initial=0)) for v in views) + 1
    covers = []
    for view in views:
        cover = np.zeros((len(view["cls_neighbors"]), size), dtype=np.float32)
        rows = np.arange(len(cover))[:, None]
        cover[rows, view["ids"][view["cls_neighbors"]]] = 1
        covers.append(cover)
    pairs = [np.zeros((0, 4), dtype=np.int64)]
    for a, b in combinations(range(len(views)), 2):
        if len(covers[a]) == 0 or len(covers[b]) == 0:
            continue
        shared = covers[a] @ covers[b].T
        union = covers[a].sum(1)[:, None] + covers[b].sum(1)[None] - shared
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
