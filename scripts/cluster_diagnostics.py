"""Whether content-aware cells would sharpen the boundaries of the regions.

Whole Radboud slides are encoded with a checkpoint, with its own cells
("model"). Its final nucleus tokens are also clustered per slide with SLIC
here: starting from the initial cells of the data loader, each nucleus is reassigned to the nearby cell
center closest in position and token, and the centers are recomputed. The
weight of the tokens against the position is `lambda`, and 0 is spatial
k-means.

The cells are scored by their label purity, and by linear probes on the final
token of a nucleus concatenated with the mean token of its cell, by the
distance of the nucleus to the nearest nucleus of the other label.

    uv run python scripts/cluster_diagnostics.py checkpoint.ckpt [overrides]
"""

import sys
import time

import numpy as np
import torch
from nucleus_diagnostics import (
    balanced_threshold,
    boundary_distance,
    fit_probe,
    setup,
    slide_loader,
)
from sklearn.metrics import roc_auc_score
from torch import Tensor

from nfm.modeling.pooling import segment_sum
from nfm.ssl_meta_arch import SSLMetaArch


TRAIN_SLIDES, VALIDATION_SLIDES = 100, 50
TRAIN_NUCLEI, VALIDATION_NUCLEI = 1000, 3000  # sampled per slide
LAMBDAS = [0.0, 1.0, 4.0, 16.0, 64.0, 256.0]
ITERATIONS = 10
BINS = [(0, 10), (10, 20), (20, 50), (50, np.inf), (np.inf, np.inf)]


def segment_mean(values: Tensor, segments: Tensor, count: int) -> Tensor:
    sums = segment_sum(values, segments, count)
    sizes = torch.bincount(segments, minlength=count).clamp_min(1)
    return sums / sizes[:, None]


def slic(
    positions: Tensor, features: Tensor, cells: Tensor, count: int, weight: float
) -> Tensor:
    """Cells of the nuclei after SLIC iterations from the initial `cells`.

    Positions are scaled by the typical spacing of the initial cell centers and
    features so that two random nuclei are about 2 apart, so `weight` balances
    the two. A nucleus is only reassigned to centers within twice the spacing.
    """
    center_pos = segment_mean(positions, cells, count)
    center_feat = segment_mean(features, cells, count)
    gaps = torch.cdist(center_pos, center_pos)
    gaps.fill_diagonal_(torch.inf)
    spacing = gaps.min(1).values.median().clamp_min(1.0)
    for _ in range(ITERATIONS):
        spatial = torch.cdist(positions, center_pos).square() / spacing**2
        distance = spatial + weight * torch.cdist(features, center_feat).square()
        distance = distance.masked_fill(spatial > 4, torch.inf)  # beyond 2 spacings
        nearest = spatial.argmin(1)  # always reachable
        cells = torch.where(distance.min(1).values.isinf(), nearest, distance.argmin(1))
        sizes = torch.bincount(cells, minlength=count)
        moved_pos = segment_mean(positions, cells, count)
        moved_feat = segment_mean(features, cells, count)
        kept = (sizes > 0)[:, None]  # empty cells keep their centers
        center_pos = torch.where(kept, moved_pos, center_pos)
        center_feat = torch.where(kept, moved_feat, center_feat)
    return cells


def purity(cells: np.ndarray, labels: np.ndarray, distance: np.ndarray) -> dict:
    """How well the cells follow the labels."""
    count = cells.max() + 1
    positive = np.bincount(cells, labels, minlength=count)
    size = np.bincount(cells, minlength=count)
    share = positive / np.maximum(size, 1)
    majority = (share >= 0.5)[cells]
    mixed = ((share > 0) & (share < 1))[cells]
    near = distance < 20
    return {
        "cells": int((size > 0).sum()),
        "mixed": mixed,
        "agree": majority == (labels == 1),
        "agree_near": (majority == (labels == 1))[near],
    }


@torch.no_grad()
def encode(model: SSLMetaArch, loader, per_slide: int) -> dict:
    """Features and cell statistics of a sample of the nuclei of each slide."""
    names = ["model"] + [f"slic {w:g}" for w in LAMBDAS]
    rows: dict[str, list] = {}
    stats: dict[str, dict[str, list]] = {name: {} for name in names}
    start = time.time()
    for i, batch in enumerate(loader):
        labels = batch["labels"].numpy()
        distance = boundary_distance(batch["positions"].numpy(), labels)
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = model.encode(batch)
        simpool, tokens = encoded["regions"], encoded["nuclei"].float()
        features = (tokens - tokens.mean(0)) / (tokens.std(0) + 1e-6)
        features = features / features.size(1) ** 0.5
        count = int(batch["cell_count"])
        assignments = {"model": encoded["cells"]}
        for weight in LAMBDAS:
            assignments[f"slic {weight:g}"] = slic(
                batch["positions"], features, batch["cells"], count, weight
            )
        pick = np.random.default_rng(i).permutation(len(labels))[:per_slide]
        index = torch.from_numpy(pick).cuda()
        sample = {
            "labels": labels[pick],
            "distance": distance[pick],
            "final": tokens[index].half().cpu().numpy(),
            "model simpool": simpool.float()[encoded["cells"][index]]
            .half()
            .cpu()
            .numpy(),
        }
        for name, cells in assignments.items():
            means = segment_mean(tokens, cells, count)
            sample[f"{name} mean"] = means[cells[index]].half().cpu().numpy()
            for key, value in purity(cells.cpu().numpy(), labels, distance).items():
                stats[name].setdefault(key, []).append(value)
        for key, value in sample.items():
            rows.setdefault(key, []).append(value)
        if i % 10 == 0:
            print(f"  {i + 1} slides, {time.time() - start:.0f} s", flush=True)
    features = {key: np.concatenate(value) for key, value in rows.items()}
    return features, stats


def main(path: str, overrides: list[str]) -> None:
    model, config, train, validation = setup(path, overrides)
    rng = np.random.default_rng(0)
    data = {}
    for name, part, count, per_slide in (
        ("train", train, TRAIN_SLIDES, TRAIN_NUCLEI),
        ("validation", validation, VALIDATION_SLIDES, VALIDATION_NUCLEI),
    ):
        part = part.iloc[rng.permutation(len(part))[:count]]
        print(f"{name}: {len(part)} slides", flush=True)
        data[name] = encode(model, slide_loader(config, part), per_slide)
    (train_f, _), (val_f, stats) = data["train"], data["validation"]
    y_train, y_val = train_f["labels"], val_f["labels"]

    print(
        "\nCells of the validation slides, over all their nuclei: nuclei in cells "
        "of both labels, and nuclei whose cell majority has their label, overall "
        "and within 20 um of the other label"
    )
    print(f"{'cells':12s} {'count':>7s} {'mixed':>7s} {'agree':>7s} {'near':>7s}")
    for name, s in stats.items():
        print(
            f"{name:12s} {np.sum(s['cells']):7d} "
            f"{np.concatenate(s['mixed']).mean():7.1%} "
            f"{np.concatenate(s['agree']).mean():7.1%} "
            f"{np.concatenate(s['agree_near']).mean():7.1%}"
        )

    candidates = {"final": "final"} | {
        f"final + {name}": name
        for name in ["model simpool", "model mean"]
        + [f"slic {w:g} mean" for w in LAMBDAS]
    }
    distance = val_f["distance"]
    print(
        "\nProbes on the validation nuclei: AUROC, then the error rate by distance "
        "to the nearest nucleus of the other label"
    )
    header = f"{'features':24s} {'AUROC':>7s}"
    for low, high in BINS:
        header += f" {'one label' if np.isinf(low) else f'{low}-{high}':>9s}"
    print(header)
    for name, extra in candidates.items():

        def features(f: dict, extra: str = extra) -> np.ndarray:
            if extra == "final":
                return f["final"]
            return np.concatenate([f["final"], f[extra]], 1)

        predict = fit_probe(features(train_f), y_train)
        logits = predict(features(val_f))
        threshold = balanced_threshold(predict(features(train_f)), y_train)
        row = f"{name:24s} {roc_auc_score(y_val, logits):7.4f}"
        for low, high in BINS:
            inside = (
                np.isinf(distance)
                if np.isinf(low)
                else (distance >= low) & (distance < high)
            )
            error = np.mean((logits[inside] >= threshold) != y_val[inside])
            row += f" {error:9.1%}"
        print(row, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
