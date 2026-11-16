"""Where the nucleus embeddings of a checkpoint lose against supervision.

Whole Radboud slides are encoded as in validation. For a sample of the nuclei
of each slide, the residual stream after every block, the final normed tokens
and the embedding of the Voronoi cell of the nucleus are kept, with the
distance of the nucleus to the nearest nucleus of the other label.

Linear probes are trained on each of these features of training slides and
scored on validation slides: by layer, to see whether the last layer is the
best for nuclei, and by the distance to the label boundary, to see whether the
errors are at the boundaries of the annotations.

The features are cached next to the checkpoint, in checkpoint.diagnostics/.
For a checkpoint trained with fixed farthest point sampling cells, add
`model.clustering.iterations=0`.

    uv run python scripts/nucleus_diagnostics.py checkpoint.ckpt [overrides]
"""

import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from nfm.data.data_module import init_worker
from nfm.data.packing import collate_slides
from nfm.ssl_meta_arch import SSLMetaArch


TRAIN_SLIDES, VALIDATION_SLIDES = 300, 150
TRAIN_NUCLEI, VALIDATION_NUCLEI = 1000, 3000  # sampled per slide
# um to the nearest nucleus of the other label; slides of one label have none
BINS = [(0, 10), (10, 20), (20, 50), (50, 100), (100, np.inf), (np.inf, np.inf)]


def boundary_distance(positions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Distance of each nucleus to the nearest nucleus of the other label."""
    distance = np.full(len(labels), np.inf, dtype=np.float32)
    for label in (0, 1):
        own, other = labels == label, labels != label
        if own.any() and other.any():
            distance[own] = cKDTree(positions[other]).query(positions[own])[0]
    return distance


@torch.no_grad()
def encode(
    model: SSLMetaArch, loader: DataLoader, per_slide: int, cache: Path
) -> dict[str, np.ndarray]:
    """Features of a sample of the nuclei of each slide, cached in `cache`.

    The features are written to memory-mapped arrays as they come, so they need
    no memory of their own, and a finished cache is read instead of encoding.
    """
    if (cache / "count.txt").exists():
        count = int((cache / "count.txt").read_text())
        print(f"  cached, {count} nuclei", flush=True)
        return _trim(
            {f.stem: np.load(f, mmap_mode="r") for f in cache.glob("*.npy")}, count
        )
    cache.mkdir(parents=True, exist_ok=True)
    capacity = len(loader.dataset) * per_slide
    depth = len(model.backbone.blocks)
    dim = model.backbone.norm.normalized_shape[0]
    shapes = {
        "labels": ((capacity,), np.float32),
        "distance": ((capacity,), np.float32),
        "slide": ((capacity,), np.int32),
        "online_nucleus": ((capacity,), np.float32),
        "online_region": ((capacity,), np.float32),
        "cell": ((capacity, dim), np.float16),
        "final": ((capacity, dim), np.float16),
        # the residual stream after each block, by block for contiguous reads
        "layers": ((depth, capacity, dim), np.float16),
    }
    arrays = {
        name: np.lib.format.open_memmap(
            cache / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        for name, (shape, dtype) in shapes.items()
    }
    layers: list[torch.Tensor] = []
    hooks = [
        block.register_forward_hook(lambda _m, _i, out: layers.append(out))
        for block in model.backbone.blocks
    ]
    count, start = 0, time.time()
    for i, batch in enumerate(loader):
        labels = batch["labels"].numpy()
        distance = boundary_distance(batch["positions"].numpy(), labels)
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        layers.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = model.encode(batch)
            cells, tokens = encoded["regions"], encoded["nuclei"]
            region_logits = model.region_probe(cells).squeeze(1)
            nucleus_logits = model.nucleus_probe(tokens).squeeze(1)
        pick = np.random.default_rng(i).permutation(len(labels))[:per_slide]
        index = torch.from_numpy(pick).cuda()
        cell_of = encoded["cells"][index]
        rows = slice(count, count + len(pick))
        arrays["labels"][rows] = labels[pick]
        arrays["distance"][rows] = distance[pick]
        arrays["slide"][rows] = i
        arrays["online_nucleus"][rows] = nucleus_logits[index].float().cpu().numpy()
        arrays["online_region"][rows] = region_logits[cell_of].float().cpu().numpy()
        arrays["cell"][rows] = cells[cell_of].half().cpu().numpy()
        arrays["final"][rows] = tokens[index].half().cpu().numpy()
        for block, output in enumerate(layers):
            arrays["layers"][block, rows] = output[index].half().cpu().numpy()
        count += len(pick)
        if i % 25 == 0:
            print(f"  {i + 1} slides, {time.time() - start:.0f} s", flush=True)
    for hook in hooks:
        hook.remove()
    for array in arrays.values():
        array.flush()
    (cache / "count.txt").write_text(str(count))
    return _trim(arrays, count)


def _trim(arrays: dict[str, np.ndarray], count: int) -> dict[str, np.ndarray]:
    """The first `count` nuclei of each array, along its nucleus axis."""
    return {
        name: array[:, :count] if name == "layers" else array[:count]
        for name, array in arrays.items()
    }


def fit_probe(x: np.ndarray, y: np.ndarray, epochs: int = 30) -> Any:
    """Logistic regression on standardized features, on the GPU."""
    torch.manual_seed(0)
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).cuda()
    mean, std = x_t.mean(0), x_t.std(0) + 1e-6
    x_t = (x_t - mean) / std
    y_t = torch.from_numpy(np.array(y, dtype=np.float32)).cuda()
    linear = torch.nn.Linear(x.shape[1], 1).cuda()
    optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        for chunk in torch.randperm(len(x_t), device="cuda").split(4096):
            loss = F.binary_cross_entropy_with_logits(
                linear(x_t[chunk]).squeeze(1), y_t[chunk]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def predict(features: np.ndarray) -> np.ndarray:
        z = torch.from_numpy(np.asarray(features, dtype=np.float32)).cuda()
        z = (z - mean) / std
        return linear(z).squeeze(1).cpu().numpy()

    return predict


def balanced_threshold(logits: np.ndarray, labels: np.ndarray) -> float:
    """The logit threshold of the best balanced accuracy."""
    candidates = np.quantile(logits, np.linspace(0.01, 0.99, 197))
    scores = [
        ((logits >= t) & (labels == 1)).sum() / max((labels == 1).sum(), 1)
        + ((logits < t) & (labels == 0)).sum() / max((labels == 0).sum(), 1)
        for t in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


def setup(
    path: str, overrides: list[str]
) -> tuple[SSLMetaArch, DictConfig, pd.DataFrame, pd.DataFrame]:
    """The model of a checkpoint, the config and the Radboud slide split."""
    with initialize_config_dir(f"{sys.path[0]}/../configs", version_base=None):
        config = compose(
            "nfm",
            overrides=[
                "mode=fit",
                "metadata.run_name=-",
                "+model.backbone.compile=false",  # flex attention stays compiled
                *overrides,
            ],
        )
    model = instantiate(config.model, _target_=SSLMetaArch)
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state)
    model.cuda().eval()

    labeled = config.data.datasets.labeled
    slides = pd.read_parquet(download_artifacts(labeled.slides_path))
    slides = slides[slides["nuclei_count"] >= 512]
    # The split of training, which the data module draws after seeding.
    np.random.seed(config.seed)
    train, validation = train_test_split(
        slides,
        test_size=labeled.split_size,
        stratify=slides[labeled.split_stratify_col],
    )
    return model, config, train, validation


def slide_loader(config: DictConfig, slides: pd.DataFrame) -> DataLoader:
    """Whole slides with their labels, one per batch."""
    dataset = instantiate(config.data.datasets.labeled.dataset, slides=slides)
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=3,
        worker_init_fn=init_worker,
        collate_fn=partial(collate_slides, block_size=config.data.block_size),
    )


def main(path: str, overrides: list[str]) -> None:
    model, config, train, validation = setup(path, overrides)
    rng = np.random.default_rng(0)
    features = {}
    for name, part, count, per_slide in (
        ("train", train, TRAIN_SLIDES, TRAIN_NUCLEI),
        ("validation", validation, VALIDATION_SLIDES, VALIDATION_NUCLEI),
    ):
        part = part.iloc[rng.permutation(len(part))[:count]]
        loader = slide_loader(config, part)
        print(f"{name}: {len(part)} slides", flush=True)
        cache = Path(path).with_suffix(".diagnostics") / name
        features[name] = encode(model, loader, per_slide, cache)
    train_f, val_f = features["train"], features["validation"]
    y_train, y_val = train_f["labels"], val_f["labels"]
    print(
        f"\nnuclei: {len(y_train)} train ({y_train.mean():.1%} positive), "
        f"{len(y_val)} validation ({y_val.mean():.1%} positive)"
    )

    depth = train_f["layers"].shape[0]
    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {
        f"block {i + 1}": (train_f["layers"][i], val_f["layers"][i])
        for i in range(depth)
    }
    candidates["final (normed)"] = (train_f["final"], val_f["final"])
    candidates["mean of last 4 blocks"] = (
        train_f["layers"][-4:].astype(np.float32).mean(0),
        val_f["layers"][-4:].astype(np.float32).mean(0),
    )
    candidates["cell embedding"] = (train_f["cell"], val_f["cell"])
    candidates["final + cell"] = (
        np.concatenate([train_f["final"], train_f["cell"]], 1),
        np.concatenate([val_f["final"], val_f["cell"]], 1),
    )

    print(f"\n{'features':24s} {'AUROC':>7s} {'AUPRC':>7s}")
    for name in ("online nucleus probe", "online region probe"):
        logits = val_f[name.replace("online ", "online_").replace(" probe", "")]
        print(
            f"{name:24s} {roc_auc_score(y_val, logits):7.4f} "
            f"{average_precision_score(y_val, logits):7.4f}"
        )
    predictions = {}
    for name, (x_train, x_val) in candidates.items():
        predict = fit_probe(x_train, y_train)
        logits = predict(x_val)
        threshold = balanced_threshold(predict(x_train), y_train)
        predictions[name] = (logits, threshold)
        print(
            f"{name:24s} {roc_auc_score(y_val, logits):7.4f} "
            f"{average_precision_score(y_val, logits):7.4f}",
            flush=True,
        )

    best = max(
        (n for n in predictions if n.startswith("block")),
        key=lambda n: roc_auc_score(y_val, predictions[n][0]),
    )
    shown = ["final (normed)", best, "cell embedding", "final + cell"]
    distance = val_f["distance"]
    print(
        "\nBy distance to the nearest nucleus of the other label "
        "(error rate at the threshold of best balanced accuracy on train; "
        "AUROC within the bin)"
    )
    header = f"{'distance':>12s} {'nuclei':>8s} {'positive':>8s}"
    print(header + "".join(f" | {n[:20]:>20s}" for n in shown))
    for low, high in BINS:
        if np.isinf(low):
            inside, label = np.isinf(distance), "one label"
        else:
            inside = (distance >= low) & (distance < high)
            label = f"{low}-{high} um" if np.isfinite(high) else f">{low} um"
        if not inside.any():
            continue
        row = f"{label:>12s} {inside.sum():8d} {y_val[inside].mean():8.1%}"
        for name in shown:
            logits, threshold = predictions[name]
            error = np.mean((logits[inside] >= threshold) != y_val[inside])
            both = 0 < y_val[inside].mean() < 1
            auc = roc_auc_score(y_val[inside], logits[inside]) if both else np.nan
            row += f" | {error:9.1%} {auc:10.4f}"
        print(row)
    # Errors far from any boundary would mean the embedding misses tissue that a
    # region probe sees; errors only near boundaries point at label resolution.


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
