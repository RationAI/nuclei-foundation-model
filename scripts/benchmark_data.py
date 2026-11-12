"""Throughput of each data loader alone, to find which one starves the GPU.

Each loader runs with its configured workers and without the model, so its
seconds per batch can be compared with the seconds per step of training. Then one
sample of each kind is profiled in this process: a stage whose wall time is well
above its CPU time waits for storage rather than computing.

    uv run python scripts/benchmark_data.py [overrides]
"""

import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import hydra
import numpy as np
from hydra import compose, initialize_config_dir

import nfm.data.datasets.classification_dataset as labeled_module
import nfm.data.datasets.nuclei_dataset as unlabeled_module
from nfm.data import DataModule, neighborhoods


BATCHES = 20
WARMUP = 5


def throughput(name: str, loader: Any, workers: int) -> None:
    iterator = iter(loader)
    for _ in range(WARMUP):  # workers start and fill their queues
        next(iterator)
    start = time.perf_counter()
    for _ in range(BATCHES):
        next(iterator)
    seconds = (time.perf_counter() - start) / BATCHES
    print(f"{name:12s} {workers:3d} workers: {seconds:6.2f} s per batch", flush=True)


stages: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])


@contextmanager
def timed(name: str) -> Iterator[None]:
    wall, cpu = time.perf_counter(), time.process_time()
    yield
    stage = stages[name]
    stage[0] += time.perf_counter() - wall
    stage[1] += time.process_time() - cpu
    stage[2] += 1


def wrap(owner: Any, attribute: str, name: str) -> None:
    function: Callable[..., Any] = getattr(owner, attribute)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with timed(name):
            return function(*args, **kwargs)

    setattr(owner, attribute, wrapped)


def main(overrides: list[str]) -> None:
    with initialize_config_dir(f"{sys.path[0]}/../configs", version_base=None):
        config = compose(
            "nfm", overrides=["mode=fit", "metadata.run_name=-", *overrides]
        )
    data = hydra.utils.instantiate(config.data, _recursive_=False, _target_=DataModule)
    data.prepare_data()
    data.setup("fit")
    datasets = config.data.datasets

    print(f"Loaders alone, {BATCHES} batches after {WARMUP} warm-up batches:")
    loaders = data.train_dataloader().iterables
    throughput("unlabeled", loaders["unlabeled"], datasets.unlabeled.num_workers)
    throughput("labeled", loaders["labeled"], datasets.labeled.num_workers)
    throughput("validation", data.val_dataloader(), datasets.labeled.eval_num_workers)

    wrap(unlabeled_module.NucleiDataset, "load_region", "unlabeled: lance region query")
    wrap(unlabeled_module, "make_view", "unlabeled: augmentation and features")
    wrap(unlabeled_module, "match_class_tokens", "unlabeled: class token pairs")
    wrap(labeled_module.pd, "read_parquet", "labeled: parquet reads")
    wrap(labeled_module, "make_view", "labeled: features")
    wrap(neighborhoods.Neighborhoods, "__call__", "both: neighbors and class tokens")
    np.random.seed(0)
    for i in np.random.randint(len(data.train), size=8):
        with timed("unlabeled: sample"):
            data.train[i]
    for i in range(8):
        with timed("labeled: sample"):
            data.train_labeled[i]
    print(
        f"\n{'stage, in this process':42s} {'calls':>5s} {'wall s':>7s} {'cpu/wall':>8s}"
    )
    for name, (wall, cpu, calls) in stages.items():
        print(f"{name:42s} {calls:5d} {wall / calls:7.3f} {cpu / wall:8.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
