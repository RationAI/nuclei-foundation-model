"""Opt-in worker-side data timings: set NFM_DATA_TIMING=1."""

import json
import os
import socket
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter

from torch.utils.data import get_worker_info


class DataTiming:
    """Print one flushed JSON summary per sample or batch, with stage wall times."""

    def __init__(self, scope: str, *, enabled: bool | None = None, **details) -> None:
        self.enabled = (
            os.environ.get("NFM_DATA_TIMING") == "1" if enabled is None else enabled
        )
        self.scope = scope
        self.details = details
        self.stages = defaultdict(float)

    def __enter__(self):
        if self.enabled:
            self.started = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return
        elapsed = perf_counter() - self.started
        worker = get_worker_info()
        record = {
            "scope": self.scope,
            "host": socket.gethostname(),
            "rank": os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")),
            "pid": os.getpid(),
            "worker": worker.id if worker is not None else "main",
            **self.details,
            "total_ms": round(elapsed * 1000, 3),
            "stages_ms": {
                key: round(value * 1000, 3) for key, value in self.stages.items()
            },
        }
        if exc_type is not None:
            record["error"] = exc_type.__name__
        print("[data-timing] " + json.dumps(record), flush=True)

    @contextmanager
    def measure(self, stage: str):
        if not self.enabled:
            yield
            return
        start = perf_counter()
        try:
            yield
        finally:
            self.stages[stage] += perf_counter() - start
