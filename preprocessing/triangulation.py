from pathlib import Path

import numpy as np
import pandas as pd
import ray
from scipy.spatial import Delaunay
from tqdm import tqdm


@ray.remote(num_cpus=1, memory=1024**3)
def delaunay_triangulationa(path: Path) -> None:
    ds = pd.read_parquet(path)

    centroids = ds["centroids"]

    tri = Delaunay(centroids)
    adj_graph = [[] for _ in range(len(centroids))]
    for simplex in tri.simplices:
        for i in range(3):
            p1_idx, p2_idx = simplex[i], simplex[(i + 1) % 3]
            dist = np.linalg.norm(centroids[p1_idx] - centroids[p2_idx])
            adj_graph[p1_idx].append((p2_idx, dist))
            adj_graph[p2_idx].append((p1_idx, dist))

    ds["adjacency"] = adj_graph
    ds.to_parquet(path)


def main() -> None:
    slides = list(Path("slides").glob("*.parquet"))
    pending = [delaunay_triangulationa.remote(slide) for slide in slides]

    with tqdm(slides) as pbar:
        while pending:
            ready, pending = ray.wait(pending)
            pbar.update(len(ready))


if __name__ == "__main__":
    main()
