import numpy as np
from numpy.typing import NDArray

def build_adjacency_graph(
    simplices: NDArray[np.int64], distances: NDArray[np.float32], size: int
) -> list[list[tuple[int, float]]]: ...
