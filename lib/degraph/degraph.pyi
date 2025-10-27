import numpy as np
from numpy.typing import NDArray

def build_spatial_graph(
    points: NDArray[np.float32],
) -> list[list[tuple[int, float]]]: ...
