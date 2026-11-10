"""How easily views of the same field can be matched without learning anything.

Each view gets a fingerprint: quantiles of rotation and flip invariant nucleus
statistics, the log mean radius and the relative Fourier magnitudes of the
radial distances (elongation and finer shape). The score is the share of samples
whose first view has the second view of the same sample as its nearest
fingerprint. High values mean the invariance objective can be met by matching
nucleus statistics, whatever the descriptor, so it measures the augmentations.

    uv run python scripts/view_retrieval.py augmentation=weak
"""

import sys

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


SAMPLES = 256
HARMONICS = 3
QUANTILES = [10, 25, 50, 75, 90]


def fingerprint(radii: np.ndarray) -> np.ndarray:
    if len(radii) == 0:
        return np.zeros((HARMONICS + 1) * len(QUANTILES))
    spectrum = np.abs(np.fft.rfft(radii, axis=1))
    statistics = np.concatenate(
        [
            np.log(spectrum[:, :1] / radii.shape[1]),
            spectrum[:, 1 : HARMONICS + 1] / spectrum[:, :1],
        ],
        axis=1,
    )
    return np.percentile(statistics, QUANTILES, axis=0).ravel()


def main(overrides: list[str]) -> None:
    with initialize_config_dir(f"{sys.path[0]}/../configs", version_base=None):
        config = compose(
            "nfm",
            overrides=[
                "mode=fit",
                "metadata.run_name=-",
                *overrides,
                "descriptor=radial",
            ],
        )
    unlabeled = config.data.datasets.unlabeled
    dataset = instantiate(
        unlabeled.dataset, slides=pd.read_parquet(unlabeled.slides_path)
    )
    np.random.seed(0)
    samples = [dataset[i] for i in np.random.randint(len(dataset), size=SAMPLES)]

    prints = np.array([[fingerprint(v["features"]) for v in s[:2]] for s in samples])
    flat = prints.reshape(-1, prints.shape[-1])
    prints = (prints - flat.mean(0)) / (flat.std(0) + 1e-6)
    distances = np.linalg.norm(prints[:, 0, None] - prints[None, :, 1], axis=-1)
    accuracy = np.mean(distances.argmin(1) == np.arange(SAMPLES))
    print(
        f"top-1 view retrieval among {SAMPLES}: {accuracy:.1%} (chance {1 / SAMPLES:.1%})"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
