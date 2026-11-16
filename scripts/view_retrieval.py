"""How easily paired cells can be matched without learning anything.

The initial cells of the data loader are tracked as in training, and each cell
of a track is paired with the cell of its track in the first global view. The nuclei of each paired cell get a fingerprint: quantiles of
rotation and flip invariant nucleus statistics, the log mean radius and the
relative Fourier magnitudes of the radial distances (elongation and finer
shape). The score is the share of pairs whose first cell has its partner
as the nearest fingerprint among the partners of all pairs. High values mean the
invariance objective can be met by matching nucleus statistics, whatever the
descriptor, so it measures the augmentations.

    uv run python scripts/view_retrieval.py augmentation=weak
"""

import sys

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from nfm.data.packing import collate_views
from nfm.modeling.clustering import match_tracks


SAMPLES = 32
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

    def pairs(sample: dict) -> list[np.ndarray]:
        """Fingerprints of each tracked cell and of its cell in the first view."""
        packed = collate_views([sample], block_size=config.data.block_size)
        cells, count = packed["cells"], packed["cell_count"]
        tracks, _ = match_tracks(
            cells,
            count,
            packed["keys"],
            packed["slots"],
            config.model.min_overlap,
            config.model.min_share,
        )
        slots = torch.zeros(count, dtype=torch.long)
        slots[cells] = packed["slots"]
        features = packed["features"].numpy()
        cells = cells.numpy()
        anchors = {
            int(tracks[c]): c for c in range(count) if slots[c] == 0 and tracks[c] >= 0
        }
        return [
            np.stack(
                [
                    fingerprint(features[cells == anchors[int(tracks[c])]]),
                    fingerprint(features[cells == c]),
                ]
            )
            for c in range(count)
            if slots[c] > 0 and tracks[c] >= 0
        ]

    prints = np.array([pair for s in samples for pair in pairs(s)])
    flat = prints.reshape(-1, prints.shape[-1])
    prints = (prints - flat.mean(0)) / (flat.std(0) + 1e-6)
    distances = np.linalg.norm(prints[:, 0, None] - prints[None, :, 1], axis=-1)
    count = len(prints)
    accuracy = np.mean(distances.argmin(1) == np.arange(count))
    print(
        f"top-1 cell retrieval among {count} pairs: {accuracy:.1%} "
        f"(chance {1 / count:.2%})"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
