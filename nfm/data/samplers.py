import pandas as pd
from torch.utils.data import WeightedRandomSampler


class SourceBalancedRandomSampler(WeightedRandomSampler):
    """Half slide-uniform, half sqrt(source size), with replacement."""

    def __init__(self, sources: pd.Series, source_fraction: float = 0.5) -> None:
        sizes = sources.value_counts()
        probabilities = sizes.pow(0.5) / sizes.pow(0.5).sum()
        weights = (1 - source_fraction) / len(sources) + source_fraction * sources.map(
            probabilities / sizes
        )
        super().__init__(weights.to_numpy(copy=True), len(sources), replacement=True)
