import torch
from torch.utils.data import WeightedRandomSampler

from nfm.data.datasets.classification_dataset import ClassificationNucleiDataset


class SourceBalancedRandomSampler(WeightedRandomSampler):
    """Half slide-uniform, half sqrt(source size), with replacement."""

    def __init__(self, dataset, source_fraction: float = 0.5) -> None:
        if not 0 <= source_fraction <= 1:
            raise ValueError("source_fraction must be in [0, 1]")
        sources = dataset.slides.dataset
        sizes = sources.value_counts()
        probabilities = sizes.pow(0.5) / sizes.pow(0.5).sum()
        weights = (1 - source_fraction) / len(sources) + source_fraction * sources.map(
            probabilities / sizes
        )
        super().__init__(weights.to_numpy(), len(sources), replacement=True)


class AutoWeightedRandomSampler(WeightedRandomSampler):
    def __init__(
        self,
        dataset: ClassificationNucleiDataset,
        positivity_thr: float = 0.05,
        replacement: bool = True,
        pos_slide_ratio: float = 0.75,
    ) -> None:
        positive = torch.tensor(
            [
                dataset.positivity[slide_id] > positivity_thr
                for slide_id in dataset.slides.slide_id
            ]
        )
        weights = torch.empty(len(dataset), dtype=torch.double)
        weights[positive] = pos_slide_ratio / max(int(positive.sum()), 1)
        weights[~positive] = (1 - pos_slide_ratio) / max(int((~positive).sum()), 1)
        super().__init__(weights, num_samples=len(dataset), replacement=replacement)
