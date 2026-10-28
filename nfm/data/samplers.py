import torch
from torch.utils.data import WeightedRandomSampler

from nfm.data.datasets.classification_dataset import ClassificationNucleiDataset


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
