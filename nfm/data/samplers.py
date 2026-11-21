import pandas as pd
from torch.utils.data import WeightedRandomSampler


class GroupBalancedRandomSampler(WeightedRandomSampler):
    """Half slide-uniform, half sqrt(group size), with replacement.

    A group is a combination of the given columns, such as organ and source, so
    an organ held by a single large source is not shrunk along with it, and the
    sources of an organ are still spread.
    """

    def __init__(self, slides: pd.DataFrame, group_fraction: float = 0.5) -> None:
        sizes = slides.groupby(list(slides.columns), observed=True).size()
        probabilities = sizes.pow(0.5) / sizes.pow(0.5).sum()
        per_slide = (probabilities / sizes).rename("weight")
        group = slides.join(per_slide, on=list(slides.columns))["weight"]
        weights = (1 - group_fraction) / len(slides) + group_fraction * group
        super().__init__(weights.to_numpy(copy=True), len(slides), replacement=True)
