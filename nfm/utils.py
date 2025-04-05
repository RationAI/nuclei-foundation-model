# Modified by Matěj Pekár from https://github.com/facebookresearch/dinov2/blob/main/dinov2/utils/utils.py

import numpy as np


class CosineScheduler:
    def __init__(
        self,
        base_value: float,
        final_value: float,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float = 0,
        freeze_iters: int = 0,
    ) -> None:
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros(freeze_iters)

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters))
        )
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it: int) -> float:
        return self.final_value if it >= self.total_iters else self.schedule[it]
