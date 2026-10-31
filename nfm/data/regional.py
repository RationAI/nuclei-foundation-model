"""Visible-only region batches and a fixed, slide-held-out SSL validation stream."""

from functools import partial

import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from nfm.data.data_module import DataModule


def regional_collate(batch, block_size=128, k=16, max_sequence_length=131072):
    lengths = torch.tensor([len(sample["efds"]) for sample in batch])
    if lengths.sum() > max_sequence_length:
        raise ValueError(
            "Visible cells exceed max_sequence_length; increase the budget"
        )
    return {
        "inputs": {
            "efds": torch.from_numpy(np.concatenate([s["efds"] for s in batch])),
            "pos": torch.from_numpy(np.concatenate([s["pos"] for s in batch])),
            "window_ids": torch.repeat_interleave(torch.arange(len(batch)), lengths),
            "mask": torch.from_numpy(np.stack([s["mask"] for s in batch])),
        },
        "targets": {
            name: torch.from_numpy(np.stack([s["targets"][name] for s in batch]))
            for name in ["counts", "morphology"]
        },
        "slide_ids": [s["slide_id"] for s in batch],
    }


def regional_labeled_collate(batch, block_size=128, k=16, max_sequence_length=131072):
    lengths = torch.tensor([len(s["efds"]) for s in batch])
    if lengths.sum() > max_sequence_length:
        raise ValueError(
            "Labeled cells exceed max_sequence_length; increase the budget"
        )
    labels = torch.cat(
        [torch.as_tensor(s["labels"]).reshape(-1, 1) for s in batch]
    ).float()
    return {
        "efds": torch.cat([torch.as_tensor(s["efds"]) for s in batch]).float(),
        "pos": torch.from_numpy(np.concatenate([s["pos"] for s in batch])).float(),
        "document_ids": torch.repeat_interleave(torch.arange(len(batch)), lengths),
        "labels": labels.nan_to_num(),
        "score": torch.isfinite(labels[:, 0]),
        "slide_ids": [s["slide_id"] for s in batch],
    }


class RegionalDataModule(DataModule):
    ssl_collate_fn = staticmethod(regional_collate)
    labeled_collate_fn = staticmethod(regional_labeled_collate)

    def __init__(
        self,
        datasets,
        block_size,
        k,
        max_sequence_length,
        ssl_holdout_fraction=0.02,
        ssl_validation_slides=64,
        split_seed=0,
    ):
        super().__init__(datasets, block_size, k, max_sequence_length, split_seed)
        if not 0 < ssl_holdout_fraction < 1 or ssl_validation_slides < 1:
            raise ValueError("A nonempty slide-held-out SSL validation set is required")
        self.ssl_holdout_fraction = ssl_holdout_fraction
        self.ssl_validation_slides = ssl_validation_slides
        self.split_seed = split_seed

    def setup(self, stage):
        super().setup(stage)
        slides = self.train.slides
        slides = slides[slides.nuclei_count > 0].reset_index(drop=True)
        if len(slides) < 2 or slides.id.duplicated().any():
            raise ValueError(
                "SSL metadata requires at least two uniquely identified slides"
            )
        rng = np.random.default_rng(self.split_seed)
        order = rng.permutation(len(slides))
        count = min(
            len(slides) - 1, max(1, round(len(slides) * self.ssl_holdout_fraction))
        )
        heldout = slides.iloc[order[:count]]
        self.train = instantiate(
            self.datasets.unlabeled.dataset, slides=slides.iloc[order[count:]]
        )
        self.ssl_validation = instantiate(
            self.datasets.unlabeled.dataset,
            slides=heldout.iloc[: self.ssl_validation_slides],
            deterministic=True,
            seed=self.split_seed,
        )

    def val_dataloader(self):
        workers = self.datasets.unlabeled.get("eval_num_workers", 2)
        ssl = DataLoader(
            self.ssl_validation,
            batch_size=self.datasets.unlabeled.batch_size,
            num_workers=workers,
            persistent_workers=workers > 0,
            pin_memory=True,
            multiprocessing_context="spawn" if workers else None,
            collate_fn=partial(
                self.ssl_collate_fn,
                block_size=self.block_size,
                k=self.k,
                max_sequence_length=self.max_sequence_length,
            ),
        )
        return [super().val_dataloader(), ssl]
