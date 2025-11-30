from collections.abc import Iterable

from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader


class DataModule(LightningDataModule):
    def __init__(
        self, batch_size: int, num_workers: int = 0, **datasets: DictConfig
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = datasets

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self.train = instantiate(self.datasets["train"])
            case "test":
                self.test = instantiate(self.datasets["test"])

    def train_dataloader(self) -> Iterable[dict[str, Tensor]]:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            in_order=False,
            prefetch_factor=4,
        )

    def test_dataloader(self) -> Iterable[dict[str, Tensor]]:
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )
