from lightning import LightningModule
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection

from nfm.modeling import Criterion
from nfm.typing import Input, Outputs


class DinoModel(LightningModule):
    def __init__(self, backbone: nn.Module, decode_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.criterion = Criterion()

        self.val_metrics = MetricCollection(
            {...},  # TODO add metrics you want to compute
            prefix="validation/",
        )

        self.test_mterics = self.val_metrics.clone(prefix="test/")

    def forward(self, x: Input) -> Outputs:
        features = self.backbone(x)
        return self.decode_head(features)

    def training_step(self, batch: Input) -> Tensor:
        inputs, targets = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Input) -> None:
        inputs, targets = batch
        outputs = self(inputs)

        loss = self.criterion(outputs, targets)
        self.log("validation/loss", loss, on_epoch=True, prog_bar=True)

        self.val_metrics.update(outputs, targets)
        self.log_dict(self.val_metrics, on_epoch=True)

    def test_step(self, batch: Input) -> None:
        inputs, targets = batch
        outputs = self(inputs)
        self.test_metrics.update(outputs, targets)
        self.log_dict(self.test_metrics, on_epoch=True)

    def configure_optimizers(self) -> Optimizer:
        # TODO add your optimizer
        ...
