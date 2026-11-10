import faulthandler

import hydra
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from rationai.mlkit import Trainer, autolog

from nfm.data import DataModule
from nfm.supervised_meta_arch import SupervisedMetaArch


# Print native stack traces on crashes, also in spawned DataLoader workers.
faulthandler.enable()


@hydra.main(config_path="../configs", config_name="nfm", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger) -> None:
    torch.set_float32_matmul_precision("high")
    seed_everything(config.seed)

    data = hydra.utils.instantiate(
        config.data,
        _recursive_=False,  # the datasets are instantiated in setup
        _target_=DataModule,
    )
    model = hydra.utils.instantiate(config.model, _target_=SupervisedMetaArch)

    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger)
    getattr(trainer, config.mode)(model, datamodule=data, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
