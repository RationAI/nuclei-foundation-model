import hydra
import torch
from lightning import seed_everything
from omegaconf import DictConfig
from rationai.mlkit import Trainer

from nfm.data import DataModule
from nfm.ssl_meta_arch import SSLMetaArch


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    torch.set_float32_matmul_precision("medium")
    seed_everything(config.seed, workers=True)

    data = hydra.utils.instantiate(
        config.data,
        _recursive_=False,  # to avoid instantiating all the datasets
        _target_=DataModule,
    )
    model = hydra.utils.instantiate(config.model, _target_=SSLMetaArch)

    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer)
    getattr(trainer, config.mode)(model, datamodule=data, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
