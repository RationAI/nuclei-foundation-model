"""Check production config construction and optimizer checkpoint ordering."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from nfm.data import DataModule
from nfm.ssl_meta_arch import SSLMetaArch


@pytest.mark.parametrize("experiment", ["nucleus_ssl", "LUMI"])
def test_production_data_config_instantiates(experiment):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="nfm", overrides=[f"+experiment={experiment}"])
        data = instantiate(config.data, _recursive_=False, _target_=DataModule)

    assert isinstance(data, DataModule)
    assert data.max_sequence_length == config.data.max_sequence_length


def optimizer_bundle():
    model = SSLMetaArch(
        dim=16,
        hidden_dim=32,
        num_heads=2,
        num_layers=1,
        efd_order=4,
        slices=16,
        projector_dim=8,
        warmup_steps=0,
    )
    model.trainer = SimpleNamespace(estimated_stepping_batches=12)
    bundle = model.configure_optimizers()
    return model, bundle["optimizer"], bundle["lr_scheduler"]["scheduler"]


def group_names(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [
        [names[id(parameter)] for parameter in group["params"]]
        for group in optimizer.param_groups
    ]


def optimization_step(model, optimizer, scheduler):
    # Distinct, weight-dependent gradients exercise Adam moments for every
    # parameter without needing slide data or an attention kernel.
    optimizer.zero_grad(set_to_none=True)
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = parameter.detach().sin() + (index + 1) * 0.01
    optimizer.step()
    scheduler.step()


def test_optimizer_restores_moments_to_the_same_named_parameters(tmp_path):
    model, optimizer, scheduler = optimizer_bundle()
    groups = group_names(model, optimizer)
    flattened = [name for group in groups for name in group]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == {name for name, _ in model.named_parameters()}
    assert optimizer.param_groups[0]["weight_decay"] == model.weight_decay
    assert optimizer.param_groups[1]["weight_decay"] == 0
    assert "model.backbone.layers.0.attn.qkv.weight" in groups[0]
    assert "model.backbone.layers.0.attn.rope.freqs" in groups[1]
    assert "probe.bias" in groups[1]

    for _ in range(3):
        optimization_step(model, optimizer, scheduler)
    checkpoint = tmp_path / "optimizer.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint,
    )

    restored, restored_optimizer, restored_scheduler = optimizer_bundle()
    assert group_names(restored, restored_optimizer) == groups
    payload = torch.load(checkpoint, weights_only=True)
    restored.load_state_dict(payload["model"])
    restored_optimizer.load_state_dict(payload["optimizer"])
    restored_scheduler.load_state_dict(payload["scheduler"])

    for parameter, restored_parameter in zip(
        model.parameters(), restored.parameters(), strict=True
    ):
        torch.testing.assert_close(
            optimizer.state[parameter]["exp_avg"],
            restored_optimizer.state[restored_parameter]["exp_avg"],
        )
        torch.testing.assert_close(
            optimizer.state[parameter]["exp_avg_sq"],
            restored_optimizer.state[restored_parameter]["exp_avg_sq"],
        )

    optimization_step(model, optimizer, scheduler)
    optimization_step(restored, restored_optimizer, restored_scheduler)
    for parameter, restored_parameter in zip(
        model.parameters(), restored.parameters(), strict=True
    ):
        torch.testing.assert_close(parameter, restored_parameter)
    assert scheduler.get_last_lr() == restored_scheduler.get_last_lr()
