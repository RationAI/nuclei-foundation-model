"""Behavioral checks for fixed targets, feature retention, and masking isolation."""

import copy
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning import LightningDataModule, Trainer
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from nfm.data.data_module import inference_collate_fn, morphology_collate_fn
from nfm.data.datasets.morphology_dataset import (
    MorphologyNucleiDataset,
    morphology_targets,
)
from nfm.data.efd import elliptic_fourier_descriptors
from nfm.data.geometry import radial_polygons
from nfm.modeling import transformer
from nfm.modeling.morphology import FrozenStandardizer
from nfm.morphology_meta_arch import MorphologyMetaArch


torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def cpu_attention(monkeypatch):
    compiled = transformer.flex_attention

    def kernel(*, query, key, value, block_mask):
        if query.device.type == "cuda":
            return compiled(query=query, key=key, value=value, block_mask=block_mask)
        rows, cols = block_mask.BLOCK_SIZE
        length = block_mask.seq_lengths[0]
        owners = block_mask.mask_mod.doc_ids[:length]
        allowed = (
            block_mask.to_dense()[0, 0]
            .bool()
            .repeat_interleave(rows, 0)
            .repeat_interleave(cols, 1)
        )
        allowed = (
            allowed[:length, :length]
            & (owners[:, None] == owners[None, :])
            & (owners[:, None] >= 0)
        )
        return F.scaled_dot_product_attention(query, key, value, attn_mask=allowed)

    monkeypatch.setattr(transformer, "flex_attention", kernel)


def geometry(n=64, seed=0, order=4):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-20, 20, (n, 2)).astype(np.float32)
    label = rng.integers(0, 2, n)
    angle = np.arange(64) * 2 * np.pi / 64
    a = 2 + label * 2 + rng.uniform(0, 0.25, n)
    b = 1.5 + label + rng.uniform(0, 0.15, n)
    radii = 1 / np.sqrt(
        (np.sin(angle)[None] / a[:, None]) ** 2
        + (np.cos(angle)[None] / b[:, None]) ** 2
    )
    polygons = radial_polygons(points, radii)
    efds = (
        elliptic_fourier_descriptors(polygons, order).reshape(n, -1).astype(np.float32)
    )
    return {
        "pos": points,
        "efds": efds,
        "morphology": morphology_targets(polygons, efds),
        "labels": torch.from_numpy(label.astype(np.float32))[:, None],
        "slide_id": str(seed),
    }, polygons


def batches(block_size=16):
    samples = [geometry(32, seed)[0] for seed in [0, 1]]
    return {
        "unlabeled": morphology_collate_fn(samples, block_size, 8, 256),
        "labeled": inference_collate_fn(samples, block_size, 8, 256),
    }


def model(**kwargs):
    settings = {
        "dim": 16,
        "hidden_dim": 32,
        "num_heads": 2,
        "num_layers": 1,
        "efd_order": 4,
        "intrinsic_dim": 16,
        "normalization_batches": 1,
        "warmup_steps": 0,
    }
    settings.update(kwargs)
    return MorphologyMetaArch(**settings)


def test_geometry_targets_preserve_size_and_ignore_rigid_orientation():
    sample, polygons = geometry()
    target = sample["morphology"]
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    changed = polygons @ rotation.T + [5000, -2000]
    changed[..., 0] *= -1
    changed_efds = elliptic_fourier_descriptors(changed, 4).reshape(len(polygons), -1)
    np.testing.assert_allclose(
        morphology_targets(changed, changed_efds), target, atol=2e-5
    )
    scaled = morphology_targets(polygons * 2, sample["efds"] * 2)
    np.testing.assert_allclose(scaled[:, 0], target[:, 0] + 2 * np.log(2), atol=2e-6)
    np.testing.assert_allclose(scaled[:, 1], target[:, 1] + np.log(2), atol=2e-6)
    np.testing.assert_allclose(scaled[:, 2:], target[:, 2:], atol=2e-6)


def test_streaming_normalization_freezes_and_is_sample_independent():
    torch.manual_seed(1)
    data = torch.randn(30, 4) * 3 + 2
    normalizer = FrozenStandardizer(4, 0.001)
    normalizer.observe(data[:7])
    normalizer.observe(data[7:])
    normalizer.freeze()
    torch.testing.assert_close(normalizer.mean.float(), data.mean(0))
    torch.testing.assert_close(normalizer.scale, data.std(0, unbiased=False))
    before = {name: value.clone() for name, value in normalizer.state_dict().items()}
    first = normalizer(data)[0]
    changed = data.clone()
    changed[1:] += 1e5
    torch.testing.assert_close(first, normalizer(changed)[0], atol=0, rtol=0)
    assert all(
        torch.equal(before[name], value)
        for name, value in normalizer.state_dict().items()
    )
    with pytest.raises(RuntimeError, match="frozen"):
        normalizer.observe(data)


def test_single_crop_dataset_and_packing_keep_targets_aligned(monkeypatch):
    sample, polygons = geometry()
    dataset = MorphologyNucleiDataset(
        pd.DataFrame([{"id": "slide"}]), "unused", crop_size=32, efd_order=4
    )
    monkeypatch.setattr(
        dataset, "_candidates", lambda _: (sample["pos"], polygons, np.arange(64), 0)
    )
    item = dataset[0]
    assert len(item["efds"]) == 32
    assert item["morphology"].shape == (32, 8)
    packed = morphology_collate_fn([item, item], 16, 8, 256)
    assert packed["efds"].shape == (64, 16)
    assert packed["region_ids"].tolist() == [0] * 32 + [1] * 32
    # EFD coefficients identify each row independently of packing order.
    for row, target in zip(packed["efds"], packed["morphology"], strict=True):
        nearest = np.argmin(np.linalg.norm(item["efds"] - row.numpy(), axis=1))
        torch.testing.assert_close(
            target, torch.from_numpy(item["morphology"][nearest])
        )


def test_single_crop_reads_radial_lance_store(tmp_path):
    sample, polygons = geometry()
    points = sample["pos"].astype(np.float32)
    radii = np.linalg.norm(polygons - points[:, None], axis=-1).astype(np.float16)
    path = tmp_path / "organ=test/dataset=test/slide_id=slide/nuclei.lance"
    path.parent.mkdir(parents=True)
    table = pa.table(
        {
            "points": pa.FixedSizeListArray.from_arrays(pa.array(points.ravel()), 2),
            "radial_distances": pa.FixedSizeListArray.from_arrays(
                pa.array(radii.ravel()), 64
            ),
        }
    )
    lance.write_dataset(table, str(path))
    slides = pd.DataFrame(
        [
            {
                "id": "slide",
                "organ": "test",
                "dataset": "test",
                "mpp_x": 0.5,
                "mpp_y": 0.25,
            }
        ]
    )
    dataset = MorphologyNucleiDataset(slides, str(tmp_path), crop_size=32, efd_order=4)
    item = dataset[0]
    assert item["efds"].shape == (32, 16)
    assert item["morphology"].shape == (32, 8)
    assert np.isfinite(item["efds"]).all() and np.isfinite(item["morphology"]).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_masked_shapes_have_no_effect_or_gradient_on_context_predictions(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires the remote H100/CUDA runtime")
    torch.manual_seed(2)
    batch = batches(block_size=128)["unlabeled"]
    batch = {
        name: value.to(device) if hasattr(value, "to") else value
        for name, value in batch.items()
    }
    net = model(dim=64, hidden_dim=128).to(device).eval()
    net.calibrate(batch)
    mask = net.sample_mask(batch["region_ids"])
    original = batch["efds"].clone().requires_grad_()
    changed = original.detach().clone()
    changed[mask] += 10000

    def predict(values):
        intrinsic = net.model.shape_encoder(net.model.standardizer(values))
        context = net.model.contextualize(
            intrinsic, batch["pos"], batch["block_mask"], mask
        )
        return net.context_decoder(context)[mask]

    with torch.autocast(device, dtype=dtype, enabled=dtype == torch.bfloat16):
        first = predict(original)
        second = predict(changed)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    first.float().square().sum().backward()
    assert (original.grad[mask] == 0).all()
    assert original.grad[~mask].abs().sum() > 0
    assert torch.isfinite(original.grad).all()


def test_probe_labels_do_not_change_representation_update_or_clipping():
    torch.manual_seed(3)
    data = batches()
    original = model()
    original.calibrate(data["unlabeled"])
    other = copy.deepcopy(original)
    other_labels = {**data["labeled"], "labels": 1 - data["labeled"]["labels"]}
    for net, labeled in [(original, data["labeled"]), (other, other_labels)]:
        net.log = lambda *args, **kwargs: None
        torch.manual_seed(9)
        ssl, _ = net.reconstruction(data["unlabeled"])
        loss = ssl + net.train_probes(labeled)
        loss.backward()
        net.configure_gradient_clipping(None, 0.01, "norm")
    for (name, first), (_, second) in zip(
        original.named_parameters(), other.named_parameters(), strict=True
    ):
        if not name.startswith("probes."):
            torch.testing.assert_close(first.grad, second.grad, atol=0, rtol=0)
    assert not torch.equal(
        original.probes["raw"].weight.grad, other.probes["raw"].weight.grad
    )


def test_intrinsic_denoising_retains_linearly_readable_shape_information():
    torch.manual_seed(4)
    train, _ = geometry(128, 10)
    heldout, _ = geometry(64, 11)
    net = model(context_weight=0)
    batch = morphology_collate_fn([train], 16, 8, 256)
    net.calibrate(batch)
    x = net.model.standardizer(torch.from_numpy(train["efds"]))
    parameters = [
        *net.model.shape_encoder.parameters(),
        *net.intrinsic_decoder.parameters(),
    ]
    optimizer = torch.optim.Adam(parameters, lr=0.005)
    losses = []
    for _ in range(150):
        intrinsic = net.model.shape_encoder(x + torch.randn_like(x) * 0.05)
        loss = F.mse_loss(net.intrinsic_decoder(intrinsic), x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert np.mean(losses[-10:]) < 0.1 * np.mean(losses[:10])
    with torch.no_grad():
        train_features = net.model.shape_encoder(x).numpy()
        heldout_features = net.model.shape_encoder(
            net.model.standardizer(torch.from_numpy(heldout["efds"]))
        ).numpy()
    probe = LogisticRegression(max_iter=1000).fit(
        train_features, train["labels"].numpy().ravel()
    )
    assert probe.score(heldout_features, heldout["labels"].numpy().ravel()) > 0.95


def test_lightning_training_validation_and_portable_checkpoint(tmp_path):
    torch.manual_seed(5)
    batch = batches()
    net = model()
    net.calibrate(batch["unlabeled"])
    before = net.model.shape_encoder[0].weight.detach().clone()
    loader = DataLoader([batch] * 4, batch_size=None)
    val = DataLoader([batch["labeled"]], batch_size=None)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=3,
        max_epochs=-1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        val_check_interval=3,
        check_val_every_n_epoch=None,
        gradient_clip_val=1.0,
        default_root_dir=tmp_path,
    )
    trainer.fit(net, train_dataloaders=loader, val_dataloaders=val)
    assert not torch.equal(before, net.model.shape_encoder[0].weight)
    assert torch.isfinite(trainer.callback_metrics["validation/contextual_AUPRC"])
    checkpoint = tmp_path / "last.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = MorphologyMetaArch.load_from_checkpoint(checkpoint)
    assert restored.model.standardizer.ready and restored.calibration_batches == 1
    restored.eval()
    net.eval()
    with torch.no_grad():
        expected, actual = net(batch["labeled"]), restored(batch["labeled"])
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name])


def test_new_config_uses_original_main_dispatch():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="morphology")
        data = instantiate(config.data, _recursive_=False)
        net = instantiate(config.model)
    assert data.ssl_collate_fn is morphology_collate_fn
    assert isinstance(net, MorphologyMetaArch)
    assert (
        config.trainer.callbacks.model_checkpoint.monitor
        == "validation/contextual_AUPRC"
    )


class CachedMorphologyData(LightningDataModule):
    def setup(self, stage):
        self.batch = batches()

    def train_dataloader(self):
        return DataLoader([self.batch] * 6, batch_size=None)

    def val_dataloader(self):
        return DataLoader([self.batch["labeled"]], batch_size=None)


def test_original_main_calibrates_then_trains_and_checkpoints(tmp_path):
    from nfm.__main__ import main

    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
        version_base=None,
    ):
        config = compose(config_name="morphology")
    config.data = OmegaConf.create({"_target_": f"{__name__}.CachedMorphologyData"})
    config.model = OmegaConf.create(
        {
            "_target_": "nfm.morphology_meta_arch.MorphologyMetaArch",
            "dim": 16,
            "hidden_dim": 32,
            "num_heads": 2,
            "num_layers": 1,
            "efd_order": 4,
            "intrinsic_dim": 16,
            "normalization_batches": 1,
            "warmup_steps": 0,
        }
    )
    config.trainer = OmegaConf.create(
        {
            "accelerator": "cpu",
            "devices": 1,
            "max_steps": 4,
            "max_epochs": -1,
            "precision": "32-true",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
            "val_check_interval": 2,
            "check_val_every_n_epoch": None,
            "gradient_clip_val": 1.0,
            "default_root_dir": str(tmp_path),
            "callbacks": {
                "model_checkpoint": {
                    "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "dirpath": str(tmp_path),
                    "save_last": True,
                    "save_top_k": 1,
                    "every_n_train_steps": 2,
                    "monitor": "validation/contextual_AUPRC",
                    "mode": "max",
                }
            },
        }
    )
    main.__wrapped__.__wrapped__(config, logger=False)
    payload = torch.load(tmp_path / "last.ckpt", weights_only=True)
    assert payload["global_step"] == 4
    assert payload["state_dict"]["calibration_batches"] == 1
    assert payload["state_dict"]["model.standardizer.ready"]
    optimizer_steps = [
        state["step"].item()
        for state in payload["optimizer_states"][0]["state"].values()
    ]
    assert optimizer_steps and set(optimizer_steps) == {3}
