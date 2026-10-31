"""Physical-window completeness, target isolation, controls, and spatial learning."""

import copy
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning import LightningDataModule, Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nfm.data.datasets.regional_dataset import (
    RegionalNucleiDataset,
    block_mask,
    make_sample,
)
from nfm.data.geometry import radial_polygons
from nfm.data.regional import (
    RegionalDataModule,
    regional_collate,
    regional_labeled_collate,
)
from nfm.frozen_benchmark import extract, random_reference
from nfm.regional_meta_arch import (
    RegionalMetaArch,
    multiscale_deviance,
    poisson_deviance,
)


torch.set_num_threads(1)


def sample(seed=0, points=None, mask=None):
    rng = np.random.default_rng(seed)
    if points is None:
        points = rng.uniform(1, 63, (96, 2))
    radii = np.full((len(points), 32), 1.5)
    polygons = radial_polygons(points, radii)
    if mask is None:
        mask = block_mask(rng, 4, 0.4, 2, 0)
    return make_sample(
        points,
        polygons,
        np.zeros(2),
        mask,
        grid_size=4,
        tile_size_um=16,
        fine_bins=2,
        efd_order=2,
        slide_id=str(seed),
    )


def batches():
    samples = [sample(0), sample(1)]
    labeled = [
        {
            "efds": s["efds"],
            "pos": s["pos"],
            "labels": (np.arange(len(s["efds"])) % 2)[:, None],
            "slide_id": s["slide_id"],
        }
        for s in samples
    ]
    return {
        "unlabeled": regional_collate(samples),
        "labeled": regional_labeled_collate(labeled),
    }


def model(**kwargs):
    settings = {
        "dim": 32,
        "hidden_dim": 64,
        "num_heads": 2,
        "num_layers": 1,
        "efd_order": 2,
        "intrinsic_dim": 8,
        "grid_size": 4,
        "tile_size_um": 16,
        "fine_bins": 2,
        "normalization_batches": 1,
        "warmup_steps": 0,
    }
    settings.update(kwargs)
    return RegionalMetaArch(**settings)


def test_hidden_population_only_changes_targets_not_encoder_or_calibration():
    original = sample(4)
    mask = original["mask"]
    hidden_tile = int(np.flatnonzero(mask)[0])
    added = np.array([[hidden_tile % 4 * 16 + 5, hidden_tile // 4 * 16 + 7]])
    # Retain all original visible cells exactly, add a previously unseen hidden cell.
    changed = sample(4, points=np.concatenate([original["pos"], added]), mask=mask)
    for key in ["pos", "efds", "mask"]:
        np.testing.assert_array_equal(original[key], changed[key])
    assert changed["targets"]["counts"].sum() != original["targets"]["counts"].sum()
    first, second = regional_collate([original]), regional_collate([changed])
    torch.manual_seed(3)
    net = model().eval()
    other = copy.deepcopy(net)
    net.calibrate(first)
    other.calibrate(second)
    for key, value in net.state_dict().items():
        torch.testing.assert_close(value, other.state_dict()[key], atol=0, rtol=0)
    with torch.no_grad():
        a, b = (
            net.model.encode_windows(first["inputs"]),
            net.model.encode_windows(second["inputs"]),
        )
    torch.testing.assert_close(a["regions"], b["regions"], atol=0, rtol=0)


def test_hidden_cells_cannot_be_smuggled_into_encoder_inputs():
    batch = regional_collate([sample()])
    net = model()
    net.calibrate(batch)
    inputs = batch["inputs"]
    hidden = int(inputs["mask"][0].nonzero()[0])
    bad = {**inputs, "pos": inputs["pos"].clone()}
    bad["pos"][0] = torch.tensor([hidden % 4 * 16 + 1, hidden // 4 * 16 + 1])
    with pytest.raises(ValueError, match="Hidden cells"):
        net.model.encode_windows(bad)


def test_multiscale_targets_are_count_conserving_and_zero_safe():
    rate = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0], [2.0, 1.0]]]])
    mask = torch.tensor([[True]])
    expected = (
        poisson_deviance(rate, target).mean()
        + poisson_deviance(rate.sum(), target.sum())
    ) / 2
    torch.testing.assert_close(multiscale_deviance(rate, target, mask), expected)
    perfect = multiscale_deviance(target + 1e-6, target, mask)
    assert perfect < 1e-5
    empty = multiscale_deviance(rate, torch.zeros_like(target), mask)
    empty.backward()
    assert torch.isfinite(rate.grad).all() and (rate.grad > 0).all()


def test_certified_lance_window_matches_full_census_and_is_repeatable(tmp_path):
    axis = np.linspace(10000, 10200, 21)
    xx, yy = np.meshgrid(axis, axis)
    pixels = np.stack([xx.ravel(), yy.ravel()], -1).astype(np.float32)
    radii = np.repeat((1 + np.arange(len(pixels)) % 4)[:, None], 32, axis=1).astype(
        np.float16
    )
    path = tmp_path / "organ=test/dataset=test/slide_id=one/nuclei.lance"
    path.parent.mkdir(parents=True)
    lance.write_dataset(
        pa.table(
            {
                "points": pa.FixedSizeListArray.from_arrays(
                    pa.array(pixels.ravel()), 2
                ),
                "radial_distances": pa.FixedSizeListArray.from_arrays(
                    pa.array(radii.ravel()), 32
                ),
            }
        ),
        str(path),
    )
    slides = pd.DataFrame(
        [
            {
                "id": "one",
                "organ": "test",
                "dataset": "test",
                "mpp_x": 0.5,
                "mpp_y": 0.25,
                "nuclei_count": len(pixels),
            }
        ]
    )
    options = {
        "grid_size": 4,
        "tile_size_um": 16,
        "fine_bins": 2,
        "efd_order": 2,
        "initial_query": 2,
        "max_query": len(pixels),
        "deterministic": True,
    }
    dataset = RegionalNucleiDataset(slides, str(tmp_path), **options)
    result = dataset[0]
    physical = pixels.astype(np.float64) * [0.5, 0.25]
    origin = result["window_origin_um"]
    inside = np.all((physical >= origin) & (physical < origin + 64), axis=1)
    assert result["targets"]["counts"].sum() == inside.sum()
    # Persistent row-ID contour fetching must preserve point/shape alignment.
    from nfm.data.efd import elliptic_fourier_descriptors

    visible = result["pos"] + origin
    indices = np.linalg.norm(visible[:, None] - physical[None], axis=-1).argmin(1)
    expected = elliptic_fourier_descriptors(
        radial_polygons(pixels[indices], radii[indices]) * [0.5, 0.25], 2
    ).reshape(len(visible), -1)
    np.testing.assert_allclose(result["efds"], expected, rtol=1e-5, atol=1e-6)
    again = dataset[0]
    for key in ["pos", "efds", "mask"]:
        np.testing.assert_array_equal(result[key], again[key])
    limited = RegionalNucleiDataset(
        slides, str(tmp_path), **{**options, "max_query": 2}
    )
    with pytest.raises(ValueError, match="Uncertified"):
        limited[0]


def test_slide_features_cover_cells_without_cross_slide_or_chunk_leakage():
    net = model(inference_window_batch=1).eval()
    net.calibrate(regional_collate([sample()]))
    values = batches()["labeled"]
    values["pos"] = values["pos"] + torch.arange(len(values["pos"]))[:, None] % 3 * 64
    other = {**values, "efds": values["efds"].clone()}
    other["efds"][values["document_ids"] == 1] += 10
    with torch.no_grad():
        a = net(values)
        b = net(other)
        net.model.inference_window_batch = 32
        c = net(values)
        regions = net.model.encode_slide(values, return_regions=True)
    keep = values["document_ids"] == 0
    for name in a:
        torch.testing.assert_close(a[name][keep], b[name][keep], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(a[name], c[name], atol=1e-5, rtol=1e-5)
    assert a["contextual"].shape == (len(values["efds"]), 40)
    assert regions["regions"]["counts"].sum() == len(values["efds"])
    assert a["local_summary"].shape[-1] == 29


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_spatial_backward_and_labels_are_isolated(dtype):
    data = batches()
    net = model()
    net.calibrate(data["unlabeled"])
    other = copy.deepcopy(net)
    reference_before = {
        key: value.clone() for key, value in net.reference.state_dict().items()
    }
    flipped = {**data["labeled"], "labels": 1 - data["labeled"]["labels"]}
    for current, labeled in [(net, data["labeled"]), (other, flipped)]:
        current.log = lambda *args, **kwargs: None
        torch.manual_seed(2)
        with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
            loss, _metrics = current.spatial_loss(data["unlabeled"])
            loss = loss + current.train_probes(labeled)
        loss.backward()
        current.configure_gradient_clipping(None, 1.0, "norm")
        assert torch.isfinite(loss)
        assert all(
            torch.isfinite(p.grad).all()
            for p in current.parameters()
            if p.grad is not None
        )
    for (name, p), (_, q) in zip(
        net.named_parameters(), other.named_parameters(), strict=True
    ):
        if not name.startswith("probes.") and p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    assert all(
        torch.equal(value, reference_before[key])
        for key, value in net.reference.state_dict().items()
    )
    reference = random_reference(net, 0)
    for p, q in zip(reference.parameters(), net.reference.parameters(), strict=True):
        torch.testing.assert_close(p, q)


def test_frozen_benchmark_handles_regional_controls_without_a_cell_graph():
    net = model().eval()
    net.calibrate(batches()["unlabeled"])
    original = sample()
    labeled = {
        **original,
        "labels": (np.arange(len(original["efds"])) % 2)[:, None],
    }
    config = OmegaConf.create(
        {"data": {"block_size": 16, "k": 4, "max_sequence_length": 512}}
    )
    features, labels, online = extract(
        labeled, net, random_reference(net, 0), config, "fp32"
    )
    assert set(features) == set(online) == set(net.probes)
    np.testing.assert_array_equal(labels, labeled["labels"].ravel())
    for name in ["intrinsic", "contextual"]:
        # Before any update, the saved random reference is exactly matched.
        np.testing.assert_allclose(features[name], features[f"random_{name}"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_native_attention_backward(dtype):
    def cuda_dictionary(values):
        return {
            key: cuda_dictionary(value)
            if isinstance(value, dict)
            else value.cuda()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in values.items()
        }

    data = cuda_dictionary(batches())
    net = model().cuda()
    net.calibrate(data["unlabeled"])
    net.log = lambda *args, **kwargs: None
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad])
    # The zero-initialized decoders first learn a readout, then update the backbone.
    for _ in range(2):
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
            loss, _ = net.spatial_loss(data["unlabeled"])
            loss = loss + net.train_probes(data["labeled"])
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None
        )
        optimizer.step()
    assert net.model.layers[0].qkv.weight.grad.abs().sum() > 0


def patterned_sample(seed, mode):
    """Four equiprobable layouts; the mode is inferable only from visible cells."""
    rng = np.random.default_rng(seed)
    points = []
    for y in range(4):
        for x in range(4):
            coordinate = x if mode < 2 else y
            dense = (coordinate < 2) == (mode % 2 == 0)
            for fy in range(2):
                for fx in range(2):
                    n = 6 if dense else 1
                    origin = np.array([x * 16 + fx * 8, y * 16 + fy * 8])
                    points.extend(origin + rng.uniform(1, 7, (n, 2)))
    points = np.asarray(points)
    return sample(
        points=points,
        mask=block_mask(rng, 4, 0.4, 2, 0),
    )


def test_objective_learns_spatial_context_on_unseen_windows_and_masks():
    torch.manual_seed(0)
    training = [
        regional_collate([patterned_sample(i * 16 + j, j % 4) for j in range(8)])
        for i in range(12)
    ]
    validation = regional_collate(
        [patterned_sample(10000 + j, j % 4) for j in range(16)]
    )
    contrary = regional_collate(
        [patterned_sample(10000 + j, (j % 4) ^ 1) for j in range(16)]
    )
    assert torch.equal(validation["inputs"]["mask"], contrary["inputs"]["mask"])
    net = model(num_layers=2, noise_std=0, intrinsic_weight=0, morphology_weight=0)
    net.calibrate(training[0])
    optimizer = torch.optim.AdamW(net.model.parameters(), lr=0.003)
    optimizer.add_param_group({"params": net.density_decoder.parameters()})
    optimizer.add_param_group({"params": net.occupancy_decoder.parameters()})
    with torch.no_grad():
        _, before = net.spatial_loss(validation, add_noise=False)
    for step in range(100):
        optimizer.zero_grad()
        loss, _ = net.spatial_loss(training[step % len(training)])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
    with torch.no_grad():
        _, after = net.spatial_loss(validation, add_noise=False)
        _, wrong_context = net.spatial_loss(
            {"inputs": contrary["inputs"], "targets": validation["targets"]},
            add_noise=False,
        )
    assert after["geometry_deviance"] < before["geometry_deviance"] * 0.25
    assert after["spatial_skill"] > 0.75
    assert wrong_context["geometry_deviance"] > after["geometry_deviance"] + 1


def test_real_data_module_holds_out_ssl_slides_and_validates_full_slides(tmp_path):
    ssl, labeled, annotations = [], [], []
    x, y = np.meshgrid(np.linspace(0, 63, 32), np.linspace(0, 63, 16))
    points = np.stack([x.ravel(), y.ravel()], -1).astype(np.float32)
    radii = np.full((len(points), 32), 1.5, dtype=np.float16)
    polygons = radial_polygons(points, radii)
    for i in range(6):
        slide_id = f"ssl{i}"
        path = tmp_path / f"organ=test/dataset=test/slide_id={slide_id}/nuclei.lance"
        path.parent.mkdir(parents=True)
        lance.write_dataset(
            pa.table(
                {
                    "points": pa.FixedSizeListArray.from_arrays(
                        pa.array(points.ravel()), 2
                    ),
                    "radial_distances": pa.FixedSizeListArray.from_arrays(
                        pa.array(radii.ravel()), 32
                    ),
                }
            ),
            str(path),
        )
        ssl.append(
            {
                "id": slide_id,
                "organ": "test",
                "dataset": "test",
                "mpp_x": 1.0,
                "mpp_y": 1.0,
                "nuclei_count": len(points),
            }
        )
    for i in range(12):
        slide_id = f"labeled{i}"
        path = tmp_path / f"{slide_id}.parquet"
        pd.DataFrame(
            {
                "id": np.arange(len(points)),
                "centroid": list(points),
                "polygon": list(polygons.reshape(len(points), -1)),
            }
        ).to_parquet(path)
        labeled.append(
            {
                "slide_id": slide_id,
                "slide_nuclei_path": str(path),
                "nuclei_count": len(points),
                "mpp_x": 1.0,
                "mpp_y": 1.0,
                "is_carcinoma": True,
                "gleason_score": i % 2,
            }
        )
        annotations.extend(
            {"slide_id": slide_id, "id": j, "annot_label": j % 2}
            for j in range(len(points))
        )
    pd.DataFrame(ssl).to_parquet(tmp_path / "ssl.parquet")
    pd.DataFrame(labeled).to_parquet(tmp_path / "labeled.parquet")
    pd.DataFrame(annotations).to_parquet(tmp_path / "annotations.parquet")
    config = OmegaConf.create(
        {
            "unlabeled": {
                "slides_path": str(tmp_path / "ssl.parquet"),
                "batch_size": 2,
                "num_workers": 0,
                "eval_num_workers": 0,
                "dataset": {
                    "_target_": "nfm.data.datasets.regional_dataset.RegionalNucleiDataset",
                    "nuclei_path": str(tmp_path),
                    "grid_size": 4,
                    "tile_size_um": 16,
                    "fine_bins": 2,
                    "efd_order": 2,
                },
            },
            "labeled": {
                "slides_path": str(tmp_path / "labeled.parquet"),
                "split_size": 0.25,
                "split_stratify_col": "gleason_score",
                "batch_size": 2,
                "num_workers": 0,
                "eval_num_workers": 0,
                "sampler": {"_target_": "nfm.data.samplers.AutoWeightedRandomSampler"},
                "dataset": {
                    "_target_": "nfm.data.datasets.ClassificationNucleiDataset",
                    "annotation_path": str(tmp_path / "annotations.parquet"),
                    "crop_size": 128,
                    "efd_order": 2,
                },
            },
        }
    )

    def data_module():
        return RegionalDataModule(
            config, 16, 4, 2048, ssl_holdout_fraction=0.5, ssl_validation_slides=2
        )

    data = data_module()
    data.setup("fit")
    assert len(data.train) == 3 and len(data.ssl_validation) == 2
    trained = set(data.train.slides.id)
    heldout = set(data.ssl_validation.slides.id)
    assert not trained & heldout
    # All held-out slides, including those beyond the evaluation subset, are excluded.
    assert len(trained | heldout) == 5
    np.random.seed(999)
    other = data_module()
    other.setup("fit")
    assert set(data.validation.slides.slide_id) == set(other.validation.slides.slide_id)
    assert set(data.ssl_validation.slides.id) == set(other.ssl_validation.slides.id)
    first, second = data.val_dataloader(), other.val_dataloader()
    a, b = next(iter(first[1])), next(iter(second[1]))
    for key in a["inputs"]:
        torch.testing.assert_close(a["inputs"][key], b["inputs"][key], atol=0, rtol=0)
    assert len(next(iter(first[0]))["efds"]) == len(points)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=3,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        val_check_interval=3,
        check_val_every_n_epoch=None,
        limit_val_batches=1,
    )
    trainer.fit(model(), datamodule=data)
    assert "ssl_validation/geometry_deviance" in trainer.callback_metrics
    assert "validation/random_contextual_AUPRC" in trainer.callback_metrics


class CachedRegionalData(LightningDataModule):
    def setup(self, stage):
        self.batch = batches()

    def train_dataloader(self):
        return DataLoader([self.batch] * 6, batch_size=None)

    def val_dataloader(self):
        return [
            DataLoader([self.batch["labeled"]], batch_size=None),
            DataLoader([self.batch["unlabeled"]], batch_size=None),
        ]


def test_original_entrypoint_trains_validates_and_resumes_portably(tmp_path):
    from nfm.__main__ import main

    with initialize_config_dir(
        config_dir=str(Path("configs").resolve()), version_base=None
    ):
        config = compose(config_name="regional")
    assert config.data.datasets.labeled.batch_size == 32
    assert config.model.num_layers == 8
    assert (
        instantiate(config.data, _recursive_=False).ssl_collate_fn is regional_collate
    )
    config.data = OmegaConf.create({"_target_": f"{__name__}.CachedRegionalData"})
    settings = dict(model().hparams)
    config.model = OmegaConf.create(
        {"_target_": "nfm.regional_meta_arch.RegionalMetaArch", **settings}
    )
    config.trainer = OmegaConf.create(
        {
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32-true",
            "max_steps": 4,
            "max_epochs": -1,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
            "val_check_interval": 2,
            "check_val_every_n_epoch": None,
            "default_root_dir": str(tmp_path),
            "gradient_clip_val": 1.0,
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
    checkpoint = tmp_path / "last.ckpt"
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["global_step"] == 4
    restored = RegionalMetaArch.load_from_checkpoint(checkpoint)
    assert restored.reference.standardizer.ready
    assert not any(p.requires_grad for p in restored.reference.parameters())
    resumed = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=5,
        max_epochs=-1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        val_check_interval=2,
        check_val_every_n_epoch=None,
    )
    resumed.fit(restored, datamodule=CachedRegionalData(), ckpt_path=checkpoint)
    assert resumed.global_step == 5
