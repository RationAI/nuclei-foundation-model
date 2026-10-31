"""Behavioral checks for correspondence, isolation, loss semantics, and the assay."""

import copy
from unittest.mock import patch

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
from lightning import Trainer, seed_everything
from omegaconf import OmegaConf

from nfm.configuration import Config
from nfm.data import DataModule
from nfm.data.augmentations import ClusterDrop, Compose, FieldRotation, RandomDrop
from nfm.data.data_module import _pack_crops, inference_collate_fn, train_collate_fn
from nfm.data.datasets import NucleiDataset
from nfm.data.efd import elliptic_fourier_descriptors, normalize_efd_for_starting_point
from nfm.data.geometry import radial_polygons
from nfm.modeling import transformer
from nfm.modeling.objectives import NucleusObjective
from nfm.modeling.transformer import NFM
from nfm.ssl_meta_arch import SSLMetaArch


torch.set_num_threads(1)


def allowed_tokens(mask):
    """Expand a small test mask; never used by the production pipeline."""
    rows, cols = mask.BLOCK_SIZE
    allowed = (
        mask.to_dense()[0, 0]
        .bool()
        .repeat_interleave(rows, 0)
        .repeat_interleave(cols, 1)
    )
    length = mask.seq_lengths[0]
    owners = mask.mask_mod.doc_ids[:length]
    allowed = allowed[:length, :length]
    return allowed & (owners[:, None] >= 0) & (owners[:, None] == owners[None, :])


def attention_oracle(*, query, key, value, block_mask):
    """Explicit attention arithmetic for tests, independent of fused kernels."""
    with torch.autocast(query.device.type, enabled=False):
        allowed = allowed_tokens(block_mask)
        q, k, v = query.double(), key.double(), value.double()
        scores = (q @ k.transpose(-2, -1)) / query.shape[-1] ** 0.5
        scores = scores.masked_fill(~allowed, float("-inf"))
        scores = torch.where(allowed.any(-1, keepdim=True), scores, 0)
        probabilities = scores.softmax(-1) * allowed
        return (probabilities @ v).to(query.dtype)


@pytest.fixture(autouse=True)
def cpu_attention_kernel(monkeypatch):
    """Exercise model logic on CPU; real FlexAttention is checked on CUDA."""
    compiled = transformer.flex_attention

    def kernel(**kwargs):
        if kwargs["query"].device.type == "cpu":
            return attention_oracle(**kwargs)
        return compiled(**kwargs)

    monkeypatch.setattr(transformer, "flex_attention", kernel)


def contours(n=64, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-60, 60, (n, 2)).astype(np.float32)
    label = rng.integers(2, size=n)
    theta = np.arange(64) * 2 * np.pi / 64
    a = 2.0 + label * 2.5 + rng.uniform(0, 0.3, n)
    b = 1.5 + label * 1.5 + rng.uniform(0, 0.3, n)
    radii = 1 / np.sqrt(
        (np.sin(theta)[None] / a[:, None]) ** 2
        + (np.cos(theta)[None] / b[:, None]) ** 2
    )
    return points, radii.astype(np.float32), label


@pytest.fixture(scope="module")
def source_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("nucleus-data")
    cells = root / "cells"
    ssl, probe, annotations = [], [], []
    for i in range(6):
        points, radii, _ = contours(512, i)
        slide_id = f"ssl{i}"
        table = pa.table(
            {
                "points": pa.FixedSizeListArray.from_arrays(
                    pa.array(points.ravel()), 2
                ),
                "radial_distances": pa.FixedSizeListArray.from_arrays(
                    pa.array(radii.astype(np.float16).ravel()), 64
                ),
            }
        )
        path = cells / f"organ=synthetic/dataset=synthetic/slide_id={slide_id}.lance"
        path.parent.mkdir(parents=True, exist_ok=True)
        lance.write_dataset(table, str(path))
        ssl.append(
            {
                "id": slide_id,
                "organ": "synthetic",
                "dataset": "synthetic",
                "mpp_x": 0.5,
                "mpp_y": 0.25,
                "nuclei_count": len(points),
                "slide_path": f"/{slide_id}.tif",
                "level": 0,
            }
        )
    for i in range(12):
        points, radii, labels = contours(769, 100 + i)
        polygons = radial_polygons(points, radii)
        slide_id = f"probe{i}"
        path = root / f"{slide_id}.parquet"
        pd.DataFrame(
            {
                "id": np.arange(len(points)),
                "centroid": list(points),
                "polygon": list(polygons.reshape(len(points), -1)),
            }
        ).to_parquet(path)
        probe.append(
            {
                "slide_id": slide_id,
                "slide_nuclei_path": str(path),
                "is_carcinoma": True,
                "mpp_x": 0.5,
                "mpp_y": 0.25,
                "patient_id": f"patient{i // 2}",
                "nuclei_count": len(points),
                "gleason_score": i % 2,
            }
        )
        annotations.extend(
            {"slide_id": slide_id, "id": j, "annot_label": int(label)}
            for j, label in enumerate(labels)
        )
    ssl_path, probe_path, annotation_path = (
        root / "ssl.parquet",
        root / "probe.parquet",
        root / "annotations.parquet",
    )
    pd.DataFrame(ssl).to_parquet(ssl_path)
    pd.DataFrame(probe).to_parquet(probe_path)
    pd.DataFrame(annotations).to_parquet(annotation_path)
    cfg = OmegaConf.create(
        {
            "block_size": 16,
            "k": 8,
            "max_sequence_length": 1024,
            "datasets": {
                "unlabeled": {
                    "slides_path": str(ssl_path),
                    "batch_size": 2,
                    "num_workers": 0,
                    "sampler": {
                        "_target_": "nfm.data.samplers.SourceBalancedRandomSampler"
                    },
                    "dataset": {
                        "_target_": "nfm.data.datasets.NucleiDataset",
                        "nuclei_path": str(cells),
                        "crop_size": 64,
                        "efd_order": 4,
                        "alpha": 0.85,
                    },
                },
                "labeled": {
                    "slides_path": str(probe_path),
                    "split_size": 0.25,
                    "split_stratify_col": "gleason_score",
                    "batch_size": 2,
                    "num_workers": 0,
                    "eval_num_workers": 0,
                    "sampler": {
                        "_target_": "nfm.data.samplers.AutoWeightedRandomSampler"
                    },
                    "dataset": {
                        "_target_": "nfm.data.datasets.ClassificationNucleiDataset",
                        "annotation_path": str(annotation_path),
                        "crop_size": 64,
                        "efd_order": 4,
                        "alpha": 0.8,
                    },
                },
            },
        }
    )
    return root, cfg


@pytest.mark.parametrize("kind", ["sigreg", "visreg", "vicreg"])
def test_loss_rewards_actual_correspondence(kind):
    base = torch.randn(4, 16, 8, generator=torch.Generator().manual_seed(4))
    aligned = torch.stack([base, base])
    shuffled = torch.stack([base, base.flip(1)])
    torch.manual_seed(11)
    good, parts = NucleusObjective(kind, slices=32)(aligned)
    torch.manual_seed(11)
    bad, _ = NucleusObjective(kind, slices=32)(shuffled)
    assert parts["alignment"] == 0
    assert good < bad


@pytest.mark.parametrize("kind", ["sigreg", "visreg", "vicreg"])
@pytest.mark.parametrize("scale", [0.0, 1e-7, 1.0])
def test_loss_has_finite_backward_near_collapse(kind, scale):
    z = (torch.randn(2, 3, 8, 12) * scale).requires_grad_()
    loss, _ = NucleusObjective(kind, slices=16)(z)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(z.grad).all()


def test_efd_only_inputs_match_original_and_preserve_size():
    points, radii, _ = contours()
    polygons = radial_polygons(points, radii)
    dataset = NucleiDataset(pd.DataFrame(), nuclei_path="unused", efd_order=16)
    _, original = dataset.polygon_to_efd(polygons)
    _, translated = dataset.polygon_to_efd(polygons + np.array([1000, -500]))
    _, scaled = dataset.polygon_to_efd(polygons * 2)
    assert original.shape == (len(polygons), 64)
    np.testing.assert_allclose(
        original, elliptic_fourier_descriptors(polygons, 16).reshape(-1, 64), atol=1e-6
    )
    np.testing.assert_allclose(original, translated, atol=1e-5)
    np.testing.assert_allclose(scaled, 2 * original, atol=1e-5)


def test_efd_handles_partial_degeneracy_and_repeated_vertices():
    coefficients = np.ones((3, 4, 4))
    coefficients[1, 0] = 0
    coefficients[0, 0, 0] = -2
    assert np.isfinite(normalize_efd_for_starting_point(coefficients)).all()
    points, radii, _ = contours(4)
    polygons = radial_polygons(points, radii)
    repeated = np.concatenate([polygons[:, :1], polygons], axis=1)
    np.testing.assert_allclose(
        elliptic_fourier_descriptors(polygons, 4),
        elliptic_fourier_descriptors(repeated, 4),
        atol=1e-8,
    )


def pack_test_crops(views, *, block_size, k):
    return _pack_crops(
        [(v["pos"], v["efds"]) for v in views],
        block_size,
        k,
        max_sequence_length=1024,
    )[0]


def paired_sample(views, ids, slide_id):
    return {
        "pos": [v["pos"] for v in views],
        "efds": [v["efds"] for v in views],
        "indices": [v["ids"] for v in views],
        "matched_ids": ids,
        "slide_id": slide_id,
    }


def tagged_view(seed, tag, n=25):
    points, _, _ = contours(n, seed)
    return {
        "pos": points,
        "efds": np.stack(
            [np.arange(n) + tag, np.ones(n), np.zeros(n), np.zeros(n)], -1
        ).astype(np.float32),
        "ids": np.arange(n),
    }


def test_correspondence_survives_sorting_and_reused_ids_on_other_slides():
    batch = [
        paired_sample(
            [tagged_view(b, tag), tagged_view(b + 20, tag)],
            np.array([0, 7, 19]),
            str(b),
        )
        for b, tag in enumerate([100, 1000])
    ]
    packed = train_collate_fn(batch, block_size=16, k=8)
    selected = packed["efds"][packed["pair_indices"]][..., 0]
    torch.testing.assert_close(selected[0], selected[1])
    torch.testing.assert_close(selected[0, :3], torch.tensor([100, 107, 119.0]))
    torch.testing.assert_close(selected[0, 3:], torch.tensor([1000, 1007, 1019.0]))


def test_blocks_are_disjoint_and_other_crops_cannot_be_keys():
    packed = pack_test_crops(
        [tagged_view(0, 0), tagged_view(1, 100)],
        block_size=16,
        k=8,
    )
    mask = packed["block_mask"]
    for row, count in enumerate(mask.kv_num_blocks[0, 0]):
        partial = set(mask.kv_indices[0, 0, row, : int(count)].tolist())
        full = set(
            mask.full_kv_indices[
                0, 0, row, : int(mask.full_kv_num_blocks[0, 0, row])
            ].tolist()
        )
        assert not partial & full
    reference = allowed_tokens(mask)
    assert reference.shape == (50, 50)
    assert not reference[:25, 25:50].any()
    assert not reference[25:50, :25].any()


def test_encoder_has_no_cross_crop_leakage_at_a_shared_block_boundary():
    torch.manual_seed(0)
    encoder = NFM(
        Config(dim=16, hidden_dim=32, num_layers=2, num_heads=2, efd_order=1)
    ).eval()
    batch = pack_test_crops(
        [tagged_view(0, 0), tagged_view(1, 100)], block_size=16, k=8
    )
    other = batch["efds"].clone()
    other[25:50] += 1e5
    with torch.no_grad():
        first = encoder(batch["efds"], batch["pos"], batch["block_mask"])
        second = encoder(other, batch["pos"], batch["block_mask"])
    torch.testing.assert_close(first[:25], second[:25], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("lengths", [[25, 25], [1, 24], [73, 139]])
def test_batch_norm_uses_the_entire_unpadded_sequence(lengths):
    torch.manual_seed(3)
    encoder = NFM(
        Config(dim=16, hidden_dim=32, num_layers=1, num_heads=2, efd_order=1)
    ).train()
    batch = pack_test_crops(
        [tagged_view(i, i * 100, n) for i, n in enumerate(lengths)],
        block_size=16,
        k=8,
    )
    assert len(batch["efds"]) == sum(lengths)
    x = batch["efds"].clone().requires_grad_()
    reference = copy.deepcopy(encoder.bn)
    expected = reference(x)
    captured = []
    hook = encoder.polygon_proj.register_forward_pre_hook(
        lambda module, args: captured.append(args[0])
    )
    result = encoder(x, batch["pos"], batch["block_mask"])
    hook.remove()
    torch.testing.assert_close(captured[0], expected)
    torch.testing.assert_close(encoder.bn.running_mean, reference.running_mean)
    torch.testing.assert_close(encoder.bn.running_var, reference.running_var)
    assert encoder.bn.num_batches_tracked == 1
    result.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(encoder.polygon_proj.weight.grad).all()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="Requires CUDA"
            ),
        ),
    ],
)
def test_variable_length_stack_backward_survives_an_interleaved_probe(device):
    """Match gradients when another batch runs before the SSL backward."""
    torch.manual_seed(7)
    encoder = (
        NFM(Config(dim=64, hidden_dim=128, num_layers=2, num_heads=2, efd_order=1))
        .to(device)
        .train()
    )
    reference = copy.deepcopy(encoder)
    probe = pack_test_crops([tagged_view(9, 200, 41)], block_size=128, k=8)
    probe_mask = probe["block_mask"].to(device)

    for lengths in ((17, 18), (73, 139), (128, 129)):
        batch = pack_test_crops(
            [tagged_view(i, i * 100, n) for i, n in enumerate(lengths)],
            block_size=128,
            k=8,
        )
        mask = batch["block_mask"].to(device)
        assert mask.mask_mod.doc_ids.shape == probe_mask.mask_mod.doc_ids.shape
        encoder.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        x = batch["efds"].to(device).requires_grad_()
        other_x = x.detach().clone().requires_grad_()
        positions = batch["pos"].to(device)
        actual = encoder(x, positions, mask)
        assert actual.shape == (sum(lengths), 64)
        upstream = torch.randn_like(actual)

        encoder.eval()
        with torch.no_grad():
            encoder(probe["efds"].to(device), probe["pos"].to(device), probe_mask)
        encoder.train()
        actual.backward(upstream)

        with patch.object(transformer, "flex_attention", attention_oracle):
            expected = reference(other_x, positions, mask)
        expected.backward(upstream)
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-3)
        torch.testing.assert_close(x.grad, other_x.grad, atol=2e-4, rtol=1e-3)
        for parameter, other in zip(
            encoder.parameters(), reference.parameters(), strict=True
        ):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-4, rtol=1e-3)
        torch.testing.assert_close(encoder.bn.running_mean, reference.bn.running_mean)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires the remote CUDA/H100 runtime"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_flex_matches_attention_oracle(dtype):
    torch.manual_seed(0)
    encoder = (
        NFM(Config(dim=64, hidden_dim=128, num_layers=1, num_heads=2, efd_order=1))
        .cuda()
        .eval()
    )
    views = [tagged_view(0, 0, 150), tagged_view(1, 100, 150)]
    outputs, gradients = [], []
    valid = torch.arange(300).cuda()
    for kernel in [transformer.flex_attention, attention_oracle]:
        batch = pack_test_crops(views, block_size=128, k=8)
        encoder.zero_grad(set_to_none=True)
        with (
            patch.object(transformer, "flex_attention", kernel),
            torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
        ):
            result = encoder(
                batch["efds"].cuda(),
                batch["pos"].cuda(),
                batch["block_mask"].to("cuda"),
            )[valid]
            loss = result.float().square().mean()
        loss.backward()
        outputs.append(result.detach().float())
        gradients.append(encoder.polygon_proj.weight.grad.detach().float().clone())
        assert torch.isfinite(gradients[-1]).all()
    tolerance = 0.03 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(outputs[0], outputs[1], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        gradients[0], gradients[1], atol=tolerance, rtol=tolerance
    )


def make_data(cfg):
    return DataModule(**cfg)


def make_model(kind="sigreg"):
    return SSLMetaArch(
        dim=16,
        hidden_dim=32,
        num_heads=2,
        num_layers=1,
        efd_order=4,
        regularizer=kind,
        slices=16,
        projector_dim=8,
        warmup_steps=0,
    )


def test_graph_pair_seeds_overlap_and_independent_augmented_ids(source_data):
    _, cfg = source_data
    data = make_data(cfg)
    data.setup("fit")
    dataset = data.train
    calls = []
    original = dataset.find_component

    def record(seed, *args):
        result = original(seed, *args)
        calls.append((seed, result))
        return result

    dataset.find_component = record
    dataset.augmentation = Compose([FieldRotation(), RandomDrop(0.2), ClusterDrop(0.1)])
    seed_everything(17, workers=True)
    sample = dataset[0]
    _, first_members = calls[0]
    second_seed, _ = calls[-1]
    assert second_seed in first_members
    assert len(sample["pos"]) == 2
    expected = np.intersect1d(*sample["indices"])
    np.testing.assert_array_equal(sample["matched_ids"], expected)
    assert len(expected) > 0
    for ids, pos, features in zip(
        sample["indices"], sample["pos"], sample["efds"], strict=True
    ):
        assert set(sample["matched_ids"]) <= set(ids)
        assert len(ids) == len(pos) == len(features) <= 64
        assert features.shape[1] == 16
    assert not np.array_equal(sample["indices"][0], sample["indices"][1])
    batch = train_collate_fn([sample], 16, 8)
    assert batch["pair_indices"].shape == (2, len(expected))
    assert batch["pair_counts"].tolist() == [len(expected)]
    assert len(batch["efds"]) == sum(len(pos) for pos in sample["pos"])


def test_shared_feature_values_match_original_intersection_after_sorting(source_data):
    _, cfg = source_data
    data = make_data(cfg)
    data.setup("fit")
    data.train.augmentation = Compose([])
    samples = [data.train[i] for i in range(2)]
    batch = train_collate_fn(samples, 16, 8)
    paired = batch["efds"][batch["pair_indices"]]
    torch.testing.assert_close(paired[0], paired[1], atol=1e-5, rtol=1e-5)
    assert torch.isfinite(paired).all()


def test_fixed_labeled_training_and_entire_slide_validation(source_data):
    _, cfg = source_data
    data = make_data(cfg)
    data.setup("fit")
    assert data.train_labeled.crop_size == 64
    assert data.validation.crop_size is None
    for i in range(len(data.train_labeled)):
        assert len(data.train_labeled[i]["efds"]) == 64
    batch = next(iter(data.val_dataloader()))
    assert len(batch["slide_ids"]) == 1
    assert len(batch["efds"]) == 769
    assert batch["block_mask"].seq_lengths == (769, 769)
    assert batch["block_mask"].mask_mod.doc_ids.shape == (cfg.max_sequence_length,)
    assert batch["score"].sum() == 769
    # Whole-slide validation is repeatable despite training RNG changes.
    np.random.seed(999)
    again = next(iter(data.val_dataloader()))
    torch.testing.assert_close(batch["efds"], again["efds"])
    train_ids = set(data.train_labeled.slides.slide_id)
    val_ids = set(data.validation.slides.slide_id)
    assert not train_ids & val_ids
    another = make_data(cfg)
    another.setup("validate")
    assert set(another.validation.slides.slide_id) == val_ids


def test_unknown_labels_are_not_scored():
    view = tagged_view(0, 0, 25)
    sample = {
        "pos": view["pos"],
        "efds": torch.tensor(view["efds"]),
        "labels": torch.zeros(25, 1),
        "slide_id": "one",
    }
    sample["labels"][7] = float("nan")
    batch = inference_collate_fn([sample], 16, 64)
    assert len(batch["efds"]) == 25
    assert batch["score"].sum() == 24
    assert not batch["score"][batch["efds"][:, 0] == 7].any()
    assert torch.isfinite(batch["labels"]).all()


def test_continuous_probe_does_not_update_encoder_or_batchnorm(source_data):
    _, cfg = source_data
    data = make_data(cfg)
    data.setup("fit")
    batch = inference_collate_fn([data.train_labeled[0]], 16, 8)
    model = make_model().train()
    state = {name: value.clone() for name, value in model.model.state_dict().items()}
    loss = model.forward_labeled(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.model.training
    assert all(p.grad is None for p in model.model.parameters())
    assert model.probe.weight.grad.abs().sum() > 0
    assert all(torch.equal(state[k], v) for k, v in model.model.state_dict().items())


@pytest.mark.parametrize("regularizer", ["sigreg", "visreg", "vicreg"])
def test_integrated_training_continuous_probe_validation_and_resume(
    source_data, tmp_path, regularizer
):
    seed_everything(0, workers=True)
    _, cfg = source_data
    data = make_data(cfg)
    model = make_model(regularizer)
    assert (
        model.model.bn.num_features == 16
    )  # Four coefficients per harmonic, no scalars.
    encoder_before = model.model.polygon_proj.weight.detach().clone()
    probe_before = model.probe.weight.detach().clone()
    options = {
        "accelerator": "cpu",
        "devices": 1,
        "max_steps": 3,
        "max_epochs": -1,
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "default_root_dir": tmp_path,
        "val_check_interval": 3,
        "check_val_every_n_epoch": None,
        "num_sanity_val_steps": 0,
        "limit_val_batches": 2,
    }
    trainer = Trainer(**options)
    trainer.fit(model, datamodule=data)
    assert trainer.global_step == 3
    assert model.model.bn.num_batches_tracked == 3  # SSL only, not labeled batches
    assert not torch.equal(encoder_before, model.model.polygon_proj.weight)
    assert not torch.equal(probe_before, model.probe.weight)
    assert np.isfinite(float(trainer.callback_metrics["validation/AUPRC"]))
    encoder = {name: value.clone() for name, value in model.model.state_dict().items()}
    probe = model.probe.weight.detach().clone()
    trainer.validate(model, datamodule=data, verbose=False)
    assert all(torch.equal(encoder[k], v) for k, v in model.model.state_dict().items())
    torch.testing.assert_close(probe, model.probe.weight)
    checkpoint = tmp_path / "last.ckpt"
    trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint)
    assert payload["global_step"] == 3
    portable = SSLMetaArch.load_from_checkpoint(checkpoint)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(portable.state_dict()[key], value)
    resumed = Trainer(**{**options, "max_steps": 4})
    resumed.fit(portable, datamodule=data, ckpt_path=checkpoint)
    assert resumed.global_step == 4
    assert portable.model.bn.num_batches_tracked == 4


def test_configs_compose_through_original_entrypoint():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(Path("configs").resolve()), version_base=None
    ):
        config = compose(config_name="nfm", overrides=["+experiment=nucleus_ssl"])
        assert config.data.datasets.unlabeled.dataset.crop_size == 2048
        assert config.data.datasets.labeled.dataset.crop_size == 4096
        assert config.trainer.callbacks.model_checkpoint.monitor == "validation/AUPRC"


def test_original_main_and_mlkit_checkpoint_monitor(source_data, tmp_path):
    from nfm.__main__ import main

    _, data_config = source_data
    config = OmegaConf.create(
        {
            "seed": 0,
            "mode": "fit",
            "checkpoint": None,
            "data": data_config,
            "model": {
                "dim": 16,
                "hidden_dim": 32,
                "num_heads": 2,
                "num_layers": 1,
                "efd_order": 4,
                "regularizer": "sigreg",
                "slices": 16,
                "projector_dim": 8,
                "warmup_steps": 0,
            },
            "trainer": {
                "accelerator": "cpu",
                "devices": 1,
                "precision": "32-true",
                "max_steps": 2,
                "max_epochs": -1,
                "sync_batchnorm": True,
                "enable_progress_bar": False,
                "enable_model_summary": False,
                "default_root_dir": str(tmp_path),
                "limit_val_batches": 2,
                "val_check_interval": 2,
                "check_val_every_n_epoch": None,
                "num_sanity_val_steps": 0,
                "callbacks": {
                    "model_checkpoint": {
                        "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                        "dirpath": str(tmp_path),
                        "save_top_k": 1,
                        "save_last": True,
                        "every_n_epochs": None,
                        "every_n_train_steps": 2,
                        "monitor": "validation/AUPRC",
                        "mode": "max",
                    },
                },
            },
        }
    )
    # Execute the original main body and its real MLkit Trainer, bypassing only
    # Hydra CLI dispatch and the external MLflow autolog wrapper.
    main.__wrapped__.__wrapped__(config, logger=False)
    assert (tmp_path / "last.ckpt").exists()
    payload = torch.load(tmp_path / "last.ckpt")
    assert payload["global_step"] == 2
    callback = next(
        v for k, v in payload["callbacks"].items() if "ModelCheckpoint" in k
    )
    assert callback["best_model_path"]
    assert torch.isfinite(callback["best_model_score"])
    assert payload["state_dict"]["model.bn.num_batches_tracked"] == 2


def test_source_sampler_matches_documented_mixture():
    from types import SimpleNamespace

    from nfm.data.samplers import SourceBalancedRandomSampler

    dataset = SimpleNamespace(
        slides=pd.DataFrame({"dataset": ["large"] * 9 + ["small"]})
    )
    sampler = SourceBalancedRandomSampler(dataset)
    # Source probabilities are 3/4 and 1/4 in the source-tempered component.
    expected = torch.tensor(
        [0.5 / 10 + 0.5 * (3 / 4) / 9] * 9 + [0.5 / 10 + 0.5 / 4], dtype=torch.float64
    )
    torch.testing.assert_close(sampler.weights, expected)
    assert sampler.replacement and sampler.num_samples == 10


@pytest.mark.parametrize("counts", [[2, 5, 0], [0, 0, 0], [0, 1, 0]])
def test_ragged_intersections_preserve_ids_and_counts(counts):
    samples = [
        paired_sample(
            [tagged_view(i, i * 100), tagged_view(i + 20, i * 100)],
            np.arange(count),
            str(i),
        )
        for i, count in enumerate(counts)
    ]
    batch = train_collate_fn(samples, block_size=16, k=8)
    assert batch["pair_counts"].tolist() == counts
    assert batch["pair_indices"].shape == (2, sum(counts))
    values = batch["efds"][batch["pair_indices"]][..., 0]
    torch.testing.assert_close(values[0], values[1])
    expected = torch.tensor(
        [i * 100 + j for i, c in enumerate(counts) for j in range(c)],
        dtype=torch.float32,
    )
    torch.testing.assert_close(values[0], expected)


@pytest.mark.parametrize("kind", ["sigreg", "visreg", "vicreg"])
@pytest.mark.parametrize("count", [0, 1, 7, 31])
def test_flat_objective_variable_population_and_safe_degenerate_batches(kind, count):
    z = torch.randn(2, count, 8, requires_grad=True)
    objective = NucleusObjective(kind, slices=16)
    loss, parts = objective(z)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert parts["ssl_skipped"] == int(count < 2)
    if count >= 2:
        torch.testing.assert_close(parts["alignment"], (z[0] - z[1]).square().mean())
        # SIGReg maintains its own projection RNG, independently of manual_seed.
        rectangular_objective = copy.deepcopy(objective)
        torch.manual_seed(4)
        flat, _ = objective(z)
        torch.manual_seed(4)
        rectangular, _ = rectangular_objective(z[:, None])
        torch.testing.assert_close(flat, rectangular)


def test_seed_location_uses_physical_units_and_lance_row_ids(source_data):
    from unittest.mock import patch

    _, cfg = source_data
    data = make_data(cfg)
    data.setup("fit")
    dataset = data.train
    import pyarrow as pa

    known = pa.table({"points": [[10.0, -20.0]]})
    with patch.object(lance.LanceDataset, "sample", return_value=known):
        points, polygons, ids, seed = dataset._candidates(dataset.slides.iloc[0])
    target = np.array([10.0 * 0.5, -20.0 * 0.25])
    assert seed == np.argmin(np.linalg.norm(points - target, axis=1))
    assert len(ids) == len(points) == len(polygons)
    assert points.dtype == np.float32


@pytest.mark.parametrize("level", ["nucleus", "region"])
@pytest.mark.parametrize("counts", [[2, 5, 0], [0, 0, 0], [0, 1, 0]])
def test_ragged_model_loss_backward_in_mixed_precision(level, counts):
    samples = []
    for i, count in enumerate(counts):
        views = [tagged_view(i, i * 100), tagged_view(i + 20, i * 100)]
        for view in views:
            view["efds"] = np.tile(view["efds"], (1, 4))
        samples.append(paired_sample(views, np.arange(count), str(i)))
    batch = train_collate_fn(samples, block_size=16, k=8)
    model = make_model()
    model.hparams.alignment_level = level
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model.forward_unlabeled(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.proj[0].weight.grad is not None
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_packing_handles_mixed_crop_sizes_smaller_than_k():
    packed = pack_test_crops(
        [tagged_view(0, 0, 1), tagged_view(1, 100, 25)],
        block_size=16,
        k=8,
    )
    assert packed["pos"].shape == (26, 2)
    allowed = allowed_tokens(packed["block_mask"])
    assert allowed[0, 0]
    assert not allowed[0, 1:].any()
    assert not allowed[1:26, 0].any()
    assert allowed.shape == (26, 26)


def test_packing_uses_exact_lengths_and_contiguous_offsets():
    views = [tagged_view(i, i * 100) for i in range(3)]
    packed, layout = _pack_crops(
        [(v["pos"], v["efds"]) for v in views],
        16,
        8,
        max_sequence_length=128,
    )
    assert [offset for _, offset in layout] == [0, 25, 50]
    assert "valid_tokens" not in packed
    assert "valid_mask" not in packed
    assert packed["pos"].shape == (75, 2)
    assert packed["efds"].shape == (75, 4)
    assert packed["block_mask"].seq_lengths == (75, 75)
    owners = packed["block_mask"].mask_mod.doc_ids
    assert owners.shape == (128,)
    torch.testing.assert_close(
        owners[:75], torch.arange(3, dtype=owners.dtype).repeat_interleave(25)
    )
    assert (owners[75:] == -1).all()
    for view, (order, offset) in zip(views, layout, strict=True):
        torch.testing.assert_close(
            packed["efds"][offset : offset + 25], torch.from_numpy(view["efds"][order])
        )
    with pytest.raises(ValueError, match="exceeds doc_ids capacity"):
        _pack_crops(
            [(v["pos"], v["efds"]) for v in views],
            16,
            8,
            max_sequence_length=74,
        )


def test_continuous_probe_uses_the_unpadded_sequence():
    view = tagged_view(0, 0)
    sample = {
        "pos": view["pos"],
        "efds": np.tile(view["efds"], (1, 4)),
        "labels": torch.arange(25).remainder(2).float()[:, None],
        "slide_id": "one",
    }
    batch = inference_collate_fn([sample], 16, 8)
    model = make_model().train()
    before = {name: value.clone() for name, value in model.model.state_dict().items()}
    loss = model.forward_labeled(batch)
    loss.backward()
    assert torch.isfinite(loss) and model.model.training
    assert model.probe.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.model.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.model.state_dict().items())
    model.eval()
    with torch.no_grad():
        embeddings = model(batch)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            model.probe(embeddings), batch["labels"]
        )
    torch.testing.assert_close(loss, expected)
