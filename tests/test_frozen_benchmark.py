"""Checks for a paired, converged, leakage-free frozen-feature comparison."""

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from nfm.frozen_benchmark import (
    extract,
    fit_probe,
    prepare_cohorts,
    random_reference,
    score_predictions,
)
from nfm.morphology_meta_arch import MorphologyMetaArch


def test_reference_restores_initial_weights_and_only_copies_calibration():
    torch.manual_seed(123)
    trained = MorphologyMetaArch(
        dim=16, hidden_dim=32, num_heads=2, num_layers=0, efd_order=4, intrinsic_dim=8
    )
    initial = {
        name: parameter.detach().clone()
        for name, parameter in trained.model.named_parameters()
    }
    trained.model.standardizer.observe(torch.randn(64, 16))
    trained.model.standardizer.freeze()
    with torch.no_grad():
        trained.model.shape_encoder[0].weight.add_(1)
    rng = torch.random.get_rng_state().clone()
    reference = random_reference(trained, 123)
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(parameter, initial[name], atol=0, rtol=0)
        assert not parameter.requires_grad
    for name, value in reference.standardizer.state_dict().items():
        torch.testing.assert_close(value, trained.model.standardizer.state_dict()[name])
    assert not torch.equal(
        reference.shape_encoder[0].weight, trained.model.shape_encoder[0].weight
    )


def test_converged_probe_uses_training_scaling_only_and_preserves_ranking():
    rng = np.random.default_rng(2)
    train = rng.normal(size=(512, 4))
    labels = (train[:, 0] > 0).astype(np.int64)
    probe = fit_probe(train, labels)
    scaler = probe.named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, train.mean(0))
    heldout = rng.normal(size=(256, 4))
    target = (heldout[:, 0] > 0).astype(np.int64)
    logits = probe.decision_function(heldout)
    metrics = score_predictions(target, logits)
    assert metrics["AUROC"] > 0.99 and metrics["AUPRC"] > 0.99
    shifted = score_predictions(target, logits - 100)
    assert shifted["AUPRC"] == metrics["AUPRC"]
    assert shifted["recall_at_0.5"] == 0
    assert shifted["precision_at_recall_0.8"] > 0.9
    with pytest.raises(ValueError, match="both classes"):
        fit_probe(train, np.zeros(len(train)))


def test_reconstructed_split_matches_original_seeded_setup(tmp_path):
    slides = pd.DataFrame(
        [
            {
                "slide_id": f"slide{i}",
                "nuclei_count": 1024,
                "gleason_score": i % 2,
                "is_carcinoma": True,
                "slide_nuclei_path": "/unused",
                "mpp_x": 0.5,
                "mpp_y": 0.5,
            }
            for i in range(12)
        ]
    )
    metadata = tmp_path / "slides.parquet"
    annotations = tmp_path / "annotations.parquet"
    slides.to_parquet(metadata)
    pd.DataFrame(
        [
            {"slide_id": slide, "id": nucleus, "annot_label": nucleus}
            for slide in slides.slide_id
            for nucleus in [0, 1]
        ]
    ).to_parquet(annotations)
    config = OmegaConf.create(
        {
            "seed": 0,
            "data": {
                "datasets": {
                    "labeled": {
                        "slides_path": str(metadata),
                        "split_size": 0.25,
                        "split_stratify_col": "gleason_score",
                        "dataset": {
                            "_target_": "nfm.data.datasets.ClassificationNucleiDataset",
                            "annotation_path": str(annotations),
                            "crop_size": 512,
                            "efd_order": 4,
                        },
                    }
                }
            },
        }
    )
    np.random.seed(0)
    expected_train, expected_val = train_test_split(
        slides, test_size=0.25, stratify=slides.gleason_score
    )
    np.random.seed(999)
    training, validation, hashes = prepare_cohorts(config, "unused", 4)
    assert set(validation.slides.slide_id) == set(expected_val.slide_id)
    assert set(training.slides.slide_id) <= set(expected_train.slide_id)
    assert not set(training.slides.slide_id) & set(validation.slides.slide_id)
    assert len(training) == 4 and hashes["train"] != hashes["validation"]


def test_paired_extraction_keeps_all_feature_sets_aligned_with_labels():
    class Encoder(torch.nn.Module):
        def __init__(self, factor):
            super().__init__()
            self.factor = factor
            self.probes = torch.nn.ModuleDict(
                {
                    name: torch.nn.Linear(4, 1)
                    for name in ["raw", "intrinsic", "contextual"]
                }
            )

        def forward(self, batch):
            values = batch["efds"] * self.factor
            return {"raw": values, "intrinsic": 2 * values, "contextual": 3 * values}

    rng = np.random.default_rng(3)
    sample = {
        "pos": rng.normal(size=(20, 2)).astype(np.float32),
        "efds": np.tile(np.arange(20, dtype=np.float32)[:, None], (1, 4)),
        "labels": torch.tensor(np.arange(20) % 2, dtype=torch.float32)[:, None],
        "slide_id": "one",
    }
    config = OmegaConf.create(
        {"data": {"block_size": 16, "k": 4, "max_sequence_length": 128}}
    )
    features, labels, online = extract(sample, Encoder(1), Encoder(10), config, "fp32")
    np.testing.assert_array_equal(labels, features["raw"][:, 0].astype(np.int64) % 2)
    np.testing.assert_allclose(
        features["random_contextual"], features["contextual"] * 10
    )
    np.testing.assert_allclose(features["random_intrinsic"], features["intrinsic"] * 10)
    assert set(online) == {"raw", "intrinsic", "contextual"}
    assert all(values.dtype == np.float32 for values in online.values())
