"""Evaluate fixed nucleus/region features against a matched random encoder.

Run on the data/GPU host: python -m nfm.frozen_benchmark --run-id RUN_ID
Only aggregate results are written locally; the MLflow run is read-only.
"""

import argparse
import copy
import hashlib
import json
import random
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import get_class, instantiate
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from omegaconf import OmegaConf
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from nfm.data.data_module import inference_collate_fn
from nfm.morphology_meta_arch import MorphologyMetaArch


def random_reference(trained, seed: int):
    """Use a saved initial encoder or recreate its seeded, calibrated initialization."""
    if hasattr(trained, "reference"):
        return copy.deepcopy(trained.reference).eval().requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = MorphologyMetaArch(**dict(trained.hparams)).model
    reference.standardizer.load_state_dict(trained.model.standardizer.state_dict())
    return reference.eval().requires_grad_(False)


def fit_probe(features, labels, regularization: float = 1.0, max_iter: int = 1000):
    """Fit train-only scaling and a linear classifier; reject unconverged fits."""
    if len(np.unique(labels)) != 2:
        raise ValueError("The probe-training sample must contain both classes")
    for iterations in [max_iter, 4 * max_iter]:
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=regularization, max_iter=iterations, tol=1e-5),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(features, labels)
        if not any(issubclass(item.category, ConvergenceWarning) for item in caught):
            return pipeline
    raise RuntimeError(
        "Probe did not converge; increase --max-iter before interpreting its score"
    )


def score_predictions(labels, logits):
    if len(np.unique(labels)) != 2:
        raise ValueError("Evaluation requires both classes")
    precision, recall, _ = precision_recall_curve(labels, logits)
    return {
        "AUROC": float(roc_auc_score(labels, logits)),
        "AUPRC": float(average_precision_score(labels, logits)),
        "precision_at_0.5": float(precision_score(labels, logits > 0, zero_division=0)),
        "recall_at_0.5": float(recall_score(labels, logits > 0, zero_division=0)),
        "positive_fraction_at_0.5": float((logits > 0).mean()),
        # Operating-curve diagnostics, not deployment thresholds selected on test data.
        "precision_at_recall_0.8": float(precision[recall >= 0.8].max()),
        "recall_at_precision_0.8": float(recall[precision >= 0.8].max(initial=0)),
    }


def cohort_hash(slides):
    identifiers = sorted(str(value) for value in slides.slide_id)
    return hashlib.sha256("\n".join(identifiers).encode()).hexdigest()


def prepare_cohorts(config, tracking_uri, train_slides):
    labeled = config.data.datasets.labeled
    path = labeled.slides_path
    if path.startswith("mlflow-artifacts:"):
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        path = download_artifacts(artifact_uri=path)
    slides = pd.read_parquet(path)
    slides = slides[slides.nuclei_count >= 512]
    # The original main seeds NumPy before setup. Reproduce its slide split
    # explicitly, independently of feature extraction and probe RNG activity.
    split_seed = config.data.get("split_seed")
    train, validation = train_test_split(
        slides,
        test_size=labeled.split_size,
        stratify=slides[labeled.split_stratify_col],
        random_state=int(config.seed if split_seed is None else split_seed),
    )
    train = train[train.nuclei_count >= labeled.dataset.crop_size]
    hashes = {"train": cohort_hash(train), "validation": cohort_hash(validation)}
    count = min(train_slides, len(train))
    if count < 2:
        raise ValueError("At least two labeled training slides are required")
    selected = train.sample(n=count, random_state=int(config.seed)).reset_index(
        drop=True
    )
    training_dataset = instantiate(labeled.dataset, slides=selected)
    validation_dataset = instantiate(labeled.dataset, slides=validation, crop_size=None)
    return training_dataset, validation_dataset, hashes


@torch.inference_mode()
def extract(sample, trained, reference, config, precision):
    collate = getattr(trained, "labeled_collate_fn", inference_collate_fn)
    packed = collate(
        [sample], config.data.block_size, config.data.k, config.data.max_sequence_length
    )
    device = next(trained.parameters()).device
    batch = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in packed.items()
    }
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        learned = trained(batch)
        initial = reference(batch)
    score = batch["score"]
    combined = {
        **learned,
        "random_intrinsic": initial["intrinsic"],
        "random_contextual": initial["contextual"],
    }
    features = {
        name: value[score].float().cpu().numpy() for name, value in combined.items()
    }
    # Evaluate saved online heads in FP32 to avoid BF16 sigmoid probability ties.
    online = {}
    with torch.autocast("cuda", enabled=False):
        for name, head in trained.probes.items():
            online[name] = (
                head(combined[name][score].float()).float().cpu().numpy().ravel()
            )
    return (
        features,
        batch["labels"][score].cpu().numpy().ravel().astype(np.int64),
        online,
    )


def benchmark(trained, reference, config, training, validation, args):
    cache, labels = {}, []
    for index in tqdm(range(len(training)), desc="Cache training features"):
        for crop in range(args.crops_per_slide):
            seed = int(config.seed) + index * 1009 + crop
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            features, target, _ = extract(
                training[index], trained, reference, config, args.precision
            )
            rng = np.random.default_rng(seed)
            selected = rng.choice(
                len(target), min(args.nuclei_per_crop, len(target)), replace=False
            )
            for name, values in features.items():
                cache.setdefault(name, []).append(values[selected])
            labels.append(target[selected])
    labels = np.concatenate(labels)
    cache = {name: np.concatenate(parts) for name, parts in cache.items()}
    probes, training_scores = {}, {}
    for name, features in tqdm(cache.items(), desc="Fit converged probes"):
        probe = fit_probe(features, labels, args.regularization, args.max_iter)
        probes[name] = probe
        training_scores[name] = score_predictions(
            labels, probe.decision_function(features)
        )
    del cache
    predictions, online_predictions, validation_labels = {}, {}, []
    for index in tqdm(range(len(validation)), desc="Evaluate whole validation slides"):
        features, target, online = extract(
            validation[index], trained, reference, config, args.precision
        )
        for name, values in features.items():
            predictions.setdefault(name, []).append(
                probes[name].decision_function(values)
            )
        for name, values in online.items():
            online_predictions.setdefault(name, []).append(values)
        validation_labels.append(target)
    target = np.concatenate(validation_labels)
    return {
        "training_slides": len(training),
        "training_nuclei": len(labels),
        "validation_slides": len(validation),
        "validation_nuclei": len(target),
        "training_prevalence": float(labels.mean()),
        "validation_prevalence": float(target.mean()),
        "training_metrics": training_scores,
        "fresh_probe_metrics": {
            name: score_predictions(target, np.concatenate(parts))
            for name, parts in predictions.items()
        },
        "saved_online_probe_metrics": {
            name: score_predictions(target, np.concatenate(parts))
            for name, parts in online_predictions.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--tracking-uri", default="https://mlflow.rationai.cloud.e-infra.cz"
    )
    parser.add_argument(
        "--checkpoint-artifact", default="checkpoints/last/checkpoint.ckpt"
    )
    parser.add_argument("--train-slides", type=int, default=128)
    parser.add_argument("--crops-per-slide", type=int, default=2)
    parser.add_argument("--nuclei-per-crop", type=int, default=256)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        min(
            args.train_slides, args.crops_per_slide, args.nuclei_per_crop, args.max_iter
        )
        < 1
        or args.regularization <= 0
    ):
        parser.error("Sample sizes, iterations and regularization must be positive")
    if not torch.cuda.is_available():
        parser.error(
            "Run this evaluator on the data/GPU host; whole-cohort evaluation expects CUDA"
        )
    torch.set_float32_matmul_precision("high")
    client = MlflowClient(tracking_uri=args.tracking_uri)
    run = client.get_run(args.run_id)
    with tempfile.TemporaryDirectory(prefix="nfm-frozen-") as directory:
        configuration = client.download_artifacts(
            args.run_id, "configs/config-resolved.yaml", directory
        )
        checkpoint = client.download_artifacts(
            args.run_id, args.checkpoint_artifact, directory
        )
        config = OmegaConf.load(configuration)
        model_type = get_class(config.model._target_)
        trained = (
            model_type.load_from_checkpoint(
                checkpoint, map_location="cpu", weights_only=True
            )
            .eval()
            .requires_grad_(False)
        )
        reference = random_reference(trained, int(config.seed))
        checkpoint_step = torch.load(checkpoint, map_location="cpu", weights_only=True)[
            "global_step"
        ]
        training, validation, hashes = prepare_cohorts(
            config, args.tracking_uri, args.train_slides
        )
        result = benchmark(
            trained.cuda(), reference.cuda(), config, training, validation, args
        )
    expected = run.data.metrics.get("validation/prevalence")
    result.update(
        {
            "run_id": args.run_id,
            "checkpoint_artifact": args.checkpoint_artifact,
            "checkpoint_step": checkpoint_step,
            "run_seed": int(config.seed),
            "source_commit": run.data.tags.get("mlflow.source.git.commit"),
            "cohort_hashes": hashes,
            "logged_validation_prevalence": expected,
            "validation_prevalence_matches_run": expected is not None
            and abs(expected - result["validation_prevalence"]) < 1e-6,
            "regularization_C": args.regularization,
            "precision": args.precision,
        }
    )
    output = (
        args.output
        or Path("artifacts/nucleus_ssl/frozen_benchmark") / f"{args.run_id}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved aggregate comparison: {output}")
    if not result["validation_prevalence_matches_run"]:
        raise RuntimeError(
            "Reconstructed validation prevalence differs from the run; verify the split before comparing metrics"
        )


if __name__ == "__main__":
    main()
