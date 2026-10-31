# Morphology retention and masked-context pilot

The previous correspondence-only checkpoint failed both the online probe and a
fresh probe. This experiment uses fixed reconstruction targets to make feature
retention and context prediction directly measurable. It is a proposed remedy;
real-data transfer remains to be measured on the H100.

```bash
uv run python -m nfm --config-name=morphology
```

It uses the same Hydra/MLkit/MLflow entry point, manifests, physical coordinates,
graph-crop traversal, supervised labels, and variable-length FlexAttention as the
existing pipeline. The separate config selects new dataset/model classes. It
runs on one device for 3,000 steps and validates every 500 steps. A batch contains
16 single 2,048-nucleus crops, rather than two 4,096-nucleus views per slide.

The first 16 input batches calibrate input and target means/scales from SSL data
only. No representation or probe weights are updated during calibration. These
batches count toward Lightning's step counter and LR schedule. Statistics are
then frozen and saved in checkpoints. Input scales have a 0.001-micrometer floor;
target scales have a 0.02 floor to avoid amplifying nearly constant channels.
Do not enable validation sanity steps before calibration. Checkpoints include
the architecture, normalization, probes, and optimizer state; this experiment
starts from scratch and does not load the earlier correspondence checkpoint.

## Objectives and outputs

The intrinsic path is a pointwise 64 -> 256 -> 128 MLP. Add Gaussian noise of
standard deviation 0.05 in standardized input space and linearly decode its
128-dimensional output back to the clean EFDs using smooth-L1 loss. This creates
an explicit pressure to retain nuclear morphology in a linearly readable form.
There are no field rotations, flips, scale changes, or correspondence losses in
this first pilot. Physical size remains available.

The contextual path projects the intrinsic features to dimension 256 and uses
the existing four-layer spatial transformer. Before any message passing, replace
all shape-derived channels of 30% of nuclei in each crop with a learned mask
token. Each crop retains at least one visible cell. A linear decoder predicts
fixed morphology targets for masked nuclei only:

- log polygon area and log perimeter in physical units;
- log covariance-axis ratio, circularity, and radial coefficient of variation;
- log amplitudes of harmonics 2–8 relative to the first EFD harmonic.

These are deterministic descriptors, not nucleus-type labels. Absolute cell
locations remain visible, so this is morphology prediction conditioned on known
positions. The clean masked-cell features used by the intrinsic objective never
enter the contextual branch: the shape encoder is pointwise, normalization is
frozen, and masking occurs before the first interaction between cells.

```text
SSL loss = intrinsic reconstruction + context_weight * masked-context reconstruction
```

Both terms are feature/sample means. There is no count-scaled Gaussian
regularizer, EMA teacher, contrastive objective, or latent collapse target.
This is an adaptation motivated by [GraphMAE](https://arxiv.org/abs/2205.10803),
not a reproduction of that method or a demonstrated pathology result.

`model(batch)` returns three clean feature tensors for every input nucleus:

| Key | Default dimension | Meaning |
| --- | ---: | --- |
| `raw` | 64 | Fixed-standardized EFD reference |
| `intrinsic` | 128 | Denoising shape representation |
| `contextual` | 384 | Intrinsic features concatenated with contextual features |

The intrinsic features are retained explicitly in the fused representation;
the contextual transformer cannot overwrite them. The raw input is a comparison
branch, not an extra supervised input. Learned region/slide embeddings are not
part of this pilot.

## Read the run as a controlled comparison

Three detached probes train on exactly the same labeled crops, with separate
heads for the three feature sets. They use LR 0.001, no weight decay, no warmup,
and clipping separate from the representation parameters. Labels cannot change
encoder gradients or its clipping multiplier. The encoder uses LR 0.0003,
100-step warmup, cosine decay, and weight decay 0.05. Validation uses full slides
and scores known real nuclei only. Logged prevalence and predicted-positive
fractions make recall changes easier to interpret.

Watch these metrics rather than only `train/ssl_loss`:

- `train/intrinsic_reconstruction` should fall well below
  `train/intrinsic_mean_baseline`. Its decoder is linear, so this checks retention.
- `validation/intrinsic_AUPRC` versus `validation/raw_AUPRC` checks whether the
  learned shape path retains or improves task information.
- `train/context_reconstruction` and `train/context_mse` should beat the
  corresponding visible-mean baselines. The baseline uses only unmasked nuclei
  in the same crop; hidden-cell targets are excluded.
- `validation/contextual_AUPRC` versus `validation/intrinsic_AUPRC` is the key
  test of added contextual utility. Also compare AUROC, recall, positive
  fraction, and prevalence. Checkpoints are selected by contextual AUPRC.

If denoising improves but contextual features do not add transfer, that does not
establish useful context learning. If neither learned branch beats its reference
by 3,000 steps, retain the failed result and investigate targets/inputs instead
of scaling this recipe to the entire corpus. Reconstruction alone is not proof
of downstream utility.

The first ablation is intrinsic reconstruction only:

```bash
uv run python -m nfm --config-name=morphology model.context_weight=0
```

For this ablation, evaluate the intrinsic probe; the contextual branch has no
training objective and its output is not evidence of learned context.

For a short I/O/forward/backward/validation check:

```bash
uv run python -m nfm --config-name=morphology \
  model.normalization_batches=1 +trainer.fast_dev_run=3
```

The smoke check intentionally uses a small calibration sample. Use the default
16 calibration batches for the experiment. Override dataset paths with the same
`data.datasets.*` keys as the existing pipeline. Resume this experiment with
`checkpoint=/path/to/last.ckpt` and matching model settings.

```bash
uv run --with pytest python -m pytest \
  tests/test_morphology_pretraining.py tests/test_training_startup.py -q
```

Tests use synthetic geometry and a CPU attention reference. The mask and compiled
CUDA checks in `tests/test_block_mask.py` still apply to the shared attention
backend; real-data quality and H100 memory/throughput are not locally verified.
The masked-shape isolation test also runs against compiled FlexAttention on CUDA
in FP32/BF16 when a GPU is available.

On the H100, also require the shared compiled-kernel checks:

```bash
NFM_REQUIRE_CUDA=1 uv run --with pytest python -m pytest \
  tests/test_block_mask.py tests/test_morphology_pretraining.py -q
```

## Paired frozen-checkpoint comparison

The online raw/intrinsic/contextual heads do not establish that pretraining
outperforms an initially random spatial encoder. Before changing the objective
again, run this read-only comparison on the data/GPU host:

```bash
uv run python -m nfm.frozen_benchmark \
  --run-id 4c521e1d942440d0acb2da52ca646e5d
```

The evaluator downloads the saved configuration and checkpoint. It reconstructs
the original seeded slide split, samples 128 training slides with two fixed crops
per slide, and caches 256 nuclei per crop. The same nuclei, labels, neighborhoods,
and normalization are used for all representations. Overlapping training crops
can contain repeated nuclei; validation remains slide-disjoint and covers every
known nucleus in each validation slide. Nothing is written to the MLflow run.

It compares raw EFDs, trained intrinsic/contextual features, and random
intrinsic/contextual features with the same architecture and run-seed
initialization. The random reference copies only input calibration buffers from
the checkpoint, not learned weights. Classifier feature scaling is fitted on the
probe-training sample only. Each logistic regression is fitted to convergence;
an unconverged classifier causes an error rather than a reported weak score.
No encoder parameters or normalization buffers are updated.

The five classifiers are then evaluated on the full validation cohort. Saved
online heads are evaluated on the same examples for comparison, with FP32
readouts to avoid BF16 sigmoid ties. The report includes AUPRC/AUROC, threshold
0.5 precision/recall, and precision–recall operating-curve diagnostics. These
curve summaries do not choose deployment thresholds or establish final-test
performance. The evaluator verifies validation prevalence against the run and
records cohort hashes to detect a different split.

The JSON report is saved under
`artifacts/nucleus_ssl/frozen_benchmark/<run-id>.json`; individual predictions
are not saved. Use `--train-slides 32` for a faster preliminary check, or
`--checkpoint-artifact checkpoints/epoch=0-step=2000/checkpoint.ckpt` to select
a particular available checkpoint instead of `last`. The report records the
actual checkpoint step. Dataset paths come from the run's saved configuration.

Interpret the trained-versus-random difference before attributing a contextual
gain to SSL. Reconstruction can preserve EFDs while adding little task signal,
and a random nonlinear/spatial feature map can outperform a linear raw-EFD
classifier. A weak raw linear probe does not establish that raw morphology is
uninformative to a nonlinear classifier.
