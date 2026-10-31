# Masked physical-region pretraining

Run the new experiment through the existing Hydra / MLkit / MLflow entry point:

```bash
uv run python -m nfm --config-name=regional
```

This is a new architecture and objective; start a fresh run. The morphology
experiment remains available for comparison. Paths use the same manifests,
Lance stores, labeled Parquets, and annotations as the existing pipeline:

```bash
uv run python -m nfm --config-name=regional \
  data.datasets.unlabeled.slides_path=/path/to/ssl_manifest.parquet \
  data.datasets.unlabeled.dataset.nuclei_path=/path/to/cells \
  data.datasets.labeled.slides_path=/path/to/labeled_metadata.parquet \
  data.datasets.labeled.dataset.annotation_path=/path/to/annotations
```

The supplied config runs 3,000 steps on one H100 with BF16, 16 SSL windows per
batch, an eight-layer 256-dimensional regional transformer, and labeled probe
batches of 32 crops with 4,096 nuclei each. It starts from random weights.
The first 16 steps calibrate input normalization and decoder priors using only
visible SSL cells/tiles, without representation or probe updates. Calibration
is saved in checkpoints. Resume with `checkpoint=/path/to/last.ckpt` and matching
model/data settings.

For a three-step data/forward/backward/validation check:

```bash
uv run python -m nfm --config-name=regional \
  model.normalization_batches=1 +trainer.fast_dev_run=3
```

## What is hidden

Each sample is a complete 512 × 512 micrometer square, divided into 8 × 8 tiles
of 64 micrometers. A uniformly chosen detection remains visible as an anchor;
the window origin is jittered around it. Approximately 40% of tiles are hidden,
using adjacent 2 × 2 blocks, with the anchor tile excluded. The mask specifies
coarse physical areas, including empty ones. It does not identify hidden nuclei.

Hidden nuclei are removed before encoding: their coordinates, EFDs, counts,
IDs, and edges are absent from the input dictionary. This path builds no cell
graph. The encoder receives visible cells, their window-relative coordinates,
and the fixed tile grid. Changing hidden geometry with visible cells and mask
fixed leaves encoder outputs and calibration unchanged.

The loader converts both coordinate axes and contours using their respective
MPP. Exact nearest-neighbor retrieval expands until it certifies coverage of
the entire square, or retrieves the full slide census. Only then are contours
fetched by persistent Lance row ID. This prevents a retrieval limit from turning
missing detections into empty-space targets. Retrieval and cell budgets fail
explicitly rather than truncating or rejecting dense samples. Increase
`data.datasets.unlabeled.dataset.max_query` or `max_cells` if the corresponding
error appears. Also increase `data.max_sequence_length` if a packed batch
exceeds its visible-cell budget (default 131,072).

Exact retrieval currently scans the slide's point column with the vector index
disabled. Measure data loading time on your storage; remote I/O may limit
throughput. Sampled windows are anchor-conditioned, rather than uniform over
the slide's area. They can include background beyond the tissue boundary.

## Encoder and objective

The intrinsic path is a pointwise shape MLP, 64 EFD coefficients → 256 → 128,
trained with a linear denoising decoder. Input moments are calibrated and frozen.
Visible cells are pooled into each tile using intrinsic mean/standard deviation,
an 8 × 8 count raster, and total population. A tile projection produces 256
features. Entire hidden-tile inputs are replaced with a mask token. Fixed local
physical positional encodings and eight native SDPA transformer layers model
the 64 tiles. Attention has no dependence on hidden-cell geometry.

The hidden-region heads predict:

- Count rates at 8 micrometer resolution, trained with Poisson deviance at
  8, 16, 32, and 64 micrometers by count-preserving sum pooling.
- Occupancy of the 8 micrometer bins, trained with binary cross entropy.
- Regional histograms of nuclear area, contour-axis ratio, and circularity,
  trained with KL divergence on occupied hidden tiles. Fixed histogram edges
  (area in square micrometers, dimensionless ratios) are declared in
  `nfm/data/datasets/regional_dataset.py`; no labels construct them.

The total loss is count deviance + occupancy BCE + 0.25 × morphology KL +
intrinsic denoising loss. Empty tiles contribute to count and occupancy losses;
their undefined morphology distributions are excluded. The logged Poisson
deviance omits the conventional factor of two. Fine cell locations are
stochastic; low error or exact reconstruction is not a prerequisite for useful
conditional intensity predictions.

## Controls and decision criteria

Two percent of SSL slides are held out entirely, including from calibration.
Validation evaluates a fixed sample and mask on up to 64 of those slides every
500 steps. All held-out slides remain excluded from training, even when only a
subset is evaluated. The split uses `data.split_seed`, tied to `seed` by default.
For repeated model seeds on the same cohorts, use `seed=1 data.split_seed=0`.
This is slide-held-out SSL validation; related slides may share a source or case.
The prepared SSL manifest's upstream filtering and overlap checks are retained.
The existing labeled stratified slide split uses the same explicit split seed.
Its one-slide-per-case property comes from your prepared labeled manifest.

Monitor:

- `ssl_validation/geometry_deviance` against both
  `geometry_interpolation_baseline` and `geometry_visible_mean_baseline` under
  the same prefix. Each baseline uses only visible tiles in that window.
- `ssl_validation/spatial_skill = 1 - deviance / interpolation_deviance`:
  positive values mean the model beats local interpolation on hidden regions.
  Also compare `morphology_KL` with `morphology_interpolation_baseline`.
- `validation/contextual_AUPRC` against `random_contextual_AUPRC` and
  `local_summary_AUPRC`, with AUROC and operating-curve diagnostics as well.
  The local summary includes each nucleus's raw EFDs, tile EFD moments, the
  count raster, and population. Random features use an exact frozen copy of the
  initial encoder with identical input calibration, saved in every checkpoint.
- `validation/intrinsic_AUPRC` versus `raw_AUPRC` and `random_intrinsic_AUPRC`
  for the independent shape path.

Six linear heads train continuously on the same labeled examples, with detached
features, FP32 readouts, LR 0.001, no warmup, and no weight decay. Encoder LR is
0.0003 with warmup/cosine decay. Probe gradients are clipped separately and cannot
update or rescale encoder gradients. Full-slide labeled validation scores every
known nucleus once. Checkpoint selection uses contextual AUPRC, so this is a
development validation cohort, not an independent final-test estimate.

After training, fit converged, train-standardized classifiers using the extended
frozen benchmark:

```bash
uv run python -m nfm.frozen_benchmark --run-id NEW_RUN_ID
```

It automatically loads the regional model, its saved initial reference, and
all six feature sets. It retains the same 128 training slides × two crops × 256
sampled nuclei and whole-validation-cohort protocol as the previous comparison.
It writes only a local aggregate JSON report. First require a persistent
held-out spatial advantage and a fresh contextual-probe gain over matched random
features and the explicit summary. An improvement over a raw linear EFD probe
alone is insufficient. If those gains are absent by 3,000 steps, retain the result
and inspect the targets and architecture before scaling the experiment.

## Inference and scope

`RegionalMetaArch.forward(batch)` returns normalized raw EFDs, intrinsic nucleus
features (128), contextual nucleus features (128 + 256), and local summaries
(257). A contextual nucleus feature combines its own shape with its tile's
regional representation. Nuclei in a tile share the regional component.

`model.model.encode_slide(batch, return_regions=True)` additionally returns
64-micrometer tile embeddings, counts, physical centers, and document IDs.
Inputs need `efds`, physical-micrometer `pos`, and per-cell `document_ids`.
Use `regional_labeled_collate` for the existing labeled samples; labels are not
consumed by the encoder. Outputs preserve input nucleus order.

Slide inference assigns every nucleus to one fixed 512-micrometer window and
processes batches of 32 independent windows, with all cells visible. Attention
memory is bounded by that window batch. Cell features and output memory still
grow with the slide census. This supplies nucleus and physical-region features
in one slide API call; context is limited to the window, with boundaries and no
learned whole-slide embedding or adaptive cluster discovery. Region embeddings
can be pooled downstream, but their task utility requires measurement.

## Verification

```bash
uv run --with pytest python -m pytest \
  tests/test_regional_pretraining.py tests/test_frozen_benchmark.py -q
```

Tests cover certified physical retrieval and contour alignment, target isolation,
calibration, empty-region losses, FP32/BF16 gradients, label isolation, initial
reference preservation, slide/chunk isolation, both real-data validation streams,
the original entry point, checkpoint resume, and frozen-benchmark compatibility.
A synthetic four-layout task must learn on unseen windows/masks, beat visible
interpolation, and worsen when visible context contradicts hidden targets.
This verifies a learnable spatial path, not pathology transfer. Native CUDA
attention checks run when a GPU is available; H100 throughput/memory and actual
downstream performance must be measured on the training host.
