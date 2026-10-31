# Nucleus correspondence SSL in the main training pipeline

Run through the original Hydra / MLkit / MLflow entry point:

```bash
uv run python -m nfm +experiment=nucleus_ssl
```

The old `nfm.experiments` training module has been merged into `nfm.data`,
`nfm.modeling.objectives`, and `nfm.ssl_meta_arch`. It is no longer a separate
entry point. Existing `+experiment=LUMI` paths, logging,
trainer strategy, and augmentation configuration remain available.

## Sampling and inputs

`NucleiDataset` now produces two graph-grown crops:

1. Draw a slide using the configured sampler. The nucleus SSL and LUMI configs
   use replacement sampling: 50% slide-uniform plus 50% source-tempered, where
   source probability is proportional to sqrt(number of slides).
2. Choose a detection uniformly from the slide and retrieve pixel-space
   nearest neighbors. The candidate budget is `2 * crop_size / (1 - alpha)`,
   capped at the slide census. Deduplicate physical
   positions at 1e-4 micrometers for Delaunay graph construction.
3. Convert to physical coordinates using both pixel spacings. Grow crop A from
   the retrieved center nucleus using the original graph traversal/cost rule:
   `alpha * edge_length + (1 - alpha) * distance_from_seed`.
4. Choose the second seed with the configured triangular rule over crop A's
   graph-traversal order (mode near 90% of the order). Grow crop B on the same
   graph. This favors later traversal positions; it is not uniform sampling or
   a guarantee of physical distance. The original seed can occasionally recur.
5. Independently augment both crops, propagating IDs through deletion. Use **all**
   nucleus IDs surviving in both views. There is no fixed-count subsample and no
   overlap-based rejection/resampling. Intersection sizes vary across regions.

The nearest-neighbor pool is a bounded candidate graph, as in the original
sampler; it is not a certified reconstruction of the full-slide Delaunay graph.
There is no physical core/halo restriction in this version: the scored population
is the surviving original intersection. Identical crop membership remains possible, especially on very small slides.

The portable experiment uses 2,048 nuclei per SSL crop and 16 region draws per
batch. LUMI retains 4,096-nucleus SSL crops. Augmentation can reduce encoded crop
counts; each batch encodes exactly the nuclei that survive.
The default augmentation is flip, field rotation, and mild shape distortion;
LUMI keeps its explicitly configured augmentations and applies them independently.

Each nucleus has the original 64 EFD coefficients (order 16), with no extra
handcrafted scalar features. EFDs preserve physical scale; nucleus positions
are passed separately. The EFD function is shared by SSL and labeled datasets. NFM's
input BatchNorm, linear input projection, and contextual transformer are used
directly. Positions enter RoVE; slide IDs and nucleus IDs do not enter the model.
Packing lives directly in `nfm/data/data_module.py`: one shared sorting/packing
helper, SSL correspondence collation, and labeled collation. It uses the dataset
`efds` fields directly. Crops are packed contiguously without padding features,
positions, or labels, including at the final partial attention block. BatchNorm
processes the whole packed sequence; no valid-prefix lengths are passed through
the model. Shared blocks use document masks to prevent cross-crop attention.

Only the attention callback's `doc_ids` buffer has a fixed length, controlled by
`data.max_sequence_length` (default 1,048,576). Its unused suffix is `-1`. This is
a metadata capacity, not an encoded token budget: attention and block-mask
sequence lengths remain equal to the actual packed length. Each batch owns its
callback and buffer, so a labeled forward cannot overwrite document IDs needed
by a pending SSL backward. Batches above the capacity fail explicitly; increase
this setting to accommodate larger slides. FlexAttention uses
`torch.compile(dynamic=True)` with CUDA graph capture disabled.

The SSL manifest is filtered upstream to calibrated human level-0 records and
checked for overlap with labeled validation slides, including alternate IDs.
The training pipeline consumes that prepared manifest directly. The original
labeled stratified train/validation split is retained and uses the global RNG.
The current labeled metadata contains one slide per patient/case. Unlike the
retired pilot, no prepared protocol folder, separate final-test cohort, or fresh
logistic-probe subset is created.

## Loss and continuously trained probe

The projector is `dim -> 512 -> GELU -> 128`, without output BatchNorm or L2
normalization. Matching projections have shape `[2, total_matches, dim]`.
Ragged region intersections are concatenated, with counts retaining boundaries.
Agreement is mean squared error per matched nucleus across the whole batch;
larger intersections contribute proportionally more weight. The regularizer
uses all matched nuclei separately per view, without fake/padded samples.
Log `train/matched_nuclei`, `train/matches_per_region`, and
`train/empty_intersections` to track the population. If fewer than two matches
exist in the entire batch, SSL contributes zero and `train/ssl_skipped=1`;
the continuous probe can still learn. Region-pooled controls average within
each nonempty intersection before projection, giving equal weight per region.

```text
ssl_loss = alignment + regularizer_weight * regularizer
training_loss = ssl_loss + probe_loss
```

`model.regularizer` selects `sigreg`, `visreg`, or `vicreg`. Default weights are
0.1 for SIGReg/VISReg and 1.0 for the normalized VICReg variance/covariance ratio
of 1:1/25. These are starting weights, not equivalent calibrated loss scales.
`model.alignment_level=region` is an optional same-matched-support pooled control.

The original linear probe learns **continuously**, with an optimizer update
alongside every SSL step. Its labeled examples are fixed-count training crops
(`data.datasets.labeled.dataset.crop_size=4096` in the supplied configs), with
existing positive-slide and positive-seed sampling. Training slides too small
for that count are excluded; graph/duplicate degeneracy fails explicitly rather
than silently returning a shorter training crop.

Labeled embeddings are computed with the encoder in eval mode and detached:
probe labels update only the probe, including no changes to encoder BatchNorm
statistics. There is no fresh-probe fitting or probe reset during validation.
Missing cancer-slide annotation IDs are ignored, not treated as negatives.
The existing benign-slide convention (all zero labels) is retained.

**Only validation uses whole slides.** The validation dataset has `crop_size=None`,
batch size one, no positive-seed sampling, no random augmentation, and a dynamic
sequence length. Every nucleus is encoded; all known labels are scored once.
Metrics are pooled nucleus-level metrics. `mode=test` currently evaluates the
same validation split and is not an independent final-test estimate.

Whole-slide validation uses the existing block attention implementation. Its
mask construction has quadratic storage in the number of blocks; inference
activation memory also grows with slide size. Actual full-slide H100 capacity
must be measured, especially for million-nucleus slides. It never silently
switches to crop validation or truncates a slide to fit the document-ID capacity.

## Running

Override paths on your training machine as needed:

```bash
uv run python -m nfm +experiment=nucleus_ssl \
  data.datasets.unlabeled.slides_path=/path/to/ssl_manifest.parquet \
  data.datasets.unlabeled.dataset.nuclei_path=/path/to/cells \
  data.datasets.labeled.slides_path=/path/to/labeled_metadata.parquet \
  data.datasets.labeled.dataset.annotation_path=/path/to/annotations
```

The default labeled manifest is an MLflow artifact and needs the existing
connection. Training uses the original MLkit logging configuration. Compare
regularizers by changing just the regularizer and retaining all other settings:

```bash
uv run python -m nfm +experiment=nucleus_ssl model.regularizer=sigreg
uv run python -m nfm +experiment=nucleus_ssl model.regularizer=visreg
uv run python -m nfm +experiment=nucleus_ssl model.regularizer=vicreg
```

The portable config runs 10,000 steps and validates every 1,000 steps. Checkpoint
selection uses `validation/AUPRC` from the continuously trained probe. Repeat
with `seed=1` and `seed=2`; changing the seed also changes the labeled split.
For an initial three-step data/forward/backward/whole-slide smoke run:

```bash
uv run python -m nfm +experiment=nucleus_ssl +trainer.fast_dev_run=3
```

Reduce training memory without changing SSL crop size:

```bash
uv run python -m nfm +experiment=nucleus_ssl \
  data.datasets.unlabeled.batch_size=8
```

For checkpoint continuation, pass `checkpoint=/path/to/last.ckpt` with the same
model/data configuration. New checkpoints include the probe, projector, and
encoder. Old eight-crop or retired-pilot checkpoints are not compatible with
the current training layout. Checkpoints from the earlier 69-input variant
also have incompatible input weights; start a new run.

## Checks

```bash
uv run --with pytest python -m pytest tests/test_nucleus_experiment.py tests/test_block_mask.py -q
```

On the H100, require compiled CUDA correctness checks instead of allowing skips:

```bash
NFM_REQUIRE_CUDA=1 uv run --with pytest python -m pytest \
  tests/test_block_mask.py \
  tests/test_nucleus_experiment.py::test_cuda_flex_matches_attention_oracle -q -rs
```

Tests cover same-nucleus correspondence through sorting/deletion, all three
objectives, continuous-probe isolation, fixed labeled training crops, whole-slide
validation exceeding training crop sizes, checkpoint loading/resume, input
BatchNorm, and the independent sparse-attention oracle. They use synthetic stores;
real-data quality and GPU capacity still require the remote experiment.

The production pipeline uses FlexAttention exclusively. CPU model tests replace
only the attention kernel with explicit test-only attention arithmetic; compiled
FlexAttention forward/backward checks require CUDA.
