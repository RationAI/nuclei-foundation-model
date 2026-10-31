# Nuclei Foundation Model

Nucleus-level SSL runs through the original training pipeline:

```bash
uv run python -m nfm +experiment=nucleus_ssl
```

It uses overlapping graph-grown crops, a continuously trained detached linear
probe on fixed-size labeled crops, and whole-slide validation. SIGReg, VISReg,
and VICReg are available. See [the run guide](docs/nucleus_ssl.md).

The reconstruction-based alternative runs with:

```bash
uv run python -m nfm --config-name=morphology
```

It trains denoising intrinsic features and masked contextual prediction, with
raw/intrinsic/contextual probe comparisons. See [the experiment guide](docs/morphology_pretraining.md).

The spatial-region experiment removes hidden nuclei entirely and predicts
multiscale counts, occupancy, and regional morphology:

```bash
uv run python -m nfm --config-name=regional
```

It exports nucleus and physical-region embeddings, with slide-held-out spatial
validation, matched random encoders, and a local-geometry control. See
[the regional experiment guide](docs/regional_pretraining.md).
