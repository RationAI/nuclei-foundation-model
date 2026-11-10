# Nuclei Foundational Model: supervised baseline

The nuclei transformer of the `token-lejepa` branch, trained only on PANDA Radboud with tile-level supervision, to compare against the self-supervised model.

Each nucleus is a token: its contour as distances from the centroid at 64 fixed angles (or elliptic Fourier descriptors), embedded by an MLP, and its centroid as a position in micrometers encoded with RoVE. A transformer with a class token encodes each tile, and all tiles of a batch are packed into one sequence with full attention within each tile (flex attention with a document mask).

- **Training**: random 128 um tiles centered on nuclei, augmented as the self-supervised views (rotation, flips, zoom, nucleus drop, dropout holes, per-nucleus shape noise and rescaling). A tile is positive if at least 50 % of its remaining nuclei are. The tile head on the class token is trained end to end.
- **Nucleus probe**: detached, as in the self-supervised model, so nucleus labels never reach the encoder.
- **Validation**: identical to the self-supervised model, on the same slides for the same seed. Each validation slide is divided entirely into 128 um tiles, reporting AUROC, AUPRC, precision and recall for tiles and nuclei under the same metric names.

```bash
uv sync
uv run python -m nfm +experiment=supervised
uv run python -m nfm +experiment=supervised augmentation=weak descriptor=efd
```

| Module | Role |
| --- | --- |
| `nfm/data/nuclei.py` | Shape features, tile crops |
| `nfm/data/augmentations.py` | Polygon augmentations |
| `nfm/data/packing.py` | Packing tiles into one sequence |
| `nfm/modeling/transformer.py` | Transformer, document block mask |
| `nfm/supervised_meta_arch.py` | Tile head, nucleus probe, metrics |
