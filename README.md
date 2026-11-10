# Nuclei Foundational Model

[LeJEPA](https://github.com/galilai-group/lejepa) self-supervised training directly on nuclei. Each nucleus is a token: its contour as distances from the centroid at 64 fixed angles (or elliptic Fourier descriptors), embedded by an MLP, and its centroid as a position in micrometers encoded with RoVE. A transformer with a class token encodes each view, and all views of a batch are packed into one sequence with full attention within each view (flex attention with a document mask).

- **Views**: random square crops of one 128 um field, covering 30-100 % of its area, augmented on the polygons: rotation, flips and zoom of the view, nucleus drop, dropout holes, and per-nucleus shape noise and rescaling.
- **Objective**: LeJEPA on the class tokens, every view predicts the mean of all views.
- **Probes**: online, detached probes on PANDA Radboud, one on the class token (a tile is positive if at least 50 % of its nuclei are) and one on each nucleus token. Validation divides each validation slide entirely into 128 um tiles and reports AUROC, AUPRC, precision and recall for both.

```bash
uv sync
uv run python -m nfm +experiment=lejepa
# the first setup: Fourier descriptors, a linear embedding and weak augmentations
uv run python -m nfm +experiment=lejepa descriptor=efd augmentation=weak model.backbone.mlp_embed=false
# how easily views can be matched by nucleus statistics alone
uv run python scripts/view_retrieval.py augmentation=weak
```

| Module | Role |
| --- | --- |
| `nfm/data/nuclei.py` | Contours, shape features, view crops |
| `nfm/data/augmentations.py` | Polygon augmentations |
| `nfm/data/packing.py` | Packing views into one sequence |
| `nfm/modeling/transformer.py` | Transformer, document block mask |
| `nfm/ssl_meta_arch.py` | Objectives and probes |
