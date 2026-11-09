# Nuclei Foundational Model

[LeJEPA](https://github.com/galilai-group/lejepa) self-supervised training directly on nuclei. Each nucleus is a token: its contour as elliptic Fourier descriptors, its centroid as a position in micrometers encoded with RoVE. A transformer with a class token encodes each view, and all views of a batch are packed into one sequence with full attention within each view (flex attention with a document mask).

- **Views**: random square crops of one 128 um field, covering 30-100 % of its area, augmented on the polygons (rotation, flips, dropout holes, shape noise).
- **Objective**: LeJEPA on the class tokens, every view predicts the mean of all views.
- **Probes**: online, detached probes on PANDA Radboud, one on the class token (a tile is positive if at least 50 % of its nuclei are) and one on each nucleus token. Validation divides each validation slide entirely into 128 um tiles and reports AUROC, AUPRC, precision and recall for both.

```bash
uv sync
uv run python -m nfm +experiment=lejepa
```

| Module | Role |
| --- | --- |
| `nfm/data/nuclei.py` | Contours, shape descriptors, view crops |
| `nfm/data/augmentations.py` | Polygon augmentations |
| `nfm/data/packing.py` | Packing views into one sequence |
| `nfm/modeling/transformer.py` | Transformer, document block mask |
| `nfm/ssl_meta_arch.py` | Objectives and probes |
