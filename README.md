# Nuclei Foundational Model

[LeJEPA](https://github.com/galilai-group/lejepa) self-supervised training directly on nuclei. Each nucleus is a token: its contour as distances from the centroid at 64 fixed angles (or elliptic Fourier descriptors), embedded by an MLP, and its centroid as a position in micrometers encoded with RoVE. Views of thousands of nuclei are encoded with neighborhood attention, and local class tokens summarize their surroundings.

- **Views**: the 1024-4096 nuclei closest to a center near a shared seed, so views overlap. Augmented on the polygons: rotation, flips and zoom of the view, nucleus drop, dropout holes, and for each nucleus an affine map and shape noise.
- **Local class tokens**: one per 32 nuclei, placed by farthest point sampling, each covering its 128 nearest nuclei.
- **Attention**: within a view, a nucleus attends to the blocks of 128 spatially sorted nuclei that hold its nearest neighbors, and to the class tokens that cover its block. A class token attends to the nuclei it covers and to itself, not to other class tokens. Block pairs of nuclei are full blocks of flex attention.
- **Objective**: LeJEPA on pairs of class tokens from different views that cover the same nuclei, by a Jaccard index of at least 0.4.
- **Probes**: online, detached probes on PANDA Radboud, one on the class tokens (a region is positive if at least 50 % of the nuclei the token covers are) and one on each nucleus token. Validation encodes every validation slide whole and reports AUROC, AUPRC, precision and recall for both.

```bash
uv sync
uv run python -m nfm +experiment=lejepa
# the first nucleus setup: Fourier descriptors, a linear embedding and weak augmentations
uv run python -m nfm +experiment=lejepa descriptor=efd augmentation=weak model.backbone.mlp_embed=false
# how easily paired class tokens can be matched by nucleus statistics alone
uv run python scripts/view_retrieval.py augmentation=weak
```

| Module | Role |
| --- | --- |
| `nfm/data/nuclei.py` | Contours, shape features, views |
| `nfm/data/neighborhoods.py` | Nearest neighbors, class tokens, class token pairs |
| `nfm/data/augmentations.py` | Polygon augmentations |
| `nfm/data/packing.py` | Spatial sorting and packing of views into one sequence |
| `nfm/modeling/block_mask.py` | Neighborhood attention mask |
| `nfm/modeling/transformer.py` | Transformer |
| `nfm/ssl_meta_arch.py` | Objectives and probes |
