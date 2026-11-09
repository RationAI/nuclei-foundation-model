# Nuclei Foundational Model

[LeJEPA](https://github.com/galilai-group/lejepa) self-supervised training directly on nuclei. Each nucleus is a token: its contour as elliptic Fourier descriptors, its centroid as a position in micrometers encoded with RoVE. A transformer with a class token encodes each view, and all views of a batch are packed into one sequence with full attention within each view (flex attention with a document mask).

- **Views**: Virchow2-style global and local crops of one context at a random field of view, augmented on the polygons (rotation, flips, scaling, nucleus drop, jitter, shape noise).
- **Objectives**: LeJEPA on the class tokens, and an iBOT-style term where masked nuclei predict the same nucleus in the other global view.
- **Probes**: online, detached probes on PANDA Radboud, one on the class token (a tile is positive if at least 50 % of its nuclei are) and one on each nucleus token.

```bash
uv sync
uv run python -m nfm +experiment=lejepa
```

| Module | Role |
| --- | --- |
| `nfm/data/nuclei.py` | Contours, shape descriptors, view crops |
| `nfm/data/augmentations.py` | Polygon augmentations |
| `nfm/data/packing.py` | Packing views into one sequence, masking, nucleus pairs |
| `nfm/modeling/transformer.py` | Transformer, document block mask |
| `nfm/ssl_meta_arch.py` | Objectives and probes |
