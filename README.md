# Nuclei Foundational Model

[LeJEPA](https://github.com/galilai-group/lejepa) self-supervised training on grayscale tiles rendered from nuclei polygons, with an online linear probe on PANDA Radboud (a tile is positive if at least 50 % of its nuclei are).

```bash
uv sync
uv run python -m nfm +experiment=lejepa
```
