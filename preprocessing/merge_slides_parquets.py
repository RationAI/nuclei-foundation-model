from pathlib import Path

import pyarrow.parquet as pq

df = pq.read_table(
    "/flash/project_465002057/nuclei/slides",
    partitioning="hive",
).to_pandas()


def has_cells_dir(row):
    return Path(
        "/flash/project_465002057/nuclei/cells",
        f"organ={row.organ}",
        f"dataset={row.dataset}",
        f"slide_id={row.id}",
    ).is_dir()


df = df[df.apply(has_cells_dir, axis=1)]
df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

df.to_parquet("/flash/project_465002057/nuclei/slides.parquet", index=False)
