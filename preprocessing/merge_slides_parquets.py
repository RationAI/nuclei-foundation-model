import pyarrow as pa
import pyarrow.parquet as pq

# Read the slides table
slides_df = pq.read_table(
    "/flash/project_465002057/nuclei/slides",
    partitioning="hive",
).to_pandas()

# Read the cells dataset metadata
cells_dataset = pq.ParquetDataset(
    "/flash/project_465002057/nuclei/cells", partitioning="hive"
)

# Get row counts for each partition (slide) in the cells dataset
counts = [
    {
        "organ": p.partition_keys["organ"],
        "dataset": p.partition_keys["dataset"],
        "id": p.partition_keys["slide_id"],
        "num_rows": p.count_rows(),
    }
    for p in cells_dataset.fragments
]

# Convert counts to a DataFrame
counts_df = pa.Table.from_pylist(counts).to_pandas()

# Filter for slides with more than 5000 cells
valid_slides_df = counts_df[counts_df["num_rows"] > 5000]

# Merge with the original slides DataFrame to keep only valid slides
# Use a left merge with indicator=True to find matches
merged_df = slides_df.merge(
    valid_slides_df[["organ", "dataset", "id"]],
    on=["organ", "dataset", "id"],
    how="left",
    indicator=True,
)

# Keep only the rows that were present in both DataFrames
df = merged_df[merged_df["_merge"] == "both"].drop(columns=["_merge"])

df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

df.to_parquet("/flash/project_465002057/nuclei/slides.parquet", index=False)
