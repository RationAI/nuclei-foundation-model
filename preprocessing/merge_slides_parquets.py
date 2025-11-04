import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATASET_REGEX = re.compile(r"dataset=([^/]+)")
ORGAN_REGEX = re.compile(r"organ=([^/]+)")
SLIDE_ID_REGEX = re.compile(r"slide_id=([^/]+)")


def get_file_info_with_partitions(file_path):
    parquet_file = pq.ParquetFile(file_path)
    dataset_match = DATASET_REGEX.search(file_path)
    organ_match = ORGAN_REGEX.search(file_path)
    slide_id_match = SLIDE_ID_REGEX.search(file_path)

    return {
        "num_rows": parquet_file.metadata.num_rows,
        "dataset": dataset_match.group(1),
        "organ": organ_match.group(1),
        "id": slide_id_match.group(1),
    }


# Read the slides table
slides_df = pq.read_table(
    "/flash/project_465002057/nuclei/slides",
    partitioning="hive",
).to_pandas()

all_files = list(Path("/flash/project_465002057/nuclei/cells").rglob("slide_id=*"))
print(f"Found {len(all_files)} parquet files to process.")

results_list = []
max_workers = 32  # Adjust based on your system's (network) I/O capacity

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_file = {
        executor.submit(get_file_info_with_partitions, file): file for file in all_files
    }

    # Process results as they complete
    for i, future in enumerate(as_completed(future_to_file)):
        result = future.result()
        results_list.append(result)

        # Optional: Print progress
        if (i + 1) % 10000 == 0:
            print(f"Processed {i+1}/{len(all_files)} files...")


counts_df = pd.DataFrame(results_list)

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
