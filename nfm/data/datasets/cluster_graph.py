import tempfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch
from mlflow.artifacts import download_artifacts
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import ClusterData

from nfm.typing import Transforms


class ClusterGraph(ConcatDataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        metadata_uri: str,
        graphs_uri: str,
        subgraph_size: int,
        pre_transforms: Transforms,
    ) -> None:
        """Initialize the dataset.

        Args:
            metadata_uri: MLFlow URI to dataset metadata.
            graphs_uri: MLFlow URI to graph .pt files.
            subgraph_size: Number of nodes in each subgraph.
            pre_transforms: Optional transforms applied before graph clustering.
        """
        self.slides = pd.read_parquet(download_artifacts(metadata_uri))
        self._load_graphs(graphs_uri)

        self.subgraph_size = subgraph_size
        self.pre_transforms = pre_transforms
        super().__init__(self.generate_datasets())

    def _load_graphs(self, graphs_uri: str | None) -> None:
        graphs = list(Path(download_artifacts(graphs_uri)).rglob("*.pt"))
        self.graphs = {graph.stem: graph for graph in graphs}

    def generate_datasets(self) -> Iterable[Dataset[tuple[Tensor, Tensor]]]:
        return [
            METISClusterDataset(
                path=self.graphs[slide.slide_id],
                subgraph_size=self.subgraph_size,
                pre_transforms=self.pre_transforms,
            )
            for slide in self.slides.itertuples()
        ]


class METISClusterDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self, path: Path, subgraph_size: int, pre_transforms: Transforms
    ) -> None:
        """Initialize the dataset.

        Args:
            path (Path): Graph file path.
            subgraph_size (int): Number of nodes in each partitioned subgraph.
            pre_transforms (None | Transforms, optional): Transforms used before partitioning. Defaults to None.
        """
        super().__init__()
        self.path = path
        self.subgraph_size = subgraph_size
        self.pre_transforms = pre_transforms
        self.generate()
        self.subgraphs = torch.load(self.file, weights_only=False, mmap=True)

    def generate(self) -> None:
        """Creates METIS clusters from the graph."""
        data = torch.load(self.path, weights_only=False, mmap=True)
        data = self.pre_transforms(data)

        data = ClusterData(
            data,
            num_parts=max(2, data.x.shape[0] // self.subgraph_size),
            log=False,
        )

        data = Batch.from_data_list(
            [Data(x=cluster.x, pos=cluster.pos) for cluster in data]
        )

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            torch.save(data, temp_file.name)
            self.file = temp_file.name

    def __len__(self) -> int:
        return len(self.subgraphs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.subgraphs.get_example(idx)
