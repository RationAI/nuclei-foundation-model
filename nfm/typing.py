from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


Input: type = ...  # TODO define the model input type

Outputs: type = ...  # TODO define the model output type

Transforms: type = list[BaseTransform] | BaseTransform

Sample: type = Data
