from typing import Any

from transformers import PretrainedConfig


class Config(PretrainedConfig):
    model_type = "nfm"

    def __init__(
        self,
        dim: int = 384,
        hidden_dim: int = 384 * 4,
        num_heads: int = 12,
        num_layers: int = 24,
        **kwargs: Any,
    ) -> None:
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        super().__init__(**kwargs)
