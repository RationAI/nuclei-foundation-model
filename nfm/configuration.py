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
        rope_theta: float = 10000,
        efd_order: int = 16,
        proj_dim: int = 128,
        proj_hidden_dim: int = 128 * 4,
        **kwargs: Any,
    ) -> None:
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.rope_theta = rope_theta
        self.efd_order = efd_order
        self.proj_dim = proj_dim
        self.proj_hidden_dim = proj_hidden_dim
        super().__init__(**kwargs)
