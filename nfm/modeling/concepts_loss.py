import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SAE(nn.Module):
    def __init__(self, dim: int, num_features: int, k: int = 32):
        """
        Args:
            dim: Dimension of the input embeddings.
            num_features: Total number of concepts/features in the dictionary.
            k: The strict number of features to activate per token.
        """
        super().__init__()
        self.dim = dim
        self.num_features = num_features
        self.k = k

        self.encoder = nn.Linear(dim, num_features)
        self.decoder = nn.Linear(num_features, dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Standard initialization to ensure healthy variance at the start."""
        bound_enc = 1.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.encoder.weight, -bound_enc, bound_enc)
        nn.init.uniform_(self.encoder.bias, -bound_enc, bound_enc)

        bound_dec = 1.0 / math.sqrt(self.num_features)
        nn.init.uniform_(self.decoder.weight, -bound_dec, bound_dec)
        nn.init.uniform_(self.decoder.bias, -bound_dec, bound_dec)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # 1. Get raw scores for all concepts
        pre_activations = self.encoder(x)

        # 2. Extract the Top-K values and their indices
        topk_vals, topk_indices = torch.topk(pre_activations, self.k, dim=-1)

        # 3. Apply ReLU (we only want positive concept activations)
        topk_acts = F.relu(topk_vals)

        # 4. Scatter the Top-K activations back into a zeroed tensor of shape (batch, num_features)
        features = torch.zeros_like(pre_activations)
        features.scatter_(-1, topk_indices, topk_acts)

        # 5. Reconstruct the original input
        x_reconstructed = self.decoder(features)

        return x_reconstructed, features

    def compute_loss(self, x: Tensor, x_reconstructed: Tensor) -> Tensor:
        """
        TopK requires no sparsity penalty. The sparsity is strictly enforced
        by the forward pass. We only minimize reconstruction error.
        """
        reconstruction_error = x - x_reconstructed
        mse_loss = torch.mean(reconstruction_error**2, dim=-1)

        return mse_loss.mean()


class SpatialConceptLoss(nn.Module):
    def __init__(self, dim: int, num_concepts: int):
        super().__init__()
        self.sae = SAE(dim, num_concepts)
        self.routing_logits = nn.Parameter(torch.randn(num_concepts, 2))

    def forward(self, embed: torch.Tensor, knn_indices: torch.Tensor):
        x_reconstructed, concepts = self.sae(embed)
        sae_loss = self.sae.compute_loss(embed, x_reconstructed)

        N, C = concepts.shape
        k = knn_indices.shape[1]
        device = concepts.device

        # ---------------------------------------------------------
        # THE SPEED FIX: Build a Sparse Adjacency Matrix
        # This replaces the 2GB (N, k, C) dense broadcasting tensors
        # ---------------------------------------------------------
        row = torch.arange(N, device=device).unsqueeze(1).expand(N, k).reshape(-1)
        col = knn_indices.reshape(-1)

        indices = torch.stack([row, col], dim=0)
        values = torch.ones(N * k, device=device)

        # A is an (N, N) sparse matrix.
        # A @ X efficiently sums the features of all neighbors for every node.
        A = torch.sparse_coo_tensor(
            indices, values, (N, N), dtype=torch.float32
        ).coalesce()

        # --- 1. Dispersion Loss (Moran's I) ---
        mu = concepts.mean(dim=0, keepdim=True)
        v_centered = concepts - mu

        # THE FIX: Upcast -> SpMM -> Downcast
        with torch.autocast(device_type=device.type, enabled=False):
            v_centered_f32 = v_centered.to(torch.float32)
            neighbor_sum_centered = torch.sparse.mm(A, v_centered_f32).to(
                concepts.dtype
            )

        covariance = (v_centered * neighbor_sum_centered).sum(dim=0)
        variance = (v_centered**2).sum(dim=0).clamp(min=1e-4)

        W = N * k
        morans_i = (N / W) * (covariance / variance)
        dispersion_loss = morans_i**2

        # --- 2. Clustering Loss (L2 / Dirichlet Energy) ---
        # THE FIX: Upcast -> SpMM -> Downcast
        with torch.autocast(device_type=device.type, enabled=False):
            concepts_f32 = concepts.to(torch.float32)
            neighbor_sum_raw = torch.sparse.mm(A, concepts_f32).to(concepts.dtype)

        sum_vi_vj = (concepts * neighbor_sum_raw).sum(dim=0)
        sum_vi_sq = (concepts**2).sum(dim=0)

        cluster_loss_l2 = (2 * k * sum_vi_sq - 2 * sum_vi_vj) / (N * k)

        # --- 3. Routing ---
        routing_weights = F.softmax(self.routing_logits, dim=-1)
        w_cluster = routing_weights[:, 0]
        w_disperse = routing_weights[:, 1]

        concept_spatial_loss = (w_cluster * cluster_loss_l2) + (
            w_disperse * dispersion_loss
        )

        return {
            "spatial_loss": concept_spatial_loss.mean(),
            "routing": w_cluster.mean(),
            "sae_loss": sae_loss,
        }
