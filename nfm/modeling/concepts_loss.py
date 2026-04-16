import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def rectangle(x):
    """Rectangle kernel function."""
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)


class JumpReLUFunction(torch.autograd.Function):
    """Implementation of JumpReLU with custom backward (STE) for both x and threshold."""

    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, output_grad):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        # Pseudo-derivative of the step function H(x - theta)
        ste_term = (1.0 / bandwidth) * rectangle((x - threshold) / bandwidth)

        # 1. Flow gradient through x:
        # d/dx [x * H(x - theta)] = H(x - theta) + x * delta(x - theta)
        x_grad = ((x > threshold).to(x.dtype) + x * ste_term) * output_grad

        # 2. Flow gradient through threshold:
        # d/d_theta [x * H(x - theta)] = -x * delta(x - theta)
        threshold_grad = (
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * output_grad
        )

        return x_grad, threshold_grad, None


class SAE(nn.Module):
    def __init__(self, dim: int, num_features: int, bandwidth: float = 1.0):
        super().__init__()
        self.dim = dim
        self.num_features = num_features
        self.bandwidth = bandwidth

        # Using nn.Linear replaces the manual W and b parameter matrices
        self.encoder = nn.Linear(dim, num_features)
        self.decoder = nn.Linear(num_features, dim)

        # Initial threshold set to 0.03
        self.log_threshold = nn.Parameter(torch.full((num_features,), math.log(0.03)))

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights using the specific uniform distributions."""
        # Encoder: U(-1/n_features, 1/n_features)
        bound_enc = 1.0 / self.num_features
        nn.init.uniform_(self.encoder.weight, -bound_enc, bound_enc)
        nn.init.uniform_(self.encoder.bias, -bound_enc, bound_enc)

        # Decoder: U(-1/(n_layers*d_model), 1/(n_layers*d_model))
        # d_out represents the combined n_layers * d_model size
        bound_dec = 1.0 / self.dim
        nn.init.uniform_(self.decoder.weight, -bound_dec, bound_dec)
        nn.init.uniform_(self.decoder.bias, -bound_dec, bound_dec)

    def forward(self, x):
        """Forward pass using nn.Linear layers."""
        # Pre-activations are now computed directly via the linear layer
        pre_activations = self.encoder(x)
        threshold = torch.exp(self.log_threshold)

        features = JumpReLUFunction.apply(pre_activations, threshold, self.bandwidth)

        # Reconstruct directly via the linear layer
        x_reconstructed = self.decoder(features)

        return x_reconstructed, features, pre_activations

    def compute_loss(
        self,
        x: Tensor,
        x_reconstructed: Tensor,
        features: Tensor,
        pre_activations: Tensor,
        sparsity_coefficient: float,
        pre_act_coeff: float = 3e-6,
    ):
        """Computes the combined CLT loss: MSE + Tanh Sparsity Penalty + Pre-Activation Loss."""
        # 1. Mean-Squared Error Reconstruction Loss (e.g., against MLP outputs)
        reconstruction_error = x - x_reconstructed
        reconstruction_loss = torch.mean(reconstruction_error**2, dim=-1)

        # 2. Tanh Sparsity Penalty
        sparsity_loss = sparsity_coefficient * torch.sum(torch.tanh(features), dim=-1)

        # 3. Pre-Activation Loss: sum(ReLU(-h_f)) to prevent dead features
        pre_act_loss = pre_act_coeff * torch.sum(F.relu(-pre_activations), dim=-1)

        # Return the batch-wise mean total loss
        return torch.mean(reconstruction_loss + sparsity_loss + pre_act_loss, dim=0)


class SpatialConceptLoss(nn.Module):
    def __init__(self, dim: int, num_concepts: int):
        super().__init__()
        self.sae = SAE(dim, num_concepts)
        self.routing_logits = nn.Parameter(torch.randn(num_concepts, 2))

    def forward(self, embed: torch.Tensor, knn_indices: torch.Tensor):
        x_reconstructed, concepts, pre_activations = self.sae(embed)
        sae_loss = self.sae.compute_loss(
            embed, x_reconstructed, concepts, pre_activations, sparsity_coefficient=0.1
        )

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
