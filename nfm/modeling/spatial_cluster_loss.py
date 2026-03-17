import torch
from torch import Tensor, nn


class SpatialClusterLoss(nn.Module):
    def __init__(self, lambda_flat=1.0, lambda_cov=1.0, lambda_div=1.0, lambda_bin=1.0):
        """
        Args:
            lambda_flat: Weight for the Graph Total Variation loss (flat regions).
            lambda_cov: Weight for the Coverage Hinge loss (sum >= 1).
            lambda_div: Weight for the Covariance penalty (uncorrelated dimensions).
            lambda_bin: Weight for the Binarization / Entropy loss (push to 0 or 1).
        """
        super().__init__()
        self.lambda_flat = lambda_flat
        self.lambda_cov = lambda_cov
        self.lambda_div = lambda_div
        self.lambda_bin = lambda_bin

    def graph_total_variation(self, P: Tensor, edge_index: Tensor) -> Tensor:
        """Graph Total Variation Loss: Penalizes differences in probabilities between neighboring points.

        Encourages spatial smoothness and discourages flat regions where all points have the same
        cluster assignment.

        Args:
            P: Tensor of shape (B, M, N) containing probabilities in [0, 1].
            edge_index: Tensor of shape (B, E, 2) representing the edges between points in each batch.
        """
        B = P.shape[0]
        E = edge_index.shape[1]
        batch_idx = torch.arange(B, device=P.device).view(B, 1).expand(B, E)

        # Extract source and destination node indices: shape (B, E)
        src_idx = edge_index[:, :, 0]
        dst_idx = edge_index[:, :, 1]

        # Gather probabilities: shape (B, E, N)
        p_i = P[batch_idx, src_idx, :]
        p_j = P[batch_idx, dst_idx, :]

        # Sum over dimensions (N), average over edges (E) and batch (B)
        return torch.mean(torch.sum(torch.abs(p_i - p_j), dim=2))

    def coverage_loss(self, P: Tensor) -> Tensor:
        """Coverage Loss: Penalizes points that are not sufficiently covered by clusters.

        Encourages the sum of probabilities for each point to be at least 1.0.

        Args:
            P: Tensor of shape (B, M, N) containing probabilities in [0, 1].
        """
        sum_p = torch.sum(P, dim=-1)
        return torch.mean(torch.clamp(1.0 - sum_p, min=0.0))

    def covariance_penalty(self, P: Tensor) -> Tensor:
        """Covariance Penalty: Encourages different cluster dimensions to be uncorrelated.

        Computes the covariance matrix of the probabilities across points and penalizes
        the squared sum of off-diagonal elements.

        Args:
            P: Tensor of shape (M, N) containing probabilities in [0, 1].
        """
        _, M, N = P.shape
        mu = torch.mean(P, dim=1, keepdim=True)

        # Mean-centered probabilities: shape (B, M, N)
        P_centered = P - mu

        # Batched covariance matrix: (B, N, M) @ (B, M, N) -> (B, N, N)
        cov_matrix = torch.einsum("bmi,bmj->bij", P_centered, P_centered) / (M - 1.0)

        # Create a boolean mask for off-diagonal elements: shape (1, N, N)
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=P.device).unsqueeze(0)

        # Square the off-diagonal elements, sum over NxN, average over B
        return torch.mean(torch.sum((cov_matrix * off_diag_mask) ** 2, dim=(1, 2)))

    def cluster_certainty(self, P: Tensor) -> Tensor:
        """Cluster Certainty Loss: Encourages probabilities to be close to 0 or 1.

        Uses a parabola that peaks at P=0.5 and is zero at P=0 and P=1.

        Args:
            P: Tensor of shape (B, M, N) containing probabilities in [0, 1].
        """
        return torch.mean(P * (1.0 - P))

    def forward(self, P: Tensor, edge_index: Tensor):
        """
        Args:
            P: Tensor of shape (B, M, N) containing probabilities in [0, 1].
               B is the batch size, M is the number of points per batch, N is the number of dimensions/clusters.
            edge_index: Tensor of shape (B, E, 2) representing the spatial
                        KNN or radius graph edges between the M points in each batch.

        Returns:
            total_loss: The combined scalar loss.
            loss_dict: Dictionary containing the individual detached loss components
                       for logging/monitoring.
        """
        l_tv = self.graph_total_variation(P, edge_index)
        l_cov = self.coverage_loss(P)
        l_div = self.covariance_penalty(P)
        l_bin = self.cluster_certainty(P)

        total_loss = (
            self.lambda_flat * l_tv
            + self.lambda_cov * l_cov
            + self.lambda_div * l_div
            + self.lambda_bin * l_bin
        )

        loss_dict = {
            "loss_flat": l_tv.item(),
            "loss_cov": l_cov.item(),
            "loss_div": l_div.item(),
            "loss_bin": l_bin.item(),
        }

        return total_loss, loss_dict
