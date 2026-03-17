import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    """
    Reference:
        Taming Transformers for High-Resolution Image Synthesis
        https://arxiv.org/pdf/2012.09841.pdf
    """

    def __init__(self, n_e, e_dim, beta=1.0):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            # shape specifying (batch, height, width, channel)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
            (z_q - z.detach()) ** 2
        )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        min_encoding_indices = min_encoding_indices.view(z.shape[:-1])

        return z_q, loss, min_encoding_indices


class ResidualVectorQuantizer(nn.Module):
    """References:
    SoundStream: An End-to-End Neural Audio Codec
    https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, n_e, e_dim, num_quantizers):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.num_quantizers = num_quantizers
        self.vq_layers = nn.ModuleList(
            [VectorQuantizer(n_e, e_dim) for _ in range(num_quantizers)]
        )

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, z):
        all_losses = []
        all_min_encoding_indices = []

        z_q = 0
        residual = z
        for quantizer in self.vq_layers:
            z_res, loss, min_encoding_indices = quantizer(residual)
            residual = residual - z_res
            z_q = z_q + z_res

            all_losses.append(loss)
            all_min_encoding_indices.append(min_encoding_indices)

        mean_losses = torch.stack(all_losses).mean()
        all_min_encoding_indices = torch.stack(all_min_encoding_indices, dim=1)

        return z_q, mean_losses, all_min_encoding_indices
