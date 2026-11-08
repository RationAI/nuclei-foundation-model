"""LeJEPA with an iBOT-style masked patch objective and a detached Radboud probe."""

import math
from typing import Any

import timm
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchvision.ops import MLP

from nfm.modeling.sigreg import SIGReg


class SSLMetaArch(LightningModule):
    def __init__(
        self,
        backbone: str = "vit_small_patch14_reg4_dinov2",
        image_size: int = 224,
        embed_dim: int = 512,
        projector_dim: int = 16,
        drop_path_rate: float = 0.4,
        lamb: float = 0.02,
        ibot_weight: float = 1.0,
        mask_ratio: tuple[float, float] = (0.1, 0.5),
        mask_probability: float = 0.5,
        sigreg_patches: int = 4096,
        lr: float = 5e-4,
        probe_lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.lamb = lamb
        self.ibot_weight = ibot_weight
        self.mask_ratio = tuple(mask_ratio)
        self.mask_probability = mask_probability
        self.sigreg_patches = sigreg_patches
        self.lr = lr
        self.probe_lr = probe_lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.backbone = timm.create_model(
            backbone,
            pretrained=False,
            in_chans=1,
            img_size=image_size,
            dynamic_img_size=True,  # for local views
            num_classes=embed_dim,
            drop_path_rate=drop_path_rate,
        )
        self.projector = MLP(
            embed_dim, [2048, 2048, projector_dim], norm_layer=nn.BatchNorm1d
        )
        dim = self.backbone.embed_dim
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.patch_projector = MLP(
            dim, [2048, 2048, projector_dim], norm_layer=nn.BatchNorm1d
        )
        self.sigreg = SIGReg()
        self.probe = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.val_metrics = MetricCollection(
            {"AUROC": BinaryAUROC(), "AUPRC": BinaryAveragePrecision()},
            prefix="validation/",
        )

    def encode(self, images: Tensor, mask: Tensor | None = None) -> Tensor:
        """Returns all tokens, with masked patches replaced by the mask token."""
        b = self.backbone
        x = b.patch_embed((images.float() / 255 - 0.5) / 0.5)  # [batch, h, w, dim]
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        x = b.norm_pre(b.patch_drop(b._pos_embed(x)))
        return b.norm(b.blocks(x))

    def forward(self, images: Tensor) -> Tensor:
        return self.backbone.forward_head(self.encode(images))

    def corresponding_tokens(
        self, tokens: Tensor, source: Tensor, target: Tensor, grid: int
    ) -> tuple[Tensor, Tensor]:
        """Samples target-view tokens at the source-view patch centers.

        Args:
            tokens: Patch tokens of the target views, [n, grid * grid, dim].
            source: Context to source-view pixel maps, [n, 2, 3].
            target: Context to target-view pixel maps, [n, 2, 3].
            grid: Patches per side.

        Returns:
            Tokens aligned with the source patches and whether each patch is visible
            in the target view.
        """
        size = grid * self.backbone.patch_embed.patch_size[0]
        bottom = source.new_tensor([0, 0, 1]).expand(len(source), 1, 3)
        to_context = torch.linalg.inv(torch.cat([source.float(), bottom], 1))
        to_target = target.float() @ to_context  # source px -> target px, [n, 2, 3]
        centers = (torch.arange(grid, device=self.device) + 0.5) * size / grid
        y, x = torch.meshgrid(centers, centers, indexing="ij")
        points = torch.stack([x, y, torch.ones_like(x)]).flatten(1)  # [3, grid^2]
        coords = (to_target @ points).transpose(1, 2) / size * 2 - 1  # [-1, 1]
        maps = tokens.float().transpose(1, 2).unflatten(2, (grid, grid))
        sampled = F.grid_sample(maps, coords[:, None], align_corners=False)
        return sampled[:, :, 0].transpose(1, 2), (coords.abs() < 1).all(-1)

    def random_mask(self, n: int, grid: int) -> Tensor:
        """Blob-shaped masks: the lowest values of smoothed noise, iBOT-style ratios."""
        # Valid pooling over a padded grid keeps the statistics equal at the borders.
        noise = torch.rand(n, 1, grid + 4, grid + 4, device=self.device)
        noise = F.avg_pool2d(noise, 5, 1).flatten(1)
        ratio = torch.empty(n, device=self.device).uniform_(*self.mask_ratio)
        ranks = noise.argsort(1).argsort(1)
        return (ranks < (ratio * grid**2).long()[:, None]).view(n, grid, grid)

    def training_step(self, batch: dict[str, Any]) -> Tensor:
        # Both are [batch, views, height, width], uint8. Reordered to views first.
        global_views = batch["unlabeled"]["global"].transpose(0, 1)
        local_views = batch["unlabeled"]["local"].transpose(0, 1)
        n, v = global_views.size(1), global_views.size(0) + local_views.size(0)
        global_images = global_views.flatten(0, 1).unsqueeze(1)
        global_tokens = self.encode(global_images)
        global_embeddings = self.backbone.forward_head(global_tokens)
        local_embeddings = self(local_views.flatten(0, 1).unsqueeze(1))
        embeddings = torch.cat([global_embeddings, local_embeddings])
        proj = self.projector(embeddings).reshape(v, n, -1)  # [views, batch, dim]
        # All views predict the center of the global views.
        centers = proj[: global_views.size(0)].mean(0)
        invariance = (centers - proj).square().mean()
        sigreg = self.sigreg(proj)
        lejepa = self.lamb * sigreg + (1 - self.lamb) * invariance

        # iBOT-style: masked patches predict the tokens at the same tissue location in
        # the other, unmasked global view. The views are shifted and flipped differently,
        # so the patch position cannot be the answer. SIGReg on the patch projections
        # prevents collapse without a teacher.
        prefix = self.backbone.num_prefix_tokens
        grid = int((global_tokens.size(1) - prefix) ** 0.5)
        # As in DINOv2, only a part of the global views is masked.
        count = round(self.mask_probability * len(global_images))
        chosen = torch.randperm(len(global_images), device=self.device)[:count]
        partner = (chosen + n) % len(global_images)  # same sample, next global view
        affines = batch["unlabeled"]["global_affine"].transpose(0, 1).flatten(0, 1)
        targets, valid = self.corresponding_tokens(
            global_tokens[partner, prefix:], affines[chosen], affines[partner], grid
        )
        mask = self.random_mask(count, grid)
        masked_tokens = self.encode(global_images[chosen], mask)
        selected = mask.flatten(1) & valid
        patches = torch.cat([targets[selected], masked_tokens[:, prefix:][selected]])
        target, prediction = self.patch_projector(patches).chunk(2)
        patch_prediction = (target - prediction).square().mean()
        subset = torch.randperm(len(patches), device=self.device)[: self.sigreg_patches]
        patch_sigreg = self.sigreg(torch.cat([target, prediction])[subset][None])
        ibot = self.lamb * patch_sigreg + (1 - self.lamb) * patch_prediction

        # Labels must not reach the encoder, and the probe sees no drop path.
        self.backbone.eval()
        with torch.no_grad():
            features = self(batch["labeled"]["images"])
        self.backbone.train()
        logits = self.probe(features).squeeze(1)
        probe = F.binary_cross_entropy_with_logits(logits, batch["labeled"]["labels"])

        self.log_dict(
            {
                "train/lejepa_loss": lejepa,
                "train/invariance_loss": invariance,
                "train/sigreg_loss": sigreg,
                "train/ibot_loss": ibot,
                "train/patch_prediction_loss": patch_prediction,
                "train/patch_sigreg_loss": patch_sigreg,
                "train/predicted_patches": selected.float().sum(1).mean(),
                "train/probe_loss": probe,
                "train/embedding_std": global_embeddings.detach().std(0).mean(),
            },
            batch_size=n,
            prog_bar=True,
        )
        return lejepa + self.ibot_weight * ibot + probe

    def validation_step(self, batch: dict[str, Tensor]) -> None:
        logits = self.probe(self(batch["images"])).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
        self.val_metrics.update(logits.sigmoid(), batch["labels"].long())
        self.log("validation/probe_loss", loss, batch_size=len(logits), sync_dist=True)
        self.log_dict(self.val_metrics, batch_size=len(logits))

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        *self.backbone.parameters(),
                        *self.projector.parameters(),
                        *self.patch_projector.parameters(),
                        self.mask_token,
                    ],
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": self.probe.parameters(),
                    "lr": self.probe_lr,
                    "weight_decay": 1e-7,
                },
            ],
        )
        total = int(self.trainer.estimated_stepping_batches)

        def schedule(step: int) -> float:
            if step < self.warmup_steps:
                return 0.01 + 0.99 * step / self.warmup_steps
            progress = (step - self.warmup_steps) / (total - self.warmup_steps)
            return 1e-3 + (1 - 1e-3) * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
