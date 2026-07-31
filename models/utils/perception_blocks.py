"""
perception_blocks.py
====================
Unified perception building blocks for Helioskrill BEV pipeline.

Contains:
- ResidualBlock2DGroupNorm: 2D residual block with GroupNorm for numerical stability at small batch sizes.
- DINOv2EncoderLoRA: Meta DINOv2 Small backbone with LoRA fine-tuning on MHSA layers.
- CameraBEVProjectionV2: IPM-based projection from 2D camera feature maps to 3D Bird's Eye View space.
- CameraBEVNeck: Lightweight convolutional neck for post-projection BEV refinement.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# =============================================================================
# Convolutional Building Blocks
# =============================================================================

class ResidualBlock2DGroupNorm(nn.Module):
    """
    2D Residual block using GroupNorm instead of BatchNorm2d for numerical stability with small batch sizes (B=1).
    """
    def __init__(self, in_channels, out_channels, stride=1, num_groups=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_channels)
            )

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# =============================================================================
# DINOv2 Visual Backbone with LoRA Adaptation
# =============================================================================

class DINOv2EncoderLoRA(nn.Module):
    """
    DINOv2 Small (dinov2_vits14) feature extractor with traditional LoRA adaptation.
    Outputs 384-channel 2D spatial feature maps from input RGB images.
    """
    def __init__(self, model_name: str = "dinov2_vits14", patch_size: int = 14, lora_r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = 384  # dinov2_vits14 embedding dimension

        # Load pre-trained DINOv2 backbone from PyTorch Hub
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)

        # Freeze original backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Configure traditional LoRA for Multi-Head Self-Attention layers
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],
            lora_dropout=0.05,
            bias="none"
        )
        self.backbone = get_peft_model(self.backbone, lora_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B_total, 3, H, W] (where B_total = B * N)
        Returns:
            feature_map_2d: Tensor of shape [B_total, 384, H_grid, W_grid]
        """
        B_total, C, H, W = x.shape

        # Ensure height and width are divisible by DINOv2 patch size (14)
        pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            H_padded, W_padded = H + pad_h, W + pad_w
        else:
            H_padded, W_padded = H, W

        H_grid = H_padded // self.patch_size
        W_grid = W_padded // self.patch_size

        # Extract patch features using DINOv2 forward_features
        features_dict = self.backbone.forward_features(x)
        patch_features = features_dict["x_norm_patchtokens"]  # Shape: [B_total, H_grid * W_grid, 384]

        # Reshape flat patch tokens to 2D spatial feature map [B_total, 384, H_grid, W_grid]
        feature_map_2d = patch_features.permute(0, 2, 1).contiguous().view(B_total, self.embed_dim, H_grid, W_grid)
        return feature_map_2d


# =============================================================================
# Camera BEV Projection (IPM-based)
# =============================================================================

class CameraBEVProjectionV2(nn.Module):
    """
    Projects 2D feature maps (64 channels) into a unified 3D Bird's Eye View (BEV) space
    using Inverse Perspective Mapping (IPM) with camera intrinsics and extrinsics.
    """
    def __init__(self, bev_height=400, bev_width=400, grid_resolution=0.25, z_slices=[-1.0, 0.0, 1.0, 2.0]):
        super().__init__()
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.grid_resolution = grid_resolution
        self.z_slices = z_slices

        x_coords = torch.linspace(-bev_height / 2 * grid_resolution, bev_height / 2 * grid_resolution, bev_height)
        y_coords = torch.linspace(-bev_width / 2 * grid_resolution, bev_width / 2 * grid_resolution, bev_width)
        z_coords = torch.tensor(z_slices)

        grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1)

        self.register_buffer("grid_points", grid_points)

    def forward(self, cam_features, extrinsics, intrinsics, original_img_size=(406, 308)):
        B, N, C, H_feat, W_feat = cam_features.shape
        W_orig, H_orig = original_img_size

        grid_pts = self.grid_points.unsqueeze(0).unsqueeze(1).repeat(B, N, 1, 1, 1, 1)
        grid_pts_flat = grid_pts.view(B, N, -1, 3)

        ones = torch.ones_like(grid_pts_flat[..., :1])
        pts_homo = torch.cat([grid_pts_flat, ones], dim=-1)

        pts_cam = torch.matmul(extrinsics.unsqueeze(2), pts_homo.unsqueeze(-1)).squeeze(-1)[..., :3]

        z = pts_cam[..., 2:3]
        z_valid = z > 0.1

        pts_2d_homo = torch.matmul(intrinsics.unsqueeze(2), pts_cam.unsqueeze(-1)).squeeze(-1)
        u = pts_2d_homo[..., 0] / torch.clamp(pts_2d_homo[..., 2], min=1e-5)
        v = pts_2d_homo[..., 1] / torch.clamp(pts_2d_homo[..., 2], min=1e-5)

        u_norm = (u / W_orig) * 2.0 - 1.0
        v_norm = (v / H_orig) * 2.0 - 1.0

        in_bounds = (u_norm >= -1.0) & (u_norm <= 1.0) & (v_norm >= -1.0) & (v_norm <= 1.0) & z_valid.squeeze(-1)

        grid_uv = torch.stack([u_norm, v_norm], dim=-1)

        cam_feat_flat = cam_features.view(B * N, C, H_feat, W_feat)
        grid_uv_flat = grid_uv.view(B * N, -1, 1, 2)

        sampled_feat_flat = F.grid_sample(cam_feat_flat, grid_uv_flat, mode='bilinear', align_corners=False)
        sampled_feat = sampled_feat_flat.squeeze(-1).view(B, N, C, self.bev_height, self.bev_width, len(self.z_slices))

        in_bounds_expanded = in_bounds.view(B, N, 1, self.bev_height, self.bev_width, len(self.z_slices)).float()
        sampled_feat = sampled_feat * in_bounds_expanded

        bev_feat = sampled_feat.sum(dim=(1, 5))
        return bev_feat


# =============================================================================
# Camera-Only BEV Neck (Replaces Camera+LiDAR fusion neck)
# =============================================================================

class CameraBEVNeck(nn.Module):
    """
    Lightweight convolutional neck that refines camera-only BEV features (64ch -> 128ch).
    Replaces the previous BEVFusionNeckGroupNorm that required LiDAR input.
    """
    def __init__(self, in_channels=64, out_channels=128, num_groups=16):
        super().__init__()
        self.neck = nn.Sequential(
            ResidualBlock2DGroupNorm(in_channels, out_channels, stride=1, num_groups=num_groups),
            ResidualBlock2DGroupNorm(out_channels, out_channels, stride=1, num_groups=num_groups)
        )

    def forward(self, cam_bev):
        return self.neck(cam_bev)
