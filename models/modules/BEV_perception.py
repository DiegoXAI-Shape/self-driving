import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.blocks import VisionMambaEncoder, TemporalMamba
from models.modules.BEV_planning import BEVPlanningHead


# ==============================================================================
# 1. CAMERA 2D BACKBONE (Feature Extractor)
# ==============================================================================

class ResidualBlock2D(nn.Module):
    """
    Standard 2D residual block for feature extraction.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# ==============================================================================
# 2. LIDAR ENCODER (BEV Feature Extractor)
# ==============================================================================

class LidarBEVEncoder(nn.Module):
    """
    Processes a pre-projected LiDAR Bird's Eye View grid of shape [B, C_lidar, H_bev, W_bev].
    Expects C_lidar = 5 statistical feature channels: (Z_max, Z_diff, Z_mean, density, intensity).
    """
    def __init__(self, in_channels=5, feat_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer = nn.Sequential(
            ResidualBlock2D(32, 64, stride=1),
            ResidualBlock2D(64, feat_channels, stride=1)
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer(out)
        return out


# ==============================================================================
# 3. 2D-TO-BEV PROJECTION MODULE
# ==============================================================================

class CameraBEVProjection(nn.Module):
    """
    Projects 2D multi-view image features into a unified 3D Bird's Eye View (BEV) space
    using geometry-guided backward projection.
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

    def forward(self, cam_features, extrinsics, intrinsics, original_img_size=(800, 608)):
        """
        Projects multi-view 2D feature maps onto the precomputed BEV 3D grid.
        Args:
            cam_features: [B, N, C, H_feat, W_feat]
            extrinsics: [B, N, 4, 4]
            intrinsics: [B, N, 3, 3]
            original_img_size: (W, H)
        Returns:
            bev_features: [B, C, H_bev, W_bev]
        """
        B, N, C, H_feat, W_feat = cam_features.shape
        W_orig, H_orig = original_img_size
        
        grid_pts = self.grid_points.unsqueeze(0).unsqueeze(1).repeat(B, N, 1, 1, 1, 1)
        grid_pts_flat = grid_pts.view(B, N, -1, 3)
        
        ones = torch.ones_like(grid_pts_flat[..., :1])
        pts_homo = torch.cat([grid_pts_flat, ones], dim=-1)
        
        pts_cam = torch.matmul(extrinsics.unsqueeze(2), pts_homo.unsqueeze(-1)).squeeze(-1)
        
        z_cam = pts_cam[..., 2:3]
        z_cam_clamped = torch.clamp(z_cam, min=1e-3)
        pts_proj = torch.matmul(intrinsics.unsqueeze(2), pts_cam[..., :3].unsqueeze(-1)).squeeze(-1)
        
        u = pts_proj[..., 0:1] / z_cam_clamped
        v = pts_proj[..., 1:2] / z_cam_clamped
        
        u_norm = (u / (W_orig - 1)) * 2.0 - 1.0
        v_norm = (v / (H_orig - 1)) * 2.0 - 1.0
        
        grid_uv = torch.cat([u_norm, v_norm], dim=-1)
        
        cam_features_flat = cam_features.view(B * N, C, H_feat, W_feat)
        grid_uv_flat = grid_uv.view(B * N, self.bev_height * self.bev_width * len(self.z_slices), 1, 2)
        
        sampled_feats = F.grid_sample(cam_features_flat, grid_uv_flat, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_feats = sampled_feats.squeeze(-1).view(B, N, C, self.bev_height, self.bev_width, len(self.z_slices))
        
        valid_mask = (z_cam > 0.1) & (u_norm >= -1.0) & (u_norm <= 1.0) & (v_norm >= -1.0) & (v_norm <= 1.0)
        valid_mask = valid_mask.view(B, N, 1, self.bev_height, self.bev_width, len(self.z_slices)).float()
        
        sampled_feats = sampled_feats * valid_mask
        
        bev_feats = sampled_feats.sum(dim=1).mean(dim=-1)
        return bev_feats


# ==============================================================================
# 4. BEV FUSION NECK
# ==============================================================================

class BEVFusionNeck(nn.Module):
    """
    Fuses camera-projected BEV feature maps with processed LiDAR BEV features.
    """
    def __init__(self, cam_channels=64, lidar_channels=64, out_channels=128):
        super().__init__()
        in_channels = cam_channels + lidar_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.block1 = ResidualBlock2D(out_channels, out_channels, stride=1)
        self.block2 = ResidualBlock2D(out_channels, out_channels, stride=1)

    def forward(self, cam_bev, lidar_bev):
        x = torch.cat([cam_bev, lidar_bev], dim=1)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        return out


# ==============================================================================
# 5. MAIN SENSOR FUSION PERCEPTION NETWORK
# ==============================================================================

class BEVPerceptionNet(nn.Module):
    """
    End-to-end space-temporal multi-modal sensor fusion planning network.
    Inputs:
        - Camera sequence: [B, S, N, 3, H_img, W_img]
        - LiDAR BEV grid: [B, 5, H_bev, W_bev]
    Outputs:
        - Predicted Future Waypoints: [B, num_waypoints, 4] (rel_x, rel_y, rel_z, rel_yaw)
    """
    def __init__(self, num_waypoints=10, 
                 bev_height=400, bev_width=400, grid_resolution=0.25, img_size=(608, 800)):
        super().__init__()
        self.cam_backbone = VisionMambaEncoder(
            img_size=img_size,
            in_channels=3,
            L_blocks=4,
            D_hidden=64,
            N_ssm=32,
            Expand=3
        )
        self.lidar_backbone = LidarBEVEncoder(in_channels=5, feat_channels=64)
        self.projection = CameraBEVProjection(bev_height=bev_height, bev_width=bev_width, grid_resolution=grid_resolution)
        
        self.temporal_mamba = TemporalMamba(dim=64, L_blocks=2)
        self.fusion_neck = BEVFusionNeck(cam_channels=64, lidar_channels=64, out_channels=128)
        self.planning_head = BEVPlanningHead(in_channels=128, num_waypoints=num_waypoints)

    def forward(self, camera_imgs, lidar_bev, extrinsics, intrinsics, inference_params=None):
        if len(camera_imgs.shape) == 5:
            camera_imgs = camera_imgs.unsqueeze(1)
            
        B, S, N, C, H, W = camera_imgs.shape
        
        x_flat = camera_imgs.contiguous().view(B * S * N, C, H, W)
        
        pad_h = (16 - (H % 16)) % 16
        pad_w = (16 - (W % 16)) % 16
        if pad_h > 0 or pad_w > 0:
            x_flat = F.pad(x_flat, (0, pad_w, 0, pad_h))
        
        cam_features_2d_flat = self.cam_backbone(x_flat)
        
        C_feat, H_feat, W_feat = cam_features_2d_flat.shape[1], cam_features_2d_flat.shape[2], cam_features_2d_flat.shape[3]
        cam_features_2d = cam_features_2d_flat.view(B * S, N, C_feat, H_feat, W_feat)
        
        if len(extrinsics.shape) == 3:
            extrinsics = extrinsics.unsqueeze(0).unsqueeze(1).expand(B, S, -1, -1, -1)
        elif len(extrinsics.shape) == 4:
            extrinsics = extrinsics.unsqueeze(1).expand(-1, S, -1, -1, -1)
            
        if len(intrinsics.shape) == 3:
            intrinsics = intrinsics.unsqueeze(0).unsqueeze(1).expand(B, S, -1, -1, -1)
        elif len(intrinsics.shape) == 4:
            intrinsics = intrinsics.unsqueeze(1).expand(-1, S, -1, -1, -1)
            
        extrinsics_flat = extrinsics.contiguous().view(B * S, N, 4, 4)
        intrinsics_flat = intrinsics.contiguous().view(B * S, N, 3, 3)
        
        cam_features_bev_flat = self.projection(cam_features_2d, extrinsics_flat, intrinsics_flat, original_img_size=(W + pad_w, H + pad_h))
        cam_features_bev_seq = cam_features_bev_flat.view(B, S, C_feat, cam_features_bev_flat.shape[2], cam_features_bev_flat.shape[3])
        
        cam_features_bev_seq = self.temporal_mamba(cam_features_bev_seq, inference_params=inference_params)
        cam_features_bev = cam_features_bev_seq[:, -1, :, :, :]
        
        lidar_features_bev = self.lidar_backbone(lidar_bev)
        fused_bev = self.fusion_neck(cam_features_bev, lidar_features_bev)
        
        waypoints = self.planning_head(fused_bev)
        return waypoints
