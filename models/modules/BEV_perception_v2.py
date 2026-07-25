import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.utils.DINOv2_blocks import DINOv2EncoderLoRA
from models.utils.vim_blocks import TemporalMamba
from models.modules.BEV_planning import BEVPlanningHead, PolynomialBEVPlanningHead


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


class LidarBEVEncoderV2(nn.Module):
    """
    Processes 5-channel LiDAR BEV grid (Z_max, Z_diff, Z_mean, density, intensity) using GroupNorm.
    """
    def __init__(self, in_channels=5, feat_channels=64, num_groups=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups // 2, 32)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer = nn.Sequential(
            ResidualBlock2DGroupNorm(32, 64, stride=1, num_groups=num_groups),
            ResidualBlock2DGroupNorm(64, feat_channels, stride=1, num_groups=num_groups)
        )

    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.layer(out)
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


class CameraBEVProjectionV2(nn.Module):
    """
    Projects 2D feature maps (64 channels) into a unified 3D Bird's Eye View (BEV) space.
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
        """
        Args:
            cam_features: [B, N=8, C=64, H_feat, W_feat]
            extrinsics: [B, N=8, 4, 4]
            intrinsics: [B, N=8, 3, 3]
            original_img_size: (W, H)
        Returns:
            bev_features: [B, C=64, H_bev, W_bev]
        """
        B, N, C, H_feat, W_feat = cam_features.shape
        W_orig, H_orig = original_img_size
        
        grid_pts = self.grid_points.unsqueeze(0).unsqueeze(1).repeat(B, N, 1, 1, 1, 1)
        grid_pts_flat = grid_pts.view(B, N, -1, 3)
        
        ones = torch.ones_like(grid_pts_flat[..., :1])
        pts_homo = torch.cat([grid_pts_flat, ones], dim=-1)
        
        pts_cam = torch.matmul(extrinsics.unsqueeze(2), pts_homo.unsqueeze(-1)).squeeze(-1)
        
        z_cam = pts_cam[..., 2:3]
        z_cam_clamped = torch.clamp(z_cam, min=0.1)
        pts_proj = torch.matmul(intrinsics.unsqueeze(2), pts_cam[..., :3].unsqueeze(-1)).squeeze(-1)
        
        u = pts_proj[..., 0:1] / z_cam_clamped
        v = pts_proj[..., 1:2] / z_cam_clamped
        
        u_norm = (u / (W_orig - 1)) * 2.0 - 1.0
        v_norm = (v / (H_orig - 1)) * 2.0 - 1.0
        
        u_norm = torch.clamp(u_norm, min=-2.0, max=2.0)
        v_norm = torch.clamp(v_norm, min=-2.0, max=2.0)
        
        grid_uv = torch.cat([u_norm, v_norm], dim=-1)
        
        cam_features_flat = cam_features.view(B * N, C, H_feat, W_feat)
        grid_uv_flat = grid_uv.view(B * N, self.bev_height * self.bev_width * len(self.z_slices), 1, 2)
        
        sampled_feats = F.grid_sample(cam_features_flat, grid_uv_flat, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled_feats = sampled_feats.squeeze(-1).view(B, N, C, self.bev_height, self.bev_width, len(self.z_slices))
        
        valid_mask = (z_cam > 0.1) & (u_norm >= -1.0) & (u_norm <= 1.0) & (v_norm >= -1.0) & (v_norm <= 1.0)
        valid_mask = valid_mask.view(B, N, 1, self.bev_height, self.bev_width, len(self.z_slices)).float()
        
        sampled_feats = sampled_feats * valid_mask
        bev_feats = sampled_feats.sum(dim=1).mean(dim=-1)
        
        return torch.nan_to_num(bev_feats, nan=0.0, posinf=1.0, neginf=-1.0)


class BEVFusionNeckV2(nn.Module):
    """
    Fuses camera-projected BEV features with processed LiDAR BEV features using GroupNorm for stability.
    """
    def __init__(self, cam_channels=64, lidar_channels=64, out_channels=128, num_groups=16):
        super().__init__()
        in_channels = cam_channels + lidar_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.block1 = ResidualBlock2DGroupNorm(out_channels, out_channels, stride=1, num_groups=num_groups)
        self.block2 = ResidualBlock2DGroupNorm(out_channels, out_channels, stride=1, num_groups=num_groups)

    def forward(self, cam_bev, lidar_bev):
        x = torch.cat([cam_bev, lidar_bev], dim=1)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


class BEVPerceptionNetV2(nn.Module):
    """
    Industrial BEV Perception Architecture V2 for Helioskrill ViM Project.
    Integrates DINOv2 + LoRA visual backbone, IPM projection, Temporal Mamba in BEV,
    GroupNorm fusion neck, and Quintic Polynomial Planning Head.
    """
    def __init__(
        self,
        num_waypoints: int = 10,
        bev_height: int = 400,
        bev_width: int = 400,
        grid_resolution: float = 0.25,
        lora_r: int = 8,
        use_polynomial_head: bool = True
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.use_polynomial_head = use_polynomial_head
        
        print(f"[BEVPerceptionNetV2] Loading Pre-trained DINOv2 Small backbone (dinov2_vits14) + LoRA (r={lora_r})...")
        self.cam_backbone = DINOv2EncoderLoRA(lora_r=lora_r, lora_alpha=16)
        
        # Channel reducer from 384 DINOv2 features to 64 BEV features
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True)
        )
        
        self.lidar_backbone = LidarBEVEncoderV2(in_channels=5, feat_channels=64)
        self.projection = CameraBEVProjectionV2(bev_height=bev_height, bev_width=bev_width, grid_resolution=grid_resolution)
        
        self.temporal_mamba = TemporalMamba(dim=64, L_blocks=2)
        self.fusion_neck = BEVFusionNeckV2(cam_channels=64, lidar_channels=64, out_channels=128)
        
        if use_polynomial_head:
            self.planning_head = PolynomialBEVPlanningHead(in_channels=128, num_waypoints=num_waypoints)
        else:
            self.planning_head = BEVPlanningHead(in_channels=128, num_waypoints=num_waypoints)

    def forward(self, camera_imgs, lidar_bev, extrinsics, intrinsics, inference_params=None):
        """
        Args:
            camera_imgs: [B, S=5, N=8, 3, H, W] or [B, N=8, 3, H, W]
            lidar_bev:   [B, 5, 400, 400]
            extrinsics:  [B, N=8, 4, 4]
            intrinsics:  [B, N=8, 3, 3]
        """
        if len(camera_imgs.shape) == 6:
            B, S, N, C, H, W = camera_imgs.shape
            curr_camera_imgs = camera_imgs[:, -1, :, :, :, :]  # [B, N=8, 3, H, W]
        else:
            B, N, C, H, W = camera_imgs.shape
            S = 1
            curr_camera_imgs = camera_imgs
            
        # 1. Extract 2D visual features on N=8 current cameras
        x_flat = curr_camera_imgs.contiguous().view(B * N, C, H, W)
        cam_features_384 = self.cam_backbone(x_flat)  # [B*N=8, 384, H_grid, W_grid]
        cam_features_384 = torch.nan_to_num(cam_features_384, nan=0.0, posinf=1.0, neginf=-1.0)
        
        cam_features_64_flat = self.channel_reducer(cam_features_384)  # [B*N=8, 64, H_grid, W_grid]
        
        C_feat, H_feat, W_feat = cam_features_64_flat.shape[1], cam_features_64_flat.shape[2], cam_features_64_flat.shape[3]
        cam_features_2d = cam_features_64_flat.view(B, N, C_feat, H_feat, W_feat)
        
        if len(extrinsics.shape) == 4 and extrinsics.shape[1] == S:
            extrinsics = extrinsics[:, -1, :, :]
        if len(intrinsics.shape) == 4 and intrinsics.shape[1] == S:
            intrinsics = intrinsics[:, -1, :, :]
            
        if len(extrinsics.shape) == 3:
            extrinsics = extrinsics.unsqueeze(0)
        if len(intrinsics.shape) == 3:
            intrinsics = intrinsics.unsqueeze(0)
            
        # 2. Project N=8 current cameras to 3D BEV Space 
        cam_bev_current = self.projection(cam_features_2d, extrinsics, intrinsics, original_img_size=(W, H))  # [B, 64, 400, 400]
        
        # Expand along sequence length S=5 for Temporal Mamba state-space recurrence
        cam_bev_seq = cam_bev_current.unsqueeze(1).repeat(1, S, 1, 1, 1)  # [B, S=5, 64, 400, 400]
        
        # 3. Apply Temporal Mamba recurrence over time in lightweight BEV space
        cam_bev_seq = self.temporal_mamba(cam_bev_seq, inference_params=inference_params)
        cam_features_bev = cam_bev_seq[:, -1, :, :, :]
        cam_features_bev = torch.nan_to_num(cam_features_bev, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 4. Fuse with 5-channel LiDAR BEV map and estimate trajectory waypoints
        lidar_features_bev = self.lidar_backbone(lidar_bev)
        fused_bev = self.fusion_neck(cam_features_bev, lidar_features_bev)
        
        if self.use_polynomial_head:
            waypoints, coeffs = self.planning_head(fused_bev)
            waypoints = torch.nan_to_num(waypoints, nan=0.0, posinf=1.0, neginf=-1.0)
            return waypoints, coeffs
        else:
            waypoints = self.planning_head(fused_bev)
            return torch.nan_to_num(waypoints, nan=0.0, posinf=1.0, neginf=-1.0)


def test_bev_perception_v2():
    print("==================================================================")
    print("   Testing BEVPerceptionNetV2 (Industrial SOTA 8-Cam Pipeline)    ")
    print("==================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BEVPerceptionNetV2(num_waypoints=10, bev_height=400, bev_width=400, lora_r=8).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    B, S, N = 1, 5, 8
    camera_imgs = torch.randn(B, S, N, 3, 308, 406).to(device)
    lidar_bev = torch.randn(B, 5, 400, 400).to(device)
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(1).expand(B, N, -1, -1).to(device)
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(1).expand(B, N, -1, -1).to(device)

    model.eval()
    with torch.no_grad():
        out = model(camera_imgs, lidar_bev, extrinsics, intrinsics)
    if isinstance(out, tuple):
        wps, (cx, cy) = out
    else:
        wps = out
        
    assert wps.shape == (B, 10, 4), f"Output shape mismatch! Expected {(B, 10, 4)}, got {wps.shape}"
    print("[OK] BEVPerceptionNetV2 forward pass test passed successfully!")


if __name__ == "__main__":
    test_bev_perception_v2()
