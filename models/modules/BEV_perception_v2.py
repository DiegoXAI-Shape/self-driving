import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.utils.DINOv2_blocks import DINOv2EncoderLoRA
from models.utils.vim_blocks import TemporalMamba
from models.modules.BEV_planning import BEVPlanningHead, MultiHeadBEVPlanningHead, CommandEncoder


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


class BEVFusionNeckGroupNorm(nn.Module):
    """
    Fuses 64-channel Camera BEV features and 64-channel LiDAR BEV features into 128 channels.
    """
    def __init__(self, in_cam_channels=64, in_lidar_channels=64, out_channels=128, num_groups=16):
        super().__init__()
        self.fusion = nn.Sequential(
            ResidualBlock2DGroupNorm(in_cam_channels + in_lidar_channels, out_channels, stride=1, num_groups=num_groups),
            ResidualBlock2DGroupNorm(out_channels, out_channels, stride=1, num_groups=num_groups)
        )

    def forward(self, cam_bev, lidar_bev):
        x = torch.cat([cam_bev, lidar_bev], dim=1)
        return self.fusion(x)


class BEVPerceptionNetV2(nn.Module):
    """
    Multi-Head Industrial BEV Planning Network for Helioskrill Experiment 4.
    """
    def __init__(self, num_waypoints=10, bev_height=400, bev_width=400, lora_r=8, use_polynomial_head=True):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.use_polynomial_head = use_polynomial_head

        # 1. 2D Visual Backbone: Meta DINOv2 Small + LoRA
        self.cam_backbone = DINOv2EncoderLoRA(lora_r=lora_r, lora_alpha=16)
        
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )
        
        # 2. Camera IPM Projection to 3D BEV
        self.projection = CameraBEVProjectionV2(bev_height=bev_height, bev_width=bev_width)
        
        # 3. Temporal Mamba SSM for sequence recurrence
        self.temporal_mamba = TemporalMamba(dim=64, d_state=16, d_conv=4, expand=2)
        
        # 4. LiDAR BEV Encoder
        self.lidar_backbone = LidarBEVEncoderV2(in_channels=5, feat_channels=64)
        
        # 5. Fusion Neck
        self.fusion_neck = BEVFusionNeckGroupNorm(in_cam_channels=64, in_lidar_channels=64, out_channels=128)
        
        # 6. Navigation Command Encoder
        self.command_encoder = CommandEncoder(num_commands=6, embed_dim=64)
        
        # 7. Multi-Head Planning Head
        if use_polynomial_head:
            self.planning_head = MultiHeadBEVPlanningHead(in_channels=128, num_waypoints=num_waypoints)
        else:
            self.planning_head = BEVPlanningHead(in_channels=128, num_waypoints=num_waypoints)

    def forward(self, camera_imgs, lidar_bev, extrinsics, intrinsics, command=None, inference_params=None):
        if len(camera_imgs.shape) == 6:
            B, S, N, C, H, W = camera_imgs.shape
            curr_camera_imgs = camera_imgs[:, -1, :, :, :, :]
        else:
            B, N, C, H, W = camera_imgs.shape
            S = 1
            curr_camera_imgs = camera_imgs
            
        if command is None:
            command = torch.ones(B, dtype=torch.long, device=camera_imgs.device) # Default command 1 (LANE_FOLLOW)
            
        # 1. Extract 2D visual features on N=8 current cameras
        x_flat = curr_camera_imgs.contiguous().view(B * N, C, H, W)
        cam_features_384 = self.cam_backbone(x_flat)
        cam_features_384 = torch.nan_to_num(cam_features_384, nan=0.0, posinf=1.0, neginf=-1.0)
        
        cam_features_64_flat = self.channel_reducer(cam_features_384)
        
        C_feat, H_feat, W_feat = cam_features_64_flat.shape[1], cam_features_64_flat.shape[2], cam_features_64_flat.shape[3]
        cam_features_2d = cam_features_64_flat.view(B, N, C_feat, H_feat, W_feat)
        
        if len(extrinsics.shape) == 4 and extrinsics.shape[1] == S:
            extrinsics = extrinsics[:, -1, :, :]
        if len(intrinsics.shape) == 4 and intrinsics.shape[1] == S:
            intrinsics = intrinsics[:, -1, :, :]
            
        if len(extrinsics.shape) == 3:
            extrinsics = extrinsics.unsqueeze(0)
        # 2. Project N=8 current cameras to 3D BEV Space 
        cam_bev_current = self.projection(cam_features_2d, extrinsics, intrinsics, original_img_size=(W, H))
        
        cam_bev_seq = cam_bev_current.unsqueeze(1).repeat(1, S, 1, 1, 1)
        
        # 3. Apply Temporal Mamba recurrence (Downscaled to 100x100 for 16x speedup & low memory footprint)
        B, S, C_bev, H_bev, W_bev = cam_bev_seq.shape
        cam_bev_seq_flat = cam_bev_seq.view(B * S, C_bev, H_bev, W_bev)
        cam_bev_seq_small = F.adaptive_avg_pool2d(cam_bev_seq_flat, (100, 100))
        cam_bev_seq_small = cam_bev_seq_small.view(B, S, C_bev, 100, 100)
        
        cam_bev_mamba = self.temporal_mamba(cam_bev_seq_small, inference_params=inference_params)
        cam_features_mamba = cam_bev_mamba[:, -1, :, :, :]
        
        # Upsample back to 400x400 for high-resolution LiDAR fusion
        cam_features_bev = F.interpolate(cam_features_mamba, size=(H_bev, W_bev), mode='bilinear', align_corners=False)
        cam_features_bev = torch.nan_to_num(cam_features_bev, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 4. Fuse with 5-channel LiDAR BEV map
        lidar_features_bev = self.lidar_backbone(lidar_bev)
        fused_bev = self.fusion_neck(cam_features_bev, lidar_features_bev)
        
        # 5. Encode Navigation Command
        command_embed = self.command_encoder(command) # [B, 64]
        
        # 6. Predict Multi-Head Outputs
        if self.use_polynomial_head:
            outputs = self.planning_head(fused_bev, command_embed)
            return outputs
        else:
            waypoints = self.planning_head(fused_bev)
            return torch.nan_to_num(waypoints, nan=0.0, posinf=1.0, neginf=-1.0)


def test_bev_perception_v2():
    print("==================================================================")
    print("   Testing BEVPerceptionNetV2 (Experiment 4 Multi-Head Pipeline)  ")
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
    command = torch.tensor([1]).to(device) # LANE_FOLLOW

    model.eval()
    with torch.no_grad():
        outputs = model(camera_imgs, lidar_bev, extrinsics, intrinsics, command=command)
        
    wps = outputs["pred_waypoints"]
    trig_yaw = outputs["trig_yaw"]
    pred_speed = outputs["pred_speed"]
    pred_pedals = outputs["pred_pedals"]
    
    assert wps.shape == (B, 10, 4), f"Output shape mismatch! Expected {(B, 10, 4)}, got {wps.shape}"
    assert trig_yaw.shape == (B, 10, 2), f"Trig Yaw shape mismatch! Expected {(B, 10, 2)}, got {trig_yaw.shape}"
    assert pred_speed.shape == (B,), f"Speed shape mismatch! Expected {(B,)}, got {pred_speed.shape}"
    assert pred_pedals.shape == (B, 2), f"Pedals shape mismatch! Expected {(B, 2)}, got {pred_pedals.shape}"
    
    print("[OK] BEVPerceptionNetV2 Multi-Head forward pass test passed successfully!")


if __name__ == "__main__":
    test_bev_perception_v2()
