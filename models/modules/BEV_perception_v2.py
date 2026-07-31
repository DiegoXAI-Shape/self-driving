"""
BEV_perception_v2.py
====================
Multi-Head BEV Planning Network for Helioskrill Experiment 4+.

Architecture (Camera-Only BEV):
  1. DINOv2 Small + LoRA  →  384ch 2D feature maps per camera
  2. Channel Reducer (384 → 64)
  3. CameraBEVProjectionV2  →  IPM projection to 400×400 BEV grid
  4. TemporalMamba (downscaled 100×100)  →  temporal recurrence
  5. CameraBEVNeck (64 → 128)  →  refinement without LiDAR
  6. CommandEncoder + MultiHeadBEVPlanningHead  →  waypoints, yaw, speed, pedals

LiDAR fusion has been removed (archived in trash/lidar_bev_modules.py).
BEV depth will come from CARLA Ground Truth via Pseudo-LiDAR projection.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.utils.perception_blocks import DINOv2EncoderLoRA, CameraBEVProjectionV2, CameraBEVNeck
from models.utils.mamba_blocks import TemporalMamba
from models.modules.BEV_planning import BEVPlanningHead, MultiHeadBEVPlanningHead, CommandEncoder


class BEVPerceptionNetV2(nn.Module):
    """
    Multi-Head Industrial BEV Planning Network for Helioskrill Experiment 4+.
    Camera-only architecture — no LiDAR fusion.
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

        # 4. Camera-Only BEV Neck (replaces LiDAR fusion neck)
        self.bev_neck = CameraBEVNeck(in_channels=64, out_channels=128)

        # 5. Navigation Command Encoder
        self.command_encoder = CommandEncoder(num_commands=6, embed_dim=64)

        # 6. Multi-Head Planning Head
        if use_polynomial_head:
            self.planning_head = MultiHeadBEVPlanningHead(in_channels=128, num_waypoints=num_waypoints)
        else:
            self.planning_head = BEVPlanningHead(in_channels=128, num_waypoints=num_waypoints)

    def forward(self, camera_imgs, extrinsics, intrinsics, command=None, inference_params=None):
        if len(camera_imgs.shape) == 6:
            B, S, N, C, H, W = camera_imgs.shape
            curr_camera_imgs = camera_imgs[:, -1, :, :, :, :]
        else:
            B, N, C, H, W = camera_imgs.shape
            S = 1
            curr_camera_imgs = camera_imgs

        if command is None:
            command = torch.ones(B, dtype=torch.long, device=camera_imgs.device)  # Default: LANE_FOLLOW

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

        # 3. Apply Temporal Mamba recurrence (Downscaled to 100x100 for 16x speedup & low memory)
        B, S, C_bev, H_bev, W_bev = cam_bev_seq.shape
        cam_bev_seq_flat = cam_bev_seq.view(B * S, C_bev, H_bev, W_bev)
        cam_bev_seq_small = F.adaptive_avg_pool2d(cam_bev_seq_flat, (100, 100))
        cam_bev_seq_small = cam_bev_seq_small.view(B, S, C_bev, 100, 100)

        cam_bev_mamba = self.temporal_mamba(cam_bev_seq_small, inference_params=inference_params)
        cam_features_mamba = cam_bev_mamba[:, -1, :, :, :]

        # Upsample back to 400x400 for high-resolution planning
        cam_features_bev = F.interpolate(cam_features_mamba, size=(H_bev, W_bev), mode='bilinear', align_corners=False)
        cam_features_bev = torch.nan_to_num(cam_features_bev, nan=0.0, posinf=1.0, neginf=-1.0)

        # 4. Camera-Only BEV Neck
        fused_bev = self.bev_neck(cam_features_bev)

        # 5. Encode Navigation Command
        command_embed = self.command_encoder(command)  # [B, 64]

        # 6. Predict Multi-Head Outputs
        if self.use_polynomial_head:
            outputs = self.planning_head(fused_bev, command_embed)
            return outputs
        else:
            waypoints = self.planning_head(fused_bev)
            return torch.nan_to_num(waypoints, nan=0.0, posinf=1.0, neginf=-1.0)


def test_bev_perception_v2():
    print("==================================================================")
    print("   Testing BEVPerceptionNetV2 (Camera-Only, No LiDAR Fusion)     ")
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
    extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(1).expand(B, N, -1, -1).to(device)
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(1).expand(B, N, -1, -1).to(device)
    command = torch.tensor([1]).to(device)  # LANE_FOLLOW

    model.eval()
    with torch.no_grad():
        outputs = model(camera_imgs, extrinsics, intrinsics, command=command)

    wps = outputs["pred_waypoints"]
    trig_yaw = outputs["trig_yaw"]
    pred_speed = outputs["pred_speed"]
    pred_pedals = outputs["pred_pedals"]

    assert wps.shape == (B, 10, 4), f"Output shape mismatch! Expected {(B, 10, 4)}, got {wps.shape}"
    assert trig_yaw.shape == (B, 10, 2), f"Trig Yaw shape mismatch! Expected {(B, 10, 2)}, got {trig_yaw.shape}"
    assert pred_speed.shape == (B,), f"Speed shape mismatch! Expected {(B,)}, got {pred_speed.shape}"
    assert pred_pedals.shape == (B, 2), f"Pedals shape mismatch! Expected {(B, 2)}, got {pred_pedals.shape}"

    print("[OK] BEVPerceptionNetV2 Camera-Only forward pass test passed successfully!")


if __name__ == "__main__":
    test_bev_perception_v2()
