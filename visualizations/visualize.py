#!/usr/bin/env python3
"""
visualize.py
============
Unified Visualization & Evaluation Suite for Helioskrill.

MODES:
  --mode bev      : Generates Top-Down BEV trajectory comparison figures (Ground Truth vs Prediction).
  --mode data     : Interactive multi-camera RGB + LiDAR dataset inspection tool.
  --mode pca      : DINOv2 PCA Feature Map visualizer (inspects 384D visual representations in RGB).
  --mode metrics  : Generates 4-panel training loss/metrics curves and exports history.json.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.modules.BEV_perception_v2 import BEVPerceptionNetV2
from models.dataset import CARLADataset, compute_planning_metrics, compute_temporal_metrics_complete


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODE: BEV TOP-DOWN VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_NAMES = [
    "Cam 0: Front Main", "Cam 1: Front Wide", "Cam 2: Front Narrow", "Cam 3: Left B-Pillar",
    "Cam 4: Right B-Pillar", "Cam 5: Left Repeater", "Cam 6: Right Repeater", "Cam 7: Rear"
]

def run_bev_visualization(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model = BEVPerceptionNetV2(num_waypoints=10, bev_height=400, bev_width=400, lora_r=8, use_polynomial_head=True).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    episodes = [f"episode_{int(ep):04d}" for ep in args.episodes.split(",")]
    dataset = CARLADataset(args.data_dir, seq_len=5, stride=5, episodes=episodes)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    out_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Evaluating] Generating 8-Camera Composite BEV figures for {len(dataset)} samples...")
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= args.max_samples:
                break
            cams, lidar, ext, intrins, target_wps, command, telem, weight = batch
            cams, lidar = cams.to(device), lidar.to(device)
            ext, intrins = ext.to(device), intrins.to(device)
            target_wps = target_wps.to(device)
            command = command.to(device)

            outputs = model(cams, lidar, ext, intrins, command=command)
            pred_wps = outputs["pred_waypoints"][0].cpu().numpy()
            gt_wps = target_wps[0].cpu().numpy()

            ade, fde = compute_planning_metrics(outputs["pred_waypoints"], target_wps)
            temp_m = compute_temporal_metrics_complete(outputs["pred_waypoints"], target_wps)

            # Dark theme 8-camera panel figure
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(16, 12), facecolor='#111111')
            gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.3], hspace=0.35, wspace=0.15)

            # Render 8 Cameras
            cams_last = cams[0, -1].cpu().numpy()  # [8, 3, H, W]
            for c_idx in range(8):
                row = c_idx // 4
                col = c_idx % 4
                ax_cam = fig.add_subplot(gs[row, col])
                img_rgb = cams_last[c_idx].transpose(1, 2, 0)
                ax_cam.imshow(img_rgb)
                ax_cam.set_title(CAMERA_NAMES[c_idx], fontsize=10, fontweight='bold', color='white')
                ax_cam.axis('off')

            # Render Centered BEV Map
            ax_bev = fig.add_subplot(gs[2, 1:3])
            ax_bev.set_facecolor('#1a1a1a')
            
            # Ground truth (Green) & Prediction (Red)
            ax_bev.plot(gt_wps[:, 1], gt_wps[:, 0], 'o-', color='#00FF00', label="Ground Truth", linewidth=2.5, markersize=6)
            ax_bev.plot(pred_wps[:, 1], pred_wps[:, 0], 's--', color='#FF2222', label="VIM Prediction", linewidth=2.5, markersize=6)

            # Ego Vehicle Box
            ego_rect = plt.Rectangle((-0.9, -2.0), 1.8, 4.0, linewidth=2, edgecolor='cyan', facecolor='cyan', alpha=0.3)
            ax_bev.add_patch(ego_rect)
            ax_bev.text(0.0, -1.0, "Ego Vehicle", color='cyan', fontsize=9, fontweight='bold', ha='center')

            ax_bev.set_xlim(-12, 12)
            ax_bev.set_ylim(-6, 35)
            ax_bev.set_xlabel("Lateral Position (Left / Right) [meters]", fontsize=9)
            ax_bev.set_ylabel("Longitudinal Distance (Ahead) [meters]", fontsize=9)
            ax_bev.grid(True, color='#333333', linestyle='--', alpha=0.5)
            ax_bev.legend(loc='upper right', facecolor='#222222', edgecolor='none')

            # Metric Header Banner
            header_str = (
                f"Helioskrill ViM Evaluation — Sample: {idx:03d}\n"
                f"ADE: {ade:.2f}m | FDE: {fde:.2f}m | VelErr: {temp_m['vel_error_mps']:.2f}m/s | YawErr: {temp_m['yaw_error_deg']:.1f}°"
            )
            fig.suptitle(header_str, fontsize=14, fontweight='bold', color='white', y=0.98)

            fig_path = os.path.join(out_dir, f"sample_{idx:03d}.png")
            plt.savefig(fig_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close()

    print(f"[SUCCESS] Saved 8-camera composite BEV figures to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODE: INTERACTIVE DATASET INSPECTION
# ─────────────────────────────────────────────────────────────────────────────
def run_dataset_inspection(args):
    ep_name = f"episode_{int(args.episode):04d}"
    dataset = CARLADataset(args.data_dir, seq_len=5, stride=1, episodes=[ep_name])
    print(f"[Dataset Inspector] Loaded {ep_name} with {len(dataset)} frames.")
    print("Controls: Close plot window to finish.")

    for i in range(min(5, len(dataset))):
        cams, lidar, ext, intrins, target_wps, command, telem, weight = dataset[i]
        cam_img = cams[-1, 0].numpy().transpose(1, 2, 0)
        gt_wps = target_wps.numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(cam_img)
        ax1.set_title(f"Front Camera (Frame {i})")
        ax1.axis("off")

        ax2.plot(gt_wps[:, 1], gt_wps[:, 0], 'go-', label="Ground Truth Trajectory")
        ax2.set_title(f"LiDAR BEV Trajectory | Command: {command.item()}")
        ax2.set_xlabel("Y (m)")
        ax2.set_ylabel("X (m)")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODE: DINOv2 PCA FEATURE MAP VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────
def run_dinov2_pca(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    dataset = CARLADataset(args.data_dir, seq_len=5, episodes=["episode_0000"])
    cams, _, _, _, _, _, _, _ = dataset[0]
    front_cam = cams[-1, 0].unsqueeze(0).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state"] if (isinstance(checkpoint, dict) and "model_state" in checkpoint) else checkpoint

    model = BEVPerceptionNetV2(num_waypoints=10, bev_height=400, bev_width=400, lora_r=8, use_polynomial_head=True).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    with torch.no_grad():
        feat_384 = model.cam_backbone(front_cam)[0]

    C, H_f, W_f = feat_384.shape
    feat_flat = feat_384.view(C, -1).T
    U, S, V = torch.pca_lowrank(feat_flat, q=3)
    pca_features = torch.matmul(feat_flat, V[:, :3]).cpu().numpy()

    pca_min, pca_max = pca_features.min(axis=0), pca_features.max(axis=0)
    pca_rgb = (pca_features - pca_min) / (pca_max - pca_min + 1e-6)
    pca_img = cv2.resize(pca_rgb.reshape(H_f, W_f, 3), (400, 300), interpolation=cv2.INTER_CUBIC)

    orig_img = front_cam[0].cpu().numpy().transpose(1, 2, 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(orig_img)
    axes[0].set_title("Input Camera Image (CARLA)")
    axes[0].axis("off")

    axes[1].imshow(pca_img)
    axes[1].set_title("DINOv2 PCA Feature Map (What DINOv2 Sees)")
    axes[1].axis("off")

    plt.tight_layout()
    out_path = "./checkpoints/experimento_4/dinov2_feature_pca.png"
    plt.savefig(out_path, dpi=200)
    print(f"[SUCCESS] Saved DINOv2 PCA feature map to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODE: METRICS & TRAINING CURVES GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def run_metrics_generation(args):
    json_path = os.path.join(args.model_dir, "history.json")
    if not os.path.exists(json_path):
        print(f"Error: history.json not found in {args.model_dir}")
        return

    with open(json_path, "r") as f:
        history = json.load(f)

    epochs = [r["epoch"] for r in history]
    train_losses = [r.get("train_loss") for r in history]
    val_losses = [r.get("val_loss") for r in history]
    val_ades = [r.get("val_ade") for r in history]
    val_yaws = [r.get("val_yaw_err_deg") for r in history]
    val_speeds = [r.get("val_speed_err_kmh") for r in history]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    axes[0].plot(epochs, train_losses, 'o-', label="Train Loss", color="crimson")
    axes[0].plot(epochs, val_losses, 's--', label="Val Loss", color="dodgerblue")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(epochs, val_ades, '^-', color="seagreen", label="Val ADE (m)")
    axes[1].set_title("Validation ADE (Position Error)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ADE (meters)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    axes[2].plot(epochs, val_yaws, 'd-', color="darkorange", label="Val Yaw Error (°)")
    axes[2].set_title("Validation Yaw Angle Error")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Degrees (°)")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    if any(v is not None for v in val_speeds):
        axes[3].plot(epochs, val_speeds, 'p-', color="purple", label="Val Speed Error (km/h)")
        axes[3].set_title("Speed Prediction Error (Mamba)")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("Error (km/h)")
        axes[3].grid(True, linestyle="--", alpha=0.5)
        axes[3].legend()
    else:
        axes[3].axis("off")

    plt.tight_layout()
    plot_path = os.path.join(args.model_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=200)
    print(f"[SUCCESS] Saved training curves to: {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLI DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Visualization Suite for Helioskrill")
    parser.add_argument("--mode", choices=["bev", "data", "pca", "metrics"], default="metrics", help="Visualization mode")
    parser.add_argument("--data_dir", default="./data/", help="Path to CARLA dataset")
    parser.add_argument("--checkpoint", default="./checkpoints/experimento_4/best_model.pth", help="Checkpoint path")
    parser.add_argument("--model_dir", default="./checkpoints/experimento_4", help="Model checkpoint directory")
    parser.add_argument("--episodes", default="12,13", help="Comma-separated episode indices for BEV evaluation")
    parser.add_argument("--episode", type=int, default=0, help="Episode index for dataset inspection")
    parser.add_argument("--max_samples", type=int, default=10, help="Max samples for BEV visualization")
    parser.add_argument("--output_dir", default="./eval_results", help="Directory to save output figures")

    args = parser.parse_args()

    if args.mode == "bev":
        run_bev_visualization(args)
    elif args.mode == "data":
        run_dataset_inspection(args)
    elif args.mode == "pca":
        run_dinov2_pca(args)
    elif args.mode == "metrics":
        run_metrics_generation(args)
