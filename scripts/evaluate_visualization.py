#!/usr/bin/env python3
"""
evaluate_visualization.py
=========================
Visual evaluation and cross-validation script for Helioskrill (ViM + Temporal Mamba).

FUNCTIONALITY:
--------------
1. Loads trained model checkpoint (best_model.pth or last_model.pth).
2. Runs evaluation on validation episodes (unseen route split).
3. Computes quantitative evaluation metrics: ADE, FDE, Velocity Error, Acceleration Error, Yaw Error.
4. Generates composite Matplotlib panel visualizations with:
   - 8 RGB camera images for the current frame.
   - Bird's Eye View (BEV) map with Ground Truth (GREEN) vs Model Prediction (RED).
   - Metrics text box per sample.
5. Saves test output figures to `eval_results/visualizations/` for qualitative analysis.
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import cv2

cv2.setNumThreads(0)

# Add project root to sys.path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.modules.BEV_perception import BEVPerceptionNet
from scripts.train_vim import CARLADataset, compute_planning_metrics, compute_temporal_metrics_complete

# Tesla 8-camera configuration labels
CAMERA_NAMES = [
    "Cam 0: Front Main",
    "Cam 1: Front Wide",
    "Cam 2: Front Narrow",
    "Cam 3: Left B-Pillar",
    "Cam 4: Right B-Pillar",
    "Cam 5: Left Repeater",
    "Cam 6: Right Repeater",
    "Cam 7: Rear"
]


def load_model(checkpoint_path, device, img_size=(304, 400)):
    """
    Instantiates BEVPerceptionNet and loads weights from checkpoint.
    """
    model = BEVPerceptionNet(
        num_waypoints=10,
        bev_height=400,
        bev_width=400,
        grid_resolution=0.25,
        img_size=img_size
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    print(f"[Loading] Loading model weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=False)
        print(f"[Info] Restored full checkpoint (Epoch {checkpoint.get('epoch', 'N/A') + 1}).")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("[Info] Restored direct state_dict weights.")

    model.eval()
    return model


def visualize_sample(camera_imgs_tensor, target_wps, pred_wps, metrics, sample_idx, output_dir, ep_name, frame_id):
    """
    Generates and saves a composite 9-subplot visual figure (8 cameras + 1 BEV trajectory map).
    """
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Helioskrill ViM Evaluation — Episode: {ep_name} | Frame: {frame_id:06d}\n"
                 f"ADE: {metrics['ade']:.2f}m | FDE: {metrics['fde']:.2f}m | "
                 f"VelErr: {metrics['vel_err']:.2f}m/s | YawErr: {metrics['yaw_err']:.1f}°",
                 fontsize=14, fontweight='bold')

    # Plot 8 RGB Cameras (2x4 grid)
    last_frame_cams = camera_imgs_tensor[-1].cpu().numpy()

    for i in range(8):
        ax = fig.add_subplot(3, 4, i + 1)
        img = np.transpose(last_frame_cams[i], (1, 2, 0))
        img = np.clip(img, 0.0, 1.0)
        
        ax.imshow(img)
        ax.set_title(CAMERA_NAMES[i], fontsize=10, fontweight='semibold')
        ax.axis('off')

    # Plot BEV Trajectory Map (Bottom grid position)
    ax_bev = fig.add_subplot(3, 4, (9, 12))
    ax_bev.set_facecolor('#1a1a1a')
    ax_bev.grid(True, color='#333333', linestyle='--', linewidth=0.7)

    gt_x = target_wps[:, 0].cpu().numpy()
    gt_y = target_wps[:, 1].cpu().numpy()
    
    pred_x = pred_wps[:, 0].cpu().numpy()
    pred_y = pred_wps[:, 1].cpu().numpy()

    # Draw Ego Vehicle at origin (0, 0)
    ego_rect = patches.Rectangle((-1.0, -2.0), 2.0, 4.0, linewidth=1.5, edgecolor='cyan', facecolor='cyan', alpha=0.3, label="Ego Vehicle")
    ax_bev.add_patch(ego_rect)
    ax_bev.plot(0, 0, 'go', markersize=6)

    # Ground Truth trajectory (Green)
    ax_bev.plot(gt_y, gt_x, 'g-o', linewidth=2.5, markersize=6, label="Ground Truth")
    
    # Model Predicted trajectory (Red)
    ax_bev.plot(pred_y, pred_x, 'r--s', linewidth=2.5, markersize=6, label="ViM Prediction")

    # Yaw heading arrows at final waypoint
    gt_yaw = target_wps[-1, 3].item()
    pred_yaw = pred_wps[-1, 3].item()
    
    ax_bev.arrow(gt_y[-1], gt_x[-1], 2.0 * np.sin(gt_yaw), 2.0 * np.cos(gt_yaw), head_width=0.8, color='limegreen')
    ax_bev.arrow(pred_y[-1], pred_x[-1], 2.0 * np.sin(pred_yaw), 2.0 * np.cos(pred_yaw), head_width=0.8, color='red')

    ax_bev.set_xlim([-25.0, 25.0])
    ax_bev.set_ylim([-5.0, 55.0])
    ax_bev.set_xlabel("Y-Axis Lateral (m)", fontsize=10, color='white')
    ax_bev.set_ylabel("X-Axis Forward (m)", fontsize=10, color='white')
    ax_bev.tick_params(colors='white')
    ax_bev.set_title("Bird's Eye View Trajectory (BEV)", fontsize=12, color='white', fontweight='bold')
    ax_bev.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='none', labelcolor='white')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"sample_{sample_idx:03d}_{ep_name}_f{frame_id:06d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def run_evaluation(args):
    """
    Main evaluation pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Evaluating on: {device}")

    location_root = os.path.join(args.data_dir, "Location")
    if not os.path.exists(location_root):
        raise FileNotFoundError(f"Location directory not found at: {location_root}")

    all_episodes = sorted([d for d in os.listdir(location_root) if d.startswith("episode_") and os.path.isdir(os.path.join(location_root, d))])
    
    if args.episodes:
        val_episodes = [ep.strip() for ep in args.episodes.split(",")]
    else:
        num_val_episodes = max(1, int(0.15 * len(all_episodes)))
        val_episodes = all_episodes[-num_val_episodes:]

    print(f"[Dataset] Evaluating on validation episodes ({len(val_episodes)}): {val_episodes}")

    val_dataset = CARLADataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        resize_factor=args.resize_factor,
        stride=args.stride,
        episodes=val_episodes
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    H_res = int(600 * args.resize_factor)
    W_res = int(800 * args.resize_factor)
    H_padded = H_res + (16 - (H_res % 16)) % 16
    W_padded = W_res + (16 - (W_res % 16)) % 16
    img_size = (H_padded, W_padded)

    model = load_model(args.checkpoint, device, img_size=img_size)
    criterion = nn.HuberLoss(delta=1.0)

    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_vel_err = 0.0
    total_accel_err = 0.0
    total_yaw_err = 0.0
    horizon_ade_sum = None

    samples_to_save = max(1, args.num_samples)
    step_save_interval = max(1, len(val_loader) // samples_to_save)
    saved_images = []

    print("\n" + "="*60)
    print("  EVALUATING TRAJECTORY PLANNING MODEL")
    print("="*60)

    with torch.no_grad():
        for idx, (camera_imgs, lidar_bev, extrinsics, intrinsics, target_waypoints) in enumerate(tqdm(val_loader, desc="Evaluating")):
            camera_imgs = camera_imgs.to(device)
            lidar_bev = lidar_bev.to(device)
            extrinsics = extrinsics.to(device)
            intrinsics = intrinsics.to(device)
            target_waypoints = target_waypoints.to(device)

            with autocast(device_type='cuda'):
                pred_wps = model(camera_imgs, lidar_bev, extrinsics, intrinsics)
                loss = criterion(pred_wps, target_waypoints)

            total_loss += loss.item()

            ade, fde = compute_planning_metrics(pred_wps, target_waypoints)
            temp_m = compute_temporal_metrics_complete(pred_wps, target_waypoints)

            total_ade += ade
            total_fde += fde
            total_vel_err += temp_m["vel_error_mps"]
            total_accel_err += temp_m["accel_error_mps2"]
            total_yaw_err += temp_m["yaw_error_deg"]

            if horizon_ade_sum is None:
                horizon_ade_sum = [0.0] * len(temp_m["horizon_ade"])
            for step_i, h_val in enumerate(temp_m["horizon_ade"]):
                horizon_ade_sum[step_i] += h_val

            if (idx % step_save_interval == 0 or idx == len(val_loader) - 1) and len(saved_images) < samples_to_save:
                ep_name, start_frame = val_dataset.samples[idx]
                last_frame_id = start_frame + args.seq_len - 1
                
                metrics_sample = {
                    "ade": ade,
                    "fde": fde,
                    "vel_err": temp_m["vel_error_mps"],
                    "yaw_err": temp_m["yaw_error_deg"]
                }
                
                img_path = visualize_sample(
                    camera_imgs[0],
                    target_waypoints[0],
                    pred_wps[0],
                    metrics_sample,
                    len(saved_images) + 1,
                    vis_dir,
                    ep_name,
                    last_frame_id
                )
                saved_images.append(img_path)

    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_ade = total_ade / num_batches
    avg_fde = total_fde / num_batches
    avg_vel = total_vel_err / num_batches
    avg_accel = total_accel_err / num_batches
    avg_yaw = total_yaw_err / num_batches
    avg_horizon = [h / num_batches for h in horizon_ade_sum] if horizon_ade_sum else []

    print("\n" + "="*60)
    print("  CROSS-VALIDATION RESULTS — HELIOSKRILL VIM")
    print("="*60)
    print(f"  Mean Huber Loss:                {avg_loss:.5f}")
    print(f"  Mean ADE (Average Error):       {avg_ade:.2f} meters")
    print(f"  Mean FDE (Final Error 5.0s):    {avg_fde:.2f} meters")
    print(f"  Mean Velocity Error:            {avg_vel:.2f} m/s")
    print(f"  Mean Acceleration Error:        {avg_accel:.2f} m/s²")
    print(f"  Mean Yaw Error:                 {avg_yaw:.2f} degrees")
    print("-" * 60)
    print("  Horizon ADE per step:")
    for s_idx, h_err in enumerate(avg_horizon):
        t_sec = (s_idx + 1) * 0.5
        print(f"    t = {t_sec:.1f}s (Step {s_idx+1:02d}): {h_err:.2f} meters")
    print("="*60)

    csv_report_path = os.path.join(args.output_dir, "eval_summary.csv")
    with open(csv_report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["huber_loss", f"{avg_loss:.6f}"])
        writer.writerow(["ade_m", f"{avg_ade:.4f}"])
        writer.writerow(["fde_m", f"{avg_fde:.4f}"])
        writer.writerow(["vel_error_mps", f"{avg_vel:.4f}"])
        writer.writerow(["accel_error_mps2", f"{avg_accel:.4f}"])
        writer.writerow(["yaw_error_deg", f"{avg_yaw:.4f}"])
        for s_idx, h_err in enumerate(avg_horizon):
            writer.writerow([f"horizon_ade_step_{s_idx+1}_m", f"{h_err:.4f}"])

    print(f"\n[Saved] CSV summary report saved to: {csv_report_path}")
    print(f"[Saved] {len(saved_images)} figures saved to: {vis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Evaluation Script for Helioskrill")
    parser.add_argument("--checkpoint",     default="./checkpoints/experimento_1/best_model.pth", help="Path to checkpoint file (.pth)")
    parser.add_argument("--data_dir",       default="./data/", help="Root data directory")
    parser.add_argument("--seq_len",        type=int, default=5, help="Sequence length S")
    parser.add_argument("--stride",         type=int, default=5, help="Stride step between samples")
    parser.add_argument("--resize_factor",  type=float, default=0.5, help="Image scaling factor")
    parser.add_argument("--num_samples",    type=int, default=10, help="Number of visual sample figures to generate")
    parser.add_argument("--episodes",       default=None, help="Comma-separated episode names to evaluate (optional)")
    parser.add_argument("--output_dir",     default="./eval_results/", help="Directory to save evaluation outputs")

    args = parser.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    run_evaluation(args)
