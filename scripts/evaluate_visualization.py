#!/usr/bin/env python3
"""
evaluate_visualization.py
=========================
Visual evaluation and cross-validation script for Helioskrill (ViM + Temporal Mamba + DINOv2).

FUNCTIONALITY:
--------------
1. Auto-detects model head architecture (Polynomial vs Linear Planning Head) from checkpoint weights.
2. Formats episode numbers cleanly (e.g. `--episodes 14` -> `episode_0014`).
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
from models.modules.BEV_perception_v2 import BEVPerceptionNetV2 as BEVPerceptionNet
from models.dataset import CARLADataset, compute_planning_metrics, compute_temporal_metrics_complete

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


def load_model(checkpoint_path, device):
    """
    Auto-detects model head shape from checkpoint weights, instantiates BEVPerceptionNet,
    and loads weights cleanly.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    print(f"[Loading] Inspecting checkpoint weights at: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint["model_state"] if (isinstance(checkpoint, dict) and "model_state" in checkpoint) else checkpoint
    
    # Auto-detect if checkpoint uses Polynomial Head (22 outputs) or Linear Head (40 outputs)
    use_polynomial = False
    for k in state_dict.keys():
        if "planning_head.fc" in k and "weight" in k:
            out_dim = state_dict[k].shape[0]
            if out_dim == 22:
                use_polynomial = True
            break
            
    print(f"[Model] Detected Architecture: {'Quintic Polynomial Planning Head' if use_polynomial else 'Linear 40-Waypoint Head'}")

    model = BEVPerceptionNet(
        num_waypoints=10,
        bev_height=400,
        bev_width=400,
        grid_resolution=0.25,
        use_polynomial_head=use_polynomial
    ).to(device)

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
    Generates and saves a composite visual figure (8 cameras + 1 nicely formatted BEV trajectory map).
    """
    fig = plt.figure(figsize=(18, 12), facecolor='#121212')
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.2], hspace=0.3, wspace=0.2)
    
    fig.suptitle(f"Helioskrill ViM Evaluation — Episode: {ep_name} | Frame: {frame_id:06d}\n"
                 f"ADE: {metrics['ade']:.2f}m | FDE: {metrics['fde']:.2f}m | "
                 f"VelErr: {metrics['vel_err']:.2f}m/s | YawErr: {metrics['yaw_err']:.1f}°",
                 fontsize=14, fontweight='bold', color='white')

    # Plot 8 RGB Cameras (2x4 grid)
    last_frame_cams = camera_imgs_tensor[-1].cpu().numpy()

    for i in range(8):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        img = np.transpose(last_frame_cams[i], (1, 2, 0))
        img = np.clip(img, 0.0, 1.0)
        
        ax.imshow(img)
        ax.set_title(CAMERA_NAMES[i], fontsize=10, fontweight='bold', color='white')
        ax.axis('off')

    # Plot BEV Trajectory Grid (Bottom Middle: columns 1 and 2)
    ax_bev = fig.add_subplot(gs[2, 1:3])
    ax_bev.set_facecolor('#1a1a1a')
    ax_bev.grid(True, color='#333333', linestyle='--', linewidth=0.5)

    # Plot Ego Vehicle Bounding Box at Origin (0,0) (Width=2m, Length=4m)
    # Lateral X is [-1.0, 1.0], Longitudinal Y is [-2.0, 2.0]
    ego_rect = patches.Rectangle((-1.0, -2.0), 2.0, 4.0, linewidth=1.5, edgecolor='cyan', facecolor='teal', alpha=0.6)
    ax_bev.add_patch(ego_rect)
    ax_bev.text(0.0, -2.8, "Ego Vehicle", color='cyan', fontsize=9, ha='center', fontweight='bold')

    # Extract Coordinates (Ground Truth vs Prediction)
    # gt: [10, 4] -> (rel_x_forward, rel_y_lateral, rel_z, rel_yaw)
    gt = target_wps.cpu().numpy()
    pr = pred_wps.cpu().numpy()

    # Correct Mapping to Top-Down View:
    # Horizontal axis (X) = Lateral Position (rel_y, right is +Y, left is -Y)
    # Vertical axis (Y) = Longitudinal Forward Distance (rel_x, ahead is +X)
    gt_lat, gt_fwd = gt[:, 1], gt[:, 0]
    pr_lat, pr_fwd = pr[:, 1], pr[:, 0]

    # Ground Truth Trajectory (GREEN)
    ax_bev.plot(gt_lat, gt_fwd, 'o-', color='#00ff44', linewidth=2.5, markersize=5, label='Ground Truth')
    gt_yaw = gt[-1, 3]
    # Yaw angle relative to forward axis
    ax_bev.arrow(gt_lat[-1], gt_fwd[-1], np.sin(gt_yaw) * 1.5, np.cos(gt_yaw) * 1.5,
                 head_width=0.6, head_length=0.8, fc='#00ff44', ec='#00ff44')

    # Predicted Trajectory (RED)
    ax_bev.plot(pr_lat, pr_fwd, 's-', color='#ff2200', linewidth=2.5, markersize=5, label='VIM Prediction')
    pr_yaw = pr[-1, 3]
    ax_bev.arrow(pr_lat[-1], pr_fwd[-1], np.sin(pr_yaw) * 1.5, np.cos(pr_yaw) * 1.5,
                 head_width=0.6, head_length=0.8, fc='#ff2200', ec='#ff2200')

    # Format Axes Limits & Grid Labels
    all_lat = np.concatenate([gt_lat, pr_lat, [-5, 5]])
    all_fwd = np.concatenate([gt_fwd, pr_fwd, [-5, 30]])

    margin_lat = max(10, np.max(np.abs(all_lat)) + 5)
    max_fwd = max(25, np.max(all_fwd) + 5)
    min_fwd = min(-5, np.min(all_fwd) - 3)

    ax_bev.set_xlim(-margin_lat, margin_lat)
    ax_bev.set_ylim(min_fwd, max_fwd)
    ax_bev.set_aspect('equal')
    ax_bev.legend(loc='upper right', facecolor='#2b2b2b', edgecolor='none', labelcolor='white')
    ax_bev.set_xlabel("Lateral Position (Left / Right) [meters]", color='white')
    ax_bev.set_ylabel("Longitudinal Distance (Ahead) [meters]", color='white')
    ax_bev.tick_params(colors='white')

    plt.tight_layout()
    output_filename = f"sample_{sample_idx:03d}_{ep_name}_f{frame_id:06d}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Evaluating on: {device}")

    # Parse and format validation episodes cleanly
    if args.episodes:
        raw_eps = [e.strip() for e in args.episodes.split(",")]
        val_episodes = []
        for ep in raw_eps:
            if ep.isdigit():
                val_episodes.append(f"episode_{int(ep):04d}")
            elif not ep.startswith("episode_"):
                val_episodes.append(f"episode_{ep}")
            else:
                val_episodes.append(ep)
        print(f"[Dataset] Evaluating on user-specified episodes ({len(val_episodes)}): {val_episodes}")
    else:
        location_root = os.path.join(args.data_dir, "Location")
        all_episodes = sorted([d for d in os.listdir(location_root) if d.startswith("episode_") and os.path.isdir(os.path.join(location_root, d))])
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
    
    if len(val_dataset) == 0:
        print(f"[Error] No valid sequences found for episodes: {val_episodes}. Check episode folder names in {args.data_dir}/Location/")
        return

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = load_model(args.checkpoint, device)
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
                pred_out = model(camera_imgs, lidar_bev, extrinsics, intrinsics)
                if isinstance(pred_out, tuple):
                    pred_wps, _ = pred_out
                else:
                    pred_wps = pred_out
                    
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
    # Determine default checkpoint path intelligently
    default_ckpt = "./checkpoints/experimento_2/best_model.pth"
    if not os.path.exists(default_ckpt):
        default_ckpt = "./checkpoints/experimento_1/best_model.pth"

    parser = argparse.ArgumentParser(description="Visual Evaluation Script for Helioskrill")
    parser.add_argument("--checkpoint",     default=default_ckpt, help="Path to checkpoint file (.pth)")
    parser.add_argument("--data_dir",       default="./data/", help="Root data directory")
    parser.add_argument("--seq_len",        type=int, default=5, help="Sequence length S")
    parser.add_argument("--stride",         type=int, default=5, help="Stride step between samples")
    parser.add_argument("--resize_factor",  type=float, default=0.5, help="Image scaling factor")
    parser.add_argument("--num_samples",    type=int, default=10, help="Number of visual sample figures to generate")
    parser.add_argument("--episodes",       default=None, help="Comma-separated episode names or numbers (e.g., --episodes 14 or --episodes episode_0014)")
    parser.add_argument("--output_dir",     default="./eval_results/", help="Directory to save evaluation outputs")

    args = parser.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    run_evaluation(args)
