import os
import sys
import csv
import json
import time
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2

cv2.setNumThreads(0)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.modules.BEV_perception_v2 import BEVPerceptionNetV2
from models.dataset import CARLADataset, compute_planning_metrics, compute_temporal_metrics_complete, EarlyStopping


def train_dinov2(args):
    """
    Main training pipeline for Experiment No. 2 (DINOv2 + Traditional LoRA + Temporal Mamba + GroupNorm).
    Runs in FP32 precision with prefetching DataLoader to maximize GPU 3D engine utilization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Training Experiment 2 on: {device} (FP32 Native Precision)")
    
    # Episodic Split
    location_root = os.path.join(args.data_dir, "Location")
    all_episodes = sorted([d for d in os.listdir(location_root) if d.startswith("episode_") and os.path.isdir(os.path.join(location_root, d))])
    
    num_val_episodes = max(1, int(0.15 * len(all_episodes)))
    train_episodes = all_episodes[:-num_val_episodes]
    val_episodes = all_episodes[-num_val_episodes:]
    
    print(f"[Dataset] Train Episodes ({len(train_episodes)}): {train_episodes}")
    print(f"[Dataset] Val Episodes ({len(val_episodes)}): {val_episodes}")
    
    train_dataset = CARLADataset(args.data_dir, seq_len=args.seq_len, resize_factor=args.resize_factor, stride=args.stride, episodes=train_episodes)
    val_dataset = CARLADataset(args.data_dir, seq_len=args.seq_len, resize_factor=args.resize_factor, stride=args.stride, episodes=val_episodes)
    
    # Accelerated DataLoader with prefetching and persistent workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    print(f"[Model] Initializing DINOv2 + LoRA BEVPerceptionNetV2...")
    model = BEVPerceptionNetV2(
        num_waypoints=10,
        bev_height=400,
        bev_width=400,
        grid_resolution=0.25,
        lora_r=args.lora_r
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total Params: {total_params:,} | Trainable Params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    criterion = nn.HuberLoss(delta=args.huber_delta)
    
    trainable_model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_model_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    
    tb_writer = SummaryWriter(log_dir=os.path.join(args.model_dir, "tensorboard"))
    os.makedirs(args.model_dir, exist_ok=True)
    
    best_val_loss = float("inf")
    start_epoch = 0

    if args.resume:
        ckpt_path = args.checkpoint_path if args.checkpoint_path else os.path.join(args.model_dir, "last_model.pth")
        if os.path.exists(ckpt_path):
            print(f"[Resume] Resuming Experiment 2 training from: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            if isinstance(checkpoint, dict):
                if 'optimizer_state' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                if 'scheduler_state' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state'])
                    
                start_epoch = checkpoint.get('epoch', -1) + 1
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                early_stopping.counter = checkpoint.get('early_stopping_counter', 0)
                early_stopping.best_loss = best_val_loss
            
            print(f"[Resume] State successfully restored. Resuming at epoch {start_epoch + 1}.")
        else:
            print(f"[Resume] WARNING: Checkpoint not found at '{ckpt_path}'. Starting from scratch.")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} (Experiment 2: DINOv2 + LoRA) ---")
        
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_ade = 0.0
        train_fde = 0.0
        train_vel = 0.0
        train_accel = 0.0
        train_yaw = 0.0
        train_horizon = None
        train_bar = tqdm(train_loader, desc="Training DINOv2")
        optimizer.zero_grad()
        
        for batch_idx, (camera_imgs, lidar_bev, extrinsics, intrinsics, target_waypoints) in enumerate(train_bar):
            camera_imgs = camera_imgs.to(device)
            lidar_bev = lidar_bev.to(device)
            extrinsics = extrinsics.to(device)
            intrinsics = intrinsics.to(device)
            target_waypoints = target_waypoints.to(device)
            
            pred_wps = model(camera_imgs, lidar_bev, extrinsics, intrinsics)
            loss = criterion(pred_wps, target_waypoints)
            loss_accum = loss / args.accumulation_steps
            loss_accum.backward()
            
            with torch.no_grad():
                ade_b, fde_b = compute_planning_metrics(pred_wps, target_waypoints)
                train_ade += ade_b
                train_fde += fde_b
                
                temp_m = compute_temporal_metrics_complete(pred_wps, target_waypoints)
                train_vel += temp_m["vel_error_mps"]
                train_accel += temp_m["accel_error_mps2"]
                train_yaw += temp_m["yaw_error_deg"]
                
                if train_horizon is None:
                    train_horizon = [0.0] * len(temp_m["horizon_ade"])
                for step_idx, h_err in enumerate(temp_m["horizon_ade"]):
                    train_horizon[step_idx] += h_err
            
            is_last_batch = (batch_idx + 1) == len(train_loader)
            if (batch_idx + 1) % args.accumulation_steps == 0 or is_last_batch:
                torch.nn.utils.clip_grad_norm_(trainable_model_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item()
            train_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ADE": f"{ade_b:.2f}m",
                "FDE": f"{fde_b:.2f}m"
            })
            
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_ade = train_ade / len(train_loader)
        epoch_train_fde = train_fde / len(train_loader)
        epoch_train_vel = train_vel / len(train_loader)
        epoch_train_accel = train_accel / len(train_loader)
        epoch_train_yaw = train_yaw / len(train_loader)
        epoch_train_horizon = [h / len(train_loader) for h in train_horizon] if train_horizon else []
        
        # ── Validation ──
        model.eval()
        val_loss = 0.0
        val_ade = 0.0
        val_fde = 0.0
        val_vel = 0.0
        val_accel = 0.0
        val_yaw = 0.0
        val_horizon = None
        val_bar = tqdm(val_loader, desc="Validating DINOv2")
        
        with torch.no_grad():
            for camera_imgs, lidar_bev, extrinsics, intrinsics, target_waypoints in val_bar:
                camera_imgs = camera_imgs.to(device)
                lidar_bev = lidar_bev.to(device)
                extrinsics = extrinsics.to(device)
                intrinsics = intrinsics.to(device)
                target_waypoints = target_waypoints.to(device)
                
                pred_wps = model(camera_imgs, lidar_bev, extrinsics, intrinsics)
                loss = criterion(pred_wps, target_waypoints)
                    
                val_loss += loss.item()
                ade_b, fde_b = compute_planning_metrics(pred_wps, target_waypoints)
                val_ade += ade_b
                val_fde += fde_b
                
                temp_m = compute_temporal_metrics_complete(pred_wps, target_waypoints)
                val_vel += temp_m["vel_error_mps"]
                val_accel += temp_m["accel_error_mps2"]
                val_yaw += temp_m["yaw_error_deg"]
                
                if val_horizon is None:
                    val_horizon = [0.0] * len(temp_m["horizon_ade"])
                for step_idx, h_err in enumerate(temp_m["horizon_ade"]):
                    val_horizon[step_idx] += h_err
                
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_ade = val_ade / len(val_loader)
        epoch_val_fde = val_fde / len(val_loader)
        epoch_val_vel = val_vel / len(val_loader)
        epoch_val_accel = val_accel / len(val_loader)
        epoch_val_yaw = val_yaw / len(val_loader)
        epoch_val_horizon = [h / len(val_loader) for h in val_horizon] if val_horizon else []
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:02d} Summary: Loss Train: {epoch_train_loss:.4f} | Loss Val: {epoch_val_loss:.4f} | ADE: {epoch_val_ade:.2f}m | FDE: {epoch_val_fde:.2f}m | VelErr: {epoch_val_vel:.2f}m/s | AccErr: {epoch_val_accel:.2f}m/s² | YawErr: {epoch_val_yaw:.1f}° | LR: {current_lr:.6e}")
        
        # TensorBoard logging
        tb_writer.add_scalar("Loss/train", epoch_train_loss, epoch + 1)
        tb_writer.add_scalar("Loss/val", epoch_val_loss, epoch + 1)
        tb_writer.add_scalar("Metrics/train_ADE_m", epoch_train_ade, epoch + 1)
        tb_writer.add_scalar("Metrics/val_ADE_m", epoch_val_ade, epoch + 1)
        tb_writer.add_scalar("Metrics/train_FDE_m", epoch_train_fde, epoch + 1)
        tb_writer.add_scalar("Metrics/val_FDE_m", epoch_val_fde, epoch + 1)
        tb_writer.add_scalar("Temporal/train_vel_error_mps", epoch_train_vel, epoch + 1)
        tb_writer.add_scalar("Temporal/val_vel_error_mps", epoch_val_vel, epoch + 1)
        tb_writer.add_scalar("Temporal/train_accel_error_mps2", epoch_train_accel, epoch + 1)
        tb_writer.add_scalar("Temporal/val_accel_error_mps2", epoch_val_accel, epoch + 1)
        tb_writer.add_scalar("Temporal/train_yaw_error_deg", epoch_train_yaw, epoch + 1)
        tb_writer.add_scalar("Temporal/val_yaw_error_deg", epoch_val_yaw, epoch + 1)
        
        for step_idx in range(min(len(epoch_train_horizon), len(epoch_val_horizon))):
            tb_writer.add_scalar(f"Horizon_ADE_Train/step_{step_idx+1}_m", epoch_train_horizon[step_idx], epoch + 1)
            tb_writer.add_scalar(f"Horizon_ADE_Val/step_{step_idx+1}_m", epoch_val_horizon[step_idx], epoch + 1)
            
        tb_writer.add_scalar("Learning_Rate", current_lr, epoch + 1)
        
        # Save metrics to CSV
        metrics_csv_path = os.path.join(args.model_dir, "metrics.csv")
        file_exists = os.path.exists(metrics_csv_path)
        with open(metrics_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "epoch", "train_loss", "val_loss",
                    "train_ade_m", "val_ade_m",
                    "train_fde_m", "val_fde_m",
                    "train_vel_err_mps", "val_vel_err_mps",
                    "train_accel_err_mps2", "val_accel_err_mps2",
                    "train_yaw_err_deg", "val_yaw_err_deg",
                    "learning_rate"
                ])
            writer.writerow([
                epoch + 1,
                f"{epoch_train_loss:.6f}", f"{epoch_val_loss:.6f}",
                f"{epoch_train_ade:.4f}", f"{epoch_val_ade:.4f}",
                f"{epoch_train_fde:.4f}", f"{epoch_val_fde:.4f}",
                f"{epoch_train_vel:.4f}", f"{epoch_val_vel:.4f}",
                f"{epoch_train_accel:.4f}", f"{epoch_val_accel:.4f}",
                f"{epoch_train_yaw:.4f}", f"{epoch_val_yaw:.4f}",
                f"{current_lr:.6e}"
            ])
        
        scheduler.step(epoch_val_loss)
        early_stopping(epoch_val_loss)
        
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'early_stopping_counter': early_stopping.counter
        }
        
        last_model_path = os.path.join(args.model_dir, "last_model.pth")
        torch.save(checkpoint, last_model_path)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint['best_val_loss'] = best_val_loss
            best_model_path = os.path.join(args.model_dir, "best_model.pth")
            torch.save(checkpoint, best_model_path)
            print(f"[Record] Saved new best model to: {best_model_path}")
        
        if early_stopping.early_stop:
            print(f"\n[EarlyStopping] Training stopped early at epoch {epoch+1} due to loss stagnation.")
            break
            
    tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2: DINOv2 + LoRA + Temporal Mamba Training Pipeline")
    parser.add_argument("--data_dir",           default="./data/", help="Path to collected dataset directory")
    parser.add_argument("--epochs",             type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size",         type=int, default=1, help="Batch size per GPU iteration")
    parser.add_argument("--seq_len",            type=int, default=5, help="Temporal sequence length S")
    parser.add_argument("--stride",             type=int, default=5, help="Stride step between sequences")
    parser.add_argument("--resize_factor",      type=float, default=0.5, help="Image scaling factor")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lora_r",             type=int, default=8, help="LoRA rank hyperparameter")
    parser.add_argument("--lr",                 type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--huber_delta",        type=float, default=1.0, help="Huber loss delta parameter")
    parser.add_argument("--model_dir",          default="./checkpoints/experimento_2/", help="Output directory for Experiment 2 checkpoints")
    parser.add_argument("--num_workers",        type=int, default=4, help="Data loading worker threads")
    parser.add_argument("--patience",           type=int, default=5, help="Patience epochs for Early Stopping")
    parser.add_argument("--min_delta",          type=float, default=1e-4, help="Minimum loss improvement delta")
    parser.add_argument("--resume",             action="store_true", help="Resume training loading weights from last_model.pth")
    parser.add_argument("--checkpoint_path",   default=None, help="Custom checkpoint path to load when --resume is active")
    
    args = parser.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    args.model_dir = os.path.abspath(args.model_dir)
    
    print("\n" + "="*60)
    print("  HELIOSKRILL — EXPERIMENT 2 (DINOv2 + LoRA + Prefetch Acceleration)")
    print("="*60)
    print(f"  Data:             {args.data_dir}")
    print(f"  Batch Size:       {args.batch_size}")
    print(f"  Sequence S:       {args.seq_len}")
    print(f"  Stride:           {args.stride}")
    print(f"  Resize Factor:    {args.resize_factor}")
    print(f"  Grad Accumulation:{args.accumulation_steps}")
    print(f"  LoRA Rank (r):    {args.lora_r}")
    print(f"  Learning Rate:    {args.lr}")
    print(f"  Output Dir:       {args.model_dir}")
    print("="*60 + "\n")
    
    train_dinov2(args)
