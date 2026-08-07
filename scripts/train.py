import os
import sys
import csv
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
from models.modules.BEV_perception import BEVPerceptionNetV2
from models.modules.BEV_planning import MultiHeadPlanningLoss
from models.dataset import CARLADataset, compute_planning_metrics, compute_temporal_metrics_complete, EarlyStopping


def train_exp4(args):
    """
    Main training pipeline for Experiment No. 4 (Multi-Head Architecture + Differential Learning Rates + Data Imbalance Balance).
    Features:
    - Navigation Command Conditioning (CommandEncoder)
    - Trigonometric Yaw Head (sin(yaw), cos(yaw)) on unit circle
    - Pedal & Speed Head (speed_mps, throttle, brake)
    - Differential LR: 5e-5 for DINOv2+LoRA, 3e-4 for Mamba & Multi-Head
    - Sample Loss Weighting (3.0x multiplier for turns & braking)
    - Random Horizontal Flip Data Augmentation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================================")
    print(f"  Training Experiment No. 4: Multi-Head Architecture & Balance    ")
    print(f"==================================================================")
    print(f"[Device] Running on: {device}")
    
    # Episodic Split
    location_root = os.path.join(args.data_dir, "Location")
    all_episodes = sorted([d for d in os.listdir(location_root) if d.startswith("episode_") and os.path.isdir(os.path.join(location_root, d))])
    
    num_val_episodes = max(1, int(0.15 * len(all_episodes)))
    train_episodes = all_episodes[:-num_val_episodes]
    val_episodes = all_episodes[-num_val_episodes:]
    
    print(f"[Dataset] Train Episodes ({len(train_episodes)}): {train_episodes}")
    print(f"[Dataset] Val Episodes ({len(val_episodes)}): {val_episodes}")
    
    train_dataset = CARLADataset(
        args.data_dir,
        seq_len=args.seq_len,
        resize_factor=args.resize_factor,
        stride=args.stride,
        episodes=train_episodes,
        is_train=True,
        augment=True
    )
    val_dataset = CARLADataset(
        args.data_dir,
        seq_len=args.seq_len,
        resize_factor=args.resize_factor,
        stride=args.stride,
        episodes=val_episodes,
        is_train=False,
        augment=False
    )
    
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
    
    print(f"[Model] Initializing Multi-Head BEVPerceptionNetV2...")
    model = BEVPerceptionNetV2(
        num_waypoints=10,
        bev_height=400,
        bev_width=400,
        lora_r=args.lora_r,
        use_polynomial_head=True
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total Params: {total_params:,} | Trainable Params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Differential Learning Rates Setup
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "cam_backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone}, # DINOv2 + LoRA (5e-5)
        {'params': head_params, 'lr': args.lr_head}           # Mamba + Multi-Head (3e-4)
    ], weight_decay=1e-4)
    
    print(f"[Optimizer] Differential LR configured:")
    print(f"  - DINOv2 + LoRA LR:  {args.lr_backbone:.2e}")
    print(f"  - Mamba & Heads LR:  {args.lr_head:.2e}")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = MultiHeadPlanningLoss(w_pos=1.0, w_yaw=0.5, w_speed=0.1, w_pedal=0.1, w_smooth=0.01)
    
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.model_dir, "tensorboard"))
    os.makedirs(args.model_dir, exist_ok=True)
    
    best_val_loss = float("inf")
    start_epoch = 0
    history_list = []
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    print("\n" + "="*60)
    print("  STARTING EXPERIMENT NO. 4 TRAINING LOOP")
    print("="*60)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        model.train()
        
        train_loss_total = 0.0
        train_loss_pos = 0.0
        train_loss_yaw = 0.0
        train_loss_speed = 0.0
        train_loss_pedal = 0.0
        train_ade_sum = 0.0
        
        optimizer.zero_grad()
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs:02d} [Train]")
        for i, batch in enumerate(train_pbar):
            camera_imgs, extrinsics, intrinsics, target_waypoints, command, telemetry, sample_weights = batch
            
            camera_imgs = camera_imgs.to(device, non_blocking=True)
            extrinsics = extrinsics.to(device, non_blocking=True)
            intrinsics = intrinsics.to(device, non_blocking=True)
            target_waypoints = target_waypoints.to(device, non_blocking=True)
            command = command.to(device, non_blocking=True)
            telemetry = telemetry.to(device, non_blocking=True)
            sample_weights = sample_weights.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(camera_imgs, extrinsics, intrinsics, command=command)
                loss, loss_components = criterion(outputs, target_waypoints, telemetry, sample_weights)
                scaled_loss = loss / args.accumulation_steps
                
            scaler.scale(scaled_loss).backward()
            
            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                trainable_model_params = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_model_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            pred_wps = outputs["pred_waypoints"]
            ade, _ = compute_planning_metrics(pred_wps, target_waypoints)
            
            train_loss_total += loss.item()
            train_loss_pos += loss_components["loss_pos"]
            train_loss_yaw += loss_components["loss_yaw"]
            train_loss_speed += loss_components["loss_speed"]
            train_loss_pedal += loss_components["loss_pedal"]
            train_ade_sum += ade
            
            train_pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "PosL": f"{loss_components['loss_pos']:.4f}",
                "YawL": f"{loss_components['loss_yaw']:.4f}",
                "ADE": f"{ade:.2f}m"
            })
            
        train_loss_avg = train_loss_total / len(train_loader)
        train_ade_avg = train_ade_sum / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss_total = 0.0
        val_loss_pos = 0.0
        val_loss_yaw = 0.0
        val_loss_speed = 0.0
        val_loss_pedal = 0.0
        val_ade_sum = 0.0
        val_fde_sum = 0.0
        val_yaw_err_sum = 0.0
        val_speed_err_sum = 0.0
        val_pedal_err_sum = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs:02d} [Val]")
            for batch in val_pbar:
                camera_imgs, extrinsics, intrinsics, target_waypoints, command, telemetry, sample_weights = batch
                
                camera_imgs = camera_imgs.to(device, non_blocking=True)
                extrinsics = extrinsics.to(device, non_blocking=True)
                intrinsics = intrinsics.to(device, non_blocking=True)
                target_waypoints = target_waypoints.to(device, non_blocking=True)
                command = command.to(device, non_blocking=True)
                telemetry = telemetry.to(device, non_blocking=True)
                sample_weights = sample_weights.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(camera_imgs, extrinsics, intrinsics, command=command)
                    loss, loss_components = criterion(outputs, target_waypoints, telemetry, sample_weights)
                    
                pred_wps = outputs["pred_waypoints"]
                ade, fde = compute_planning_metrics(pred_wps, target_waypoints)
                temp_m = compute_temporal_metrics_complete(pred_wps, target_waypoints)
                
                pred_speed = outputs["pred_speed"]
                target_speed = telemetry[:, 0]
                speed_err_kmh = torch.mean(torch.abs(pred_speed - target_speed) * 3.6).item()
                
                pred_pedals = outputs["pred_pedals"]
                target_pedals = telemetry[:, 1:3]
                pedal_err = torch.mean(torch.abs(pred_pedals - target_pedals)).item()
                
                val_loss_total += loss.item()
                val_loss_pos += loss_components["loss_pos"]
                val_loss_yaw += loss_components["loss_yaw"]
                val_loss_speed += loss_components["loss_speed"]
                val_loss_pedal += loss_components["loss_pedal"]
                val_ade_sum += ade
                val_fde_sum += fde
                val_yaw_err_sum += temp_m["yaw_error_deg"]
                val_speed_err_sum += speed_err_kmh
                val_pedal_err_sum += pedal_err
                
        val_loss_avg = val_loss_total / len(val_loader)
        val_ade_avg = val_ade_sum / len(val_loader)
        val_fde_avg = val_fde_sum / len(val_loader)
        val_yaw_err_avg = val_yaw_err_sum / len(val_loader)
        val_speed_err_avg = val_speed_err_sum / len(val_loader)
        val_pedal_err_avg = val_pedal_err_sum / len(val_loader)
        
        scheduler.step(val_loss_avg)
        epoch_time = time.time() - epoch_start_time
        
        print(f"\n[Epoch {epoch+1:02d}/{args.epochs:02d}] Summary ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_loss_avg:.4f} | Train ADE: {train_ade_avg:.3f}m")
        print(f"  Val Loss:   {val_loss_avg:.4f} | Val ADE:   {val_ade_avg:.3f}m | Val FDE: {val_fde_avg:.3f}m | Val YawErr: {val_yaw_err_avg:.1f}°")
        print(f"  Val SpeedErr: {val_speed_err_avg:.2f} km/h | Val PedalErr: {val_pedal_err_avg:.4f}")
        
        # Tensorboard Logging
        tb_writer.add_scalar("Loss/Train_Total", train_loss_avg, epoch)
        tb_writer.add_scalar("Loss/Val_Total", val_loss_avg, epoch)
        tb_writer.add_scalar("Loss/Val_Pos", val_loss_pos / len(val_loader), epoch)
        tb_writer.add_scalar("Loss/Val_Yaw", val_loss_yaw / len(val_loader), epoch)
        tb_writer.add_scalar("Metrics/Val_ADE", val_ade_avg, epoch)
        tb_writer.add_scalar("Metrics/Val_FDE", val_fde_avg, epoch)
        tb_writer.add_scalar("Metrics/Val_YawError_Deg", val_yaw_err_avg, epoch)
        tb_writer.add_scalar("Metrics/Val_SpeedErr_kmh", val_speed_err_avg, epoch)
        tb_writer.add_scalar("Metrics/Val_PedalErr", val_pedal_err_avg, epoch)
        
        # Save Checkpoints
        last_ckpt = os.path.join(args.model_dir, "last_model.pth")
        best_ckpt = os.path.join(args.model_dir, "best_model.pth")
        
        ckpt_dict = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss_avg,
            "val_ade": val_ade_avg,
            "val_fde": val_fde_avg,
            "val_speed_err_kmh": val_speed_err_avg,
            "val_pedal_err": val_pedal_err_avg
        }
        torch.save(ckpt_dict, last_ckpt)
        
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(ckpt_dict, best_ckpt)
            print(f"  --> [BEST MODEL SAVED] Val Loss improved to {best_val_loss:.4f} (Saved to: {best_ckpt})")
            
        # Save JSON Training History
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(float(train_loss_avg), 4),
            "train_ade": round(float(train_ade_avg), 3),
            "val_loss": round(float(val_loss_avg), 4),
            "val_ade": round(float(val_ade_avg), 3),
            "val_fde": round(float(val_fde_avg), 3),
            "val_yaw_err_deg": round(float(val_yaw_err_avg), 2),
            "val_speed_err_kmh": round(float(val_speed_err_avg), 2),
            "val_pedal_err": round(float(val_pedal_err_avg), 4),
            "epoch_time_sec": round(float(epoch_time), 1)
        }
        history_list.append(epoch_record)
        with open(os.path.join(args.model_dir, "history.json"), "w") as f:
            json.dump(history_list, f, indent=2)

        tb_writer.close()
    print(f"\n[OK] Experiment 4 Training Completed Successfully! Best Val Loss: {best_val_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Helioskrill Experiment 4 Multi-Head Network")
    parser.add_argument("--data_dir", default="./data/", help="Path to CARLA dataset directory")
    parser.add_argument("--model_dir", default="./checkpoints/experimento_4", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--seq_len", type=int, default=5, help="Sequence temporal length")
    parser.add_argument("--stride", type=int, default=5, help="Sampling stride between sequences")
    parser.add_argument("--resize_factor", type=float, default=0.5, help="Image scaling factor")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank for DINOv2")
    parser.add_argument("--lr_backbone", type=float, default=5e-5, help="Learning rate for DINOv2+LoRA")
    parser.add_argument("--lr_head", type=float, default=3e-4, help="Learning rate for Mamba & Multi-Head")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.005, help="Early stopping min delta")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_exp4(args)
