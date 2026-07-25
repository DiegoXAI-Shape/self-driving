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
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2

cv2.setNumThreads(0)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to sys.path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.modules.BEV_perception import BEVPerceptionNet


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARLA DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CARLADataset(Dataset):
    """
    Dataset class for multi-view temporal camera and LiDAR data collected from CARLA.
    Handles data scaling, waypoint caching, camera intrinsics/extrinsics computation,
    and fallback to pre-resized image directories.
    """
    def __init__(self, data_dir: str, seq_len: int = 5, resize_factor: float = 0.5, stride: int = 5, episodes=None):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.resize_factor = resize_factor
        self.stride = stride
        
        location_root = os.path.join(data_dir, "Location")
        if not os.path.exists(location_root):
            raise FileNotFoundError(f"Location directory not found at: {location_root}")
            
        all_found = sorted([d for d in os.listdir(location_root) if d.startswith("episode_") and os.path.isdir(os.path.join(location_root, d))])
        
        if episodes is not None:
            self.episodes = [ep for ep in episodes if ep in all_found]
        else:
            self.episodes = all_found
                
        # Index sequence sample windows with stride
        self.samples = []
        for ep_name in self.episodes:
            loc_csv = os.path.join(location_root, ep_name, "location.csv")
            if not os.path.exists(loc_csv):
                continue
            with open(loc_csv, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                num_frames = len(rows) - 1
            
            for start_f in range(0, num_frames - seq_len + 1, self.stride):
                self.samples.append((ep_name, start_f))
                
        # Cache waypoints in memory
        self.waypoints_cache = {}
        planning_root = os.path.join(data_dir, "Planning")
        for ep_name in self.episodes:
            planning_csv = os.path.join(planning_root, ep_name, "waypoints.csv")
            if os.path.exists(planning_csv):
                ep_wps = {}
                with open(planning_csv, "r") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    for row in rows[1:]:
                        frame_id = int(row[0])
                        wps = []
                        for i in range(10):
                            wps.append([
                                float(row[1 + i*4]), # rel_x
                                float(row[2 + i*4]), # rel_y
                                float(row[3 + i*4]), # rel_z
                                float(row[4 + i*4])  # rel_yaw
                            ])
                        ep_wps[frame_id] = torch.tensor(wps, dtype=torch.float32)
                self.waypoints_cache[ep_name] = ep_wps
                
        # Cache camera metadata per episode
        self.metadata_cache = {}
        for ep_name in self.episodes:
            metadata_path = os.path.join(location_root, ep_name, "cameras_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    meta = json.load(f)
                    
                    K = np.array(meta["intrinsics"], dtype=np.float32)
                    if self.resize_factor != 1.0:
                        K[0, 0] *= self.resize_factor  # fx
                        K[1, 1] *= self.resize_factor  # fy
                        K[0, 2] *= self.resize_factor  # cx
                        K[1, 2] *= self.resize_factor  # cy
                        
                    K_expanded = np.zeros((8, 3, 3), dtype=np.float32)
                    for i in range(8):
                        K_expanded[i] = K
                        
                    extrinsics = np.zeros((8, 4, 4), dtype=np.float32)
                    for i, cam_cfg in enumerate(meta["cameras"]):
                        pos = cam_cfg["pos"]
                        yaw_deg = cam_cfg["yaw"]
                        extrinsics[i] = self.compute_extrinsics(pos, yaw_deg)
                        
                    self.metadata_cache[ep_name] = {
                        "intrinsics": K_expanded,
                        "extrinsics": extrinsics
                    }
        
        print(f"[Dataset] Initialized. Found {len(self.episodes)} episodes.")
        print(f"[Dataset] Total sequences (stride={self.stride}): {len(self.samples)}.")

    def compute_extrinsics(self, pos, yaw_deg) -> np.ndarray:
        """
        Computes 4x4 extrinsic transformation matrix from Ego to Camera frame.
        """
        x, y, z = pos
        yaw = np.radians(yaw_deg)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        
        R_yaw = np.array([
            [cos_y, -sin_y, 0],
            [sin_y, cos_y, 0],
            [0, 0, 1]
        ])
        
        R_default = np.array([
            [0, 1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        
        R_ego_to_cam = R_default @ R_yaw.T
        t = np.array([x, y, z])
        
        ext = np.eye(4, dtype=np.float32)
        ext[:3, :3] = R_ego_to_cam
        ext[:3, 3] = -R_ego_to_cam @ t
        return ext

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_name, start_frame = self.samples[idx]
        
        use_resized = False
        cam_root = os.path.join(self.data_dir, "Perception_resized", ep_name, "cameras")
        if os.path.exists(cam_root):
            use_resized = True
            
        seq_cam_list = []
        H_res = int(600 * self.resize_factor)
        W_res = int(800 * self.resize_factor)
        
        for t in range(self.seq_len):
            frame_id = start_frame + t
            cam_list = []
            for i in range(8):
                if use_resized:
                    img_path = os.path.join(self.data_dir, "Perception_resized", ep_name, "cameras", f"cam_{i}", f"frame_{frame_id:06d}.png")
                else:
                    img_path = os.path.join(self.data_dir, "Perception", ep_name, "cameras", f"cam_{i}", f"frame_{frame_id:06d}.png")
                    
                img = cv2.imread(img_path)
                if img is None:
                    img_tensor = torch.zeros(3, H_res, W_res)
                else:
                    if not use_resized and self.resize_factor != 1.0:
                        img = cv2.resize(img, (W_res, H_res), interpolation=cv2.INTER_LINEAR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                cam_list.append(img_tensor)
            seq_cam_list.append(torch.stack(cam_list, dim=0))
        camera_imgs = torch.stack(seq_cam_list, dim=0)
        
        last_frame_id = start_frame + self.seq_len - 1
        lidar_path = os.path.join(self.data_dir, "Perception", ep_name, "lidar", f"frame_{last_frame_id:06d}.npy")
        if os.path.exists(lidar_path):
            lidar_bev = np.load(lidar_path)
            lidar_bev = torch.from_numpy(lidar_bev).float()
        else:
            lidar_bev = torch.zeros(5, 400, 400)
            
        ep_wps = self.waypoints_cache.get(ep_name, {})
        target_waypoints = ep_wps.get(last_frame_id, torch.zeros(10, 4))
                        
        meta = self.metadata_cache.get(ep_name, {
            "intrinsics": np.zeros((8, 3, 3), dtype=np.float32),
            "extrinsics": np.zeros((8, 4, 4), dtype=np.float32)
        })
        intrinsics = torch.from_numpy(meta["intrinsics"]).float()
        extrinsics = torch.from_numpy(meta["extrinsics"]).float()
        
        return camera_imgs, lidar_bev, extrinsics, intrinsics, target_waypoints


# ─────────────────────────────────────────────────────────────────────────────
# 2. METRICS & CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def compute_planning_metrics(pred_waypoints, target_waypoints):
    """
    Computes Average Displacement Error (ADE) and Final Displacement Error (FDE) in meters.
    """
    pred_xy = pred_waypoints[..., :2]
    target_xy = target_waypoints[..., :2]
    
    errors = torch.norm(pred_xy - target_xy, dim=-1)
    
    ade = torch.mean(errors).item()
    fde = torch.mean(errors[:, -1]).item()
    
    return ade, fde


def compute_temporal_metrics_complete(pred_waypoints, target_waypoints, dt=0.5):
    """
    Evaluates temporal consistency: Horizon ADE, velocity error (m/s), acceleration error (m/s^2), and yaw error (deg).
    """
    pred_xy = pred_waypoints[..., :2]
    target_xy = target_waypoints[..., :2]
    
    pos_error = torch.norm(pred_xy - target_xy, dim=-1)
    horizon_ade = torch.mean(pos_error, dim=0).tolist()
    
    pred_vel = (pred_xy[:, 1:] - pred_xy[:, :-1]) / dt
    target_vel = (target_xy[:, 1:] - target_xy[:, :-1]) / dt
    vel_error = torch.mean(torch.abs(torch.norm(pred_vel, dim=-1) - torch.norm(target_vel, dim=-1))).item()
    
    pred_accel = (pred_vel[:, 1:] - pred_vel[:, :-1]) / dt
    target_accel = (target_vel[:, 1:] - target_vel[:, :-1]) / dt
    accel_error = torch.mean(torch.abs(torch.norm(pred_accel, dim=-1) - torch.norm(target_accel, dim=-1))).item()
    
    yaw_error_deg = torch.mean(torch.abs(pred_waypoints[..., 3] - target_waypoints[..., 3])).item() * (180.0 / 3.141592)
    
    return {
        "horizon_ade": horizon_ade,
        "vel_error_mps": vel_error,
        "accel_error_mps2": accel_error,
        "yaw_error_deg": yaw_error_deg
    }


class EarlyStopping:
    """
    Early Stopping callback to halt training when validation loss stops improving.
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] Patience: {self.counter}/{self.patience} with no improvement.")
            if self.counter >= self.patience:
                self.early_stop = True


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_model(args):
    """
    Main training function with AMP, gradient accumulation, episodic split, and full checkpointing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Training on: {device}")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    H_res = int(600 * args.resize_factor)
    W_res = int(800 * args.resize_factor)
    H_padded = H_res + (16 - (H_res % 16)) % 16
    W_padded = W_res + (16 - (W_res % 16)) % 16
    img_size = (H_padded, W_padded)
    
    print(f"[Model] Initializing input resolution: {img_size}")
    
    model = BEVPerceptionNet(
        num_waypoints=10,
        bev_height=400,
        bev_width=400,
        grid_resolution=0.25,
        img_size=img_size
    ).to(device)
    
    criterion = nn.HuberLoss(delta=args.huber_delta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    
    tb_writer = SummaryWriter(log_dir=os.path.join(args.model_dir, "tensorboard"))
    
    os.makedirs(args.model_dir, exist_ok=True)
    best_val_loss = float("inf")
    start_epoch = 0

    if args.resume:
        ckpt_path = args.checkpoint_path if args.checkpoint_path else os.path.join(args.model_dir, "last_model.pth")
        if os.path.exists(ckpt_path):
            print(f"[Resume] Resuming training from: {ckpt_path}")
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
                if scaler is not None and checkpoint.get('scaler_state') is not None:
                    scaler.load_state_dict(checkpoint['scaler_state'])
                    
                start_epoch = checkpoint.get('epoch', -1) + 1
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                early_stopping.counter = checkpoint.get('early_stopping_counter', 0)
                early_stopping.best_loss = best_val_loss
            
            print(f"[Resume] State successfully restored. Resuming at epoch {start_epoch + 1}.")
        else:
            print(f"[Resume] WARNING: Checkpoint not found at '{ckpt_path}'. Starting from scratch.")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_ade = 0.0
        train_fde = 0.0
        train_vel = 0.0
        train_accel = 0.0
        train_yaw = 0.0
        train_horizon = None
        train_bar = tqdm(train_loader, desc="Training")
        optimizer.zero_grad()
        
        for batch_idx, (camera_imgs, lidar_bev, extrinsics, intrinsics, target_waypoints) in enumerate(train_bar):
            camera_imgs = camera_imgs.to(device)
            lidar_bev = lidar_bev.to(device)
            extrinsics = extrinsics.to(device)
            intrinsics = intrinsics.to(device)
            target_waypoints = target_waypoints.to(device)
            
            if scaler is not None:
                with autocast(device_type='cuda'):
                    pred_wps = model(camera_imgs, lidar_bev, extrinsics, intrinsics)
                    loss = criterion(pred_wps, target_waypoints)
                    loss = loss / args.accumulation_steps
                scaler.scale(loss).backward()
            else:
                pred_wps = model(camera_imgs, lidar_bev, extrinsics, intrinsics)
                loss = criterion(pred_wps, target_waypoints)
                loss = loss / args.accumulation_steps
                loss.backward()
            
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
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * args.accumulation_steps
            train_bar.set_postfix({
                "loss": f"{(loss.item() * args.accumulation_steps):.4f}",
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
        val_bar = tqdm(val_loader, desc="Validating")
        
        with torch.no_grad():
            for camera_imgs, lidar_bev, extrinsics, intrinsics, target_waypoints in val_bar:
                camera_imgs = camera_imgs.to(device)
                lidar_bev = lidar_bev.to(device)
                extrinsics = extrinsics.to(device)
                intrinsics = intrinsics.to(device)
                target_waypoints = target_waypoints.to(device)
                
                if scaler is not None:
                    with autocast(device_type='cuda'):
                        pred_wps = model(camera_imgs, lidar_bev, extrinsics, intrinsics)
                        loss = criterion(pred_wps, target_waypoints)
                else:
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
            'scaler_state': scaler.state_dict() if scaler is not None else None,
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
    parser = argparse.ArgumentParser(description="Helioskrill Space-Temporal Training Pipeline")
    parser.add_argument("--data_dir",           default="./data/", help="Path to collected dataset directory")
    parser.add_argument("--epochs",             type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size",         type=int, default=2, help="Batch size per GPU iteration")
    parser.add_argument("--seq_len",            type=int, default=5, help="Temporal sequence length S")
    parser.add_argument("--stride",             type=int, default=5, help="Stride step between sequences")
    parser.add_argument("--resize_factor",      type=float, default=0.5, help="Image scaling factor")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr",                 type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--huber_delta",        type=float, default=1.0, help="Huber loss delta parameter")
    parser.add_argument("--model_dir",          default="./checkpoints/experimento_1/", help="Output directory for model checkpoints")
    parser.add_argument("--num_workers",        type=int, default=4, help="Data loading worker threads")
    parser.add_argument("--patience",           type=int, default=5, help="Patience epochs for Early Stopping")
    parser.add_argument("--min_delta",          type=float, default=1e-4, help="Minimum loss improvement delta")
    parser.add_argument("--resume",             action="store_true", help="Resume training loading weights from last_model.pth")
    parser.add_argument("--checkpoint_path",   default=None, help="Custom checkpoint path to load when --resume is active")
    
    args = parser.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    args.model_dir = os.path.abspath(args.model_dir)
    
    print("\n" + "="*60)
    print("  HELIOSKRILL — Training Pipeline")
    print("="*60)
    print(f"  Data:             {args.data_dir}")
    print(f"  Batch Size:       {args.batch_size}")
    print(f"  Sequence S:       {args.seq_len}")
    print(f"  Stride:           {args.stride}")
    print(f"  Resize Factor:    {args.resize_factor}")
    print(f"  Grad Accumulation:{args.accumulation_steps}")
    print(f"  Learning Rate:    {args.lr}")
    print(f"  Loss Function:    Huber (delta={args.huber_delta})")
    print(f"  Resume:           {args.resume}")
    print(f"  Output Dir:       {args.model_dir}")
    print("="*60 + "\n")
    
    train_model(args)
