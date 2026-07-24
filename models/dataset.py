import os
import csv
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

cv2.setNumThreads(0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARLA DATASET CLASS WITH SCALING AND MEMORY CACHING
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
# 2. METRICS (ADE, FDE, SHORTEST ANGULAR DISTANCE YAW ERROR)
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
    Evaluates temporal consistency: Horizon ADE, velocity error (m/s), acceleration error (m/s^2),
    and shortest angular distance yaw error (deg) strictly bounded in [0°, 180°].
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
    
    # Shortest angular distance between predicted and target yaw using atan2(sin(diff), cos(diff))
    # Eliminates wrap-around explosion (e.g. 8000°) and bounds error strictly within [0°, 180°]
    pred_yaw = pred_waypoints[..., 3]
    target_yaw = target_waypoints[..., 3]
    
    diff_yaw = pred_yaw - target_yaw
    shortest_diff = torch.atan2(torch.sin(diff_yaw), torch.cos(diff_yaw))
    yaw_error_deg = torch.mean(torch.abs(shortest_diff)).item() * (180.0 / np.pi)
    
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
