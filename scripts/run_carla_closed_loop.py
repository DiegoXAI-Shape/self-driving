#!/usr/bin/env python3
"""
run_carla_closed_loop.py
========================
Real-Time Closed-Loop Autonomous Driving Inference Engine in CARLA.

FUNCTIONALITY:
--------------
1. Connects to running CARLA simulator (resolving WSL 2 host IP automatically).
2. Spawns Tesla Model 3 ego vehicle equipped with 8 real Tesla RGB Cameras + LiDAR Sensor.
3. Listens to live sensor callbacks and maintains a temporal FIFO buffer (S=5 frames).
4. Runs real-time GPU inference loop using `BEVPerceptionNetV2` (Exp 2 or Exp 3).
5. Converts model waypoints into live vehicle control signals (steer, throttle, brake) via Pure Pursuit + PID.
6. Tracks vehicle with 3rd-person spectator camera in CARLA window.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import math
import collections
import argparse
import numpy as np
import torch
import cv2

cv2.setNumThreads(0)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.modules.BEV_perception_v2 import BEVPerceptionNetV2

try:
    import carla
except ImportError:
    print("[Error] CARLA Python API not found! Ensure carla library is installed in your python environment.")
    sys.exit(1)

# 8-Camera Tesla Setup Configuration
CAMERA_CONFIGS = [
    {"name": "front_main",    "pos": (1.35,  0.00, 1.45), "yaw":    0.0},
    {"name": "front_wide",    "pos": (1.35,  0.00, 1.45), "yaw":    0.0},
    {"name": "front_narrow",  "pos": (1.35,  0.00, 1.45), "yaw":    0.0},
    {"name": "left_b_pillar", "pos": (-0.20, -0.90, 1.30), "yaw":  -60.0},
    {"name": "right_b_pillar","pos": (-0.20,  0.90, 1.30), "yaw":   60.0},
    {"name": "left_repeater", "pos": ( 0.85, -0.95, 1.10), "yaw": -150.0},
    {"name": "right_repeater","pos": ( 0.85,  0.95, 1.10), "yaw":  150.0},
    {"name": "rear",          "pos": (-2.45,  0.00, 1.15), "yaw":  180.0},
]


def resolve_carla_host(host):
    """Auto-detects Windows Host IP address when running inside WSL 2."""
    if host.lower() in ["localhost", "127.0.0.1"]:
        is_wsl = os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop") or "microsoft-standard" in os.uname().release.lower()
        if is_wsl:
            try:
                import subprocess
                res = subprocess.check_output("ip route show default", shell=True).decode("utf-8")
                parts = res.split()
                if "via" in parts:
                    win_ip = parts[parts.index("via") + 1]
                    print(f"[WSL 2] Auto-detected Windows Host IP from default gateway: {win_ip}")
                    return win_ip
            except Exception:
                pass
            try:
                with open("/etc/resolv.conf", "r") as f:
                    for line in f:
                        if line.startswith("nameserver"):
                            win_ip = line.split()[1].strip()
                            print(f"[WSL 2] Auto-detected Windows Host IP from /etc/resolv.conf: {win_ip}")
                            return win_ip
            except Exception:
                pass
    return host


class LongitudinalPIDController:
    """PID Speed Controller with Anti-Windup Clamping."""
    def __init__(self, Kp=0.4, Ki=0.01, Kd=0.05):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error_sum = 0.0
        self.last_error = 0.0

    def run_step(self, target_speed, current_speed):
        error = target_speed - current_speed
        self.error_sum = np.clip(self.error_sum + error, -10.0, 10.0)
        error_diff = error - self.last_error
        self.last_error = error
        
        output = self.Kp * error + self.Ki * self.error_sum + self.Kd * error_diff
        return output


class PurePursuitLateralController:
    """Standard Pure Pursuit Steering Controller."""
    def __init__(self, wheel_base=2.87, lookahead_distance=4.0):
        self.wheel_base = wheel_base
        self.lookahead_distance = lookahead_distance
        self.last_steer = 0.0

    def run_step(self, waypoints):
        distances = np.hypot(waypoints[:, 0], waypoints[:, 1])
        target_idx = np.argmin(np.abs(distances - self.lookahead_distance))
        
        target_fwd = max(0.5, waypoints[target_idx, 0])
        target_lat = waypoints[target_idx, 1]
        
        ld = math.hypot(target_fwd, target_lat) + 1e-5
        alpha = math.atan2(target_lat, target_fwd)
        steer_angle = math.atan2(2.0 * self.wheel_base * math.sin(alpha), ld)
        
        # In dataset frame, +Y is Left. In CARLA VehicleControl, negative steer is Left.
        raw_steer = np.clip(-steer_angle / 0.61, -1.0, 1.0)
        smooth_steer = 0.70 * self.last_steer + 0.30 * raw_steer
        self.last_steer = smooth_steer
        return float(smooth_steer)


class CarlaClosedLoopRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Device] Running Closed-Loop Inference on: {self.device}")
        
        # Load Model
        self.model = self._load_model(args.checkpoint)
        
        # Controllers
        self.lon_controller = LongitudinalPIDController(Kp=0.4, Ki=0.01, Kd=0.05)
        self.lat_controller = PurePursuitLateralController(lookahead_distance=args.lookahead)
        
        # Live Sensor Buffer
        self.cam_buffers = [None] * 8
        self.lidar_raw_points = None
        self.actor_list = []
        
        # Recovery Controller State (Stuck / Escape Routine)
        self.stuck_start_time = 0.0
        self.recovery_active = False
        self.recovery_start_time = 0.0
        self.recovery_steer_dir = -1.0
        
        # FIFO Buffer S=5
        self.fifo_cams = collections.deque(maxlen=5)
        self.fifo_lidar = collections.deque(maxlen=5)

    def _load_model(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        print(f"[Loading] Inspecting checkpoint at: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model_state"] if (isinstance(checkpoint, dict) and "model_state" in checkpoint) else checkpoint
        
        poly_keys = [k for k in state_dict.keys() if "planning_head.poly_head" in k or "planning_head.fc" in k]
        use_poly = False
        if poly_keys:
            use_poly = True
            
        print(f"[Model] Initializing BEVPerceptionNetV2 ({'Multi-Head / Quintic Polynomial' if use_poly else 'Linear 40-Waypoint'})")
        model = BEVPerceptionNetV2(
            num_waypoints=10,
            bev_height=400,
            bev_width=400,
            use_polynomial_head=use_poly
        ).to(self.device)
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def build_lidar_bev_channel(self, points, grid_size=400, resolution=0.25):
        """Converts raw LiDAR point cloud into 5-channel BEV grid [5, 400, 400]."""
        bev = np.zeros((5, grid_size, grid_size), dtype=np.float32)
        if points is None or len(points) == 0:
            return bev
            
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        intensity = points[:, 3] if points.shape[1] > 3 else np.ones_like(x)
        
        valid = (x >= 0) & (x < 100) & (np.abs(y) < 50) & (z >= -3) & (z < 5)
        x, y, z, intensity = x[valid], y[valid], z[valid], intensity[valid]
        
        px = np.clip(np.int32(x / resolution), 0, grid_size - 1)
        py = np.clip(np.int32((y + 50.0) / resolution), 0, grid_size - 1)
        
        for i in range(len(x)):
            bev[0, px[i], py[i]] = max(bev[0, px[i], py[i]], z[i])  # z_max
            bev[1, px[i], py[i]] = min(bev[1, px[i], py[i]], z[i])  # z_min
            bev[2, px[i], py[i]] += z[i]                             # z_sum
            bev[3, px[i], py[i]] += 1.0                             # density
            bev[4, px[i], py[i]] = max(bev[4, px[i], py[i]], intensity[i]) # intensity
            
        return bev

    def setup_sensors(self, world, vehicle):
        bp_lib = world.get_blueprint_library()
        
        # 1. Spawn 8 RGB Cameras
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "800")
        cam_bp.set_attribute("image_size_y", "600")
        cam_bp.set_attribute("fov", "100")
        
        for idx, cfg in enumerate(CAMERA_CONFIGS):
            tf = carla.Transform(
                carla.Location(x=cfg["pos"][0], y=cfg["pos"][1], z=cfg["pos"][2]),
                carla.Rotation(pitch=0.0, yaw=cfg["yaw"], roll=0.0)
            )
            cam_actor = world.spawn_actor(cam_bp, tf, attach_to=vehicle)
            self.actor_list.append(cam_actor)
            
            def make_callback(cam_idx):
                def callback(image):
                    array = np.frombuffer(image.raw_data, dtype=np.uint8)
                    array = np.reshape(array, (image.height, image.width, 4))
                    rgb = array[:, :, :3][:, :, ::-1]  # BGRA to RGB
                    resized = cv2.resize(rgb, (400, 304), interpolation=cv2.INTER_AREA)
                    normalized = resized.astype(np.float32) / 255.0
                    self.cam_buffers[cam_idx] = np.transpose(normalized, (2, 0, 1))  # [3, 304, 400]
                return callback
                
            cam_actor.listen(make_callback(idx))

        # 2. Spawn 1 LiDAR Sensor
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", "50.0")
        lidar_bp.set_attribute("channels", "64")
        lidar_bp.set_attribute("points_per_second", "700000")
        lidar_bp.set_attribute("upper_fov", "2.0")
        lidar_bp.set_attribute("lower_fov", "-24.8")
        
        lidar_tf = carla.Transform(carla.Location(x=1.35, y=0.0, z=1.85))
        lidar_actor = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)
        self.actor_list.append(lidar_actor)
        
        def lidar_callback(data):
            points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
            self.lidar_raw_points = points[:, :4]
            
        lidar_actor.listen(lidar_callback)
        print(f"[CARLA] Successfully attached 8 RGB Cameras + 1 LiDAR Sensor to Ego Vehicle.")

    def run(self):
        target_host = resolve_carla_host(self.args.host)
        print(f"[CARLA] Connecting to CARLA server on {target_host}:{self.args.port}...")
        client = carla.Client(target_host, self.args.port)
        client.set_timeout(20.0)
        
        try:
            world = client.get_world()
        except Exception as e:
            print(f"[CARLA Error] Could not connect to CARLA at {target_host}:{self.args.port}. Make sure CARLA UE4 window is running on Windows.")
            raise e
        
        print(f"[CARLA] Connected to CARLA server on {target_host}:{self.args.port}")
        print(f"[CARLA] World Map: {world.get_map().name}")
        
        # Synchronous Mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)
        
        blueprint_library = world.get_blueprint_library()
        ego_bp = blueprint_library.find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "hero")
        
        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        
        lane_spawns = []
        for sp in spawn_points:
            wp = carla_map.get_waypoint(sp.location)
            if wp is not None and not wp.is_junction:
                lane_spawns.append(sp)
                
        if not lane_spawns:
            lane_spawns = spawn_points
            
        spawn_idx = self.args.spawn_idx % len(lane_spawns)
        spawn_point = lane_spawns[spawn_idx]
        
        vehicle = world.spawn_actor(ego_bp, spawn_point)
        self.actor_list.append(vehicle)
        print(f"[CARLA] Ego Vehicle spawned at lane point #{spawn_idx}: {spawn_point.location}")
        
        # Attach Sensors
        self.setup_sensors(world, vehicle)
        
        try:
            print("\n" + "="*60)
            print("  STARTING HELIOSKRILL CLOSED-LOOP AUTONOMOUS DRIVING")
            print("  Press Ctrl+C to stop execution.")
            print("="*60 + "\n")
            
            # Warmup sensor buffer
            for _ in range(10):
                world.tick()
                time.sleep(0.05)
                
            while True:
                world.tick()
                
                if any(c is None for c in self.cam_buffers):
                    continue
                    
                curr_cams = np.stack(self.cam_buffers, axis=0)
                curr_lidar_bev = self.build_lidar_bev_channel(self.lidar_raw_points)
                
                self.fifo_cams.append(curr_cams)
                self.fifo_lidar.append(curr_lidar_bev)
                
                while len(self.fifo_cams) < 5:
                    self.fifo_cams.append(curr_cams)
                    self.fifo_lidar.append(curr_lidar_bev)
                    
                cams_seq_np = np.stack(list(self.fifo_cams), axis=0)
                cam_tensor = torch.from_numpy(cams_seq_np).unsqueeze(0).to(self.device)
                lidar_bev = torch.from_numpy(curr_lidar_bev).unsqueeze(0).to(self.device)
                
                # Real 8-Camera Intrinsics and Extrinsics (matching CARLADataset)
                fov_rad = np.radians(100.0)
                fx = (400.0 / 2.0) / np.tan(fov_rad / 2.0)
                fy = (304.0 / 2.0) / np.tan(fov_rad / 2.0)
                K_mat = np.array([[fx, 0, 200.0], [0, fy, 152.0], [0, 0, 1.0]], dtype=np.float32)
                intrinsics_np = np.stack([K_mat] * 8, axis=0)
                
                extrinsics_np = np.zeros((8, 4, 4), dtype=np.float32)
                R_default = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)
                for i, cfg in enumerate(CAMERA_CONFIGS):
                    x, y, z = cfg["pos"]
                    yaw = np.radians(cfg["yaw"])
                    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                    R_yaw = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]], dtype=np.float32)
                    R_ego_to_cam = R_default @ R_yaw.T
                    t_vec = np.array([x, y, z], dtype=np.float32)
                    ext = np.eye(4, dtype=np.float32)
                    ext[:3, :3] = R_ego_to_cam
                    ext[:3, 3] = -R_ego_to_cam @ t_vec
                    extrinsics_np[i] = ext
                    
                extrinsics = torch.from_numpy(extrinsics_np).unsqueeze(0).to(self.device)
                intrinsics = torch.from_numpy(intrinsics_np).unsqueeze(0).to(self.device)
                
                # Velocity & Inference
                vel = vehicle.get_velocity()
                speed_mps = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                
                # Process Navigation Command input
                command_val = getattr(self.args, "command", 1)
                command_tensor = torch.tensor([command_val], dtype=torch.long, device=self.device)
                
                with torch.no_grad():
                    pred_out = self.model(cam_tensor, lidar_bev, extrinsics, intrinsics, command=command_tensor)
                    if isinstance(pred_out, dict):
                        pred_wps = pred_out["pred_waypoints"]
                        model_speed = pred_out["pred_speed"].item()
                        model_pedals = pred_out["pred_pedals"][0].cpu().numpy()
                    elif isinstance(pred_out, tuple):
                        pred_wps = pred_out[0]
                        model_speed = None
                        model_pedals = None
                    else:
                        pred_wps = pred_out
                        model_speed = None
                        model_pedals = None
                        
                wps = pred_wps[0].cpu().numpy()  # [10, 4]
                
                # 3rd-person spectator camera tracking
                spectator = world.get_spectator()
                veh_transform = vehicle.get_transform()
                spectator_offset = carla.Transform(
                    veh_transform.location + veh_transform.get_forward_vector() * -6.0 + carla.Location(z=3.0),
                    carla.Rotation(pitch=-15.0, yaw=veh_transform.rotation.yaw, roll=0.0)
                )
                spectator.set_transform(spectator_offset)
                
                # LiDAR Emergency Safety Shield Check:
                # Detects dense obstacle/wall within front bumper corridor (0.5m < X < 3.5m, |Y| < 1.0m)
                emergency_brake = False
                if self.lidar_raw_points is not None and len(self.lidar_raw_points) > 0:
                    pts = self.lidar_raw_points
                    front_mask = (pts[:, 0] > 0.5) & (pts[:, 0] < 3.5) & (np.abs(pts[:, 1]) < 1.0) & (pts[:, 2] > -0.5) & (pts[:, 2] < 1.5)
                    if np.sum(front_mask) > 10:  # Dense wall/vehicle obstacle
                        emergency_brake = True

                # Compute Pure Pursuit Controls
                steer_cmd = self.lat_controller.run_step(wps)
                
                if model_speed is not None:
                    target_speed = min(self.args.max_speed, max(0.0, model_speed))
                else:
                    target_dist_5 = math.hypot(wps[4, 0], wps[4, 1])
                    target_speed = min(self.args.max_speed, max(0.0, target_dist_5 / 1.0))
                    
                control_output = self.lon_controller.run_step(target_speed, speed_mps)
                
                if emergency_brake:
                    throttle_cmd = 0.0
                    brake_cmd = 1.0
                    status_text = " [SHIELD ALERT: Emergency Brake <3.5m!]"
                elif model_pedals is not None:
                    model_throttle, model_brake = model_pedals[0], model_pedals[1]
                    if model_brake > 0.3 or speed_mps > target_speed:
                        throttle_cmd = 0.0
                        brake_cmd = float(np.clip(max(model_brake, (speed_mps - target_speed) * 0.35), 0.1, 0.8))
                    else:
                        throttle_cmd = float(np.clip(control_output, 0.0, 0.45))
                        brake_cmd = 0.0
                    status_text = ""
                else:
                    if speed_mps > target_speed:
                        throttle_cmd = 0.0
                        brake_cmd = float(np.clip((speed_mps - target_speed) * 0.35, 0.1, 0.8))
                    else:
                        throttle_cmd = float(np.clip(control_output, 0.0, 0.45))
                        brake_cmd = 0.0
                    status_text = ""
                
                # Recovery Controller (Stuck Escape Routine):
                # Detects if throttle > 0.05 but vehicle is stuck (<0.3 m/s) for > 2.0s
                current_time = time.time()
                if not self.recovery_active:
                    if throttle_cmd > 0.05 and speed_mps < 0.3:
                        if self.stuck_start_time == 0.0:
                            self.stuck_start_time = current_time
                        elif (current_time - self.stuck_start_time) > 2.0:
                            self.recovery_active = True
                            self.recovery_start_time = current_time
                            self.recovery_steer_dir = -1.0 if steer_cmd >= 0 else 1.0
                    else:
                        self.stuck_start_time = 0.0
                else:
                    remaining_sec = 2.5 - (current_time - self.recovery_start_time)
                    if remaining_sec > 0:
                        reverse_control = carla.VehicleControl(
                            throttle=0.35,
                            steer=float(self.recovery_steer_dir * 0.4),
                            brake=0.0,
                            reverse=True
                        )
                        vehicle.apply_control(reverse_control)
                        print(f"\r[RECOVERY MODE ACTIVE] Stuck detected! Reversing out ({remaining_sec:.1f}s remaining)...", end="")
                        time.sleep(0.02)
                        continue
                    else:
                        self.recovery_active = False
                        self.stuck_start_time = 0.0

                control = carla.VehicleControl(
                    throttle=float(throttle_cmd),
                    steer=float(steer_cmd),
                    brake=float(brake_cmd)
                )
                vehicle.apply_control(control)
                
                print(f"\r[Driving LIVE] Speed: {speed_mps*3.6:5.1f} km/h | Steer: {steer_cmd:+5.2f} | Throttle: {throttle_cmd:4.2f} | Brake: {brake_cmd:4.2f}{status_text}", end="")
                time.sleep(0.02)
                
        finally:
            print("\n[CARLA] Cleaning up sensors, vehicle, and restoring settings...")
            for actor in reversed(self.actor_list):
                if actor is not None and actor.is_alive:
                    actor.destroy()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            print("[CARLA] Closed-Loop test finished cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helioskrill Closed-Loop Autonomous Driving Engine")
    parser.add_argument("--host",        default="localhost", help="CARLA server IP address")
    parser.add_argument("--port",        type=int, default=2000, help="CARLA server TCP port")
    parser.add_argument("--checkpoint",  default="./checkpoints/experimento_4/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--max_speed",   type=float, default=8.33, help="Max speed cap (m/s) [8.33 m/s = 30 km/h]")
    parser.add_argument("--lookahead",   type=float, default=4.0, help="Pure pursuit lookahead distance (m)")
    parser.add_argument("--spawn_idx",   type=int, default=0, help="Spawn point index on continuous lane")
    parser.add_argument("--command",     type=int, default=1, help="Navigation Command (1: LANE_FOLLOW, 2: TURN_LEFT, 3: TURN_RIGHT)")
    
    args = parser.parse_args()
    runner = CarlaClosedLoopRunner(args)
    runner.run()
