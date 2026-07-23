"""
carla_data_collector.py
=======================
Synchronized multi-sensor data extraction script for CARLA Simulator with automated ambient traffic.

FUNCTIONALITY
-------------
Collects multi-sensor data in synchronous mode for training the Helioskrill space-temporal model.
All sensors per frame are aligned in time at a fixed step rate (20 FPS).
Spawns background traffic (autonomous vehicles and pedestrians) automatically.

OUTPUT DIRECTORY STRUCTURE (under data/)
----------------------------------------
  Perception/
    episode_XXXX/
      cameras/
        cam_0/ ... cam_7/    <- 8 multi-view RGB camera views (.png)
      lidar/                 <- 5-channel BEV point cloud grid (.npy, 400x400)

  Location/
    episode_XXXX/
      location.csv           <- GPS + IMU + Ego vehicle pose per frame
      cameras_metadata.json  <- Intrinsics and extrinsics matrices

  Planning/
    episode_XXXX/
      waypoints.csv          <- Future relative waypoints from CARLA autopilot

  Control/
    episode_XXXX/
      control.csv            <- Throttle, brake, steer, handbrake per frame

  Prediction/
    episode_XXXX/
      actors.csv             <- Pose, velocity, and distance of nearby actors

USAGE
-----
  1. Launch CARLA Simulator (CarlaUE4.exe or ./CarlaUE4.sh)
  2. Run collector script:
       python models/utils/carla_data_collector.py
"""

import os
import sys
import time
import csv
import math
import queue
import argparse
import json
import random
import numpy as np

try:
    import carla
except ImportError:
    print("[ERROR] Could not import 'carla' module.")
    print("        Ensure CARLA egg is added to your PYTHONPATH.")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("[ERROR] Could not import 'opencv-python'. Install via: pip install opencv-python")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Default Configuration Parameters
CONFIG = {
    "host": "localhost",
    "port": 2000,
    "timeout": 20.0,

    "town": "Town03",
    "fps": 20,
    "num_episodes": 5,
    "frames_per_episode": 600,
    "warmup_frames": 40,

    "num_vehicles": 40,
    "num_walkers": 25,

    "cam_width": 800,
    "cam_height": 600,
    "cam_fov": 100,

    "lidar_range": 50.0,
    "lidar_points_per_second": 700_000,
    "lidar_channels": 64,
    "lidar_upper_fov": 2.0,
    "lidar_lower_fov": -24.8,

    "bev_range_m": 50.0,
    "bev_resolution": 0.25,

    "num_waypoints": 10,
    "waypoint_spacing": 2.0,

    "output_root": os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data")
    ),
}

# 8-Camera Multi-View Setup (Tesla Model 3 Configuration)
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


def get_next_episode_id(output_root: str) -> int:
    """
    Finds the next episode index to avoid overwriting existing datasets.
    """
    location_dir = os.path.join(output_root, "Location")
    if not os.path.exists(location_dir):
        return 0
    ep_ids = []
    for name in os.listdir(location_dir):
        if name.startswith("episode_") and os.path.isdir(os.path.join(location_dir, name)):
            try:
                ep_id = int(name.split("_")[1])
                ep_ids.append(ep_id)
            except (IndexError, ValueError):
                continue
    if not ep_ids:
        return 0
    return max(ep_ids) + 1


def build_episode_dirs(output_root: str, episode_id: int) -> dict:
    """
    Creates episode subdirectories for Perception, Location, Planning, Control, and Prediction.
    """
    ep_str = f"episode_{episode_id:04d}"
    paths = {
        "cameras": [
            os.path.join(output_root, "Perception", ep_str, "cameras", f"cam_{i}")
            for i in range(len(CAMERA_CONFIGS))
        ],
        "lidar":      os.path.join(output_root, "Perception", ep_str, "lidar"),
        "location":   os.path.join(output_root, "Location",   ep_str),
        "planning":   os.path.join(output_root, "Planning",   ep_str),
        "control":    os.path.join(output_root, "Control",    ep_str),
        "prediction": os.path.join(output_root, "Prediction", ep_str),
    }

    for cam_dir in paths["cameras"]:
        os.makedirs(cam_dir, exist_ok=True)
    for key in ("lidar", "location", "planning", "control", "prediction"):
        os.makedirs(paths[key], exist_ok=True)

    return paths


def open_csv_writers(paths: dict) -> dict:
    """
    Opens CSV file handles for logging episode telemetry.
    """
    writers = {}

    loc_path = os.path.join(paths["location"], "location.csv")
    loc_file = open(loc_path, "w", newline="", encoding="utf-8")
    loc_writer = csv.writer(loc_file)
    loc_writer.writerow([
        "frame", "gps_lat", "gps_lon", "gps_alt",
        "ego_x", "ego_y", "ego_z", "ego_yaw", "ego_pitch", "ego_roll",
        "imu_accel_x", "imu_accel_y", "imu_accel_z",
        "imu_gyro_x",  "imu_gyro_y",  "imu_gyro_z", "speed_mps"
    ])
    writers["location"] = (loc_file, loc_writer)

    plan_path = os.path.join(paths["planning"], "waypoints.csv")
    plan_file = open(plan_path, "w", newline="", encoding="utf-8")
    plan_writer = csv.writer(plan_file)
    wp_cols = ["frame"]
    for i in range(CONFIG["num_waypoints"]):
        wp_cols += [f"wp_{i}_rel_x", f"wp_{i}_rel_y", f"wp_{i}_rel_z", f"wp_{i}_rel_yaw"]
    plan_writer.writerow(wp_cols)
    writers["planning"] = (plan_file, plan_writer)

    ctrl_path = os.path.join(paths["control"], "control.csv")
    ctrl_file = open(ctrl_path, "w", newline="", encoding="utf-8")
    ctrl_writer = csv.writer(ctrl_file)
    ctrl_writer.writerow(["frame", "throttle", "brake", "steer", "hand_brake", "reverse"])
    writers["control"] = (ctrl_file, ctrl_writer)

    pred_path = os.path.join(paths["prediction"], "actors.csv")
    pred_file = open(pred_path, "w", newline="", encoding="utf-8")
    pred_writer = csv.writer(pred_file)
    pred_writer.writerow([
        "frame", "actor_id", "actor_type", "x", "y", "z", "yaw",
        "vel_x", "vel_y", "vel_z", "speed_mps", "rel_x", "rel_y", "rel_dist_m"
    ])
    writers["prediction"] = (pred_file, pred_writer)

    return writers


def close_csv_writers(writers: dict):
    for name, (fh, _) in writers.items():
        fh.close()


def lidar_to_bev_grid_vectorized(point_cloud: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Vectorized LiDAR raw point cloud to 5-channel BEV Grid converter (Z_max, Z_diff, Z_mean, density, intensity).
    """
    bev_range = cfg["bev_range_m"]
    res = cfg["bev_resolution"]
    grid_size = int(2 * bev_range / res)

    x, y, z, intensity = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], point_cloud[:, 3]

    mask = (x > -bev_range) & (x < bev_range) & (y > -bev_range) & (y < bev_range) & (z > -3.0) & (z < 5.0)
    x, y, z, intensity = x[mask], y[mask], z[mask], intensity[mask]

    row_idx = np.clip(((bev_range - x) / res).astype(np.int32), 0, grid_size - 1)
    col_idx = np.clip(((bev_range - y) / res).astype(np.int32), 0, grid_size - 1)

    bev_grid = np.zeros((5, grid_size, grid_size), dtype=np.float32)

    if len(z) == 0:
        return bev_grid

    np.maximum.at(bev_grid[0], (row_idx, col_idx), z)
    np.add.at(bev_grid[2], (row_idx, col_idx), z)
    np.add.at(bev_grid[3], (row_idx, col_idx), 1.0)
    np.maximum.at(bev_grid[4], (row_idx, col_idx), intensity)

    has_points = bev_grid[3] > 0
    bev_grid[2] = np.where(has_points, bev_grid[2] / np.maximum(bev_grid[3], 1.0), 0.0)
    bev_grid[3] = np.clip(bev_grid[3] / 64.0, 0.0, 1.0)

    return bev_grid


def spawn_cameras(world, vehicle, cfg: dict) -> list:
    blueprint_library = world.get_blueprint_library()
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(cfg["cam_width"]))
    cam_bp.set_attribute("image_size_y", str(cfg["cam_height"]))
    cam_bp.set_attribute("fov",          str(cfg["cam_fov"]))

    cameras = []
    for cam_cfg in CAMERA_CONFIGS:
        x, y, z   = cam_cfg["pos"]
        yaw       = cam_cfg["yaw"]
        transform = carla.Transform(
            carla.Location(x=x, y=y, z=z),
            carla.Rotation(yaw=yaw)
        )
        sensor = world.spawn_actor(cam_bp, transform, attach_to=vehicle)
        cameras.append(sensor)
    return cameras


def save_camera_metadata(output_dir: str, cfg: dict):
    w, h, fov = cfg["cam_width"], cfg["cam_height"], cfg["cam_fov"]
    focal_length = w / (2.0 * math.tan(fov * math.pi / 360.0))
    
    K = [
        [focal_length, 0.0, w / 2.0],
        [0.0, focal_length, h / 2.0],
        [0.0, 0.0, 1.0]
    ]

    metadata = {
        "intrinsics": K,
        "cameras": CAMERA_CONFIGS
    }

    with open(os.path.join(output_dir, "cameras_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def spawn_lidar(world, vehicle, cfg: dict):
    blueprint_library = world.get_blueprint_library()
    lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range",              str(cfg["lidar_range"]))
    lidar_bp.set_attribute("points_per_second",  str(cfg["lidar_points_per_second"]))
    lidar_bp.set_attribute("channels",           str(cfg["lidar_channels"]))
    lidar_bp.set_attribute("upper_fov",          str(cfg["lidar_upper_fov"]))
    lidar_bp.set_attribute("lower_fov",          str(cfg["lidar_lower_fov"]))
    lidar_bp.set_attribute("rotation_frequency", str(cfg["fps"]))

    lidar_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
    return world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)


def spawn_imu(world, vehicle):
    return world.spawn_actor(world.get_blueprint_library().find("sensor.other.imu"), carla.Transform(), attach_to=vehicle)


def spawn_gnss(world, vehicle):
    return world.spawn_actor(world.get_blueprint_library().find("sensor.other.gnss"), carla.Transform(), attach_to=vehicle)


def spawn_ambient_traffic(client, world, num_vehicles: int, num_walkers: int, traffic_manager, ego_spawn_point) -> tuple:
    """
    Spawns background vehicles and pedestrian actors securely without triggering Windows C++ Boost crashes.
    """
    blueprints = world.get_blueprint_library()
    vehicle_blueprints = blueprints.filter("vehicle.*")
    walker_blueprints = blueprints.filter("walker.pedestrian.*")

    vehicle_blueprints = [x for x in vehicle_blueprints if int(x.get_attribute("number_of_wheels")) == 4]
    vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith("isetta")]
    vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith("carlacola")]
    vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith("cybertruck")]

    spawn_points = world.get_map().get_spawn_points()
    spawn_points = [sp for sp in spawn_points if sp.location.distance(ego_spawn_point.location) > 10.0]
    random.shuffle(spawn_points)

    vehicles_list = []
    walkers_list = []

    num_vehicles = min(num_vehicles, len(spawn_points))
    for i in range(num_vehicles):
        blueprint = random.choice(vehicle_blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")
        
        spawn_point = spawn_points[i]
        vehicle = world.try_spawn_actor(blueprint, spawn_point)
        if vehicle is not None:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.auto_lane_change(vehicle, True)
            traffic_manager.ignore_lights_percentage(vehicle, 10.0)
            vehicles_list.append(vehicle)

    walker_spawn_points = spawn_points[num_vehicles:]
    random.shuffle(walker_spawn_points)
    
    num_walkers = min(num_walkers, len(walker_spawn_points))
    for i in range(num_walkers):
        sp = walker_spawn_points[i]
        
        yaw_rad = np.radians(sp.rotation.yaw)
        right_x = -np.sin(yaw_rad)
        right_y = np.cos(yaw_rad)
        
        side = random.choice([-3.5, 3.5])
        spawn_loc = carla.Location(
            x=sp.location.x + right_x * side,
            y=sp.location.y + right_y * side,
            z=sp.location.z + 0.2
        )
        
        walker_bp = random.choice(walker_blueprints)
        walker = world.try_spawn_actor(walker_bp, carla.Transform(spawn_loc))
        if walker is not None:
            walkers_list.append(walker)
            
            walk_yaw = random.uniform(0, 360)
            walk_yaw_rad = np.radians(walk_yaw)
            direction = carla.Vector3D(
                x=np.cos(walk_yaw_rad),
                y=np.sin(walk_yaw_rad),
                z=0.0
            )
            speed = 1.0 + random.random() * 1.5
            walker.apply_control(carla.WalkerControl(direction=direction, speed=speed))

    print(f"[OK] Ambient traffic spawned: {len(vehicles_list)} vehicles and {len(walkers_list)} pedestrians.")
    return vehicles_list, walkers_list, []


def get_future_waypoints(world, vehicle, num_waypoints: int, spacing: float) -> list:
    amap = world.get_map()
    ego_tf = vehicle.get_transform()
    ego_loc = ego_tf.location
    ego_yaw_rad = math.radians(ego_tf.rotation.yaw)

    current_wp = amap.get_waypoint(ego_loc, project_to_road=True)
    waypoints = []
    wp = current_wp

    cos_y = math.cos(-ego_yaw_rad)
    sin_y = math.sin(-ego_yaw_rad)

    for _ in range(num_waypoints):
        next_wps = wp.next(spacing)
        if not next_wps:
            break
        wp = next_wps[0]
        
        dx = wp.transform.location.x - ego_loc.x
        dy = wp.transform.location.y - ego_loc.y
        dz = wp.transform.location.z - ego_loc.z

        rel_x = dx * cos_y - dy * sin_y
        rel_y = dx * sin_y + dy * cos_y
        rel_yaw = wp.transform.rotation.yaw - ego_tf.rotation.yaw

        waypoints.append({
            "rel_x": rel_x,
            "rel_y": rel_y,
            "rel_z": dz,
            "rel_yaw": rel_yaw,
        })

    while len(waypoints) < num_waypoints:
        fallback = waypoints[-1].copy() if waypoints else {"rel_x": 0.0, "rel_y": 0.0, "rel_z": 0.0, "rel_yaw": 0.0}
        waypoints.append(fallback)

    return waypoints


def get_nearby_actors(world, ego_vehicle, max_dist: float = 50.0) -> list:
    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = ego_tf.rotation.yaw
    actors_data = []

    all_actors = world.get_actors()
    vehicles    = all_actors.filter("vehicle.*")
    pedestrians = all_actors.filter("walker.pedestrian.*")

    yaw_rad = math.radians(ego_yaw)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    for actor_list, actor_type in [(vehicles, "vehicle"), (pedestrians, "pedestrian")]:
        for actor in actor_list:
            if actor.id == ego_vehicle.id:
                continue

            loc = actor.get_transform().location
            dist = ego_loc.distance(loc)

            if dist > max_dist:
                continue

            vel = actor.get_velocity()
            speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            yaw = actor.get_transform().rotation.yaw

            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y

            rel_x = dx * cos_y + dy * sin_y
            rel_y = -dx * sin_y + dy * cos_y

            actors_data.append({
                "actor_id":   actor.id,
                "actor_type": actor_type,
                "x": loc.x, "y": loc.y, "z": loc.z,
                "yaw": yaw,
                "vel_x": vel.x, "vel_y": vel.y, "vel_z": vel.z,
                "speed_mps": speed,
                "rel_x": rel_x, "rel_y": rel_y,
                "rel_dist_m": dist,
            })

    return actors_data


def run_episode(client, world, episode_id: int, cfg: dict):
    print(f"\n{'='*60}")
    print(f"  STARTING EPISODE {episode_id:04d}  |  {cfg['town']}")
    print(f"{'='*60}")

    paths   = build_episode_dirs(cfg["output_root"], episode_id)
    writers = open_csv_writers(paths)
    save_camera_metadata(paths["location"], cfg)

    bp_lib  = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available on this map.")

    spawn_tf = random.choice(spawn_points)
    vehicle  = world.spawn_actor(vehicle_bp, spawn_tf)
    print(f"[OK] Ego Vehicle spawned (ID={vehicle.id})")

    traffic_manager = client.get_trafficmanager()
    vehicle.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.ignore_lights_percentage(vehicle, 0.0)
    traffic_manager.distance_to_leading_vehicle(vehicle, 3.0)

    ambient_vehicles, ambient_walkers, ambient_controllers = spawn_ambient_traffic(
        client, world,
        num_vehicles=cfg["num_vehicles"],
        num_walkers=cfg["num_walkers"],
        traffic_manager=traffic_manager,
        ego_spawn_point=spawn_tf
    )

    cameras = spawn_cameras(world, vehicle, cfg)
    lidar   = spawn_lidar(world, vehicle, cfg)
    imu     = spawn_imu(world, vehicle)
    gnss    = spawn_gnss(world, vehicle)

    all_sensors = cameras + [lidar, imu, gnss]

    cam_queues  = [queue.Queue() for _ in cameras]
    lidar_queue = queue.Queue()
    imu_queue   = queue.Queue()
    gnss_queue  = queue.Queue()

    for i, cam in enumerate(cameras):
        cam.listen(cam_queues[i].put)
    lidar.listen(lidar_queue.put)
    imu.listen(imu_queue.put)
    gnss.listen(gnss_queue.put)

    total_frames  = cfg["frames_per_episode"] + cfg["warmup_frames"]
    saved_frames  = 0
    skipped_timeout = 0

    try:
        for tick in tqdm(range(total_frames), desc=f"Recording Ep {episode_id:04d}", unit="frame"):
            world.tick()

            timeout = 2.0 / cfg["fps"]
            try:
                cam_images  = [q.get(timeout=timeout) for q in cam_queues]
                lidar_data  = lidar_queue.get(timeout=timeout)
                imu_data    = imu_queue.get(timeout=timeout)
                gnss_data   = gnss_queue.get(timeout=timeout)
            except queue.Empty:
                skipped_timeout += 1
                continue

            if tick < cfg["warmup_frames"]:
                continue

            frame_id = saved_frames

            # Save Cameras PNGs
            for i, img_data in enumerate(cam_images):
                array = np.frombuffer(img_data.raw_data, dtype=np.uint8)
                array = array.reshape((cfg["cam_height"], cfg["cam_width"], 4))
                bgr = array[:, :, :3]
                filename = os.path.join(paths["cameras"][i], f"frame_{frame_id:06d}.png")
                cv2.imwrite(filename, bgr)

            # Save LiDAR 5-channel BEV Tensor
            pts_raw = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
            pts_raw = pts_raw.reshape(-1, 4)
            bev_grid = lidar_to_bev_grid_vectorized(pts_raw, cfg)
            lidar_filename = os.path.join(paths["lidar"], f"frame_{frame_id:06d}.npy")
            np.save(lidar_filename, bev_grid)

            # Telemetry logging
            ego_tf  = vehicle.get_transform()
            ego_vel = vehicle.get_velocity()
            speed   = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)

            loc_row = [
                frame_id, gnss_data.latitude, gnss_data.longitude, gnss_data.altitude,
                ego_tf.location.x, ego_tf.location.y, ego_tf.location.z,
                ego_tf.rotation.yaw, ego_tf.rotation.pitch, ego_tf.rotation.roll,
                imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z,
                imu_data.gyroscope.x,     imu_data.gyroscope.y,     imu_data.gyroscope.z, speed
            ]
            writers["location"][1].writerow(loc_row)

            future_wps = get_future_waypoints(world, vehicle, num_waypoints=cfg["num_waypoints"], spacing=cfg["waypoint_spacing"])
            plan_row = [frame_id]
            for wp in future_wps:
                plan_row += [wp["rel_x"], wp["rel_y"], wp["rel_z"], wp["rel_yaw"]]
            writers["planning"][1].writerow(plan_row)

            ctrl = vehicle.get_control()
            ctrl_row = [
                frame_id, round(ctrl.throttle, 6), round(ctrl.brake, 6), round(ctrl.steer, 6),
                int(ctrl.hand_brake), int(ctrl.reverse)
            ]
            writers["control"][1].writerow(ctrl_row)

            nearby = get_nearby_actors(world, vehicle, max_dist=cfg["bev_range_m"])
            for actor in nearby:
                pred_row = [
                    frame_id, actor["actor_id"], actor["actor_type"],
                    actor["x"], actor["y"], actor["z"], actor["yaw"],
                    actor["vel_x"], actor["vel_y"], actor["vel_z"], actor["speed_mps"],
                    actor["rel_x"], actor["rel_y"], actor["rel_dist_m"]
                ]
                writers["prediction"][1].writerow(pred_row)

            saved_frames += 1

    finally:
        close_csv_writers(writers)

        for sensor in all_sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()

        if vehicle.is_alive:
            vehicle.destroy()

        for controller in ambient_controllers:
            if controller.is_alive:
                controller.stop()
                controller.destroy()

        for walker in ambient_walkers:
            if walker.is_alive:
                walker.destroy()

        for v in ambient_vehicles:
            if v.is_alive:
                v.destroy()

        print(f"[OK] Episode {episode_id:04d} finished safely.")


def main():
    parser = argparse.ArgumentParser(description="Helioskrill — Synchronized CARLA Data Extractor")
    parser.add_argument("--host",     default=CONFIG["host"])
    parser.add_argument("--port",     default=CONFIG["port"], type=int)
    parser.add_argument("--town",     default=CONFIG["town"])
    parser.add_argument("--episodes", default=CONFIG["num_episodes"], type=int)
    parser.add_argument("--frames",   default=CONFIG["frames_per_episode"], type=int)
    parser.add_argument("--vehicles", default=CONFIG["num_vehicles"], type=int)
    parser.add_argument("--walkers",  default=CONFIG["num_walkers"], type=int)
    args = parser.parse_args()

    CONFIG["host"] = args.host
    CONFIG["port"] = args.port
    CONFIG["town"] = args.town
    CONFIG["num_episodes"] = args.episodes
    CONFIG["frames_per_episode"] = args.frames
    CONFIG["num_vehicles"] = args.vehicles
    CONFIG["num_walkers"] = args.walkers

    print("\n" + "="*60)
    print("  HELIOSKRILL — Synchronized Data Collector")
    print("="*60)
    print(f"  Town:          {CONFIG['town']}")
    print(f"  Episodes:      {CONFIG['num_episodes']}")
    print(f"  Traffic Fleet: {CONFIG['num_vehicles']} vehicles, {CONFIG['num_walkers']} walkers")
    print(f"  Output Dir:    {CONFIG['output_root']}")
    print("="*60 + "\n")

    try:
        client = carla.Client(CONFIG["host"], CONFIG["port"])
        client.set_timeout(CONFIG["timeout"])
        client.load_world(CONFIG["town"])
        world = client.get_world()
        
        print("[OK] Connected to CARLA Simulator.")
    except RuntimeError as e:
        print(f"[ERROR] Connection failed: {e}")
        sys.exit(1)

    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 1.0 / CONFIG["fps"]
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(int(time.time()))

    try:
        start_ep_id = get_next_episode_id(CONFIG["output_root"])
        print(f"[INFO] Starting data capture at Episode ID: {start_ep_id:04d}")
        for i in range(CONFIG["num_episodes"]):
            ep_id = start_ep_id + i
            run_episode(client, world, episode_id=ep_id, cfg=CONFIG)
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[INFO] Data collection interrupted by user.")

    finally:
        print("\n[Cleanup] Restoring asynchronous mode...")
        if 'world' in locals() and 'settings' in locals():
            try:
                if 'traffic_manager' in locals():
                    traffic_manager.set_synchronous_mode(False)
            except Exception as e:
                print(f"[Warning] Failed to set Traffic Manager to async: {e}")
            try:
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
            except Exception as e:
                print(f"[Warning] Failed to apply async settings to world: {e}")
        print("[OK] Finished safely.")
        os._exit(0)


if __name__ == "__main__":
    main()
