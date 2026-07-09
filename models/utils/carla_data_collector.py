"""
carla_data_collector.py
=======================
Script de extracción de datos sincronizados desde CARLA Simulator con generación de tráfico.

OBJETIVO
--------
Recolectar datos multi-sensor en modo sincrónico para entrenar el modelo
Helioskrill (ViM + Knowledge Distillation). Todos los datos de un mismo
frame están perfectamente alineados en el tiempo.
Spawnea tráfico de fondo (vehículos y peatones con IA) de forma automática.

ESTRUCTURA DE SALIDA  (relativa a src/data/)
--------------------------------------------
  Perception/CARLA/
    episode_XXXX/
      cameras/
        cam_0/  cam_1/  ...  cam_7/   <- 8 vistas RGB   (.png)
      lidar/                           <- Nube de puntos BEV (.npy, 5 canales)

  Location/
    episode_XXXX/
      location.csv                     <- GPS + IMU por frame

  Planning/
    episode_XXXX/
      waypoints.csv                    <- Waypoints futuros del autopiloto

  Control/
    episode_XXXX/
      control.csv                      <- Throttle, brake, steer por frame

  Prediction/
    episode_XXXX/
      actors.csv                       <- Pose + velocidad de todos los actores

REQUISITOS
----------
  pip install carla numpy opencv-python tqdm

CÓMO CORRER
-----------
  1. Abre CARLA Simulator (CarlaUE4.exe o ./CarlaUE4.sh)
  2. Ejecuta este script:
       python carla_data_collector.py
"""

import os
import sys
import time
import csv
import math
import queue
import argparse
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Verificación de la librería de CARLA
# ─────────────────────────────────────────────────────────────────────────────
try:
    import carla
except ImportError:
    print("[ERROR] No se encontró el módulo 'carla'.")
    print("        Asegúrate de que el egg de CARLA esté en tu PYTHONPATH.")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("[ERROR] No se encontró 'opencv-python'. Instálalo con:  pip install opencv-python")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Conexión a CARLA ──────────────────────────────────────────────────────
    "host": "localhost",
    "port": 2000,
    "timeout": 20.0,          # segundos de espera para conectar

    # ── Simulación ────────────────────────────────────────────────────────────
    "town": "Town03",          # Mapa de CARLA a cargar
    "fps": 20,                 # Frames por segundo en modo sincrónico
    "num_episodes": 5,         # Cuántos episodios (recorridos) grabar
    "frames_per_episode": 600, # Frames por episodio  (600 / 20fps = 30 segundos)
    "warmup_frames": 40,       # Frames iniciales a descartar (autopiloto e IA iniciando)

    # ── Tráfico y Peatones (Ambiental) ────────────────────────────────────────
    "num_vehicles": 40,        # Número de vehículos de fondo a generar
    "num_walkers": 25,         # Número de peatones de fondo a generar

    # ── Cámaras ───────────────────────────────────────────────────────────────
    "cam_width":  800,         # Resolución de captura
    "cam_height": 600,
    "cam_fov":    100,         # Campo de visión

    # ── LiDAR ─────────────────────────────────────────────────────────────────
    "lidar_range":      50.0,
    "lidar_points_per_second": 700_000,
    "lidar_channels":   64,
    "lidar_upper_fov":   2.0,
    "lidar_lower_fov": -24.8,

    # ── Grid BEV del LiDAR ────────────────────────────────────────────────────
    "bev_range_m":    50.0,    # metros hacia adelante/atrás/izquierda/derecha
    "bev_resolution": 0.25,    # 0.25m → grid 400×400

    # ── Waypoints de Planning ─────────────────────────────────────────────────
    "num_waypoints":    10,
    "waypoint_spacing": 2.0,

    # ── Rutas de salida ───────────────────────────────────────────────────────
    "output_root": os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data")
    ),
}

# Configuración Tesla para 8 cámaras
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

# ─────────────────────────────────────────────────────────────────────────────
# CREACIÓN DE DIRECTORIOS Y CSV
# ─────────────────────────────────────────────────────────────────────────────

def build_episode_dirs(output_root: str, episode_id: int) -> dict:
    ep_str = f"episode_{episode_id:04d}"
    paths = {
        "cameras": [
            os.path.join(output_root, "Perception", "CARLA", ep_str, "cameras", f"cam_{i}")
            for i in range(len(CAMERA_CONFIGS))
        ],
        "lidar":      os.path.join(output_root, "Perception", "CARLA", ep_str, "lidar"),
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
        wp_cols += [f"wp_{i}_x", f"wp_{i}_y", f"wp_{i}_z", f"wp_{i}_yaw"]
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

# ─────────────────────────────────────────────────────────────────────────────
# LIDAR BEV CONVERTER
# ─────────────────────────────────────────────────────────────────────────────

def lidar_to_bev_grid(point_cloud: np.ndarray, cfg: dict) -> np.ndarray:
    bev_range = cfg["bev_range_m"]
    res       = cfg["bev_resolution"]
    grid_size = int(2 * bev_range / res)

    z_max_grid   = np.full((grid_size, grid_size), -np.inf, dtype=np.float32)
    z_min_grid   = np.full((grid_size, grid_size),  np.inf, dtype=np.float32)
    z_sum_grid   = np.zeros((grid_size, grid_size),         dtype=np.float32)
    count_grid   = np.zeros((grid_size, grid_size),         dtype=np.float32)
    intens_grid  = np.zeros((grid_size, grid_size),         dtype=np.float32)

    x, y, z, intensity = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], point_cloud[:, 3]

    mask = (
        (x > -bev_range) & (x < bev_range) &
        (y > -bev_range) & (y < bev_range) &
        (z > -3.0) & (z < 5.0)
    )
    x, y, z, intensity = x[mask], y[mask], z[mask], intensity[mask]

    row_idx = ((bev_range - x) / res).astype(np.int32)
    col_idx = ((bev_range - y) / res).astype(np.int32)

    row_idx = np.clip(row_idx, 0, grid_size - 1)
    col_idx = np.clip(col_idx, 0, grid_size - 1)

    for r, c, zi, ii in zip(row_idx, col_idx, z, intensity):
        if zi > z_max_grid[r, c]:
            z_max_grid[r, c]  = zi
        if zi < z_min_grid[r, c]:
            z_min_grid[r, c]  = zi
        z_sum_grid[r, c]  += zi
        count_grid[r, c]  += 1
        if ii > intens_grid[r, c]:
            intens_grid[r, c] = ii

    has_points = count_grid > 0
    z_max_grid[~has_points]  = 0.0
    z_min_grid[~has_points]  = 0.0
    z_diff_grid = np.where(has_points, z_max_grid - z_min_grid, 0.0).astype(np.float32)
    z_mean_grid = np.where(has_points, z_sum_grid / np.maximum(count_grid, 1), 0.0).astype(np.float32)

    max_density = 64.0
    density_grid = np.clip(count_grid / max_density, 0.0, 1.0).astype(np.float32)

    bev_grid = np.stack([
        z_max_grid,
        z_diff_grid,
        z_mean_grid,
        density_grid,
        intens_grid.astype(np.float32),
    ], axis=0)

    return bev_grid

# ─────────────────────────────────────────────────────────────────────────────
# SPAWN DE SENSORES Y AUTOPILOTO EGO
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE TRÁFICO (VEHÍCULOS Y PEATONES DE FONDO)
# ─────────────────────────────────────────────────────────────────────────────

def spawn_ambient_traffic(client, world, num_vehicles: int, num_walkers: int, traffic_manager, ego_spawn_point) -> tuple:
    """
    Spawnea vehículos y peatones de fondo controlados de forma autónoma.
    Evita spawnearlos encima de la posición inicial del ego.
    """
    blueprints = world.get_blueprint_library()
    vehicle_blueprints = blueprints.filter("vehicle.*")
    walker_blueprints = blueprints.filter("walker.pedestrian.*")

    # Filtrar vehículos que puedan causar problemas de colisión en CARLA
    vehicle_blueprints = [x for x in vehicle_blueprints if int(x.get_attribute("number_of_wheels")) == 4]
    vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith("isetta")]
    vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith("carlacola")]
    vehicle_blueprints = [x for x in vehicle_blueprints if not x.id.endswith("cybertruck")]

    spawn_points = world.get_map().get_spawn_points()
    
    # Filtrar spawn points para no spawnear justo encima del ego
    spawn_points = [sp for sp in spawn_points if sp.location.distance(ego_spawn_point.location) > 10.0]
    import random
    random.shuffle(spawn_points)

    vehicles_list = []
    walkers_list = []
    controllers_list = []

    # 1. Spawnear Vehículos de Fondo
    num_vehicles = min(num_vehicles, len(spawn_points))
    for i in range(num_vehicles):
        blueprint = random.choice(vehicle_blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")
        
        spawn_point = spawn_points[i]
        vehicle = world.try_spawn_actor(blueprint, spawn_point)
        if vehicle is not None:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            # Hacer que Traffic Manager los controle de forma fluida e ignorando semáforos a veces
            traffic_manager.auto_lane_change(vehicle, True)
            traffic_manager.ignore_lights_percentage(vehicle, 10.0) # Realismo
            vehicles_list.append(vehicle)

    # 2. Spawnear Peatones de Fondo (Walkers) en aceras usando spawn points desviados
    # Esto previene el uso de get_random_location_from_navigation() que causa crashes en Windows
    walker_spawn_points = spawn_points[num_vehicles:]
    import random
    random.shuffle(walker_spawn_points)
    
    num_walkers = min(num_walkers, len(walker_spawn_points))
    for i in range(num_walkers):
        sp = walker_spawn_points[i]
        
        # Desviar la posición del spawn de vehículos lateralmente para ubicarlo en la acera
        yaw_rad = np.radians(sp.rotation.yaw)
        right_x = -np.sin(yaw_rad)
        right_y = np.cos(yaw_rad)
        
        # Desplazar 3.5m a la izquierda o derecha aleatoriamente
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
            
            # Movimiento por control físico directo sin usar la malla de navegación
            walk_yaw = random.uniform(0, 360)
            walk_yaw_rad = np.radians(walk_yaw)
            direction = carla.Vector3D(
                x=np.cos(walk_yaw_rad),
                y=np.sin(walk_yaw_rad),
                z=0.0
            )
            speed = 1.0 + random.random() * 1.5
            walker.apply_control(carla.WalkerControl(direction=direction, speed=speed))

    print(f"[OK] Tráfico ambiental generado: {len(vehicles_list)} vehículos y {len(walkers_list)} peatones.")
    return vehicles_list, walkers_list, []

# ─────────────────────────────────────────────────────────────────────────────
# OTRAS UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def get_future_waypoints(world, vehicle, num_waypoints: int, spacing: float) -> list:
    amap       = world.get_map()
    ego_tf     = vehicle.get_transform()
    current_wp = amap.get_waypoint(ego_tf.location, project_to_road=True)

    waypoints  = []
    wp         = current_wp

    for _ in range(num_waypoints):
        next_wps = wp.next(spacing)
        if not next_wps:
            break
        wp = next_wps[0]
        waypoints.append({
            "x":   wp.transform.location.x,
            "y":   wp.transform.location.y,
            "z":   wp.transform.location.z,
            "yaw": wp.transform.rotation.yaw,
        })

    if waypoints:
        while len(waypoints) < num_waypoints:
            waypoints.append(waypoints[-1].copy())
    else:
        loc = ego_tf.location
        fallback = {"x": loc.x, "y": loc.y, "z": loc.z, "yaw": ego_tf.rotation.yaw}
        waypoints = [fallback] * num_waypoints

    return waypoints


def get_nearby_actors(world, ego_vehicle, max_dist: float = 50.0) -> list:
    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = ego_tf.rotation.yaw
    actors_data = []

    all_actors = world.get_actors()
    vehicles    = all_actors.filter("vehicle.*")
    pedestrians = all_actors.filter("walker.pedestrian.*")

    # Rotation parameters for global to local transformation
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

            # Translate
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y

            # Rotate to local frame (X is forward, Y is right in CARLA)
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

# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN DEL EPISODIO
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(client, world, episode_id: int, cfg: dict):
    print(f"\n{'='*60}")
    print(f"  INICIANDO EPISODIO {episode_id:04d}  |  {cfg['town']}")
    print(f"{'='*60}")

    paths   = build_episode_dirs(cfg["output_root"], episode_id)
    writers = open_csv_writers(paths)

    # 1. Spawn Ego-Vehicle Tesla Model 3
    bp_lib  = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No hay spawn points en el mapa.")

    import random
    spawn_tf = random.choice(spawn_points)
    vehicle  = world.spawn_actor(vehicle_bp, spawn_tf)
    print(f"[OK] Ego-Vehículo creado (ID={vehicle.id})")

    # Configurar Traffic Manager para el piloto automático del Ego
    traffic_manager = client.get_trafficmanager()
    vehicle.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.ignore_lights_percentage(vehicle, 0.0) # Seguir normas de tráfico
    traffic_manager.distance_to_leading_vehicle(vehicle, 3.0)

    # 2. Spawnear Tráfico y Peatones Ambientales
    ambient_vehicles, ambient_walkers, ambient_controllers = spawn_ambient_traffic(
        client, world,
        num_vehicles=cfg["num_vehicles"],
        num_walkers=cfg["num_walkers"],
        traffic_manager=traffic_manager,
        ego_spawn_point=spawn_tf
    )

    # 3. Spawnear Sensores del Ego
    cameras = spawn_cameras(world, vehicle, cfg)
    lidar   = spawn_lidar(world, vehicle, cfg)
    imu     = spawn_imu(world, vehicle)
    gnss    = spawn_gnss(world, vehicle)

    all_sensors = cameras + [lidar, imu, gnss]

    # Registrar Colas
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
        for tick in tqdm(range(total_frames), desc=f"Captura Ep {episode_id:04d}", unit="frame"):
            # Avanzar frame sincrónico
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

            # Saltar periodo de estabilización inicial
            if tick < cfg["warmup_frames"]:
                continue

            frame_id = saved_frames

            # A. PERCEPCIÓN (Cámaras PNG y LiDAR Grid BEV)
            for i, img_data in enumerate(cam_images):
                array = np.frombuffer(img_data.raw_data, dtype=np.uint8)
                array = array.reshape((cfg["cam_height"], cfg["cam_width"], 4))
                bgr = array[:, :, :3]
                filename = os.path.join(paths["cameras"][i], f"frame_{frame_id:06d}.png")
                cv2.imwrite(filename, bgr)

            pts_raw = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
            pts_raw = pts_raw.reshape(-1, 4)
            bev_grid = lidar_to_bev_grid(pts_raw, cfg)
            lidar_filename = os.path.join(paths["lidar"], f"frame_{frame_id:06d}.npy")
            np.save(lidar_filename, bev_grid)

            # B. LOCATION
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

            # C. PLANNING
            future_wps = get_future_waypoints(world, vehicle, num_waypoints=cfg["num_waypoints"], spacing=cfg["waypoint_spacing"])
            plan_row = [frame_id]
            for wp in future_wps:
                plan_row += [wp["x"], wp["y"], wp["z"], wp["yaw"]]
            writers["planning"][1].writerow(plan_row)

            # D. CONTROL
            ctrl = vehicle.get_control()
            ctrl_row = [
                frame_id, round(ctrl.throttle, 6), round(ctrl.brake, 6), round(ctrl.steer, 6),
                int(ctrl.hand_brake), int(ctrl.reverse)
            ]
            writers["control"][1].writerow(ctrl_row)

            # E. PREDICTION (Coches y peatones detectados)
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
        # Cerrar archivos
        close_csv_writers(writers)

        # Destruir sensores
        for sensor in all_sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()

        # Destruir coche ego
        if vehicle.is_alive:
            vehicle.destroy()

        # Destruir peatones y sus controladores de IA
        for controller in ambient_controllers:
            if controller.is_alive:
                controller.stop()
                controller.destroy()

        for walker in ambient_walkers:
            if walker.is_alive:
                walker.destroy()

        # Destruir vehículos ambientales
        for v in ambient_vehicles:
            if v.is_alive:
                v.destroy()

        print(f"[OK] Episodio {episode_id:04d} finalizado. Datos limpios en memoria.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Helioskrill — Extractor de datos sincronizados con tráfico")
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
    print("  HELIOSKRILL — Extractor de Datos con Tráfico Activo")
    print("="*60)
    print(f"  Mapa:          {CONFIG['town']}")
    print(f"  Episodios:     {CONFIG['num_episodes']}")
    print(f"  Flota Tráfico: {CONFIG['num_vehicles']} coches, {CONFIG['num_walkers']} peatones")
    print(f"  Salida:        {CONFIG['output_root']}")
    print("="*60 + "\n")

    try:
        client = carla.Client(CONFIG["host"], CONFIG["port"])
        client.set_timeout(CONFIG["timeout"])
        client.load_world(CONFIG["town"])
        world = client.get_world()
        
        # Conectado a CARLA
        print(f"[OK] Conectado a CARLA.")
    except RuntimeError as e:
        print(f"[ERROR] Conexión fallida: {e}")
        sys.exit(1)

    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 1.0 / CONFIG["fps"]
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    # Semilla aleatoria para que el tráfico cambie en cada run
    traffic_manager.set_random_device_seed(int(time.time()))

    try:
        for ep_id in range(CONFIG["num_episodes"]):
            run_episode(client, world, episode_id=ep_id, cfg=CONFIG)
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[INFO] Recolección cancelada por el usuario.")

    finally:
        print("\n[Limpieza] Restaurando modo asincrónico...")
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
        print("[OK] Finalizado correctamente.")
        os._exit(0)  # Bypasses Python GC teardown to prevent CARLA's Boost.Python crash (0xC0000409) on Windows exit


if __name__ == "__main__":
    main()
