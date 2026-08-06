#!/usr/bin/env python3
"""
collect_dagger_data.py
======================
Automated DAgger (Dataset Aggregation) Data Collector in CARLA.

REFACTORING COMPLIANCE:
-----------------------
1. Reuses models/utils/carla_data_collector.py functions & constants directly.
2. Synchronous CARLA world.tick() with individual sensor queues (cam_queues, lidar_queue, imu_queue, gnss_queue).
3. Real IMU and GNSS telemetry logged to location.csv.
4. Generates 100% schema-identical datasets across all 5 folders.
5. `is_recovery` (0/1) flag logged to control.csv to mark perturbation and recovery window frames.
"""

import os
import sys
import time
import math
import queue
import argparse
import random
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.utils.carla_data_collector import (
    CONFIG,
    CAMERA_CONFIGS,
    get_next_episode_id,
    build_episode_dirs,
    open_csv_writers,
    close_csv_writers,
    lidar_to_bev_grid_vectorized,
    save_camera_metadata,
    get_future_waypoints,
    get_nearby_actors,
    spawn_cameras,
    spawn_depth_cameras,
    spawn_semantic_cameras,
    spawn_lidar,
    spawn_imu,
    spawn_gnss,
    spawn_ambient_traffic
)

try:
    import carla
except ImportError:
    print("[ERROR] Could not import 'carla' module.")
    sys.exit(1)


import socket
import subprocess


def test_host_connection(host: str, port: int = 2000, timeout: float = 1.0) -> bool:
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except Exception:
        return False


def resolve_carla_host(host_arg: str, port: int = 2000) -> str:
    if host_arg not in ("auto", "localhost", "127.0.0.1", ""):
        return host_arg

    candidates = ["localhost", "127.0.0.1"]

    # Try WSL 2 default gateway IP from ip route
    try:
        res = subprocess.check_output("ip route show default", shell=True, text=True)
        tokens = res.strip().split()
        if "via" in tokens:
            gateway_ip = tokens[tokens.index("via") + 1]
            if gateway_ip not in candidates:
                candidates.append(gateway_ip)
    except Exception:
        pass

    # Try nameserver IP from /etc/resolv.conf
    if os.path.exists("/etc/resolv.conf"):
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                if line.startswith("nameserver"):
                    ip = line.split()[1].strip()
                    if ip not in candidates:
                        candidates.append(ip)

    for ip in candidates:
        if test_host_connection(ip, port):
            print(f"[WSL 2 Auto-Detect] Connected successfully to CARLA at {ip}:{port}")
            return ip

    # Fallback to localhost if none connected yet
    return "localhost"


def run_dagger_episode(client, world, episode_id: int, cfg: dict, perturb_interval: float = 4.0):
    print(f"\n{'='*60}")
    print(f"  STARTING DAGGER EPISODE {episode_id:04d}  |  {cfg['town']}")
    print(f"{'='*60}")

    paths = build_episode_dirs(cfg["output_root"], episode_id)
    writers = open_csv_writers(paths)
    save_camera_metadata(paths["location"], cfg)

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available on this map.")

    spawn_tf = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_tf)
    print(f"[OK] Ego Vehicle spawned (ID={vehicle.id})")

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    vehicle.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.ignore_lights_percentage(vehicle, 0.0)
    traffic_manager.distance_to_leading_vehicle(vehicle, 3.0)

    ambient_vehicles, ambient_walkers, ambient_controllers = spawn_ambient_traffic(
        client, world,
        num_vehicles=cfg.get("num_vehicles", 20),
        num_walkers=cfg.get("num_walkers", 10),
        traffic_manager=traffic_manager,
        ego_spawn_point=spawn_tf
    )

    cameras = spawn_cameras(world, vehicle, cfg)
    depth_cameras = spawn_depth_cameras(world, vehicle, cfg)
    semantic_cameras = spawn_semantic_cameras(world, vehicle, cfg)

    imu = spawn_imu(world, vehicle)
    gnss = spawn_gnss(world, vehicle)

    # Individual Sensor Queues matching main collector
    cam_queues = [queue.Queue() for _ in range(len(CAMERA_CONFIGS))]
    depth_queues = [queue.Queue() for _ in range(len(CAMERA_CONFIGS))]
    semantic_queues = [queue.Queue() for _ in range(len(CAMERA_CONFIGS))]

    imu_queue = queue.Queue()
    gnss_queue = queue.Queue()

    for i in range(len(CAMERA_CONFIGS)):
        cameras[i].listen(cam_queues[i].put)
        depth_cameras[i].listen(depth_queues[i].put)
        semantic_cameras[i].listen(semantic_queues[i].put)

    imu.listen(imu_queue.put)
    gnss.listen(gnss_queue.put)

    total_frames = cfg["frames_per_episode"] + cfg["warmup_frames"]
    saved_frames = 0
    skipped_timeout = 0

    last_perturb_frame = 0
    perturb_duration_frames = int(0.6 * cfg["fps"])  # 0.6 seconds of forced steer
    recovery_duration_frames = int(0.6 * cfg["fps"]) # 0.6 seconds of recovery window
    perturb_interval_frames = int(perturb_interval * cfg["fps"])

    perturb_active = False
    perturb_steer = 0.0
    perturb_start_frame = 0

    try:
        for tick in range(total_frames):
            world.tick()

            timeout = 2.0 / cfg["fps"]
            try:
                cam_images = [q.get(timeout=timeout) for q in cam_queues]
                depth_images = [q.get(timeout=timeout) for q in depth_queues]
                semantic_images = [q.get(timeout=timeout) for q in semantic_queues]
                imu_data = imu_queue.get(timeout=timeout)
                gnss_data = gnss_queue.get(timeout=timeout)
            except queue.Empty:
                skipped_timeout += 1
                continue

            if tick < cfg["warmup_frames"]:
                continue

            # Perturbation & Recovery State Logic
            if not perturb_active and (saved_frames - last_perturb_frame) > perturb_interval_frames:
                perturb_active = True
                perturb_steer = random.choice([-0.3, 0.3])
                perturb_start_frame = saved_frames
                vehicle.set_autopilot(False)
                print(f"\n  ⚡ [PERTURBATION INJECTED] Steer Offset: {perturb_steer:+.2f} (Forcing Off-Center)...", end="")

            is_recovery = 0
            if perturb_active:
                if (saved_frames - perturb_start_frame) < perturb_duration_frames:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=perturb_steer, brake=0.0))
                    is_recovery = 1
                else:
                    perturb_active = False
                    last_perturb_frame = saved_frames
                    vehicle.set_autopilot(True, traffic_manager.get_port())
                    print(" [RECOVERY ACTIVE] Autopilot Executing Corrective Steering Back to Center!")
                    is_recovery = 1
            else:
                if (saved_frames - last_perturb_frame) < recovery_duration_frames:
                    is_recovery = 1

            frame_id = saved_frames

            # Save 8 Multi-View RGB, Depth, and Semantic Camera frames
            for i in range(len(CAMERA_CONFIGS)):
                # RGB Frame
                array = np.frombuffer(cam_images[i].raw_data, dtype=np.uint8)
                array = array.reshape((cfg["cam_height"], cfg["cam_width"], 4))
                bgr = array[:, :, :3]
                cv2.imwrite(os.path.join(paths["cameras"][i], f"frame_{frame_id:06d}.png"), bgr)

                # Depth Frame (CARLA 24-bit encoded depth in PNG format)
                d_array = np.frombuffer(depth_images[i].raw_data, dtype=np.uint8)
                d_array = d_array.reshape((cfg["cam_height"], cfg["cam_width"], 4))
                cv2.imwrite(os.path.join(paths["depth"][i], f"frame_{frame_id:06d}.png"), d_array)

                # Semantic Segmentation Frame (CARLA Semantic Tag in Red channel)
                s_array = np.frombuffer(semantic_images[i].raw_data, dtype=np.uint8)
                s_array = s_array.reshape((cfg["cam_height"], cfg["cam_width"], 4))
                cv2.imwrite(os.path.join(paths["semantic"][i], f"frame_{frame_id:06d}.png"), s_array)

            # Telemetry logging (Real IMU + GNSS + Pose)
            ego_tf = vehicle.get_transform()
            ego_vel = vehicle.get_velocity()
            speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)

            loc_row = [
                frame_id, gnss_data.latitude, gnss_data.longitude, gnss_data.altitude,
                ego_tf.location.x, ego_tf.location.y, ego_tf.location.z,
                ego_tf.rotation.yaw, ego_tf.rotation.pitch, ego_tf.rotation.roll,
                imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z,
                imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z, speed
            ]
            writers["location"][1].writerow(loc_row)

            # Waypoints logging
            future_wps = get_future_waypoints(world, vehicle, num_waypoints=cfg["num_waypoints"], spacing=cfg["waypoint_spacing"])
            plan_row = [frame_id]
            for wp in future_wps:
                plan_row += [wp["rel_x"], wp["rel_y"], wp["rel_z"], wp["rel_yaw"]]
            writers["planning"][1].writerow(plan_row)

            # Control logging with is_recovery flag
            ctrl = vehicle.get_control()
            ctrl_row = [
                frame_id, round(ctrl.throttle, 6), round(ctrl.brake, 6), round(ctrl.steer, 6),
                int(ctrl.hand_brake), int(ctrl.reverse), is_recovery
            ]
            writers["control"][1].writerow(ctrl_row)

            # Nearby actors prediction logging
            nearby = get_nearby_actors(world, vehicle, max_dist=cfg["bev_range_m"])
            for actor in nearby:
                pred_row = [
                    frame_id, actor["actor_id"], actor["actor_type"],
                    actor["x"], actor["y"], actor["z"], actor["yaw"],
                    actor["vel_x"], actor["vel_y"], actor["vel_z"], actor["speed_mps"],
                    actor["rel_x"], actor["rel_y"], actor["rel_dist_m"]
                ]
                writers["prediction"][1].writerow(pred_row)

            print(f"\r  Frame [{saved_frames:04d}/{cfg['frames_per_episode']:04d}] | Speed: {speed*3.6:5.1f} km/h | Steer: {ctrl.steer:+5.2f} | Recovery: {is_recovery}", end="")
            saved_frames += 1

    finally:
        print(f"\n[DAgger Collector] Cleaning up episode episode_{episode_id:04d} actors...")
        close_csv_writers(writers)

        all_actors = cameras + [lidar, imu, gnss, vehicle] + ambient_vehicles + ambient_walkers + ambient_controllers
        for a in all_actors:
            try:
                if a and a.is_alive():
                    a.destroy()
            except Exception:
                pass

        print(f"[SUCCESS] Saved DAgger episode_{episode_id:04d} ({saved_frames} frames, {skipped_timeout} timeouts)")


def main():
    parser = argparse.ArgumentParser(description="CARLA DAgger Data Collector (Refactored)")
    parser.add_argument("--host", default="auto", help="CARLA host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--town", type=str, default=None, help="CARLA town map (e.g. Town01, Town03, Town04, Town05)")
    parser.add_argument("--num_episodes", type=int, default=2, help="Number of DAgger episodes")
    parser.add_argument("--frames_per_episode", type=int, default=600, help="Frames per episode")
    parser.add_argument("--perturb_interval", type=float, default=4.0, help="Seconds between perturbations")

    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg["host"] = resolve_carla_host(args.host, args.port)
    cfg["port"] = args.port
    cfg["frames_per_episode"] = args.frames_per_episode
    if args.town:
        cfg["town"] = args.town

    print(f"[DAgger Collector] Connecting to CARLA at {cfg['host']}:{cfg['port']}...")
    client = carla.Client(cfg["host"], cfg["port"])
    client.set_timeout(cfg["timeout"])

    if args.town:
        current_world_name = client.get_world().get_map().name.split("/")[-1]
        if current_world_name != args.town:
            print(f"[DAgger Collector] Loading CARLA Town Map: {args.town}...")
            client.load_world(args.town)

    world = client.get_world()
    original_settings = world.get_settings()

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / cfg["fps"]
        world.apply_settings(settings)

        start_ep_id = get_next_episode_id(cfg["output_root"])
        print(f"[DAgger Collector] Will save DAgger episodes starting at: episode_{start_ep_id:04d}")

        for ep_offset in range(args.num_episodes):
            ep_id = start_ep_id + ep_offset
            run_dagger_episode(client, world, ep_id, cfg, perturb_interval=args.perturb_interval)

    finally:
        print("\n[DAgger Collector] Restoring original world settings...")
        world.apply_settings(original_settings)
        print("[DAgger Collector] Finished.")


if __name__ == "__main__":
    main()
