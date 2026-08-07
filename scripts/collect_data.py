#!/usr/bin/env python3
"""
collect_data.py
===============
Unified Master CARLA Data Collector for Helioskrill.

Modes:
  --mode normal : 100% smooth expert autopilot driving (is_recovery = 0.0).
  --mode dagger : Injects steering & reverse perturbations and logs expert recovery maneuvers (is_recovery = 1.0).
  --mode all    : Alternates between Normal and DAgger episodes across towns.

Features:
  - Multi-Town Map Rotation (Town01, Town02, Town03, Town04, Town05).
  - 8 RGB Cameras + 8 Depth Cameras + 8 Semantic Cameras per frame.
  - Asymmetric 200x200 BEV Grid spatial geometry.
  - 100% Camera-only pipeline (no physical LiDAR).
  - Collision-tolerant: minor bumps pause recording instead of aborting the whole episode.
  - CARLA Depth decoded to float16 .npy (meters).
  - CARLA Semantic Segmentation: only Red channel (CARLA tag ID) saved as uint8 grayscale.
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
    save_camera_metadata,
    get_future_waypoints,
    get_nearby_actors,
    spawn_cameras,
    spawn_depth_cameras,
    spawn_semantic_cameras,
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

    try:
        res = subprocess.check_output("ip route show default", shell=True, text=True)
        tokens = res.strip().split()
        if "via" in tokens:
            gateway_ip = tokens[tokens.index("via") + 1]
            if gateway_ip not in candidates:
                candidates.append(gateway_ip)
    except Exception:
        pass

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

    return "localhost"


def decode_carla_depth(raw_bgra: np.ndarray) -> np.ndarray:
    """
    Decodes CARLA 24-bit encoded depth buffer (BGRA uint8) to meters (float16).
    Formula: depth_m = (R + G*256 + B*256^2) / (256^3 - 1) * 1000.0
    Reference: https://carla.readthedocs.io/en/latest/ref_sensors/#depth-camera
    """
    R = raw_bgra[:, :, 2].astype(np.float32)
    G = raw_bgra[:, :, 1].astype(np.float32)
    B = raw_bgra[:, :, 0].astype(np.float32)
    depth_meters = (R + G * 256.0 + B * 65536.0) / 16777215.0 * 1000.0
    return depth_meters.astype(np.float16)


def extract_semantic_tag(raw_bgra: np.ndarray) -> np.ndarray:
    """
    Extracts CARLA semantic segmentation tag ID from the Red channel.
    CARLA stores class tag in Red channel (index 2 in BGRA order from OpenCV).
    Reference: https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
    """
    return raw_bgra[:, :, 2].copy()


def stop_and_destroy_sensors(sensor_list: list):
    """
    Safely stops sensor listening streams.
    Child sensors attached to the vehicle are automatically destroyed on CARLA server
    when vehicle.destroy() is called, avoiding C++ double-free exceptions.
    """
    for s in sensor_list:
        if s is not None:
            try:
                s.stop()
            except Exception:
                pass


def destroy_actors(actor_list: list):
    """Destroys non-sensor CARLA actors (vehicles, walkers, controllers)."""
    for a in actor_list:
        if a is not None:
            try:
                a.destroy()
            except Exception:
                pass


def run_collection_episode(client, world, episode_id: int, cfg: dict, enable_dagger: bool = True, perturb_interval: float = 4.0):
    ep_type_str = "DAGGER (RECOVERY)" if enable_dagger else "NORMAL (EXPERT)"
    print(f"\n{'='*65}")
    print(f"  STARTING {ep_type_str} EPISODE {episode_id:04d}  |  MAP: {cfg['town']}")
    print(f"{'='*65}")

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

    # --- Traffic Manager Configuration ---
    # Autopilot RESPECTS traffic lights and stop signs for clean expert driving data.
    # vehicle_percentage_speed_difference: positive = slower, negative = faster than speed limit.
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    vehicle.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.ignore_lights_percentage(vehicle, 0.0)      # RESPECT all traffic lights
    traffic_manager.ignore_signs_percentage(vehicle, 0.0)       # RESPECT all stop signs
    traffic_manager.vehicle_percentage_speed_difference(vehicle, 10.0)  # 10% SLOWER than limit
    traffic_manager.distance_to_leading_vehicle(vehicle, cfg.get("min_dist", 3.5))   # Configurable (3.5m ~ 2-3m real bumper spacing)
    traffic_manager.auto_lane_change(vehicle, True)

    # Reduced ambient traffic to avoid spawning chaos
    ambient_vehicles, ambient_walkers, ambient_controllers = spawn_ambient_traffic(
        client, world,
        num_vehicles=cfg.get("num_vehicles", 15),
        num_walkers=cfg.get("num_walkers", 8),
        traffic_manager=traffic_manager,
        ego_spawn_point=spawn_tf
    )

    cameras = spawn_cameras(world, vehicle, cfg)
    depth_cameras = spawn_depth_cameras(world, vehicle, cfg)
    semantic_cameras = spawn_semantic_cameras(world, vehicle, cfg)

    imu = spawn_imu(world, vehicle)
    gnss = spawn_gnss(world, vehicle)

    # Collision Sensor: tolerant mode (pause + continue, NOT instant abort)
    collision_bp = bp_lib.find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    collision_count = [0]
    collision_cooldown = [999]  # Start high so first frames are not in cooldown
    MAX_COLLISIONS = 5          # Abort only after 5 cumulative collisions
    COLLISION_COOLDOWN_FRAMES = 40  # 2 seconds at 20 FPS: skip saving during impact
    def _on_collision(event):
        collision_count[0] += 1
        collision_cooldown[0] = 0  # Reset cooldown counter
    collision_sensor.listen(_on_collision)

    # Keep a flat list of ALL sensors for cleanup
    all_sensors = cameras + depth_cameras + semantic_cameras + [imu, gnss, collision_sensor]

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
    stuck_frames = 0
    episode_aborted = False

    last_perturb_frame = 0
    perturb_duration_frames = int(0.6 * cfg["fps"])  # 0.6 seconds of perturbation
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

            # Check Vehicle Pose & Velocity
            ego_tf = vehicle.get_transform()
            ego_vel = vehicle.get_velocity()
            speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)

            # --- 3rd Person Spectator Chase Camera ---
            # Moves CARLA viewport camera 6m behind and 2.5m above vehicle with -15deg pitch
            try:
                spectator = world.get_spectator()
                cam_loc = ego_tf.transform(carla.Location(x=-6.0, z=2.5))
                spectator.set_transform(carla.Transform(
                    cam_loc,
                    carla.Rotation(pitch=-15.0, yaw=ego_tf.rotation.yaw, roll=0.0)
                ))
            except Exception:
                pass

            # Universal Smart Pause:
            # If vehicle is stationary (speed < 0.2 m/s) for >20 frames (1 sec), pause saving images & CSVs
            # to prevent static frame flooding in dataset, but keep simulation ticking!
            # Resumes saving automatically as soon as speed > 0.2 m/s.
            if speed < 0.2:
                stuck_frames += 1
            else:
                stuck_frames = 0

            if stuck_frames > 20:
                print(f"\r  [PAUSE] Stationary Wait ({stuck_frames*0.05:.1f}s) | Speed: {speed*3.6:5.1f} km/h | Saved: [{saved_frames:04d}/{cfg['frames_per_episode']:04d}]", end="")
                continue

            # Abort only on rollover (pitch or roll > 60 deg)
            if abs(ego_tf.rotation.pitch) > 60.0 or abs(ego_tf.rotation.roll) > 60.0:
                print(f"\n  [ABORT] Vehicle rolled over! Aborting episode_{episode_id:04d}.")
                episode_aborted = True
                break

            if collision_count[0] >= MAX_COLLISIONS:
                print(f"\n  [ABORT] {MAX_COLLISIONS} collisions reached. Aborting episode_{episode_id:04d}.")
                episode_aborted = True
                break

            # Collision cooldown: skip saving frames right after a collision
            collision_cooldown[0] += 1
            if collision_count[0] > 0 and collision_cooldown[0] < COLLISION_COOLDOWN_FRAMES:
                continue  # Don't save frames during collision cooldown

            is_recovery = 0.0

            # DAgger Perturbation & Recovery Logic (Only active if enable_dagger=True)
            if enable_dagger:
                if not perturb_active and (saved_frames - last_perturb_frame) > perturb_interval_frames:
                    perturb_active = True
                    # Smooth, realistic steer offsets (±0.15 to ±0.25 rad) for lane-keeping recovery
                    perturb_steer = random.choice([-0.25, -0.15, 0.15, 0.25])
                    perturb_start_frame = saved_frames
                    vehicle.set_autopilot(False)
                    print(f"\n  [PERTURBATION INJECTED] Steer Offset: {perturb_steer:+.2f}...", end="")

                if perturb_active:
                    if (saved_frames - perturb_start_frame) < perturb_duration_frames:
                        vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=perturb_steer, brake=0.0, reverse=False))
                        is_recovery = 1.0
                    else:
                        perturb_active = False
                        last_perturb_frame = saved_frames
                        vehicle.set_autopilot(True, traffic_manager.get_port())
                        print(" [RECOVERY ACTIVE] Autopilot Re-centering Trajectory!")
                        is_recovery = 1.0
                else:
                    if (saved_frames - last_perturb_frame) < recovery_duration_frames:
                        is_recovery = 1.0

            frame_id = saved_frames
            timestamp_sec = round(saved_frames * (1.0 / cfg["fps"]), 4)

            # Save 8 Multi-View RGB, Depth, and Semantic Camera frames
            for i in range(len(CAMERA_CONFIGS)):
                # RGB Frame (BGR uint8 PNG)
                array = np.frombuffer(cam_images[i].raw_data, dtype=np.uint8)
                array = array.reshape((cfg["cam_height"], cfg["cam_width"], 4))
                bgr = array[:, :, :3]
                cv2.imwrite(os.path.join(paths["cameras"][i], f"frame_{frame_id:06d}.png"), bgr)

                # Depth Frame (Raw CARLA 24-bit BGRA depth buffer in PNG format)
                d_array = np.frombuffer(depth_images[i].raw_data, dtype=np.uint8)
                d_array = d_array.reshape((cfg["cam_height"], cfg["cam_width"], 4))
                cv2.imwrite(os.path.join(paths["depth"][i], f"frame_{frame_id:06d}.png"), d_array)

                # Semantic Frame: extract Red channel (CARLA tag ID) -> uint8 grayscale PNG
                s_array = np.frombuffer(semantic_images[i].raw_data, dtype=np.uint8)
                s_array = s_array.reshape((cfg["cam_height"], cfg["cam_width"], 4))
                sem_tag = extract_semantic_tag(s_array)
                cv2.imwrite(os.path.join(paths["semantic"][i], f"frame_{frame_id:06d}.png"), sem_tag)

            loc_row = [
                frame_id, timestamp_sec, gnss_data.latitude, gnss_data.longitude, gnss_data.altitude,
                ego_tf.location.x, ego_tf.location.y, ego_tf.location.z,
                ego_tf.rotation.yaw, ego_tf.rotation.pitch, ego_tf.rotation.roll,
                imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z,
                imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z, speed
            ]
            writers["location"][1].writerow(loc_row)

            # Future Waypoints planning logging
            future_wps = get_future_waypoints(
                world, vehicle,
                num_waypoints=cfg["num_waypoints"],
                spacing=cfg["waypoint_spacing"]
            )
            plan_row = [frame_id, timestamp_sec]
            for wp in future_wps:
                plan_row += [wp["rel_x"], wp["rel_y"], wp["rel_z"], wp["rel_yaw"]]
            writers["planning"][1].writerow(plan_row)

            # Control logging with is_recovery flag
            ctrl = vehicle.get_control()
            ctrl_row = [
                frame_id, timestamp_sec, round(ctrl.throttle, 6), round(ctrl.brake, 6), round(ctrl.steer, 6),
                int(ctrl.hand_brake), int(ctrl.reverse), is_recovery
            ]
            writers["control"][1].writerow(ctrl_row)

            # Nearby actors prediction logging
            nearby = get_nearby_actors(world, vehicle, max_dist=cfg["bev_x_max"])
            for actor in nearby:
                pred_row = [
                    frame_id, actor["actor_id"], actor["actor_type"],
                    actor["x"], actor["y"], actor["z"], actor["yaw"],
                    actor["vel_x"], actor["vel_y"], actor["vel_z"], actor["speed_mps"],
                    actor["rel_x"], actor["rel_y"], actor["rel_dist_m"]
                ]
                writers["prediction"][1].writerow(pred_row)

            print(f"\r  Frame [{saved_frames:04d}/{cfg['frames_per_episode']:04d}] | Speed: {speed*3.6:5.1f} km/h | Steer: {ctrl.steer:+5.2f} | Collisions: {collision_count[0]} | Recovery: {is_recovery}", end="")
            saved_frames += 1

    finally:
        print(f"\n[Data Collector] Cleaning up episode_{episode_id:04d} actors...")
        close_csv_writers(writers)

        # 1. Stop sensor listening streams
        for s in all_sensors:
            if s is not None:
                try:
                    s.stop()
                except Exception:
                    pass

        # 2. Tick world once in synchronous mode so CARLA processes sensor stop commands
        try:
            world.tick()
        except Exception:
            pass

        # 3. Destroy ego vehicle (CARLA server automatically destroys all attached sensors!)
        if vehicle is not None:
            try:
                vehicle.destroy()
            except Exception:
                pass

        # 4. Destroy ambient traffic
        destroy_actors(ambient_vehicles + ambient_walkers + ambient_controllers)

        if episode_aborted or saved_frames < 100:
            import shutil
            ep_str = f"episode_{episode_id:04d}"
            reason = "ABORTED" if episode_aborted else f"TOO SHORT ({saved_frames} frames)"
            print(f"[CLEANUP] Deleting {reason} episode_{episode_id:04d} files...")
            for category in ["Control", "Location", "Perception", "Planning", "Prediction"]:
                cat_dir = os.path.join(cfg["output_root"], category, ep_str)
                if os.path.exists(cat_dir):
                    shutil.rmtree(cat_dir, ignore_errors=True)
            print(f"[DISCARDED] episode_{episode_id:04d} - {reason}")
        else:
            print(f"[SUCCESS] Saved episode_{episode_id:04d} ({saved_frames} frames, {skipped_timeout} timeouts, {collision_count[0]} collisions)")


def cleanup_all_carla_actors(world):
    """
    Scans CARLA world and forcibly destroys all sensors, vehicle actors, and ambient walkers
    spawned by Helioskrill so the simulation is left 100% clean when script stops or is interrupted.
    """
    if world is None:
        return
    print("\n[GLOBAL CLEANUP] Destroying active CARLA actors...")

    try:
        actors = world.get_actors()
        # Destroy vehicles and walkers first (CARLA server automatically cleans attached sensors)
        for a in list(actors.filter("vehicle.*")) + list(actors.filter("walker.*")) + list(actors.filter("controller.ai.walker")):
            try:
                a.destroy()
            except Exception:
                pass

        # Stop/destroy remaining standalone sensors
        for s in world.get_actors().filter("sensor.*"):
            try:
                s.stop()
            except Exception:
                pass
            try:
                s.destroy()
            except Exception:
                pass
    except Exception:
        pass
    print("[GLOBAL CLEANUP] Simulation world cleaned successfully.")


def main():
    parser = argparse.ArgumentParser(description="Unified Master CARLA Data Collector (Normal + DAgger)")
    parser.add_argument("--host", default="auto", help="CARLA host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--mode", choices=["all", "normal", "dagger"], default="all", help="Collection mode: normal (100%% expert), dagger (perturbations), all (50/50 mix)")
    parser.add_argument("--towns", type=str, default="Town01,Town02,Town03,Town04,Town05", help="Comma-separated CARLA towns for map rotation")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to collect")
    parser.add_argument("--frames_per_episode", type=int, default=1200, help="Frames per episode (default 1200 = 60 seconds at 20 FPS)")
    parser.add_argument("--num_vehicles", type=int, default=15, help="Number of ambient vehicles per episode")
    parser.add_argument("--num_walkers", type=int, default=8, help="Number of ambient walkers per episode")
    parser.add_argument("--min_dist", type=float, default=3.5, help="Distance to leading vehicle in CARLA Traffic Manager (default 3.5m = ~2-3m real bumper space)")
    parser.add_argument("--perturb_interval", type=float, default=4.0, help="Seconds between perturbations in DAgger mode")

    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg["host"] = resolve_carla_host(args.host, args.port)
    cfg["port"] = args.port
    cfg["frames_per_episode"] = args.frames_per_episode
    cfg["num_vehicles"] = args.num_vehicles
    cfg["num_walkers"] = args.num_walkers
    cfg["min_dist"] = args.min_dist

    town_list = [t.strip() for t in args.towns.split(",") if t.strip()]

    print(f"[Data Collector] Connecting to CARLA at {cfg['host']}:{cfg['port']}...")
    client = carla.Client(cfg["host"], cfg["port"])
    client.set_timeout(cfg["timeout"])

    world = None
    try:
        start_ep_id = get_next_episode_id(cfg["output_root"])
        print(f"[Data Collector] Will save episodes starting at: episode_{start_ep_id:04d}")

        for ep_offset in range(args.num_episodes):
            ep_id = start_ep_id + ep_offset
            target_town = town_list[ep_offset % len(town_list)]
            cfg["town"] = target_town

            enable_dagger = (ep_offset % 2 == 1) if args.mode == "all" else (args.mode == "dagger")

            try:
                current_world_name = client.get_world().get_map().name.split("/")[-1]
                if world is None or current_world_name != target_town:
                    print(f"\n[Data Collector] Rotating Map -> Loading CARLA Town: {target_town}...")
                    client.load_world(target_town)
                    time.sleep(3.0)  # Let CARLA stabilize after map load

                    world = client.get_world()
                    settings = world.get_settings()
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = 1.0 / cfg["fps"]
                    world.apply_settings(settings)
                else:
                    world = client.get_world()

                run_collection_episode(
                    client, world, ep_id, cfg,
                    enable_dagger=enable_dagger,
                    perturb_interval=args.perturb_interval
                )

            except KeyboardInterrupt:
                print("\n[USER INTERRUPT] Collection stopped by user (Ctrl+C). Cleaning up all actors...")
                cleanup_all_carla_actors(world)
                sys.exit(0)
            except Exception as ep_err:
                import traceback
                print(f"\n[ERROR] Episode {ep_id:04d} encountered unexpected error: {ep_err}. Skipping to next episode...")
                traceback.print_exc()
                time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n[USER INTERRUPT] Interrupted by user. Cleaning up simulation...")
        cleanup_all_carla_actors(world)
        sys.exit(0)
    finally:
        if world is not None:
            cleanup_all_carla_actors(world)

    print("\n[Data Collector] Master Collection Finished Successfully.")


if __name__ == "__main__":
    main()
