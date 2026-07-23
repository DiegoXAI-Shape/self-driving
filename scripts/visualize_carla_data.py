"""
visualize_carla_data.py
=======================
Interactive dataset inspection tool for visualizing collected CARLA multi-sensor simulation data.

FEATURES
--------
  - Displays a synchronized 8-camera multi-view RGB grid.
  - Displays LiDAR BEV grid point cloud density map.
  - Overlays nearby vehicles and pedestrian actors (Prediction).
  - Overlays planned future waypoints (Planning).
  - Displays vehicle control telemetry (throttle, brake, steer) and state (speed, IMU G-forces).
  - Playback controls:
      * Right Arrow: Next frame
      * Left Arrow: Previous frame
      * Spacebar: Play / Pause
      * 'q' key: Quit

USAGE
-----
  python scripts/visualize_carla_data.py --data_dir ./data/ --episode 0
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


CAM_NAMES = [
    "Front Main", "Front Wide", "Front Narrow",
    "Left B-Pillar", "Right B-Pillar",
    "Left Repeater", "Right Repeater", "Rear"
]


def global_to_local(ego_x, ego_y, ego_yaw_deg, points_x, points_y):
    """
    Converts global coordinates (map) to local ego-vehicle coordinates.
    In CARLA local frame:
      - X local points forward.
      - Y local points right.
    """
    yaw = np.radians(ego_yaw_deg)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)

    dx = points_x - ego_x
    dy = points_y - ego_y

    local_x = dx * cos_y + dy * sin_y
    local_y = -dx * sin_y + dy * cos_y

    return local_x, local_y


class CARLADataVisualizer:
    def __init__(self, data_root, episode_id):
        self.data_root = data_root
        self.episode_id = episode_id
        self.ep_str = f"episode_{episode_id:04d}"

        self.ep_path_perception = os.path.join(data_root, "Perception", self.ep_str)
        self.ep_path_location = os.path.join(data_root, "Location", self.ep_str)
        self.ep_path_planning = os.path.join(data_root, "Planning", self.ep_str)
        self.ep_path_control = os.path.join(data_root, "Control", self.ep_str)
        self.ep_path_prediction = os.path.join(data_root, "Prediction", self.ep_str)

        if not os.path.exists(self.ep_path_perception):
            raise FileNotFoundError(f"Episode {self.ep_str} not found in {data_root}.")

        print("[Loading] Reading telemetry CSV metadata...")
        
        self.df_location = pd.read_csv(os.path.join(self.ep_path_location, "location.csv"))
        self.df_planning = pd.read_csv(os.path.join(self.ep_path_planning, "waypoints.csv"))
        self.df_control = pd.read_csv(os.path.join(self.ep_path_control, "control.csv"))
        
        pred_file = os.path.join(self.ep_path_prediction, "actors.csv")
        if os.path.exists(pred_file) and os.path.getsize(pred_file) > 0:
            self.df_prediction = pd.read_csv(pred_file)
        else:
            self.df_prediction = pd.DataFrame()

        self.num_frames = len(self.df_location)
        self.current_frame = 0
        self.playing = False

        print(f"[OK] Loaded {self.num_frames} frames for visualization.")
        self.setup_layout()

    def setup_layout(self):
        plt.rcParams['toolbar'] = 'None'
        self.fig = plt.figure(figsize=(18, 10), facecolor='#0f172a')
        self.fig.suptitle(f"Helioskrill — Episode {self.episode_id:04d} Inspector", 
                          color='white', fontsize=18, fontweight='bold', y=0.97)

        gs = plt.GridSpec(3, 4, figure=self.fig, left=0.03, right=0.97, bottom=0.05, top=0.92, wspace=0.15, hspace=0.25)

        self.cam_axes = []
        cam_positions = [
            (0, 1), # Front Main
            (0, 0), # Front Wide
            (0, 2), # Front Narrow
            (1, 0), # Left B-Pillar
            (1, 2), # Right B-Pillar
            (2, 0), # Left Repeater
            (2, 2), # Right Repeater
            (1, 1), # Rear
        ]

        for pos in cam_positions:
            ax = self.fig.add_subplot(gs[pos[0], pos[1]])
            ax.axis('off')
            self.cam_axes.append(ax)

        self.info_ax = self.fig.add_subplot(gs[2, 1])
        self.info_ax.axis('off')

        self.bev_ax = self.fig.add_subplot(gs[0:3, 3])
        self.bev_ax.set_facecolor('#090d16')
        self.bev_ax.set_title("LiDAR BEV & Prediction Map", color='white', fontsize=14, fontweight='bold')
        
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def draw_frame(self):
        frame_idx = self.current_frame
        
        row_loc = self.df_location.iloc[frame_idx]
        row_ctrl = self.df_control.iloc[frame_idx]
        row_plan = self.df_planning.iloc[frame_idx]

        # 1. Render 8 Multi-View RGB Camera Views
        for i in range(8):
            ax = self.cam_axes[i]
            ax.clear()
            ax.axis('off')
            
            img_path = os.path.join(self.ep_path_perception, "cameras", f"cam_{i}", f"frame_{frame_idx:06d}.png")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "NO SIGNAL", color='#ef4444', ha='center', va='center', fontsize=12, fontweight='bold')
                ax.set_facecolor('#1e293b')
            
            ax.set_title(f"{CAM_NAMES[i]} (cam_{i})", color='#94a3b8', fontsize=10, pad=4)

        # 2. Render Vehicle Telemetry Panel
        self.info_ax.clear()
        self.info_ax.axis('off')
        
        speed_kmh = row_loc['speed_mps'] * 3.6
        throttle = row_ctrl['throttle']
        brake = row_ctrl['brake']
        steer = row_ctrl['steer']

        info_text = (
            f"Frame: {frame_idx:06d} / {self.num_frames - 1}\n\n"
            f"Speed: {speed_kmh:.1f} km/h ({row_loc['speed_mps']:.1f} m/s)\n"
            f"Location: X: {row_loc['ego_x']:.2f}, Y: {row_loc['ego_y']:.2f}\n"
            f"Orientation (Yaw): {row_loc['ego_yaw']:.1f}°\n\n"
            f"Control Actions:\n"
            f" ├─ Throttle: {throttle * 100:.1f}%\n"
            f" ├─ Brake:    {brake * 100:.1f}%\n"
            f" └─ Steer:    {steer:+.4f}\n\n"
            f"IMU G-Forces:\n"
            f" ├─ Accel X:  {row_loc['imu_accel_x']:+.2f} m/s²\n"
            f" └─ Accel Y:  {row_loc['imu_accel_y']:+.2f} m/s²"
        )
        self.info_ax.text(0.05, 0.95, info_text, color='white', family='monospace', 
                          fontsize=11, ha='left', va='top', transform=self.info_ax.transAxes)

        # 3. Render LiDAR BEV Map
        self.bev_ax.clear()
        self.bev_ax.set_facecolor('#090d16')
        
        npy_path = os.path.join(self.ep_path_perception, "lidar", f"frame_{frame_idx:06d}.npy")
        if os.path.exists(npy_path):
            bev_grid = np.load(npy_path)
            density_channel = bev_grid[3]
            self.bev_ax.imshow(density_channel, cmap='inferno', extent=[-50, 50, -50, 50], origin='upper')
        else:
            self.bev_ax.text(0, 0, "LiDAR grid (.npy) not found", color='gray', ha='center', va='center')

        circles = [10, 20, 30, 40, 50]
        for r in circles:
            circle = plt.Circle((0, 0), r, color='#334155', fill=False, linestyle='--', alpha=0.5)
            self.bev_ax.add_patch(circle)

        ego_marker = plt.Rectangle((-1.0, -2.4), 2.0, 4.8, color='#10b981', fill=True, label="Ego (Tesla)")
        self.bev_ax.add_patch(ego_marker)
        self.bev_ax.arrow(0, 0, 0, 6, head_width=2.5, head_length=2.5, fc='#059669', ec='#059669')

        # 4. Render Future Waypoints
        local_wp_x = []
        local_wp_y = []
        for w_i in range(10):
            local_wp_x.append(row_plan[f"wp_{w_i}_rel_x"])
            local_wp_y.append(row_plan[f"wp_{w_i}_rel_y"])
            
        local_wp_x = np.array(local_wp_x)
        local_wp_y = np.array(local_wp_y)
        
        self.bev_ax.plot(-local_wp_y, local_wp_x, color='#a78bfa', marker='o', markersize=6, 
                         linewidth=2, linestyle='-', label="Planned Waypoints")

        # 5. Render Detected Nearby Actors
        if not self.df_prediction.empty:
            frame_actors = self.df_prediction[self.df_prediction['frame'] == frame_idx]
            vehicles_plotted = False
            walkers_plotted = False
            
            for _, actor in frame_actors.iterrows():
                act_x_plot = -actor['rel_y']
                act_y_plot = actor['rel_x']
                
                if actor['actor_type'] == "vehicle":
                    self.bev_ax.plot(act_x_plot, act_y_plot, color='#3b82f6', marker='s', markersize=8, 
                                     linestyle='None', markeredgecolor='white', markeredgewidth=1)
                    vehicles_plotted = True
                else:
                    self.bev_ax.plot(act_x_plot, act_y_plot, color='#ef4444', marker='o', markersize=6, 
                                     linestyle='None', markeredgecolor='white', markeredgewidth=1)
                    walkers_plotted = True

            if vehicles_plotted:
                self.bev_ax.plot([], [], color='#3b82f6', marker='s', label="Detected Vehicle", linestyle='None')
            if walkers_plotted:
                self.bev_ax.plot([], [], color='#ef4444', marker='o', label="Detected Pedestrian", linestyle='None')

        self.bev_ax.set_xlim([-50, 50])
        self.bev_ax.set_ylim([-50, 50])
        self.bev_ax.grid(color='#1e293b', linestyle=':', alpha=0.6)
        self.bev_ax.tick_params(colors='#64748b', labelsize=9)
        self.bev_ax.legend(loc='lower right', facecolor='#0f172a', edgecolor='#1e293b', labelcolor='white', fontsize=8)

        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'right':
            self.playing = False
            self.current_frame = min(self.current_frame + 1, self.num_frames - 1)
            self.draw_frame()
        elif event.key == 'left':
            self.playing = False
            self.current_frame = max(self.current_frame - 1, 0)
            self.draw_frame()
        elif event.key == ' ':
            self.playing = not self.playing
            if self.playing:
                self.run_player()
        elif event.key == 'q':
            plt.close()

    def run_player(self):
        while self.playing and self.current_frame < self.num_frames - 1:
            self.current_frame += 1
            self.draw_frame()
            plt.pause(0.05)
        self.playing = False

    def show(self):
        self.draw_frame()
        print("\n" + "═"*60)
        print("  KEYBOARD CONTROLS:")
        print("  ├─ Right Arrow : Next frame")
        print("  ├─ Left Arrow  : Previous frame")
        print("  ├─ Spacebar    : Play / Pause")
        print("  └─ 'q' Key     : Quit tool")
        print("═"*60 + "\n")
        plt.show()


if __name__ == "__main__":
    CONFIG = {
        "output_root": os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data")
        )
    }
    
    parser = argparse.ArgumentParser(description="Helioskrill CARLA Simulation Data Inspector")
    parser.add_argument("--data_dir", default=CONFIG["output_root"], help="Path to data directory")
    parser.add_argument("--episode", type=int, default=0, help="Episode ID to visualize (e.g. 0)")
    args = parser.parse_args()

    abs_data_dir = os.path.abspath(args.data_dir)

    try:
        visualizer = CARLADataVisualizer(abs_data_dir, args.episode)
        visualizer.show()
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize visualization: {e}")
        print("Ensure you have collected dataset samples first by running:")
        print("  python models/utils/carla_data_collector.py")
