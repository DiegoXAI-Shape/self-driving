import numpy as np
import threading
from collections import deque
import time
import random

# ==============================================================================
# MODULE 1: EXTENDED KALMAN FILTER (EKF) FOR IMU + LIDAR FUSION
# ==============================================================================

class ExtendedKalmanFilter:
    """
    An Extended Kalman Filter (EKF) to fuse high-frequency IMU reads (~100Hz)
    with lower-frequency LiDAR measurements (~10-20Hz).
    
    State Vector x (8x1):
        x = [p_x, p_y, v_x, v_y, a_x, a_y, theta, omega]^T
        - p_x, p_y: Global position (meters)
        - v_x, v_y: Global velocity (meters/second)
        - a_x, a_y: Global acceleration (meters/second^2)
        - theta: Yaw angle / orientation (radians)
        - omega: Yaw rate / angular velocity (radians/second)
    """
    def __init__(self, dt=0.01, rolling_window_size=15):
        self.dt = dt
        self.M = rolling_window_size
        
        # State vector x [8, 1]
        self.x = np.zeros((8, 1))
        
        # State covariance P [8, 8] - uncertainty of the state estimation
        self.P = np.eye(8) * 1.0
        
        # Process noise covariance Q [8, 8] - uncertainty of the kinematic model
        self.Q = np.eye(8) * 0.01
        self.Q[0, 0] = 0.05  # Position X
        self.Q[1, 1] = 0.05  # Position Y
        self.Q[2, 2] = 0.1   # Velocity X
        self.Q[3, 3] = 0.1   # Velocity Y
        self.Q[4, 4] = 0.5   # Acceleration X
        self.Q[5, 5] = 0.5   # Acceleration Y
        self.Q[6, 6] = 0.01  # Yaw (orientation)
        self.Q[7, 7] = 0.05  # Yaw rate

        # Baseline (minimum) measurement noise covariance R_min
        # IMU: [a_body_x, a_body_y, omega_imu]
        self.R_imu_min = np.diag([0.2, 0.2, 0.05])
        self.R_imu = self.R_imu_min.copy()
        
        # LiDAR: [p_x, p_y, v_x, v_y]
        self.R_lidar_min = np.diag([0.1, 0.1, 0.1, 0.1])
        self.R_lidar = self.R_lidar_min.copy()
        
        # Deques to store rolling innovations (residuals) for adaptive covariance estimation
        self.imu_innovations = deque(maxlen=self.M)
        self.lidar_innovations = deque(maxlen=self.M)

    def init_state(self, px, py, vx=0.0, vy=0.0, yaw=0.0):
        """
        Initializes the state of the vehicle.
        """
        self.x = np.array([[px], [py], [vx], [vy], [0.0], [0.0], [yaw], [0.0]], dtype=np.float64)
        self.P = np.eye(8) * 0.5

    def predict(self, dt=None):
        """
        Predict step based on a constant acceleration kinematic motion model.
        """
        if dt is not None:
            self.dt = dt
            
        dt = self.dt
        dt2 = 0.5 * (dt ** 2)
        
        # Extract current state values
        px, py, vx, vy, ax, ay, theta, omega = self.x.flatten()
        
        # State Transition Function f(x)
        px_next = px + vx * dt + ax * dt2
        py_next = py + vy * dt + ay * dt2
        vx_next = vx + ax * dt
        vy_next = vy + ay * dt
        ax_next = ax
        ay_next = ay
        theta_next = theta + omega * dt
        
        # Normalize yaw to [-pi, pi]
        theta_next = (theta_next + np.pi) % (2 * np.pi) - np.pi
        omega_next = omega
        
        self.x = np.array([[px_next], [py_next], [vx_next], [vy_next], 
                           [ax_next], [ay_next], [theta_next], [omega_next]], dtype=np.float64)
        
        # Jacobian matrix F of the transition model (linear for constant acceleration model)
        F = np.array([
            [1.0, 0.0, dt,  0.0, dt2, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, dt,  0.0, dt2, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, dt,  0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, dt,  0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, dt ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Predict state covariance: P = F P F^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update_imu(self, z_imu):
        """
        Correction step for an IMU measurement.
        z_imu = [a_body_x, a_body_y, omega_imu]^T (acceleration in body-frame, yaw-rate)
        """
        px, py, vx, vy, ax, ay, theta, omega = self.x.flatten()
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # 1. Non-linear Measurement function h_imu(x)
        # Rotates global accelerations into sensor body-frame
        h_ax = ax * cos_t + ay * sin_t
        h_ay = -ax * sin_t + ay * cos_t
        h_omega = omega
        h = np.array([[h_ax], [h_ay], [h_omega]], dtype=np.float64)
        
        # 2. Innovation (residual) y
        y = z_imu.reshape(3, 1) - h
        self.imu_innovations.append(y.flatten())
        
        # 3. Dynamically adjust R_imu based on recent residuals variance
        self._adapt_R_imu()
        
        # 4. Measurement Jacobian H_imu (3x8)
        H = np.zeros((3, 8), dtype=np.float64)
        H[0, 4] = cos_t
        H[0, 5] = sin_t
        H[0, 6] = -ax * sin_t + ay * cos_t  # d_h_ax / d_theta
        
        H[1, 4] = -sin_t
        H[1, 5] = cos_t
        H[1, 6] = -ax * cos_t - ay * sin_t  # d_h_ay / d_theta
        
        H[2, 7] = 1.0                       # d_h_omega / d_omega
        
        # 5. Kalman gain and state update
        S = H @ self.P @ H.T + self.R_imu
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P
        
        # Re-normalize yaw to [-pi, pi]
        self.x[6, 0] = (self.x[6, 0] + np.pi) % (2 * np.pi) - np.pi

    def update_lidar(self, z_lidar):
        """
        Correction step for a LiDAR measurement.
        z_lidar = [px_lidar, py_lidar, vx_lidar, vy_lidar]^T (global pos and velocity)
        """
        # 1. Linear Measurement function h_lidar(x)
        h = self.x[:4]
        
        # 2. Innovation y
        y = z_lidar.reshape(4, 1) - h
        self.lidar_innovations.append(y.flatten())
        
        # 3. Dynamically adjust R_lidar based on recent residuals variance
        self._adapt_R_lidar()
        
        # 4. Measurement Jacobian H_lidar (4x8)
        H = np.zeros((4, 8), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[3, 3] = 1.0
        
        # 5. Kalman gain and state update
        S = H @ self.P @ H.T + self.R_lidar
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P
        
        # Re-normalize yaw
        self.x[6, 0] = (self.x[6, 0] + np.pi) % (2 * np.pi) - np.pi

    def _adapt_R_imu(self):
        """
        Estimates the R_imu noise covariance based on the variance of recent innovations.
        R_adaptive = max(R_min, C_y - H P H^T)
        """
        if len(self.imu_innovations) < self.M:
            return  # Need enough samples to calculate variance
            
        innovations = np.array(self.imu_innovations)
        # Sample variance of innovations
        C_y = np.var(innovations, axis=0) # [3]
        
        # Calculate theoretical H P H^T
        px, py, vx, vy, ax, ay, theta, omega = self.x.flatten()
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        H = np.zeros((3, 8), dtype=np.float64)
        H[0, 4] = cos_t; H[0, 5] = sin_t; H[0, 6] = -ax * sin_t + ay * cos_t
        H[1, 4] = -sin_t; H[1, 5] = cos_t; H[1, 6] = -ax * cos_t - ay * sin_t
        H[2, 7] = 1.0
        
        HPH = H @ self.P @ H.T
        diag_HPH = np.diag(HPH)
        
        # R = C_y - diag(HPH)
        r_adapted = C_y - diag_HPH
        
        # Clamp to baseline R_min to guarantee positive-definiteness
        r_min = np.diag(self.R_imu_min)
        r_final = np.maximum(r_min, r_adapted)
        
        self.R_imu = np.diag(r_final)

    def _adapt_R_lidar(self):
        """
        Estimates the R_lidar noise covariance based on the variance of recent innovations.
        """
        if len(self.lidar_innovations) < self.M:
            return
            
        innovations = np.array(self.lidar_innovations)
        C_y = np.var(innovations, axis=0) # [4]
        
        # H is simply projection to first 4 elements, H P H^T is the top-left 4x4 of P
        diag_HPH = np.diag(self.P[:4, :4])
        
        r_adapted = C_y - diag_HPH
        r_min = np.diag(self.R_lidar_min)
        r_final = np.maximum(r_min, r_adapted)
        
        self.R_lidar = np.diag(r_final)

    def get_state(self):
        """
        Returns the current estimated state and covariance.
        """
        px, py, vx, vy, ax, ay, theta, omega = self.x.flatten()
        return {
            "position": (px, py),
            "velocity": (vx, vy),
            "speed": np.sqrt(vx**2 + vy**2),
            "acceleration": (ax, ay),
            "yaw": theta,
            "yaw_rate": omega,
            "P": self.P.copy(),
            "R_imu": np.diag(self.R_imu).tolist(),
            "R_lidar": np.diag(self.R_lidar).tolist()
        }


# ==============================================================================
# MODULE 2: MULTI-CAMERA TEMPORAL SYNCHRONIZATION BUFFER
# ==============================================================================

class CameraSyncBuffer:
    """
    Thread-safe synchronization buffer to align frame streams from 8 independent cameras.
    Ensures frames are processed together only if their timestamps are within 5ms of each other.
    Drops old/lagging frames to prevent blocking the perception pipeline.
    """
    def __init__(self, tolerance_ms=5.0):
        self.tolerance = tolerance_ms / 1000.0  # Convert to seconds
        self.lock = threading.Lock()
        
        # Dict of deques for 8 cameras storing: (timestamp, image_data)
        self.buffers = {i: deque() for i in range(8)}
        self.total_dropped = 0
        self.total_synced = 0

    def add_frame(self, cam_id, timestamp, image_data):
        """
        Inserts a new frame from a camera stream.
        
        Args:
            cam_id: Camera index (0 to 7)
            timestamp: Event timestamp in seconds (float)
            image_data: Image tensor / raw frame array
        Returns:
            synced_group: Dict of {cam_id: (timestamp, image_data)} for all 8 cameras if synchronized,
                          otherwise None.
        """
        with self.lock:
            # Append new data
            self.buffers[cam_id].append((timestamp, image_data))
            
            # Synchronization loop
            while True:
                # 1. We need at least one frame in all 8 buffers
                if any(len(self.buffers[i]) == 0 for i in range(8)):
                    return None  # Wait for more data
                
                # 2. Inspect the oldest frame in each buffer
                oldest_timestamps = {i: self.buffers[i][0][0] for i in range(8)}
                
                t_min_cam = min(oldest_timestamps, key=oldest_timestamps.get)
                t_min = oldest_timestamps[t_min_cam]
                t_max = max(oldest_timestamps.values())
                
                # 3. Check if the timestamps align within the 5ms window
                if (t_max - t_min) <= self.tolerance:
                    # Match found! Pop the matched frames from all buffers
                    synced_group = {}
                    for i in range(8):
                        synced_group[i] = self.buffers[i].popleft()
                    self.total_synced += 1
                    return synced_group
                    
                # 4. If the delta is larger than 5ms, the oldest frame (t_min) is too old
                #    and must be discarded so it does not block the pipeline.
                else:
                    self.buffers[t_min_cam].popleft()
                    self.total_dropped += 1
                    # Loop again to check the next set of oldest frames


# ==============================================================================
# MOCK SIMULATION AND VERIFICATION TESTS
# ==============================================================================

def run_imu_lidar_ekf_test():
    print("\n" + "="*80)
    print("           RUNNING MODULE 1 TEST: EKF IMU + LIDAR SENSOR FUSION         ")
    print("="*80)
    
    # 1. Initialize filter (dt = 10ms for 100Hz IMU updates)
    dt = 0.01
    ekf = ExtendedKalmanFilter(dt=dt, rolling_window_size=15)
    ekf.init_state(px=20.0, py=0.0, vx=0.0, vy=10.0, yaw=np.pi/2)
    
    # 2. Setup ground truth trajectory (Car moving in a circle of R=20m at v=10m/s)
    # Centripetal acceleration = v^2/R = 100/20 = 5.0 m/s^2 (in body Y direction)
    # Yaw rate (omega) = v/R = 10/20 = 0.5 rad/s
    R = 20.0
    v = 10.0
    omega = 0.5
    
    true_x = 20.0
    true_y = 0.0
    true_yaw = np.pi/2
    
    print("Simulating circular motion for 5.0 seconds...")
    print("Adding a massive IMU noise spike (sensor anomaly) between t=2.0s and t=3.5s.")
    print("Testing if EKF automatically adjusts covariance R and relies on LiDAR.")
    print("-"*80)
    print(f"{'Time':<6} | {'True Pos (X, Y)':<18} | {'Est Pos (X, Y)':<18} | {'Err (m)':<7} | {'R_imu (ax)':<11} | {'R_lidar (x)':<12}")
    print("-"*80)
    
    # Run simulation steps
    for step in range(500):
        t = step * dt
        
        # Update true trajectory
        true_yaw += omega * dt
        true_yaw = (true_yaw + np.pi) % (2 * np.pi) - np.pi
        
        # Circular motion position
        true_x = R * np.cos(true_yaw - np.pi/2)
        true_y = R * np.sin(true_yaw - np.pi/2)
        
        # Predict step
        ekf.predict()
        
        # --- Sensor Generation ---
        # Baseline noise
        imu_noise_std = 0.1
        lidar_noise_std = 0.2
        
        # Inject massive IMU noise spike between 2.0s and 3.5s
        is_imu_anomalous = (2.0 <= t < 3.5)
        if is_imu_anomalous:
            imu_noise_std = 5.5 # Extreme noise/drift
            
        # 100Hz IMU update: centripetal acceleration is 5.0 in Y, 0 in X (body frame)
        ax_body = 0.0 + np.random.normal(0, imu_noise_std)
        ay_body = (v**2 / R) + np.random.normal(0, imu_noise_std)
        omega_imu = omega + np.random.normal(0, imu_noise_std * 0.1)
        
        z_imu = np.array([ax_body, ay_body, omega_imu])
        ekf.update_imu(z_imu)
        
        # 10Hz LiDAR update (every 10 steps)
        if step % 10 == 0:
            px_lidar = true_x + np.random.normal(0, lidar_noise_std)
            py_lidar = true_y + np.random.normal(0, lidar_noise_std)
            vx_lidar = -v * np.sin(true_yaw - np.pi/2) + np.random.normal(0, lidar_noise_std)
            vy_lidar = v * np.cos(true_yaw - np.pi/2) + np.random.normal(0, lidar_noise_std)
            
            z_lidar = np.array([px_lidar, py_lidar, vx_lidar, vy_lidar])
            ekf.update_lidar(z_lidar)
            
        # Log status periodically
        if step % 50 == 0 or step == 499:
            state = ekf.get_state()
            est_x, est_y = state["position"]
            err = np.sqrt((est_x - true_x)**2 + (est_y - true_y)**2)
            
            # Retrieve adapted measurement noise
            r_imu_ax = state["R_imu"][0]
            r_lidar_x = state["R_lidar"][0]
            
            status_str = "ANOMALY" if is_imu_anomalous else "NORMAL"
            print(f"{t:<5.2f}s | ({true_x:>6.2f}, {true_y:>6.2f}) | ({est_x:>6.2f}, {est_y:>6.2f}) | {err:>5.3f}m | {r_imu_ax:>9.4f}  | {r_lidar_x:>10.4f}  [{status_str}]")

    print("-"*80)
    print("EKF Test completed successfully.")


def run_camera_sync_test():
    print("\n" + "="*80)
    print("          RUNNING MODULE 2 TEST: CAMERA TIMESTAMPS SYNC BUFFER          ")
    print("="*80)
    
    # Instantiate Sync Buffer with 5ms tolerance
    sync_buf = CameraSyncBuffer(tolerance_ms=5.0)
    
    print("Simulating asynchronous frame arrivals...")
    print("Tolerance window: 5ms (0.005 seconds)")
    print("-"*80)
    
    # Helper to print queue sizes
    def print_queue_states():
        sizes = [len(sync_buf.buffers[i]) for i in range(8)]
        print(f"  Buffer sizes: {sizes} | Total Synced: {sync_buf.total_synced} | Total Dropped: {sync_buf.total_dropped}")

    # Case 1: Perfect Alignment (within 5ms)
    print("\nCase 1: Sending aligned frames at T = 0.100s (with offsets < 2ms)...")
    for cam in range(8):
        t_offset = random.uniform(-0.001, 0.001)
        sync_buf.add_frame(cam, 0.100 + t_offset, f"frame_0.1_{cam}")
        
    print_queue_states()
    
    # Case 2: One camera lags, but still within 5ms (e.g. 4ms offset)
    print("\nCase 2: Sending aligned frames at T = 0.200s (Cam 7 lags by 4ms)...")
    for cam in range(7):
        sync_buf.add_frame(cam, 0.200, f"frame_0.2_{cam}")
    # Add lagging camera 7 within threshold
    sync_buf.add_frame(7, 0.204, "frame_0.2_7")
    
    print_queue_states()
    
    # Case 3: Stale Frame Drop (Cam 7 lags by 12ms > 5ms tolerance)
    print("\nCase 3: Sending frames at T = 0.300s, but Cam 7 lags by 12ms (3.012s vs 3.000s)...")
    print("First, sending frames for Cameras 0-6 at 0.300s:")
    for cam in range(7):
        sync_buf.add_frame(cam, 0.300, f"frame_0.3_{cam}")
    print_queue_states()
    
    print("Now, adding Cam 7 at 0.312s (12ms late). Old 0.300s frames should be dropped:")
    sync_buf.add_frame(7, 0.312, "frame_0.3_7")
    print_queue_states()
    
    # Case 4: Recovering and syncing next frame
    print("\nCase 4: Sending new aligned frames at T = 0.400s (offsets < 1ms)...")
    for cam in range(8):
        sync_buf.add_frame(cam, 0.400, f"frame_0.4_{cam}")
        
    print_queue_states()
    
    print("-"*80)
    print("Camera Sync Buffer Test completed successfully.")
    print("="*80)


if __name__ == '__main__':
    run_imu_lidar_ekf_test()
    run_camera_sync_test()
