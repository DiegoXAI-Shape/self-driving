import torch
import torch.nn as nn


class BEVPlanningHead(nn.Module):
    """
    Standard linear planning head that regresses 10 raw waypoints.
    """
    def __init__(self, in_channels: int = 128, num_waypoints: int = 10):
        super().__init__()
        self.num_waypoints = num_waypoints
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_waypoints * 4)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.pool(x)
        x = x.view(B, -1)
        out = self.fc(x)
        return out.view(B, self.num_waypoints, 4)


class PolynomialBEVPlanningHead(nn.Module):
    """
    Industrial Polynomial Planning Head (Quintic Trajectory Parametrization).
    
    Instead of predicting 10 unconstrained waypoints, predicts 12 Quintic Polynomial coefficients:
      X(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
      Y(t) = b0 + b1*t + b2*t^2 + b3*t^3 + b4*t^4 + b5*t^5
      
    Waypoints (x, y, z, yaw) are generated deterministically by evaluating the polynomial:
      - (x, y) = evaluated at t = [0.25, 0.50, ..., 2.50]
      - yaw = atan2(dY/dt, dX/dt)  <-- Guaranteed 100% heading alignment with velocity!
    """
    def __init__(self, in_channels: int = 128, num_waypoints: int = 10, total_time: float = 2.5):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.total_time = total_time
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Predicts 12 polynomial coefficients (6 for X, 6 for Y) + 10 Z offsets
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 4 * 4, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 12 + num_waypoints)
        )
        
        # Time evaluation matrix T [num_waypoints, 6] for t in [dt, 2dt, ..., total_time]
        t_vals = torch.linspace(total_time / num_waypoints, total_time, num_waypoints)  # e.g., 0.25 to 2.5s
        
        # T matrix: [10, 6] -> [1, t, t^2, t^3, t^4, t^5]
        T = torch.stack([t_vals**k for k in range(6)], dim=1)
        self.register_buffer("T", T)
        
        # Derivative Time matrix dT [10, 5] -> [1, 2t, 3t^2, 4t^3, 5t^4]
        dT = torch.stack([(k + 1) * (t_vals**k) for k in range(5)], dim=1)
        self.register_buffer("dT", dT)

    def forward(self, x):
        B = x.shape[0]
        x = self.pool(x)
        x = x.view(B, -1)
        coeffs_and_z = self.fc(x)  # [B, 12 + num_waypoints]
        
        coeffs_x = coeffs_and_z[:, 0:6]   # [B, 6] -> (a0, a1, a2, a3, a4, a5)
        coeffs_y = coeffs_and_z[:, 6:12]  # [B, 6] -> (b0, b1, b2, b3, b4, b5)
        z_vals = coeffs_and_z[:, 12:]     # [B, 10]
        
        # 1. Evaluate positions: X = T @ a, Y = T @ b
        pred_x = torch.matmul(coeffs_x, self.T.T)  # [B, 10]
        pred_y = torch.matmul(coeffs_y, self.T.T)  # [B, 10]
        
        # 2. Evaluate velocities (derivatives): dX = dT @ (a1..a5), dY = dT @ (b1..b5)
        dx_dt = torch.matmul(coeffs_x[:, 1:], self.dT.T)  # [B, 10]
        dy_dt = torch.matmul(coeffs_y[:, 1:], self.dT.T)  # [B, 10]
        
        # 3. Heading Yaw is 100% mathematically aligned with trajectory tangent
        pred_yaw = torch.atan2(dy_dt, dx_dt)  # [B, 10]
        
        # Stack to format [B, 10, 4] -> (rel_x, rel_y, rel_z, rel_yaw)
        pred_wps = torch.stack([pred_x, pred_y, z_vals, pred_yaw], dim=-1)
        
        return pred_wps, (coeffs_x, coeffs_y)


class PolynomialSmoothnessLoss(nn.Module):
    """
    Smoothness and Kinematic Curvature Loss for Polynomial Trajectories.
    Penalizes acceleration (2nd deriv), jerk (3rd deriv), and higher order curvature terms.
    """
    def __init__(self, w_accel: float = 0.1, w_jerk: float = 0.05, w_curvature: float = 0.01):
        super().__init__()
        self.w_accel = w_accel
        self.w_jerk = w_jerk
        self.w_curvature = w_curvature

    def forward(self, coeffs_x, coeffs_y):
        # coeffs: [B, 6] -> (a0, a1, a2, a3, a4, a5)
        a2, a3, a4, a5 = coeffs_x[:, 2], coeffs_x[:, 3], coeffs_x[:, 4], coeffs_x[:, 5]
        b2, b3, b4, b5 = coeffs_y[:, 2], coeffs_y[:, 3], coeffs_y[:, 4], coeffs_y[:, 5]
        
        # Acceleration penalty (a2, b2)
        loss_accel = torch.mean(a2**2 + b2**2)
        
        # Jerk penalty (a3, b3)
        loss_jerk = torch.mean(a3**2 + b3**2)
        
        # High frequency curvature penalty (a4, a5, b4, b5)
        loss_curvature = torch.mean(a4**2 + a5**2 + b4**2 + b5**2)
        
        total_smooth_loss = (self.w_accel * loss_accel + 
                             self.w_jerk * loss_jerk + 
                             self.w_curvature * loss_curvature)
        return total_smooth_loss
