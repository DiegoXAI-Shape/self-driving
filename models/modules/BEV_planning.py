import torch
import torch.nn as nn
import torch.nn.functional as F


class CommandEncoder(nn.Module):
    """
    Navigation Command Encoder MLP.
    Embeds discrete navigation commands (1: LANE_FOLLOW, 2: TURN_LEFT, 3: TURN_RIGHT, 4: STRAIGHT)
    and projects them into a 64-dimensional feature vector.
    """
    def __init__(self, num_commands: int = 6, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_commands, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, command):
        x = self.embedding(command) # [B, 64]
        return self.fc(x)          # [B, 64]


class BEVPlanningHead(nn.Module):
    """
    Legacy linear planning head for backward compatibility.
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

    def forward(self, x, command_embed=None):
        B = x.shape[0]
        x = self.pool(x).view(B, -1)
        out = self.fc(x)
        return out.view(B, self.num_waypoints, 4)


class PolynomialBEVPlanningHead(nn.Module):
    """
    Polynomial Planning Head for Experiment 3 backward compatibility.
    """
    def __init__(self, in_channels: int = 128, num_waypoints: int = 10, total_time: float = 2.5):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.total_time = total_time
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 4 * 4, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 12 + num_waypoints)
        )
        t_vals = torch.linspace(total_time / num_waypoints, total_time, num_waypoints)
        T = torch.stack([t_vals**k for k in range(6)], dim=1)
        self.register_buffer("T", T)
        dT = torch.stack([(k + 1) * (t_vals**k) for k in range(5)], dim=1)
        self.register_buffer("dT", dT)

    def forward(self, x, command_embed=None):
        B = x.shape[0]
        x = self.pool(x).view(B, -1)
        coeffs_and_z = self.fc(x)
        coeffs_x = coeffs_and_z[:, 0:6]
        coeffs_y = coeffs_and_z[:, 6:12]
        z_vals = coeffs_and_z[:, 12:]
        pred_x = torch.matmul(coeffs_x, self.T.T)
        pred_y = torch.matmul(coeffs_y, self.T.T)
        dx_dt = torch.matmul(coeffs_x[:, 1:], self.dT.T)
        dy_dt = torch.matmul(coeffs_y[:, 1:], self.dT.T)
        pred_yaw = torch.atan2(dy_dt, dx_dt)
        pred_wps = torch.stack([pred_x, pred_y, z_vals, pred_yaw], dim=-1)
        return pred_wps, (coeffs_x, coeffs_y)


class MultiHeadBEVPlanningHead(nn.Module):
    """
    Industrial Multi-Head Planning Head for Experiment 4:
    1. Quintic Polynomial Position Head (X(t), Y(t) coefficients + Z offsets)
    2. Bounded Trigonometric Yaw Head (sin(yaw), cos(yaw)) on unit circle
    3. Pedal & Speed Head (target_speed_mps, throttle, brake)
    """
    def __init__(self, in_channels: int = 128, num_waypoints: int = 10, total_time: float = 2.5):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.total_time = total_time
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.shared_fc = nn.Sequential(
            nn.Linear(in_channels * 4 * 4 + 64, 256), # +64 from CommandEncoder
            nn.SiLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.poly_head = nn.Linear(256, 12 + num_waypoints)
        self.yaw_trig_head = nn.Linear(256, num_waypoints * 2)
        
        self.pedal_speed_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 3) # [speed_mps, throttle, brake]
        )
        
        t_vals = torch.linspace(total_time / num_waypoints, total_time, num_waypoints)
        T = torch.stack([t_vals**k for k in range(6)], dim=1)
        self.register_buffer("T", T)
        
        dT = torch.stack([(k + 1) * (t_vals**k) for k in range(5)], dim=1)
        self.register_buffer("dT", dT)

    def forward(self, bev_features, command_embed):
        B = bev_features.shape[0]
        x = self.pool(bev_features)
        x = x.view(B, -1)
        
        x = torch.cat([x, command_embed], dim=-1)
        feat = self.shared_fc(x)
        
        coeffs_and_z = self.poly_head(feat)
        coeffs_x = coeffs_and_z[:, 0:6]
        coeffs_y = coeffs_and_z[:, 6:12]
        z_vals = coeffs_and_z[:, 12:]
        
        pred_x = torch.matmul(coeffs_x, self.T.T)
        pred_y = torch.matmul(coeffs_y, self.T.T)
        
        raw_trig = self.yaw_trig_head(feat).view(B, self.num_waypoints, 2)
        norm_trig = F.normalize(raw_trig, p=2, dim=-1)
        pred_yaw = torch.atan2(norm_trig[..., 0], norm_trig[..., 1]) * (180.0 / torch.pi)
        
        pred_wps = torch.stack([pred_x, pred_y, z_vals, pred_yaw], dim=-1)
        
        pedal_speed_out = self.pedal_speed_head(feat)
        pred_speed = pedal_speed_out[:, 0]
        pred_pedals = torch.sigmoid(pedal_speed_out[:, 1:3])
        
        return {
            "pred_waypoints": pred_wps,
            "coeffs": (coeffs_x, coeffs_y),
            "trig_yaw": norm_trig,
            "pred_speed": pred_speed,
            "pred_pedals": pred_pedals
        }


class MultiHeadPlanningLoss(nn.Module):
    """
    Weighted Multitask Loss Function for Multi-Head Trajectory Planning.
    """
    def __init__(self, w_pos: float = 1.0, w_yaw: float = 0.5, w_speed: float = 0.1, w_pedal: float = 0.1, w_smooth: float = 0.01):
        super().__init__()
        self.w_pos = w_pos
        self.w_yaw = w_yaw
        self.w_speed = w_speed
        self.w_pedal = w_pedal
        self.w_smooth = w_smooth
        self.huber = nn.HuberLoss(delta=1.0, reduction='none')

    def forward(self, outputs, target_wps, telemetry_target, sample_weights):
        pred_wps = outputs["pred_waypoints"]
        coeffs_x, coeffs_y = outputs["coeffs"]
        trig_yaw = outputs["trig_yaw"]
        pred_speed = outputs["pred_speed"]
        pred_pedals = outputs["pred_pedals"]
        
        loss_pos_raw = self.huber(pred_wps[..., :3], target_wps[..., :3]).mean(dim=[1, 2])
        
        gt_yaw_rad = target_wps[..., 3] * (torch.pi / 180.0)
        gt_sin = torch.sin(gt_yaw_rad)
        gt_cos = torch.cos(gt_yaw_rad)
        gt_trig = torch.stack([gt_sin, gt_cos], dim=-1)
        loss_yaw_raw = self.huber(trig_yaw, gt_trig).mean(dim=[1, 2])
        
        gt_speed = telemetry_target[:, 0]
        gt_pedals = telemetry_target[:, 1:3]
        loss_speed_raw = self.huber(pred_speed, gt_speed)
        loss_pedal_raw = self.huber(pred_pedals, gt_pedals).mean(dim=-1)
        
        a2, a3, a4, a5 = coeffs_x[:, 2], coeffs_x[:, 3], coeffs_x[:, 4], coeffs_x[:, 5]
        b2, b3, b4, b5 = coeffs_y[:, 2], coeffs_y[:, 3], coeffs_y[:, 4], coeffs_y[:, 5]
        loss_smooth_raw = (a2**2 + b2**2 + 0.5*(a3**2 + b3**2) + 0.1*(a4**2 + a5**2 + b4**2 + b5**2))
        
        sample_loss = (
            self.w_pos * loss_pos_raw +
            self.w_yaw * loss_yaw_raw +
            self.w_speed * loss_speed_raw +
            self.w_pedal * loss_pedal_raw +
            self.w_smooth * loss_smooth_raw
        )
        
        weighted_total_loss = torch.mean(sample_loss * sample_weights)
        
        return weighted_total_loss, {
            "loss_pos": torch.mean(loss_pos_raw).item(),
            "loss_yaw": torch.mean(loss_yaw_raw).item(),
            "loss_speed": torch.mean(loss_speed_raw).item(),
            "loss_pedal": torch.mean(loss_pedal_raw).item(),
            "loss_smooth": torch.mean(loss_smooth_raw).item()
        }


class PolynomialSmoothnessLoss(nn.Module):
    def __init__(self, w_accel: float = 0.1, w_jerk: float = 0.05, w_curvature: float = 0.01):
        super().__init__()
        self.w_accel = w_accel
        self.w_jerk = w_jerk
        self.w_curvature = w_curvature

    def forward(self, coeffs_x, coeffs_y):
        a2, a3, a4, a5 = coeffs_x[:, 2], coeffs_x[:, 3], coeffs_x[:, 4], coeffs_x[:, 5]
        b2, b3, b4, b5 = coeffs_y[:, 2], coeffs_y[:, 3], coeffs_y[:, 4], coeffs_y[:, 5]
        loss_accel = torch.mean(a2**2 + b2**2)
        loss_jerk = torch.mean(a3**2 + b3**2)
        loss_curvature = torch.mean(a4**2 + a5**2 + b4**2 + b5**2)
        return self.w_accel * loss_accel + self.w_jerk * loss_jerk + self.w_curvature * loss_curvature
