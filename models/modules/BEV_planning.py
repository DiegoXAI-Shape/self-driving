import torch
import torch.nn as nn


class BEVPlanningHead(nn.Module):
    """
    Planning head that regresses future relative waypoints of the ego vehicle from fused BEV features.
    
    Input shape:  [B, C_fused, H_bev, W_bev] (e.g., [B, 128, 400, 400])
    Output shape: [B, num_waypoints, 4] -> (rel_x, rel_y, rel_z, rel_yaw)
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
