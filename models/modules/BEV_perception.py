import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. CAMERA 2D BACKBONE (Feature Extractor)
# ==============================================================================

class ResidualBlock2D(nn.Module):
    """
    Standard 2D residual block for feature extraction.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class CameraBackbone2D(nn.Module):
    """
    Extracts features from multi-view images of shape [B, N, C, H, W].
    Fuses stacked temporal frames at the channel level.
    Optimized for lower resolutions (e.g. 150x200) to keep memory footprint low.
    """
    def __init__(self, in_channels=15, feat_channels=64):
        super().__init__()
        # Initial downsampling layer (stride 2)
        # Input: [B*N, 15, 150, 200] -> [B*N, 32, 75, 100]
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet-like stages to extract deep features
        self.layer1 = nn.Sequential(
            ResidualBlock2D(32, 32, stride=1),
            ResidualBlock2D(32, 32, stride=1)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock2D(32, 64, stride=2), # [B*N, 64, 38, 50]
            ResidualBlock2D(64, 64, stride=1)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock2D(64, feat_channels, stride=1), # [B*N, feat_channels, 38, 50]
            ResidualBlock2D(feat_channels, feat_channels, stride=1)
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        # Flatten Batch and Camera views for 2D Conv
        x_flat = x.view(B * N, C, H, W)
        
        out = self.relu(self.bn1(self.conv1(x_flat)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Reshape back to [B, N, C_feat, H_feat, W_feat]
        C_feat, H_feat, W_feat = out.shape[1], out.shape[2], out.shape[3]
        return out.view(B, N, C_feat, H_feat, W_feat)

# ==============================================================================
# 2. LIDAR ENCODER (BEV Feature Extractor with Voxel Grid Statistics)
# ==============================================================================

class LidarBEVEncoder(nn.Module):
    """
    Processes a pre-projected LiDAR Bird's Eye View grid of shape [B, C_lidar, H_bev, W_bev].
    Expects C_lidar = 5 channels representing statistical features calculated on CPU:
      - Channel 0: Max height (Z_max) in the grid cell
      - Channel 1: Height difference (Z_max - Z_min) representing vertical span
      - Channel 2: Mean height (Z_mean) representing density center
      - Channel 3: Points density (number of points normalized)
      - Channel 4: Max intensity (laser reflectivity)
    """
    def __init__(self, in_channels=5, feat_channels=64):
        super().__init__()
        # Initial convolution on the BEV grid (e.g. 400x400)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer = nn.Sequential(
            ResidualBlock2D(32, 64, stride=1),
            ResidualBlock2D(64, feat_channels, stride=1)
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer(out)
        return out

# ==============================================================================
# 3. 2D-TO-BEV PROJECTION MODULE
# ==============================================================================

class CameraBEVProjection(nn.Module):
    """
    Projects 2D multi-view image features into a unified 3D Bird's Eye View (BEV) space
    using geometry-guided backward projection (Spatial grid querying).
    """
    def __init__(self, bev_height=400, bev_width=400, grid_resolution=0.25, z_slices=[-1.0, 0.0, 1.0, 2.0]):
        super().__init__()
        self.bev_height = bev_height
        self.bev_width = bev_width
        self.grid_resolution = grid_resolution
        self.z_slices = z_slices
        
        # Precompute 3D coordinate grid of BEV space in ego vehicle coordinate frame
        # X: Forward, Y: Left, Z: Up
        x_coords = torch.linspace(-bev_height / 2 * grid_resolution, bev_height / 2 * grid_resolution, bev_height)
        y_coords = torch.linspace(-bev_width / 2 * grid_resolution, bev_width / 2 * grid_resolution, bev_width)
        z_coords = torch.tensor(z_slices)
        
        # Generate grid coords: shape [bev_height, bev_width, len(z_slices), 3]
        grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        
        # Register buffer to automatically move it to GPU along with model
        self.register_buffer("grid_points", grid_points.reshape(-1, 3)) # [num_points, 3]

    def forward(self, x_2d, extrinsics, intrinsics, original_img_size=(800, 600)):
        """
        Args:
            x_2d: Multiview 2D features of shape [B, N, C, H_feat, W_feat]
            extrinsics: Ego-to-Camera matrices of shape [B, N, 4, 4]
                        Maps points from Ego Coordinates -> Camera Coordinates
            intrinsics: Camera Intrinsic matrices of shape [B, N, 3, 3]
            original_img_size: Tuple representing original (width, height) used for mapping intrinsics
        Returns:
            bev_features: 2D BEV features of shape [B, C, H_bev, W_bev]
        """
        B, N, C, H_feat, W_feat = x_2d.shape
        num_points = self.grid_points.shape[0]
        img_w, img_h = original_img_size
        
        # Expand BEV points for batch and camera views -> [B, N, num_points, 3]
        pts_ego = self.grid_points.unsqueeze(0).unsqueeze(1).expand(B, N, -1, -1)
        
        # 1. Transform Ego points to Camera frame: P_cam = R * P_ego + T
        R = extrinsics[..., :3, :3] # [B, N, 3, 3]
        T = extrinsics[..., :3, 3].unsqueeze(-2) # [B, N, 1, 3]
        pts_cam = torch.matmul(pts_ego, R.transpose(-1, -2)) + T # [B, N, num_points, 3]
        
        # 2. Project 3D Camera points onto 2D image plane: P_pixel = K * P_cam
        K = intrinsics # [B, N, 3, 3]
        pts_pixel = torch.matmul(pts_cam, K.transpose(-1, -2)) # [B, N, num_points, 3]
        
        # 3. Homogeneous divide (divide by Z/depth)
        depth = pts_pixel[..., 2:3]
        depth = torch.clamp(depth, min=1e-5)
        pixel_coords = pts_pixel[..., :2] / depth # [B, N, num_points, 2]
        
        # 4. Normalize to [-1, 1] for grid_sample API compatibility
        x_norm = (pixel_coords[..., 0] / img_w) * 2.0 - 1.0
        y_norm = (pixel_coords[..., 1] / img_h) * 2.0 - 1.0
        grid_sample_coords = torch.stack([x_norm, y_norm], dim=-1) # [B, N, num_points, 2]
        
        # 5. Generate validation mask: points must be in front of the camera and within FOV
        valid_mask = (x_norm >= -1.0) & (x_norm <= 1.0) & (y_norm >= -1.0) & (y_norm <= 1.0) & (pts_cam[..., 2] > 0.1)
        valid_mask = valid_mask.float().unsqueeze(2) # [B, N, 1, num_points]
        
        # 6. Sample features from 2D representations
        x_2d_flat = x_2d.view(B * N, C, H_feat, W_feat)
        grid_coords_flat = grid_sample_coords.view(B * N, 1, num_points, 2)
        
        # Sample features: output is [B*N, C, 1, num_points]
        sampled_features = F.grid_sample(x_2d_flat, grid_coords_flat, align_corners=False)
        sampled_features = sampled_features.view(B, N, C, num_points)
        
        # Mask out-of-bounds queries
        sampled_features = sampled_features * valid_mask
        
        # 7. Aggregate queries across all views (masked average)
        sum_features = torch.sum(sampled_features, dim=1) # [B, C, num_points]
        sum_masks = torch.sum(valid_mask, dim=1) # [B, 1, num_points]
        sum_masks = torch.clamp(sum_masks, min=1.0)
        
        bev_points_features = sum_features / sum_masks # [B, C, num_points]
        
        # 8. Reconstruct 3D Grid & reduce Z-dimension
        num_z = len(self.z_slices)
        bev_features_3d = bev_points_features.view(B, C, self.bev_height, self.bev_width, num_z)
        
        # Compress vertical slices to get flat 2D BEV features
        bev_features_2d = torch.mean(bev_features_3d, dim=-1) # [B, C, H_bev, W_bev]
        
        return bev_features_2d

# ==============================================================================
# 4. BEV FUSION & ENCODER-DECODER HEADS
# ==============================================================================

class BEVFusionNeck(nn.Module):
    """
    Concatenates and fuses BEV features extracted from Cameras and LiDAR.
    """
    def __init__(self, cam_channels=64, lidar_channels=64, out_channels=128):
        super().__init__()
        in_channels = cam_channels + lidar_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.block1 = ResidualBlock2D(out_channels, out_channels, stride=1)
        self.block2 = ResidualBlock2D(out_channels, out_channels, stride=1)

    def forward(self, cam_bev, lidar_bev):
        # Concatenate along the channel axis
        x = torch.cat([cam_bev, lidar_bev], dim=1)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        return out

class BEVSemanticDecoder(nn.Module):
    """
    U-Net style decoder that outputs segmentation masks (occupancy grid maps) in BEV space.
    """
    def __init__(self, in_channels=128, num_classes=4):
        super().__init__()
        self.decoder = nn.Sequential(
            ResidualBlock2D(in_channels, 64, stride=1),
            ResidualBlock2D(64, 32, stride=1),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.decoder(x)

class BEVObjectDetectionHead(nn.Module):
    """
    CenterNet/CenterPoint styled 3D bounding box prediction head in BEV space.
    Predicts class heatmaps, sub-pixel center offset, dimensions (size), and orientation (yaw).
    """
    def __init__(self, in_channels=128, num_classes=3):
        super().__init__()
        # Heatmap prediction (centers of objects)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        # Center coordinate offset regression
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )
        # Size regression (width, length, height)
        self.size_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1)
        )
        # Heading / Yaw angle regression (cos(yaw), sin(yaw))
        self.yaw_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )

    def forward(self, x):
        return {
            "heatmap": self.heatmap_head(x),
            "offset": self.offset_head(x),
            "size": self.size_head(x),
            "yaw": self.yaw_head(x)
        }

# ==============================================================================
# 5. MAIN SENSOR FUSION PERCEPTION NETWORK
# ==============================================================================

class BEVPerceptionNet(nn.Module):
    """
    End-to-end multi-modal sensor fusion perception model.
    Inputs:
        - 8 Camera views with stacked frames: [B, 8, 15, 150, 200] (optimized resolution)
        - 1 LiDAR projected BEV map: [B, 5, 400, 400] (high resolution 25cm grid)
    Outputs:
        - BEV Semantic Segmentation logits: [B, num_seg_classes, 400, 400]
        - BEV Object Detection outputs (dict of heatmaps and bounding box properties)
    """
    def __init__(self, num_seg_classes=4, num_det_classes=3, bev_height=400, bev_width=400, grid_resolution=0.25):
        super().__init__()
        self.cam_backbone = CameraBackbone2D(in_channels=15, feat_channels=64)
        self.lidar_backbone = LidarBEVEncoder(in_channels=5, feat_channels=64)
        self.projection = CameraBEVProjection(bev_height=bev_height, bev_width=bev_width, grid_resolution=grid_resolution)
        
        self.fusion_neck = BEVFusionNeck(cam_channels=64, lidar_channels=64, out_channels=128)
        
        # Dual perception tasks heads
        self.seg_head = BEVSemanticDecoder(in_channels=128, num_classes=num_seg_classes)
        self.det_head = BEVObjectDetectionHead(in_channels=128, num_classes=num_det_classes)

    def forward(self, camera_imgs, lidar_bev, extrinsics, intrinsics):
        """
        Args:
            camera_imgs: Multi-view temporal images -> [B, N, C_cam, H_img, W_img]
            lidar_bev: Pre-projected LiDAR grids -> [B, 5, H_bev, W_bev]
            extrinsics: Camera-to-Ego extrinsics -> [B, N, 4, 4]
            intrinsics: Camera intrinsics -> [B, N, 3, 3]
        """
        # 1. Extract 2D camera features
        # Input: [B, N, 15, 150, 200] -> Output: [B, N, 64, 38, 50]
        cam_features_2d = self.cam_backbone(camera_imgs)
        
        # 2. Project camera features to BEV space
        # Output: [B, 64, H_bev, W_bev] (e.g. [B, 64, 400, 400])
        cam_features_bev = self.projection(cam_features_2d, extrinsics, intrinsics)
        
        # 3. Extract LiDAR features in BEV space
        # Input: [B, 5, 400, 400] -> Output: [B, 64, 400, 400]
        lidar_features_bev = self.lidar_backbone(lidar_bev)
        
        # 4. Fuse features
        # Output: [B, 128, 400, 400]
        fused_bev = self.fusion_neck(cam_features_bev, lidar_features_bev)
        
        # 5. Run task heads
        seg_logits = self.seg_head(fused_bev)
        det_outputs = self.det_head(fused_bev)
        
        return seg_logits, det_outputs


# ==============================================================================
# 6. MOCK TEST FOR VERIFICATION
# ==============================================================================
if __name__ == '__main__':
    print("------------------------------------------------------------------")
    print("   Testing Optimized BEVPerceptionNet Sensor Fusion Model Build   ")
    print("------------------------------------------------------------------")
    
    # 1. Config dimensions matching requirements
    B = 16
    N = 8
    C_cam = 15     # Frame stacking (e.g., 5 frames of 3 channels RGB)
    H_cam, W_cam = 150, 200 # Optimized camera resolution
    
    H_bev, W_bev = 400, 400 # High-precision grid (25cm resolution)
    C_lidar = 5    # Voxel statistics channels (max_z, diff_z, mean_z, density, intensity)
    
    print(f"Creating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BEVPerceptionNet(
        num_seg_classes=4, # e.g. [Background, Road, Lane markings, Obstacles]
        num_det_classes=3, # e.g. [Car, Pedestrian, Cyclist]
        bev_height=H_bev,
        bev_width=W_bev,
        grid_resolution=0.25
    ).to(device)
    
    # 2. Generate dummy sensors input
    print("\nGenerating dummy input tensors...")
    camera_imgs = torch.randn(B, N, C_cam, H_cam, W_cam).to(device)
    lidar_bev = torch.randn(B, C_lidar, H_bev, W_bev).to(device)
    
    # Intrinsics: Calculate focal length based on FOV = 100 degrees for 800x600 resolution
    # f = W / (2 * tan(FOV/2)) = 800 / (2 * tan(50 deg)) = 800 / (2 * 1.19175) ≈ 335.6
    focal = 335.6
    intrinsics = torch.zeros(B, N, 3, 3).to(device)
    for b in range(B):
        for n in range(N):
            intrinsics[b, n] = torch.tensor([
                [focal, 0.0, 400.0],
                [0.0, focal, 300.0],
                [0.0, 0.0, 1.0]
            ])
            
    # Extrinsics: Generate camera matrices based on Tesla proportions (8 cameras)
    # (x, y, z) position and (yaw) angle in degrees relative to the ego frame
    import numpy as np
    camera_configs = [
        # 0: Front Main
        {"pos": (1.35, 0.00, 1.45), "yaw": 0.0},
        # 1: Front Wide
        {"pos": (1.35, 0.00, 1.45), "yaw": 0.0},
        # 2: Front Narrow
        {"pos": (1.35, 0.00, 1.45), "yaw": 0.0},
        # 3: Left B-Pillar
        {"pos": (-0.20, -0.90, 1.30), "yaw": -60.0},
        # 4: Right B-Pillar
        {"pos": (-0.20, 0.90, 1.30), "yaw": 60.0},
        # 5: Left Repeater
        {"pos": (0.85, -0.95, 1.10), "yaw": -150.0},
        # 6: Right Repeater
        {"pos": (0.85, 0.95, 1.10), "yaw": 150.0},
        # 7: Rear
        {"pos": (-2.45, 0.00, 1.15), "yaw": 180.0}
    ]
    
    extrinsics_np = np.zeros((B, N, 4, 4), dtype=np.float32)
    for b in range(B):
        for n in range(N):
            config = camera_configs[n]
            x, y, z = config["pos"]
            yaw_deg = config["yaw"]
            
            yaw = np.radians(yaw_deg)
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            
            # Rotation around Z axis (ego to sensor yaw)
            R_yaw = np.array([
                [cos_y, -sin_y, 0],
                [sin_y, cos_y, 0],
                [0, 0, 1]
            ])
            
            # Default sensor rotation mapping Ego to Camera (X_cam=Y_ego, Y_cam=-Z_ego, Z_cam=X_ego)
            R_default = np.array([
                [0, 1, 0],
                [0, 0, -1],
                [1, 0, 0]
            ])
            
            R_ego_to_cam = R_default @ R_yaw.T
            t = np.array([x, y, z])
            
            ext = np.eye(4)
            ext[:3, :3] = R_ego_to_cam
            ext[:3, 3] = -R_ego_to_cam @ t
            extrinsics_np[b, n] = ext
            
    extrinsics = torch.from_numpy(extrinsics_np).to(device)
            
    print(f"Camera views input shape:    {camera_imgs.shape}")
    print(f"LiDAR BEV input shape:        {lidar_bev.shape}")
    print(f"Intrinsics matrices shape:   {intrinsics.shape}")
    print(f"Extrinsics matrices shape:   {extrinsics.shape}")
    
    # 3. Run forward pass
    print("\nRunning forward pass (Late fusion & decoding)...")
    model.eval()
    with torch.no_grad():
        seg_out, det_out = model(camera_imgs, lidar_bev, extrinsics, intrinsics)
        
    print("\nForward pass completed successfully!")
    print("------------------------------------------------------------------")
    print("OUTPUT SHAPES:")
    print(f"1. Semantic Segmentation Map:   {seg_out.shape} -> [B, NumClasses, H_bev, W_bev]")
    print(f"2. Object Detection Heatmaps:  {det_out['heatmap'].shape} -> [B, NumClasses, H_bev, W_bev]")
    print(f"3. Object Center Offsets:      {det_out['offset'].shape} -> [B, 2, H_bev, W_bev]")
    print(f"4. Object 3D Sizes (w,l,h):    {det_out['size'].shape} -> [B, 3, H_bev, W_bev]")
    print(f"5. Object Yaw angle (cos,sin): {det_out['yaw'].shape} -> [B, 2, H_bev, W_bev]")
    print("------------------------------------------------------------------")
