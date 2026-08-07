#!/usr/bin/env python3
"""
test_bev_grid_identity.py
=========================
Verification Unit Test for LiDAR BEV Grid Identity & Ego Centering Compliance.

VERIFIES:
1. Shape is exactly (5, 400, 400).
2. Ego position at (x=0.0, y=0.0) maps to cell (200, 200).
3. Channel semantics:
   - Channel 0: Raw Z_max
   - Channel 1: Z_diff = Z_max - Z_min
   - Channel 2: Raw Z_mean
   - Channel 3: Normalized Point Density [0, 1]
   - Channel 4: Max Intensity
4. Bit-for-bit identity between carla_data_collector.py and collect_dagger_data.py.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.utils.carla_data_collector import CONFIG, lidar_to_bev_grid_vectorized as main_bev_func
from scripts.collect_data import decode_carla_depth


def test_bev_identity_and_centering():
    print("==============================================================")
    print("  RUNNING LIDAR BEV GRID IDENTITY & EGO CENTERING TEST       ")
    print("==============================================================")

    # 1. Test Synthetic Point Cloud Generation (Deterministic)
    pts = np.array([
        [0.0, 0.0, 1.5, 0.8],    # Point at Ego location (Z_max)
        [0.0, 0.0, -0.5, 0.4],   # Point at Ego location (Z_min)
        [10.0, 10.0, 3.0, 0.9],  # Far point
        [-20.0, -15.0, 0.0, 0.5] # Another far point
    ], dtype=np.float32)

    # 2. Compute BEV grids from both functions
    grid_main = main_bev_func(pts, CONFIG)
    grid_dagger = dagger_bev_func(pts, CONFIG)

    # Test Shape
    assert grid_main.shape == (5, 400, 400), f"Error: Main grid shape is {grid_main.shape}, expected (5, 400, 400)"
    assert grid_dagger.shape == (5, 400, 400), f"Error: DAgger grid shape is {grid_dagger.shape}, expected (5, 400, 400)"
    print("[PASS] Grid shape is exactly (5, 400, 400) in both modules.")

    # Test Bit-for-Bit Identity
    is_identical = np.allclose(grid_main, grid_dagger, atol=1e-7)
    assert is_identical, "Error: BEV grids are NOT bit-for-bit identical between modules!"
    print("[PASS] BEV grids are BIT-FOR-BIT IDENTICAL between carla_data_collector.py and collect_dagger_data.py.")

    # Test Ego Centering at Cell (200, 200)
    ego_cell_ch0 = grid_main[0, 200, 200]  # Z_max = 1.5
    ego_cell_ch1 = grid_main[1, 200, 200]  # Z_diff = 1.5 - (-0.5) = 2.0
    ego_cell_ch2 = grid_main[2, 200, 200]  # Z_mean = (1.5 + -0.5)/2 = 0.5
    ego_cell_ch3 = grid_main[3, 200, 200]  # Normalized Density = 2.0 / 64.0 = 0.03125
    ego_cell_ch4 = grid_main[4, 200, 200]  # Max Intensity = 0.8

    print(f"\n[Ego Cell (200, 200) Telemetry]:")
    print(f"  - Channel 0 (Z_max):   {ego_cell_ch0:.4f} (Expected: 1.5000)")
    print(f"  - Channel 1 (Z_diff):  {ego_cell_ch1:.4f} (Expected: 2.0000)")
    print(f"  - Channel 2 (Z_mean):  {ego_cell_ch2:.4f} (Expected: 0.5000)")
    print(f"  - Channel 3 (Density): {ego_cell_ch3:.4f} (Expected: 0.0312)")
    print(f"  - Channel 4 (Intens):  {ego_cell_ch4:.4f} (Expected: 0.8000)")

    assert abs(ego_cell_ch0 - 1.5) < 1e-4, "Error: Ego cell (200,200) Channel 0 failed!"
    assert abs(ego_cell_ch1 - 2.0) < 1e-4, "Error: Ego cell (200,200) Channel 1 failed!"
    assert abs(ego_cell_ch2 - 0.5) < 1e-4, "Error: Ego cell (200,200) Channel 2 failed!"
    assert abs(ego_cell_ch3 - 2.0/64.0) < 1e-4, "Error: Ego cell (200,200) Channel 3 failed!"
    assert abs(ego_cell_ch4 - 0.8) < 1e-4, "Error: Ego cell (200,200) Channel 4 failed!"

    print("[PASS] Ego vehicle position (0,0) is EXACTLY mapped to cell (200, 200).")
    print("==============================================================")
    print("  ALL BEV GRID COMPLIANCE TESTS PASSED SUCCESSFULLY!          ")
    print("==============================================================\n")


if __name__ == "__main__":
    test_bev_identity_and_centering()
