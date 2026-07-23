import http.server
import socketserver
import json
import urllib.parse
import os
import webbrowser
import sys
from PIL import Image
import numpy as np
import io
import torch
import torchvision.transforms as T

# Port to serve the dashboard
PORT = 8501

# Lyft Udacity Challenge Colormap mapping class_id -> (R, G, B)
COLORMAP = {
    0: (31, 41, 55),       # None / Unlabeled - Dark gray
    1: (70, 130, 180),     # Building - Steel Blue
    2: (188, 143, 143),    # Fence - Rosy Brown
    3: (112, 128, 144),    # Other - Slate Gray
    4: (239, 68, 68),      # Pedestrian - Bright Red
    5: (245, 158, 11),     # Pole - Amber / Golden
    6: (254, 240, 138),    # Road Line - Soft Yellow
    7: (99, 102, 241),     # Road - Indigo Blue
    8: (236, 72, 153),     # Sidewalk - Pink
    9: (16, 185, 129),     # Vegetation - Emerald Green
    10: (59, 130, 246),    # Car / Vehicle - Dodge Blue
    11: (120, 53, 4),      # Wall - Brown
    12: (14, 165, 233)     # Traffic Sign - Sky Blue
}

# Lyft Udacity Challenge Class names
CLASS_NAMES = {
    0: "None / Unlabeled",
    1: "Building",
    2: "Fence",
    3: "Other",
    4: "Pedestrian",
    5: "Pole",
    6: "Road Line",
    7: "Road",
    8: "Sidewalk",
    9: "Vegetation",
    10: "Car / Vehicle",
    11: "Wall",
    12: "Traffic Sign"
}

# Search for all image-mask pairs in the data directory
def find_image_pairs(data_dir):
    pairs = []
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory '{data_dir}' does not exist.")
        return pairs
        
    for root, dirs, files in os.walk(data_dir):
        if "CameraRGB" in dirs:
            rgb_dir = os.path.join(root, "CameraRGB")
            seg_dir = os.path.join(root, "CameraSeg")
            if os.path.exists(seg_dir):
                for f in os.listdir(rgb_dir):
                    if f.lower().endswith(".png"):
                        rgb_path = os.path.join(rgb_dir, f)
                        seg_path = os.path.join(seg_dir, f)
                        if os.path.exists(seg_path):
                            pairs.append({
                                "name": f,
                                "rgb_path": os.path.abspath(rgb_path),
                                "seg_path": os.path.abspath(seg_path),
                                "dataset": os.path.basename(os.path.dirname(root))
                            })
    # Sort logically
    pairs = sorted(pairs, key=lambda x: (x["dataset"], x["name"]))
    return pairs

# Create a colorized segmentation mask from the raw classes in the red channel
def colorize_mask(mask_path):
    mask = Image.open(mask_path)
    mask_np = np.array(mask)
    # Class index is in the Red channel
    class_ids = mask_np[:, :, 0]
    
    h, w = class_ids.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in COLORMAP.items():
        color_mask[class_ids == class_id] = color
        
    return Image.fromarray(color_mask)

# Scanned image pairs list
IMAGES_LIST = find_image_pairs("./src/data")

# Embedded HTML, CSS, JS Dashboard
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Helioskrill Data Inspector</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.7);
            --bg-sidebar: rgba(15, 23, 42, 0.9);
            --accent: #8b5cf6;
            --accent-glow: rgba(139, 92, 246, 0.4);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border: rgba(255, 255, 255, 0.08);
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
            scrollbar-width: thin;
            scrollbar-color: rgba(255, 255, 255, 0.1) transparent;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            overflow: hidden;
            height: 100vh;
            display: flex;
        }

        /* Sidebar Styling */
        .sidebar {
            width: 320px;
            background: var(--bg-sidebar);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            height: 100%;
            z-index: 10;
            backdrop-filter: blur(20px);
        }

        .sidebar-header {
            padding: 24px;
            border-bottom: 1px solid var(--border);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }

        .brand-logo {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #8b5cf6, #ec4899);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 18px;
            box-shadow: 0 0 15px var(--accent-glow);
        }

        .brand-name {
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(to right, #ffffff, #cbd5e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .dataset-summary {
            font-size: 13px;
            color: var(--text-muted);
            background: rgba(255, 255, 255, 0.05);
            padding: 6px 12px;
            border-radius: 20px;
            display: inline-block;
            margin-top: 8px;
            border: 1px solid var(--border);
        }

        .filter-section {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .search-box {
            position: relative;
        }

        .search-box input {
            width: 100%;
            padding: 10px 16px 10px 36px;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: #fff;
            font-size: 14px;
            outline: none;
            transition: var(--transition);
        }

        .search-box input:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 2px var(--accent-glow);
        }

        .search-icon {
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            fill: var(--text-muted);
            width: 16px;
            height: 16px;
        }

        .select-dataset {
            width: 100%;
            padding: 10px;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: #fff;
            outline: none;
            font-size: 14px;
            cursor: pointer;
        }

        .image-list-container {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
        }

        .image-item {
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: var(--transition);
            border: 1px solid transparent;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .image-item:hover {
            background: rgba(255, 255, 255, 0.03);
            border-color: var(--border);
        }

        .image-item.active {
            background: rgba(139, 92, 246, 0.15);
            border-color: var(--accent);
        }

        .image-item-info {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .image-item-name {
            font-weight: 500;
            font-size: 14px;
            color: var(--text-main);
        }

        .image-item-meta {
            font-size: 11px;
            color: var(--text-muted);
        }

        .badge {
            background: rgba(255, 255, 255, 0.08);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Main Workspace Styling */
        .workspace {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100%;
            overflow-y: auto;
            background: radial-gradient(circle at top right, rgba(139, 92, 246, 0.05), transparent 60%);
        }

        .workspace-header {
            padding: 24px 32px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .current-image-info h2 {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .current-image-info p {
            color: var(--text-muted);
            font-size: 14px;
        }

        .controls {
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text-main);
            padding: 10px 18px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            font-size: 14px;
            transition: var(--transition);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-1px);
        }

        .btn:active {
            transform: translateY(0);
        }

        .btn-primary {
            background: var(--accent);
            border-color: transparent;
            box-shadow: 0 4px 15px var(--accent-glow);
        }

        .btn-primary:hover {
            background: #7c3aed;
            box-shadow: 0 4px 20px rgba(139, 92, 246, 0.6);
        }

        /* Visualizer Layout */
        .visualizer-container {
            padding: 32px;
            display: flex;
            flex-direction: column;
            gap: 32px;
            max-width: 1600px;
            width: 100%;
            margin: 0 auto;
        }

        .view-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
        }

        .view-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }

        .view-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .view-card-title {
            font-weight: 600;
            font-size: 15px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .image-wrapper {
            position: relative;
            width: 100%;
            aspect-ratio: 4/3;
            background: #090d16;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .image-wrapper img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: opacity 0.15s ease-in-out;
        }

        .overlay-mask {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            mix-blend-mode: normal;
        }

        .slider-container {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 4px 8px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
        }

        .slider-container span {
            font-size: 12px;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            min-width: 40px;
            text-align: right;
        }

        .opacity-slider {
            flex: 1;
            height: 6px;
            -webkit-appearance: none;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            outline: none;
            cursor: pointer;
        }

        .opacity-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: var(--accent);
            box-shadow: 0 0 10px var(--accent);
            transition: var(--transition);
        }

        .opacity-slider::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }

        /* Metadata & Analysis layout */
        .analysis-grid {
            display: grid;
            grid-template-columns: 2fr 3fr;
            gap: 24px;
        }

        .analysis-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(10px);
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .analysis-card-title {
            font-size: 18px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
        }

        .tensor-info-table {
            width: 100%;
            border-collapse: collapse;
        }

        .tensor-info-table th {
            text-align: left;
            padding: 8px 12px;
            color: var(--text-muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid var(--border);
        }

        .tensor-info-table td {
            padding: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            font-size: 14px;
        }

        .tensor-info-table tr:last-child td {
            border-bottom: none;
        }

        .tensor-tag {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(139, 92, 246, 0.1);
            color: #a78bfa;
            border: 1px solid rgba(139, 92, 246, 0.2);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 13px;
        }

        /* Class Breakdown Layout */
        .class-legend {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            overflow-y: auto;
            max-height: 240px;
            padding-right: 8px;
        }

        .class-legend-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 14px;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
        }

        .class-color-info {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .class-color-box {
            width: 14px;
            height: 14px;
            border-radius: 4px;
            box-shadow: inset 0 0 4px rgba(0, 0, 0, 0.2);
        }

        .class-name {
            font-size: 13px;
            font-weight: 500;
        }

        .class-stats {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 2px;
        }

        .class-percent {
            font-size: 13px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }

        .class-pixels {
            font-size: 10px;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Scrollbars */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 80%;
            color: var(--text-muted);
            gap: 16px;
        }
    </style>
</head>
<body>
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-header">
            <div class="brand">
                <div class="brand-logo">H</div>
                <div class="brand-name">Helioskrill</div>
            </div>
            <div class="dataset-summary" id="dataset-summary">Loading dataset...</div>
        </div>

        <div class="filter-section">
            <div class="search-box">
                <svg class="search-icon" viewBox="0 0 24 24"><path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>
                <input type="text" id="search-input" placeholder="Search images..." oninput="filterImages()">
            </div>
            <select class="select-dataset" id="dataset-select" onchange="filterImages()">
                <option value="all">All Sub-datasets</option>
            </select>
        </div>

        <div class="image-list-container" id="image-list">
            <!-- Populated via Javascript -->
        </div>
    </div>

    <!-- Main Workspace -->
    <div class="workspace" id="workspace">
        <div class="workspace-header">
            <div class="current-image-info">
                <h2 id="current-title">Select an image</h2>
                <p id="current-meta">-</p>
            </div>
            <div class="controls">
                <button class="btn" onclick="navigateImage(-1)">
                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M15.41 16.59L10.83 12l4.58-4.59L14 6l-6 6 6 6 1.41-1.41z"/></svg>
                    Previous
                </button>
                <button class="btn" onclick="navigateImage(1)">
                    Next
                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6-1.41-1.41z"/></svg>
                </button>
                <button class="btn btn-primary" onclick="loadRandomImage()">
                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M10.59 9.17L5.41 4 4 5.41l5.17 5.17 1.42-1.41zM14.5 4l2.04 2.04L4 18.59 5.41 20 17.96 7.46 20 9.5V4h-5.5zm.38 10.17l-1.41 1.41 3.17 3.17L14.5 20H20v-5.5l-2.04 2.04-3.08-3.07z"/></svg>
                    Random Image
                </button>
            </div>
        </div>

        <div class="visualizer-container" id="visualizer-content" style="display: none;">
            <!-- Side-by-side Images -->
            <div class="view-grid">
                <!-- Original RGB -->
                <div class="view-card">
                    <div class="view-card-header">
                        <span class="view-card-title">Original RGB</span>
                        <span class="badge">RGB (3 channels)</span>
                    </div>
                    <div class="image-wrapper">
                        <img id="img-rgb" src="" alt="RGB image">
                    </div>
                </div>

                <!-- Overlay -->
                <div class="view-card">
                    <div class="view-card-header">
                        <span class="view-card-title">Interactive Overlay</span>
                        <div class="slider-container">
                            <input type="range" id="opacity-slider" class="opacity-slider" min="0" max="100" value="45" oninput="updateOpacity(this.value)">
                            <span id="opacity-value">45%</span>
                        </div>
                    </div>
                    <div class="image-wrapper">
                        <img id="img-overlay-bg" src="" alt="Overlay Background">
                        <img id="img-overlay-fg" class="overlay-mask" src="" alt="Overlay Mask">
                    </div>
                </div>

                <!-- Segmentation Mask -->
                <div class="view-card">
                    <div class="view-card-header">
                        <span class="view-card-title">Segmentation Mask</span>
                        <span class="badge">Red Channel (Class IDs)</span>
                    </div>
                    <div class="image-wrapper">
                        <img id="img-mask" src="" alt="Segmentation mask">
                    </div>
                </div>
            </div>

            <!-- Analysis Row -->
            <div class="analysis-grid">
                <!-- PyTorch Tensor Details -->
                <div class="analysis-card">
                    <div class="analysis-card-title">
                        <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24" style="color: var(--accent);"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-2 10h-4v4h-2v-4H7v-2h4V7h2v4h4v2z"/></svg>
                        PyTorch Tensor Representation
                    </div>
                    <table class="tensor-info-table">
                        <thead>
                            <tr>
                                <th>Variable / Tensor</th>
                                <th>Shape</th>
                                <th>DType</th>
                                <th>Val Range</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Image Tensor</strong><br><small style="color: var(--text-muted);">Standard inputs</small></td>
                                <td><span class="tensor-tag" id="tensor-img-shape">[3, 600, 800]</span></td>
                                <td><span class="tensor-tag" id="tensor-img-dtype">torch.float32</span></td>
                                <td><span class="tensor-tag" id="tensor-img-range">0.0 - 1.0</span></td>
                            </tr>
                            <tr>
                                <td><strong>Target Mask</strong><br><small style="color: var(--text-muted);">CrossEntropy target</small></td>
                                <td><span class="tensor-tag" id="tensor-mask-shape">[600, 800]</span></td>
                                <td><span class="tensor-tag" id="tensor-mask-dtype">torch.int64</span></td>
                                <td><span class="tensor-tag" id="tensor-mask-range">0 - 12</span></td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <!-- Class Breakdown / Legend -->
                <div class="analysis-card">
                    <div class="analysis-card-title">
                        <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24" style="color: #ec4899;"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H7c0-2.76 2.24-5 5-5s5 2.24 5 5c0 1.04-.42 1.99-1.07 2.75z"/></svg>
                        Classes Present in Image
                    </div>
                    <div class="class-legend" id="classes-legend">
                        <!-- Populated via Javascript -->
                    </div>
                </div>
            </div>
        </div>

        <div class="empty-state" id="empty-state">
            <svg width="64" height="64" fill="none" stroke="currentColor" stroke-width="1" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375 0 11-.75 0 .375 0 01.75 0z"/></svg>
            <p>Select an image from the sidebar to inspect its segmentation mask and tensor details.</p>
        </div>
    </div>

    <script>
        let allImages = [];
        let filteredImages = [];
        let currentIndex = -1;

        // Fetch list of images on page load
        window.addEventListener('DOMContentLoaded', async () => {
            try {
                const res = await fetch('/api/list');
                allImages = await res.json();
                
                // Populate dataset selector and image list
                populateSelectors();
                filterImages();
                
                document.getElementById('dataset-summary').innerText = `${allImages.length} Image pairs discovered`;
            } catch (err) {
                console.error("Failed to load image list:", err);
                document.getElementById('dataset-summary').innerText = "Failed to load dataset.";
            }
        });

        function populateSelectors() {
            const datasets = new Set(allImages.map(img => img.dataset));
            const selectEl = document.getElementById('dataset-select');
            
            datasets.forEach(ds => {
                const opt = document.createElement('option');
                opt.value = ds;
                opt.innerText = ds;
                selectEl.appendChild(opt);
            });
        }

        function filterImages() {
            const searchVal = document.getElementById('search-input').value.toLowerCase();
            const datasetVal = document.getElementById('dataset-select').value;
            
            filteredImages = allImages.filter(img => {
                const matchesSearch = img.name.toLowerCase().includes(searchVal);
                const matchesDataset = (datasetVal === 'all') || (img.dataset === datasetVal);
                return matchesSearch && matchesDataset;
            });
            
            renderImageList();
            
            // Auto select first image if nothing active
            if (filteredImages.length > 0) {
                if (currentIndex === -1 || !filteredImages.find(img => img.id === currentIndex)) {
                    selectImage(filteredImages[0].id);
                }
            } else {
                showEmptyState();
            }
        }

        function renderImageList() {
            const listContainer = document.getElementById('image-list');
            listContainer.innerHTML = '';
            
            filteredImages.forEach(img => {
                const item = document.createElement('div');
                item.className = `image-item ${img.id === currentIndex ? 'active' : ''}`;
                item.onclick = () => selectImage(img.id);
                
                item.innerHTML = `
                    <div class="image-item-info">
                        <span class="image-item-name">${img.name}</span>
                        <span class="image-item-meta">${img.dataset}</span>
                    </div>
                    <span class="badge">#${img.id}</span>
                `;
                listContainer.appendChild(item);
            });
        }

        async function selectImage(id) {
            currentIndex = id;
            
            // Mark active item in list
            document.querySelectorAll('.image-item').forEach(item => {
                item.classList.remove('active');
            });
            
            // Fetch images and info
            const imgData = allImages.find(img => img.id === id);
            if (!imgData) return;

            // Highlight in list
            renderImageList();
            
            // Update Headers
            document.getElementById('current-title').innerText = imgData.name;
            document.getElementById('current-meta').innerText = `Dataset: ${imgData.dataset} | RGB: ${imgData.rgb_path} | Seg: ${imgData.seg_path}`;

            // Show visualizer, hide empty state
            document.getElementById('visualizer-content').style.display = 'block';
            document.getElementById('empty-state').style.display = 'none';

            // Set images src
            document.getElementById('img-rgb').src = `/api/image?id=${id}`;
            document.getElementById('img-mask').src = `/api/mask?id=${id}`;
            
            // Set overlay images
            document.getElementById('img-overlay-bg').src = `/api/image?id=${id}`;
            document.getElementById('img-overlay-fg').src = `/api/mask?id=${id}`;

            // Fetch info
            try {
                const res = await fetch(`/api/info?id=${id}`);
                const info = await res.json();
                renderInfo(info);
            } catch (err) {
                console.error("Failed to load tensor details:", err);
            }
        }

        function renderInfo(info) {
            // Render PyTorch details
            document.getElementById('tensor-img-shape').innerText = JSON.stringify(info.image.shape);
            document.getElementById('tensor-img-dtype').innerText = info.image.dtype;
            document.getElementById('tensor-img-range').innerText = `${info.image.min.toFixed(2)} - ${info.image.max.toFixed(2)}`;

            document.getElementById('tensor-mask-shape').innerText = JSON.stringify(info.mask.shape);
            document.getElementById('tensor-mask-dtype').innerText = info.mask.dtype;
            document.getElementById('tensor-mask-range').innerText = `${info.mask.min} - ${info.mask.max}`;

            // Render Class legend breakdown
            const legendEl = document.getElementById('classes-legend');
            legendEl.innerHTML = '';
            
            info.classes.forEach(c => {
                const item = document.createElement('div');
                item.className = 'class-legend-item';
                item.innerHTML = `
                    <div class="class-color-info">
                        <div class="class-color-box" style="background-color: ${c.color}"></div>
                        <span class="class-name">${c.name}</span>
                    </div>
                    <div class="class-stats">
                        <span class="class-percent">${c.percentage}%</span>
                        <span class="class-pixels">${c.pixels.toLocaleString()} px</span>
                    </div>
                `;
                legendEl.appendChild(item);
            });
        }

        function showEmptyState() {
            document.getElementById('visualizer-content').style.display = 'none';
            document.getElementById('empty-state').style.display = 'flex';
            document.getElementById('current-title').innerText = 'Select an image';
            document.getElementById('current-meta').innerText = '-';
        }

        function navigateImage(direction) {
            if (filteredImages.length === 0) return;
            let listIdx = filteredImages.findIndex(img => img.id === currentIndex);
            if (listIdx === -1) listIdx = 0;
            
            listIdx = (listIdx + direction + filteredImages.length) % filteredImages.length;
            selectImage(filteredImages[listIdx].id);
        }

        function loadRandomImage() {
            if (filteredImages.length === 0) return;
            const randIdx = Math.floor(Math.random() * filteredImages.length);
            selectImage(filteredImages[randIdx].id);
        }

        function updateOpacity(val) {
            document.getElementById('img-overlay-fg').style.opacity = val / 100;
            document.getElementById('opacity-value').innerText = `${val}%`;
        }
    </script>
</body>
</html>
"""

# Implement Custom Request Handler for Serving API and Web Elements
class DatasetInspectorHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence default terminal logs to keep output clean, unless debug is needed
        pass

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed_url.query)
        path = parsed_url.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))
            return

        elif path == "/api/list":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            out = []
            for i, p in enumerate(IMAGES_LIST):
                out.append({
                    "id": i,
                    "name": p["name"],
                    "dataset": p["dataset"],
                    "rgb_path": p["rgb_path"],
                    "seg_path": p["seg_path"]
                })
            self.wfile.write(json.dumps(out).encode("utf-8"))
            return

        elif path == "/api/image":
            img_id = int(query.get("id", [0])[0])
            if img_id < 0 or img_id >= len(IMAGES_LIST):
                self.send_error(404, "Image ID out of bounds")
                return
                
            img_path = IMAGES_LIST[img_id]["rgb_path"]
            if not os.path.exists(img_path):
                self.send_error(404, "Image file not found")
                return

            self.send_response(200)
            self.send_header("Content-type", "image/jpeg")
            self.end_headers()
            
            # Open, convert to JPEG for lightweight serving
            img = Image.open(img_path)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            self.wfile.write(buffer.getvalue())
            return

        elif path == "/api/mask":
            img_id = int(query.get("id", [0])[0])
            if img_id < 0 or img_id >= len(IMAGES_LIST):
                self.send_error(404, "Image ID out of bounds")
                return

            seg_path = IMAGES_LIST[img_id]["seg_path"]
            if not os.path.exists(seg_path):
                self.send_error(404, "Segmentation mask file not found")
                return

            self.send_response(200)
            self.send_header("Content-type", "image/png")
            self.end_headers()

            # Generate and send colorized mask
            color_mask = colorize_mask(seg_path)
            buffer = io.BytesIO()
            color_mask.save(buffer, format="PNG")
            self.wfile.write(buffer.getvalue())
            return

        elif path == "/api/info":
            img_id = int(query.get("id", [0])[0])
            if img_id < 0 or img_id >= len(IMAGES_LIST):
                self.send_error(404, "Image ID out of bounds")
                return

            img_path = IMAGES_LIST[img_id]["rgb_path"]
            seg_path = IMAGES_LIST[img_id]["seg_path"]

            try:
                img = Image.open(img_path)
                mask = Image.open(seg_path)

                # PyTorch logic to show user exact tensor information
                img_tensor = T.functional.to_tensor(img) # FloatTensor, shape [C, H, W], range [0.0, 1.0]
                
                # Mask tensor conversion matching typical cross-entropy shape [H, W] with class indices
                mask_np = np.array(mask)
                class_mask_np = mask_np[:, :, 0] # Class index is in the R channel
                mask_tensor = torch.from_numpy(class_mask_np).long() # LongTensor, shape [H, W], values [0, 12]

                # Extract PyTorch properties
                img_shape = list(img_tensor.shape)
                img_dtype = str(img_tensor.dtype)
                img_min = float(img_tensor.min())
                img_max = float(img_tensor.max())

                mask_shape = list(mask_tensor.shape)
                mask_dtype = str(mask_tensor.dtype)
                mask_min = int(mask_tensor.min())
                mask_max = int(mask_tensor.max())

                # Calculate class distribution in mask using PyTorch
                unique_classes, counts = torch.unique(mask_tensor, return_counts=True)
                total_pixels = mask_tensor.numel()

                class_distribution = []
                for c, cnt in zip(unique_classes.tolist(), counts.tolist()):
                    class_name = CLASS_NAMES.get(c, f"Unknown Class (ID {c})")
                    percentage = (cnt / total_pixels) * 100
                    color = COLORMAP.get(c, (128, 128, 128))
                    hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                    
                    class_distribution.append({
                        "id": c,
                        "name": class_name,
                        "pixels": cnt,
                        "percentage": round(percentage, 2),
                        "color": hex_color
                    })
                
                # Sort by highest coverage first
                class_distribution = sorted(class_distribution, key=lambda x: x["percentage"], reverse=True)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                
                info = {
                    "image": {
                        "shape": img_shape,
                        "dtype": img_dtype,
                        "min": img_min,
                        "max": img_max
                    },
                    "mask": {
                        "shape": mask_shape,
                        "dtype": mask_dtype,
                        "min": mask_min,
                        "max": mask_max
                    },
                    "classes": class_distribution
                }
                self.wfile.write(json.dumps(info).encode("utf-8"))
                
            except Exception as e:
                self.send_error(500, f"Error processing PyTorch analysis: {str(e)}")
            return

        else:
            self.send_error(404, "Endpoint not found")
            return

def start_server():
    if len(IMAGES_LIST) == 0:
        print("Error: No image-mask pairs found in './src/data/'. Please verify data paths.")
        sys.exit(1)
        
    handler = DatasetInspectorHandler
    # Allow address reuse to prevent 'Address already in use' errors during rapid restarts
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        url = f"http://localhost:{PORT}"
        print("==========================================================================")
        print("          Helioskrill Dataset Inspector & PyTorch Analyzer Launched       ")
        print("==========================================================================")
        print(f" -> Found {len(IMAGES_LIST)} image-segmentation mask pairs in ./src/data")
        print(f" -> Serving dashboard web GUI locally on: {url}")
        print(" -> Press CTRL+C in this terminal window to stop the server at any time.")
        print("==========================================================================")
        
        # Open in browser automatically
        webbrowser.open(url)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server. Goodbye!")

if __name__ == "__main__":
    start_server()
