#!/usr/bin/env python3
"""
preprocess_dataset.py
=====================
Parallel image resizing script for CARLA multi-view dataset.
Pre-resizes original camera PNG images to prevent CPU and disk I/O bottlenecks during training.
"""

import os
import argparse
from multiprocessing import Pool, cpu_count
import cv2
from tqdm import tqdm


def resize_single_image(task_args):
    """
    Resizes a single image file and saves it to the destination directory.
    """
    src_path, dest_path, resize_factor = task_args
    if os.path.exists(dest_path):
        return True
    
    try:
        img = cv2.imread(src_path)
        if img is None:
            return False
        
        h, w = img.shape[:2]
        h_new = int(h * resize_factor)
        w_new = int(w * resize_factor)
        
        img_resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        cv2.imwrite(dest_path, img_resized)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def preprocess_dataset(data_dir, resize_factor):
    """
    Collects all camera images across episodes and resizes them in parallel using CPU multiprocessing.
    """
    perception_root = os.path.join(data_dir, "Perception")
    dest_root = os.path.join(data_dir, "Perception_resized")
    
    if not os.path.exists(perception_root):
        raise FileNotFoundError(f"Original perception directory not found: {perception_root}")
        
    print(f"Scanning raw images in: {perception_root}")
    print(f"Preprocessed target directory: {dest_root}")
    
    tasks = []
    episodes = [d for d in os.listdir(perception_root) if d.startswith("episode_")]
    
    for ep in episodes:
        cam_root = os.path.join(perception_root, ep, "cameras")
        if not os.path.exists(cam_root):
            continue
            
        for cam in os.listdir(cam_root):
            if not cam.startswith("cam_"):
                continue
                
            cam_dir = os.path.join(cam_root, cam)
            dest_cam_dir = os.path.join(dest_root, ep, "cameras", cam)
            
            for file in os.listdir(cam_dir):
                if file.endswith(".png"):
                    src_file_path = os.path.join(cam_dir, file)
                    dest_file_path = os.path.join(dest_cam_dir, file)
                    tasks.append((src_file_path, dest_file_path, resize_factor))
                    
    total_images = len(tasks)
    print(f"Found {total_images} images to resize.")
    
    if total_images == 0:
        print("No images found to process.")
        return
        
    num_cores = cpu_count()
    print(f"Starting parallel preprocessing with {num_cores} CPU cores...")
    
    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap_unordered(resize_single_image, tasks), total=total_images, desc="Resizing images"))
        
    successful = sum(1 for r in results if r)
    print(f"Preprocessing completed! {successful}/{total_images} images processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helioskrill dataset parallel image preprocessing script.")
    parser.add_argument("--data_dir", default="./data/", help="Path to root data directory")
    parser.add_argument("--resize_factor", type=float, default=0.5, help="Image scaling factor (0.5 = 400x300)")
    
    args = parser.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    
    preprocess_dataset(args.data_dir, args.resize_factor)
