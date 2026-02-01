"""
Part 1.2: MediaPipe Brand Pipeline

This script processes images from clothing company datasets to extract
pose landmarks and create a "Corporate Master CSV" for comparison.

Usage:
    python mediapipe_pipeline.py --input_dir /path/to/brand/images --output brand_poses.csv
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Optional, Dict, List
import json


# MediaPipe Pose setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# MediaPipe landmark indices to our naming convention
LANDMARK_MAPPING = {
    0: 'nose',
    11: 'lshoulder',
    12: 'rshoulder',
    13: 'lelbow',
    14: 'relbow',
    15: 'lwrist',
    16: 'rwrist',
    23: 'lhip',
    24: 'rhip',
    25: 'lknee',
    26: 'rknee',
    27: 'lankle',
    28: 'rankle',
    31: 'bigtoe',  # Left foot index
    32: 'rbigtoe'  # Right foot index
}


def extract_pose_from_image(image_path: str, pose_model) -> Optional[Dict]:
    """
    Extract pose landmarks from a single image using MediaPipe.
    
    Args:
        image_path: Path to image file
        pose_model: MediaPipe Pose instance
    
    Returns:
        Dict with x_* and y_* coordinates, or None if no pose detected
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  Warning: Could not read image: {image_path}")
        return None
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Process image
    results = pose_model.process(image_rgb)
    
    if not results.pose_landmarks:
        return None
    
    # Extract landmarks
    landmarks = results.pose_landmarks.landmark
    pose_data = {'source_file': str(image_path)}
    
    # Map MediaPipe landmarks to our format
    for mp_idx, our_name in LANDMARK_MAPPING.items():
        if mp_idx < len(landmarks):
            lm = landmarks[mp_idx]
            pose_data[f'x_{our_name}'] = lm.x * w  # Denormalize to pixel coords
            pose_data[f'y_{our_name}'] = lm.y * h
    
    # Compute neck (midpoint of shoulders)
    if 'x_lshoulder' in pose_data and 'x_rshoulder' in pose_data:
        pose_data['x_neck'] = (pose_data['x_lshoulder'] + pose_data['x_rshoulder']) / 2
        pose_data['y_neck'] = (pose_data['y_lshoulder'] + pose_data['y_rshoulder']) / 2
    
    # Compute midhip
    if 'x_lhip' in pose_data and 'x_rhip' in pose_data:
        pose_data['x_midhip'] = (pose_data['x_lhip'] + pose_data['x_rhip']) / 2
        pose_data['y_midhip'] = (pose_data['y_lhip'] + pose_data['y_rhip']) / 2
    
    return pose_data


def process_image_directory(input_dir: Path, brand_name: str = "unknown", 
                           gender: str = "unknown") -> List[Dict]:
    """
    Process all images in a directory and extract poses.
    
    Args:
        input_dir: Directory containing images
        brand_name: Name of the brand/source
        gender: Gender label for the images
    
    Returns:
        List of pose dictionaries
    """
    poses = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"\nProcessing {len(image_files)} images from {input_dir}...")
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:
        
        for i, image_path in enumerate(image_files):
            pose_data = extract_pose_from_image(image_path, pose)
            
            if pose_data:
                pose_data['brand'] = brand_name
                pose_data['gender'] = gender
                poses.append(pose_data)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(image_files)} images...")
    
    print(f"  Successfully extracted {len(poses)} poses")
    return poses


def normalize_poses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize poses to [0,1] scale and translate to midhip origin.
    Matches the normalization from normalize.py
    """
    df_normalized = df.copy()
    
    x_cols = [col for col in df.columns if col.startswith('x_')]
    y_cols = [col for col in df.columns if col.startswith('y_')]
    
    # Per-row normalization
    for idx in df_normalized.index:
        # Get current row values
        row_x = df_normalized.loc[idx, x_cols].values.astype(float)
        row_y = df_normalized.loc[idx, y_cols].values.astype(float)
        
        # Normalize to [0,1]
        x_min, x_max = row_x.min(), row_x.max()
        y_min, y_max = row_y.min(), row_y.max()
        
        width = x_max - x_min if x_max != x_min else 1
        height = y_max - y_min if y_max != y_min else 1
        
        row_x_norm = (row_x - x_min) / width
        row_y_norm = (row_y - y_min) / height
        
        # Translate to midhip origin
        midhip_x = row_x_norm[x_cols.index('x_midhip')]
        midhip_y = row_y_norm[y_cols.index('y_midhip')]
        
        df_normalized.loc[idx, x_cols] = row_x_norm - midhip_x
        df_normalized.loc[idx, y_cols] = row_y_norm - midhip_y
    
    return df_normalized


def main():
    parser = argparse.ArgumentParser(
        description='Extract poses from brand images using MediaPipe'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--output', type=str, default='brand_poses.csv',
                       help='Output CSV file path')
    parser.add_argument('--brand', type=str, default='unknown',
                       help='Brand name for labeling')
    parser.add_argument('--gender', type=str, default='unknown',
                       help='Gender label (male/female/non-binary)')
    parser.add_argument('--normalize', action='store_true',
                       help='Apply normalization to poses')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        return
    
    # Process images
    poses = process_image_directory(input_path, args.brand, args.gender)
    
    if not poses:
        print("No poses were extracted!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(poses)
    
    # Optionally normalize
    if args.normalize:
        print("\nApplying normalization...")
        df = normalize_poses(df)
    
    # Save to CSV
    output_path = Path(args.output)
    df.to_csv(output_path, sep=';', index=False)
    print(f"\nSaved {len(df)} poses to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Total images processed: {len(poses)}")
    print(f"Brand: {args.brand}")
    print(f"Gender: {args.gender}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
