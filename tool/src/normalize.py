"""
Part 1: Data Normalization & Skeletonization

This module converts pose coordinates into a standardized, scale-invariant format:
1. Scales coordinates to [0,1] range
2. Translates coordinates so Mid-Hip is at origin (0,0)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def parse_coordinate(value):
    """Parse coordinate values that use comma as decimal separator."""
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load raw CSV data with semicolon separator."""
    df = pd.read_csv(csv_path, sep=';')
    
    # Convert all coordinate columns (except 'gender') to float
    coord_columns = [col for col in df.columns if col != 'gender']
    for col in coord_columns:
        df[col] = df[col].apply(parse_coordinate)
    
    return df


def get_bounding_box(df: pd.DataFrame) -> dict:
    """Calculate bounding box for all x and y coordinates."""
    x_cols = [col for col in df.columns if col.startswith('x_')]
    y_cols = [col for col in df.columns if col.startswith('y_')]
    
    x_min = df[x_cols].min().min()
    x_max = df[x_cols].max().max()
    y_min = df[y_cols].min().min()
    y_max = df[y_cols].max().max()
    
    return {
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'width': x_max - x_min,
        'height': y_max - y_min
    }


def normalize_to_unit_scale(df: pd.DataFrame, image_width: float = None, image_height: float = None) -> pd.DataFrame:
    """
    Normalize coordinates to [0,1] range.
    If image dimensions not provided, uses data bounding box.
    """
    df_normalized = df.copy()
    
    x_cols = [col for col in df.columns if col.startswith('x_')]
    y_cols = [col for col in df.columns if col.startswith('y_')]
    
    if image_width is None or image_height is None:
        # Use per-row normalization based on bounding box
        for idx in df_normalized.index:
            row_x_vals = df_normalized.loc[idx, x_cols].values
            row_y_vals = df_normalized.loc[idx, y_cols].values
            
            x_min, x_max = row_x_vals.min(), row_x_vals.max()
            y_min, y_max = row_y_vals.min(), row_y_vals.max()
            
            width = x_max - x_min if x_max != x_min else 1
            height = y_max - y_min if y_max != y_min else 1
            
            # Normalize to [0, 1]
            df_normalized.loc[idx, x_cols] = (row_x_vals - x_min) / width
            df_normalized.loc[idx, y_cols] = (row_y_vals - y_min) / height
    else:
        # Use provided image dimensions
        df_normalized[x_cols] = df[x_cols] / image_width
        df_normalized[y_cols] = df[y_cols] / image_height
    
    return df_normalized


def translate_to_midhip_origin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Translate all coordinates so that Mid-Hip is at (0, 0).
    This makes poses independent of position in frame.
    """
    df_translated = df.copy()
    
    x_cols = [col for col in df.columns if col.startswith('x_')]
    y_cols = [col for col in df.columns if col.startswith('y_')]
    
    # Translate each row so midhip is at origin
    for idx in df_translated.index:
        midhip_x = df_translated.loc[idx, 'x_midhip']
        midhip_y = df_translated.loc[idx, 'y_midhip']
        
        df_translated.loc[idx, x_cols] = df_translated.loc[idx, x_cols] - midhip_x
        df_translated.loc[idx, y_cols] = df_translated.loc[idx, y_cols] - midhip_y
    
    return df_translated


def full_normalization_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete normalization pipeline:
    1. Normalize to [0,1] scale
    2. Translate to Mid-Hip origin
    """
    df_normalized = normalize_to_unit_scale(df)
    df_final = translate_to_midhip_origin(df_normalized)
    return df_final


def compute_pose_statistics(df: pd.DataFrame) -> dict:
    """Compute summary statistics for pose data by gender."""
    stats = {}
    
    x_cols = [col for col in df.columns if col.startswith('x_')]
    y_cols = [col for col in df.columns if col.startswith('y_')]
    coord_cols = x_cols + y_cols
    
    for gender in df['gender'].unique():
        gender_data = df[df['gender'] == gender]
        stats[gender] = {
            'count': len(gender_data),
            'mean': gender_data[coord_cols].mean().to_dict(),
            'std': gender_data[coord_cols].std().to_dict()
        }
    
    return stats


def export_normalized_data(df: pd.DataFrame, output_path: str):
    """Export normalized data to CSV."""
    df.to_csv(output_path, sep=';', index=False)
    print(f"Normalized data saved to: {output_path}")


# Keypoint mapping for OpenPose to MediaPipe conversion
OPENPOSE_KEYPOINTS = [
    'nose', 'neck', 'rshoulder', 'relbow', 'rwrist',
    'lshoulder', 'lelbow', 'lwrist', 'midhip',
    'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle',
    'bigtoe', 'rbigtoe'  # Additional points
]


def create_skeleton_connections():
    """Define skeleton bone connections for visualization."""
    return [
        ('nose', 'neck'),
        ('neck', 'rshoulder'), ('rshoulder', 'relbow'), ('relbow', 'rwrist'),
        ('neck', 'lshoulder'), ('lshoulder', 'lelbow'), ('lelbow', 'lwrist'),
        ('neck', 'midhip'),
        ('midhip', 'rhip'), ('rhip', 'rknee'), ('rknee', 'rankle'), ('rankle', 'rbigtoe'),
        ('midhip', 'lhip'), ('lhip', 'lknee'), ('lknee', 'lankle'), ('lankle', 'bigtoe'),
    ]


if __name__ == "__main__":
    # Example usage
    data_path = Path(__file__).parent.parent / "Unveiling_digital_mirrors.csv"
    output_path = Path(__file__).parent.parent / "data" / "normalized_poses.csv"
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading raw data...")
    df = load_raw_data(data_path)
    print(f"Loaded {len(df)} poses")
    
    print("\nApplying normalization pipeline...")
    df_normalized = full_normalization_pipeline(df)
    
    print("\nComputing statistics by gender...")
    stats = compute_pose_statistics(df_normalized)
    for gender, data in stats.items():
        print(f"  {gender}: {data['count']} poses")
    
    print("\nExporting normalized data...")
    export_normalized_data(df_normalized, output_path)
    
    print("\n✓ Part 1 Complete: Data normalization finished!")
