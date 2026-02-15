"""
Data Normalization Pipeline (Optional)
======================================
Preprocesses raw pose CSV: scale to [0,1] per bounding box, center on midhip.
Output: normalized_poses.csv. Note: cluster_poses.py uses its own paper-aligned
normalization (bounding box → bottom-left origin → unit length), so this script
is optional for the main pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def parse_coordinate(value):
    """Parse coordinate values that use comma as decimal separator (European format)."""
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return float(value)


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load raw CSV with semicolon separator and parse coordinate columns to float."""
    df = pd.read_csv(csv_path, sep=';')
    coord_columns = [col for col in df.columns if col != 'gender']
    for col in coord_columns:
        df[col] = df[col].apply(parse_coordinate)
    return df


def normalize_to_unit_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each pose's coordinates to [0,1] using its own bounding box.
    Handles poses of different sizes in the same dataset.
    """
    df_normalized = df.copy()
    x_cols = [col for col in df.columns if col.startswith('x_')]
    y_cols = [col for col in df.columns if col.startswith('y_')]

    for idx in df_normalized.index:
        row_x_vals = df_normalized.loc[idx, x_cols].values
        row_y_vals = df_normalized.loc[idx, y_cols].values
        x_min, x_max = row_x_vals.min(), row_x_vals.max()
        y_min, y_max = row_y_vals.min(), row_y_vals.max()
        width = x_max - x_min if x_max != x_min else 1
        height = y_max - y_min if y_max != y_min else 1
        df_normalized.loc[idx, x_cols] = (row_x_vals - x_min) / width
        df_normalized.loc[idx, y_cols] = (row_y_vals - y_min) / height
    return df_normalized


def translate_to_midhip_origin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Translate each pose so Mid-Hip is at (0, 0).
    Makes poses comparable regardless of where the person stood in the frame.
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
    """Run full pipeline: scale to [0,1] then center on midhip."""
    df_normalized = normalize_to_unit_scale(df)
    df_final = translate_to_midhip_origin(df_normalized)
    return df_final


def compute_pose_statistics(df: pd.DataFrame) -> dict:
    """Compute count, mean, std per gender for logging."""
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
    """Export normalized data to CSV (semicolon-separated)."""
    df.to_csv(output_path, sep=';', index=False)
    print(f"Normalized data saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    data_path = Path(__file__).parent.parent.parent / "Unveiling_digital_mirrors.csv"
    output_path = Path(__file__).parent.parent.parent / "data" / "normalized_poses.csv"
    
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
