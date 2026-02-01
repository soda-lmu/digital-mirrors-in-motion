"""
Export pose signature data to JSON for frontend consumption.
This creates the data files that power the interactive tool.
"""

import json
import pandas as pd
from pathlib import Path
from geometry import (
    generate_all_pose_signatures,
    calculate_brand_averages,
    calculate_pose_signature
)


def export_brand_averages_json(averages: dict, output_path: Path):
    """Export brand averages in a format suitable for the frontend."""
    
    frontend_data = {}
    
    for gender, data in averages.items():
        metrics = data['metrics']
        frontend_data[gender.lower().replace('-', '')] = {
            'count': data['count'],
            'headTilt': round(metrics['head_tilt']['mean'], 2),
            'headTiltStd': round(metrics['head_tilt']['std'], 2),
            'hipCanting': round(metrics['hip_canting']['mean'], 2),
            'hipCantingStd': round(metrics['hip_canting']['std'], 2),
            'shoulderTilt': round(metrics['shoulder_tilt']['mean'], 2),
            'shoulderTiltStd': round(metrics['shoulder_tilt']['std'], 2),
            'stability': round(metrics['stability_score']['mean'], 2),
            'stabilityStd': round(metrics['stability_score']['std'], 2),
            'stanceWidth': round(metrics['stance_width']['mean'], 2),
            'stanceWidthStd': round(metrics['stance_width']['std'], 2),
            'avgArmAngle': round(metrics['avg_arm_angle']['mean'], 2),
            'avgArmAngleStd': round(metrics['avg_arm_angle']['std'], 2),
            'avgLegAngle': round(metrics['avg_leg_angle']['mean'], 2),
            'avgLegAngleStd': round(metrics['avg_leg_angle']['std'], 2),
        }
    
    with open(output_path, 'w') as f:
        json.dump(frontend_data, f, indent=2)
    
    print(f"Exported brand averages to: {output_path}")


def export_signature_histograms(signatures_df: pd.DataFrame, output_path: Path):
    """
    Export histogram data for each metric, grouped by gender.
    This allows the frontend to show distribution overlays.
    """
    
    metrics = ['head_tilt', 'hip_canting', 'shoulder_tilt', 
               'stability_score', 'stance_width', 'avg_arm_angle', 'avg_leg_angle']
    
    histogram_data = {}
    
    for metric in metrics:
        histogram_data[metric] = {}
        
        for gender in signatures_df['gender'].unique():
            gender_data = signatures_df[signatures_df['gender'] == gender][metric]
            
            # Create histogram bins
            hist, bin_edges = pd.cut(gender_data, bins=20, retbins=True)
            counts = hist.value_counts().sort_index()
            
            histogram_data[metric][gender.lower().replace('-', '')] = {
                'bins': [round(float(b), 3) for b in bin_edges],
                'counts': [int(c) for c in counts.values]
            }
    
    with open(output_path, 'w') as f:
        json.dump(histogram_data, f, indent=2)
    
    print(f"Exported histogram data to: {output_path}")


def export_average_poses(df: pd.DataFrame, output_path: Path):
    """
    Export average pose coordinates for each gender.
    This can be used to draw "ghost" reference poses.
    """
    
    x_cols = [col for col in df.columns if col.startswith('x_')]
    y_cols = [col for col in df.columns if col.startswith('y_')]
    
    average_poses = {}
    
    for gender in df['gender'].unique():
        gender_data = df[df['gender'] == gender]
        
        pose = {}
        for col in x_cols:
            keypoint = col.replace('x_', '')
            pose[keypoint] = {
                'x': round(gender_data[f'x_{keypoint}'].mean(), 4),
                'y': round(gender_data[f'y_{keypoint}'].mean(), 4)
            }
        
        average_poses[gender.lower().replace('-', '')] = pose
    
    with open(output_path, 'w') as f:
        json.dump(average_poses, f, indent=2)
    
    print(f"Exported average poses to: {output_path}")


if __name__ == "__main__":
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    frontend_dir = Path(__file__).parent.parent / "frontend" / "data"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    
    # Load normalized data
    print("Loading normalized pose data...")
    df = pd.read_csv(data_dir / "normalized_poses.csv", sep=';')
    print(f"Loaded {len(df)} poses")
    
    # Generate signatures
    print("\nGenerating pose signatures...")
    signatures_df = generate_all_pose_signatures(df)
    
    # Calculate averages
    print("Calculating brand averages...")
    averages = calculate_brand_averages(df)
    
    # Export data files
    print("\nExporting data for frontend...")
    
    export_brand_averages_json(
        averages, 
        frontend_dir / "brand_averages.json"
    )
    
    export_signature_histograms(
        signatures_df,
        frontend_dir / "histograms.json"
    )
    
    export_average_poses(
        df,
        frontend_dir / "average_poses.json"
    )
    
    # Also export full signatures CSV
    signatures_df.to_csv(
        data_dir / "pose_signatures.csv",
        sep=';',
        index=False
    )
    print(f"Exported full signatures to: {data_dir / 'pose_signatures.csv'}")
    
    print("\n✓ All data exported successfully!")
