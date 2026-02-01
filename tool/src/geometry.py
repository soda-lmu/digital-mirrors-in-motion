"""
Part 2: Feature Engineering - Geometric Analysis

This module translates pose coordinates into meaningful behavioral metrics:
1. Joint angle calculations using Law of Cosines / vector math
2. Center of Gravity (CoG) and stability measurements
3. "Gendered" pose angles identified in research (Head Tilt, Hip Canting, etc.)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional


def calculate_angle_between_points(p1: Tuple[float, float], 
                                    p2: Tuple[float, float], 
                                    p3: Tuple[float, float]) -> float:
    """
    Calculate angle at p2 formed by p1-p2-p3 using vector math.
    
    Args:
        p1: First point (x, y)
        p2: Vertex point (x, y) - the angle is measured here
        p3: Third point (x, y)
    
    Returns:
        Angle in degrees (0-180)
    """
    # Create vectors from vertex to other points
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Calculate magnitudes
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Calculate angle using dot product
    cos_angle = np.dot(v1, v2) / (mag1 * mag2)
    # Clamp to avoid numerical issues with arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def calculate_angle_from_vertical(p1: Tuple[float, float], 
                                   p2: Tuple[float, float]) -> float:
    """
    Calculate the angle a line makes from vertical (useful for tilts).
    Positive = tilted right, Negative = tilted left.
    
    Args:
        p1: Top point (x, y)
        p2: Bottom point (x, y)
    
    Returns:
        Angle in degrees from vertical (-90 to 90)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    if dy == 0:
        return 90.0 if dx > 0 else -90.0
    
    angle_rad = np.arctan(dx / dy)
    return np.degrees(angle_rad)


def calculate_angle_from_horizontal(p1: Tuple[float, float], 
                                     p2: Tuple[float, float]) -> float:
    """
    Calculate the angle a line makes from horizontal.
    
    Args:
        p1: Left point (x, y)
        p2: Right point (x, y)
    
    Returns:
        Angle in degrees from horizontal (-90 to 90)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    if dx == 0:
        return 90.0 if dy > 0 else -90.0
    
    angle_rad = np.arctan(dy / dx)
    return np.degrees(angle_rad)


# =============================================================================
# CRITICAL "GENDERED" POSE ANGLES FROM RESEARCH
# =============================================================================

def calculate_head_tilt(row: pd.Series) -> float:
    """
    Calculate head tilt angle (Nose relative to Neck vertical).
    This is associated with "licensed withdrawal" in Goffman's research.
    
    Returns:
        Angle in degrees (positive = right tilt, negative = left tilt)
    """
    nose = (row['x_nose'], row['y_nose'])
    neck = (row['x_neck'], row['y_neck'])
    
    return calculate_angle_from_vertical(nose, neck)


def calculate_shoulder_tilt(row: pd.Series) -> float:
    """
    Calculate shoulder tilt (asymmetry between left and right shoulder).
    
    Returns:
        Angle in degrees from horizontal
    """
    lshoulder = (row['x_lshoulder'], row['y_lshoulder'])
    rshoulder = (row['x_rshoulder'], row['y_rshoulder'])
    
    return calculate_angle_from_horizontal(lshoulder, rshoulder)


def calculate_hip_canting(row: pd.Series) -> float:
    """
    Calculate hip canting/tilt angle.
    Hip canting is associated with "contrapposto" poses and gender performance.
    
    Returns:
        Angle in degrees from horizontal
    """
    lhip = (row['x_lhip'], row['y_lhip'])
    rhip = (row['x_rhip'], row['y_rhip'])
    
    return calculate_angle_from_horizontal(lhip, rhip)


def calculate_spine_curvature(row: pd.Series) -> float:
    """
    Calculate the curvature of the spine (Neck-MidHip alignment).
    Measures how "S-curved" vs straight the torso is.
    
    Returns:
        Angle deviation from vertical in degrees
    """
    neck = (row['x_neck'], row['y_neck'])
    midhip = (row['x_midhip'], row['y_midhip'])
    
    return calculate_angle_from_vertical(neck, midhip)


def calculate_arm_openness(row: pd.Series) -> Dict[str, float]:
    """
    Calculate how open/closed the arms are (shoulder-elbow-wrist angles).
    More open arms = more "space claiming" posture.
    
    Returns:
        Dict with left and right arm angles
    """
    # Left arm
    lshoulder = (row['x_lshoulder'], row['y_lshoulder'])
    lelbow = (row['x_lelbow'], row['y_lelbow'])
    lwrist = (row['x_lwrist'], row['y_lwrist'])
    
    # Right arm
    rshoulder = (row['x_rshoulder'], row['y_rshoulder'])
    relbow = (row['x_relbow'], row['y_relbow'])
    rwrist = (row['x_rwrist'], row['y_rwrist'])
    
    left_arm_angle = calculate_angle_between_points(lshoulder, lelbow, lwrist)
    right_arm_angle = calculate_angle_between_points(rshoulder, relbow, rwrist)
    
    return {
        'left_arm_angle': left_arm_angle,
        'right_arm_angle': right_arm_angle,
        'avg_arm_angle': (left_arm_angle + right_arm_angle) / 2
    }


def calculate_leg_stance(row: pd.Series) -> Dict[str, float]:
    """
    Calculate leg positioning (hip-knee-ankle angles).
    Bent knees vs straight legs affect perceived assertiveness.
    
    Returns:
        Dict with left and right leg angles
    """
    # Left leg
    lhip = (row['x_lhip'], row['y_lhip'])
    lknee = (row['x_lknee'], row['y_lknee'])
    lankle = (row['x_lankle'], row['y_lankle'])
    
    # Right leg
    rhip = (row['x_rhip'], row['y_rhip'])
    rknee = (row['x_rknee'], row['y_rknee'])
    rankle = (row['x_rankle'], row['y_rankle'])
    
    left_leg_angle = calculate_angle_between_points(lhip, lknee, lankle)
    right_leg_angle = calculate_angle_between_points(rhip, rknee, rankle)
    
    return {
        'left_leg_angle': left_leg_angle,
        'right_leg_angle': right_leg_angle,
        'avg_leg_angle': (left_leg_angle + right_leg_angle) / 2
    }


# =============================================================================
# CENTER OF GRAVITY & STABILITY
# =============================================================================

def calculate_torso_center(row: pd.Series) -> Tuple[float, float]:
    """
    Calculate the center of the torso (midpoint between shoulders and hips).
    
    Returns:
        (x, y) coordinates of torso center
    """
    shoulder_center_x = (row['x_lshoulder'] + row['x_rshoulder']) / 2
    shoulder_center_y = (row['y_lshoulder'] + row['y_rshoulder']) / 2
    
    hip_center_x = (row['x_lhip'] + row['x_rhip']) / 2
    hip_center_y = (row['y_lhip'] + row['y_rhip']) / 2
    
    torso_center_x = (shoulder_center_x + hip_center_x) / 2
    torso_center_y = (shoulder_center_y + hip_center_y) / 2
    
    return (torso_center_x, torso_center_y)


def calculate_support_base(row: pd.Series) -> Tuple[float, float]:
    """
    Calculate the support base (midpoint between feet).
    
    Returns:
        (x, y) coordinates of support base
    """
    # Use ankle positions as foot approximation
    support_x = (row['x_lankle'] + row['x_rankle']) / 2
    support_y = (row['y_lankle'] + row['y_rankle']) / 2
    
    return (support_x, support_y)


def calculate_stability_score(row: pd.Series) -> float:
    """
    Calculate pose stability based on CoG alignment with support base.
    Higher score = more stable/grounded pose.
    
    Returns:
        Stability score (0-100, where 100 is perfectly balanced)
    """
    torso_center = calculate_torso_center(row)
    support_base = calculate_support_base(row)
    
    # Calculate horizontal distance from CoG to support base
    horizontal_offset = abs(torso_center[0] - support_base[0])
    
    # Normalize by body width (distance between hips)
    body_width = abs(row['x_lhip'] - row['x_rhip'])
    if body_width == 0:
        body_width = 0.1  # Avoid division by zero
    
    # Convert to percentage (smaller offset = higher stability)
    normalized_offset = horizontal_offset / body_width
    stability = max(0, 100 - (normalized_offset * 100))
    
    return stability


def calculate_stance_width(row: pd.Series) -> float:
    """
    Calculate how wide the stance is (distance between ankles relative to hips).
    Wider stance = more "power posing".
    
    Returns:
        Stance width ratio (1.0 = ankles same width as hips)
    """
    hip_width = abs(row['x_lhip'] - row['x_rhip'])
    ankle_width = abs(row['x_lankle'] - row['x_rankle'])
    
    if hip_width == 0:
        return 1.0
    
    return ankle_width / hip_width


# =============================================================================
# COMPLETE POSE SIGNATURE
# =============================================================================

def calculate_pose_signature(row: pd.Series) -> Dict[str, float]:
    """
    Generate a complete "Pose Signature" with all geometric features.
    
    Returns:
        Dict containing all pose metrics
    """
    arm_angles = calculate_arm_openness(row)
    leg_angles = calculate_leg_stance(row)
    
    return {
        # Tilt angles
        'head_tilt': calculate_head_tilt(row),
        'shoulder_tilt': calculate_shoulder_tilt(row),
        'hip_canting': calculate_hip_canting(row),
        'spine_curvature': calculate_spine_curvature(row),
        
        # Arm metrics
        'left_arm_angle': arm_angles['left_arm_angle'],
        'right_arm_angle': arm_angles['right_arm_angle'],
        'avg_arm_angle': arm_angles['avg_arm_angle'],
        
        # Leg metrics
        'left_leg_angle': leg_angles['left_leg_angle'],
        'right_leg_angle': leg_angles['right_leg_angle'],
        'avg_leg_angle': leg_angles['avg_leg_angle'],
        
        # Stability metrics
        'stability_score': calculate_stability_score(row),
        'stance_width': calculate_stance_width(row),
    }


def generate_all_pose_signatures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate pose signatures for all rows in a dataframe.
    
    Returns:
        DataFrame with original data plus pose signature columns
    """
    signatures = []
    
    for idx, row in df.iterrows():
        sig = calculate_pose_signature(row)
        sig['gender'] = row['gender']
        signatures.append(sig)
    
    return pd.DataFrame(signatures)


def calculate_brand_averages(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate average pose signatures grouped by gender (or brand).
    This creates the "Brand Signature" mentioned in the project plan.
    
    Returns:
        Dict with gender/brand keys and average metrics
    """
    signature_df = generate_all_pose_signatures(df)
    
    feature_cols = [col for col in signature_df.columns if col != 'gender']
    
    averages = {}
    for gender in signature_df['gender'].unique():
        gender_data = signature_df[signature_df['gender'] == gender]
        averages[gender] = {
            'count': len(gender_data),
            'metrics': {}
        }
        for col in feature_cols:
            averages[gender]['metrics'][col] = {
                'mean': gender_data[col].mean(),
                'std': gender_data[col].std()
            }
    
    return averages


def format_brand_signature(averages: Dict, gender: str) -> str:
    """
    Format a brand signature as human-readable text.
    
    Example output:
    "Zara models average a 12-degree hip tilt and 85% stability"
    """
    if gender not in averages:
        return f"No data for {gender}"
    
    metrics = averages[gender]['metrics']
    count = averages[gender]['count']
    
    hip_tilt = metrics['hip_canting']['mean']
    stability = metrics['stability_score']['mean']
    head_tilt = metrics['head_tilt']['mean']
    stance = metrics['stance_width']['mean']
    
    return (
        f"'{gender.title()}' poses (n={count}):\n"
        f"  • Average hip canting: {hip_tilt:.1f}°\n"
        f"  • Average head tilt: {head_tilt:.1f}°\n"
        f"  • Average stability: {stability:.1f}%\n"
        f"  • Stance width ratio: {stance:.2f}x hip width"
    )


if __name__ == "__main__":
    # Load normalized data from Part 1
    data_path = Path(__file__).parent.parent / "data" / "normalized_poses.csv"
    output_path = Path(__file__).parent.parent / "data" / "pose_signatures.csv"
    
    print("Loading normalized pose data...")
    df = pd.read_csv(data_path, sep=';')
    print(f"Loaded {len(df)} poses")
    
    print("\nGenerating pose signatures...")
    signatures_df = generate_all_pose_signatures(df)
    
    print("\nCalculating brand/gender averages...")
    averages = calculate_brand_averages(df)
    
    print("\n" + "="*60)
    print("POSE SIGNATURES BY GENDER")
    print("="*60)
    
    for gender in averages.keys():
        print(f"\n{format_brand_signature(averages, gender)}")
    
    print("\n" + "="*60)
    
    # Save signatures
    signatures_df.to_csv(output_path, sep=';', index=False)
    print(f"\nPose signatures saved to: {output_path}")
    
    print("\n✓ Part 2 Complete: Feature engineering finished!")
