"""
Advanced pose clustering with proper normalization and hierarchical refinement.
Creates meaningful pose categories with prototype (centroid) and examples.

FEATURES ARE MIRROR-INVARIANT: A pose and its mirror are treated as the same.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path
import json

# Joint names we care about (excluding bigtoe which may not be reliable)
JOINTS = ['nose', 'neck', 'lshoulder', 'rshoulder', 'lelbow', 'relbow', 
          'lwrist', 'rwrist', 'midhip', 'lhip', 'rhip', 'lknee', 'rknee', 
          'lankle', 'rankle']

def normalize_pose(row):
    """
    Normalize a pose to be scale and position invariant.
    Centers on midhip, scales by torso length, keeps Y pointing down.
    Returns dict of joint: {x, y} with normalized coords.
    """
    # Extract coordinates
    coords = {}
    for joint in JOINTS:
        x_col = f'x_{joint}'
        y_col = f'y_{joint}'
        if x_col in row.index and y_col in row.index:
            coords[joint] = np.array([float(row[x_col]), float(row[y_col])])
    
    if 'midhip' not in coords or 'neck' not in coords:
        return None
    
    # Center on midhip
    center = coords['midhip']
    
    # Scale by torso length (neck to midhip)
    torso_vec = coords['neck'] - coords['midhip']
    torso_length = np.linalg.norm(torso_vec)
    if torso_length < 1:
        torso_length = 100  # fallback
    
    # Normalize
    normalized = {}
    for joint, pos in coords.items():
        norm_pos = (pos - center) / torso_length
        normalized[joint] = {'x': float(norm_pos[0]), 'y': float(norm_pos[1])}
    
    return normalized


def canonicalize_pose(norm_pose):
    """
    Make pose mirror-invariant by flipping to a canonical orientation.
    We flip so that the "active" side (more extended arm/leg) is always on the right.
    This ensures mirrored poses cluster together.
    """
    if norm_pose is None:
        return None
    
    # Calculate which side has more "activity" (further from body center)
    left_arm_extent = abs(norm_pose['lwrist']['x']) + abs(norm_pose['lwrist']['y'] - norm_pose['lshoulder']['y'])
    right_arm_extent = abs(norm_pose['rwrist']['x']) + abs(norm_pose['rwrist']['y'] - norm_pose['rshoulder']['y'])
    
    left_leg_extent = abs(norm_pose['lankle']['x']) + abs(norm_pose['lankle']['y'])
    right_leg_extent = abs(norm_pose['rankle']['x']) + abs(norm_pose['rankle']['y'])
    
    left_total = left_arm_extent + left_leg_extent
    right_total = right_arm_extent + right_leg_extent
    
    # If left side is more active, flip the pose horizontally
    if left_total > right_total:
        flipped = {}
        swap_pairs = [
            ('lshoulder', 'rshoulder'), ('lelbow', 'relbow'), ('lwrist', 'rwrist'),
            ('lhip', 'rhip'), ('lknee', 'rknee'), ('lankle', 'rankle')
        ]
        swap_dict = {}
        for l, r in swap_pairs:
            swap_dict[l] = r
            swap_dict[r] = l
        
        for joint, pos in norm_pose.items():
            new_joint = swap_dict.get(joint, joint)
            # Mirror the x-coordinate and swap the joint names
            flipped[new_joint] = {'x': -pos['x'], 'y': pos['y']}
        
        return flipped
    
    return norm_pose.copy()


def get_angle(p1, p2, p3):
    """Angle at p2 formed by p1-p2-p3 in degrees."""
    v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
    v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
    mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if mag1 == 0 or mag2 == 0:
        return 180.0
    cos_angle = np.clip(np.dot(v1, v2) / (mag1 * mag2), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def get_angle_from_vertical(p1, p2):
    """Angle of line p1->p2 from vertical (0=straight up, positive=tilted right)."""
    dx = p2['x'] - p1['x']
    dy = p2['y'] - p1['y']
    if dy == 0:
        return 90.0 if dx > 0 else -90.0
    return np.degrees(np.arctan2(dx, dy))


def extract_pose_features(norm_pose):
    """
    Extract meaningful, mirror-invariant features from a normalized pose.
    
    Features capture recognizable pose characteristics:
    - Head tilt (absolute - direction doesn't matter for clustering)
    - Body lean
    - Arm positions (hands on hips, arms raised, crossed, etc.)
    - Leg positions (crossed, wide stance, bent knees, etc.)
    - Contrapposto (hip/shoulder opposition)
    """
    if norm_pose is None:
        return None
    
    # Canonicalize to handle mirrors
    pose = canonicalize_pose(norm_pose)
    
    features = {}
    
    # === HEAD & BODY TILT ===
    # Head tilt (absolute value - we don't care which direction)
    head_tilt = get_angle_from_vertical(pose['neck'], pose['nose'])
    features['head_tilt_abs'] = abs(head_tilt)
    
    # Shoulder tilt (absolute)
    shoulder_dy = pose['rshoulder']['y'] - pose['lshoulder']['y']
    shoulder_dx = pose['rshoulder']['x'] - pose['lshoulder']['x']
    features['shoulder_tilt_abs'] = abs(np.degrees(np.arctan2(shoulder_dy, shoulder_dx))) if shoulder_dx != 0 else 0
    
    # Hip tilt (absolute)
    hip_dy = pose['rhip']['y'] - pose['lhip']['y']
    hip_dx = pose['rhip']['x'] - pose['lhip']['x']
    features['hip_tilt_abs'] = abs(np.degrees(np.arctan2(hip_dy, hip_dx))) if hip_dx != 0 else 0
    
    # Body lean (torso angle from vertical)
    body_lean = get_angle_from_vertical(pose['midhip'], pose['neck'])
    features['body_lean_abs'] = abs(body_lean)
    
    # Contrapposto (shoulder vs hip tilt opposition)
    features['contrapposto'] = abs(features['shoulder_tilt_abs'] - features['hip_tilt_abs'])
    
    # === ARM POSITIONS ===
    # Arm bend angles (elbow angles)
    left_elbow_angle = get_angle(pose['lshoulder'], pose['lelbow'], pose['lwrist'])
    right_elbow_angle = get_angle(pose['rshoulder'], pose['relbow'], pose['rwrist'])
    features['avg_elbow_angle'] = (left_elbow_angle + right_elbow_angle) / 2
    features['elbow_angle_diff'] = abs(left_elbow_angle - right_elbow_angle)  # Asymmetry
    
    # Arms raised (how high are hands relative to shoulders)
    left_arm_raised = pose['lshoulder']['y'] - pose['lwrist']['y']  # Positive = raised
    right_arm_raised = pose['rshoulder']['y'] - pose['rwrist']['y']
    features['max_arm_raised'] = max(left_arm_raised, right_arm_raised)
    features['min_arm_raised'] = min(left_arm_raised, right_arm_raised)
    
    # Hands on hips detection (wrist near hip)
    left_hand_to_hip = np.sqrt((pose['lwrist']['x'] - pose['lhip']['x'])**2 + 
                                (pose['lwrist']['y'] - pose['lhip']['y'])**2)
    right_hand_to_hip = np.sqrt((pose['rwrist']['x'] - pose['rhip']['x'])**2 + 
                                 (pose['rwrist']['y'] - pose['rhip']['y'])**2)
    features['min_hand_to_hip'] = min(left_hand_to_hip, right_hand_to_hip)
    features['hands_on_hips_score'] = 1.0 / (1.0 + features['min_hand_to_hip'])  # Higher = closer to hips
    
    # Arms crossed detection (wrists near opposite shoulders)
    left_to_right = np.sqrt((pose['lwrist']['x'] - pose['rshoulder']['x'])**2 + 
                            (pose['lwrist']['y'] - pose['rshoulder']['y'])**2)
    right_to_left = np.sqrt((pose['rwrist']['x'] - pose['lshoulder']['x'])**2 + 
                            (pose['rwrist']['y'] - pose['lshoulder']['y'])**2)
    features['arms_crossed_score'] = 1.0 / (1.0 + (left_to_right + right_to_left) / 2)
    
    # Arm spread (how far apart are hands horizontally)
    features['arm_spread'] = abs(pose['lwrist']['x'] - pose['rwrist']['x'])
    
    # === LEG POSITIONS ===
    # Leg bend angles (knee angles)
    left_knee_angle = get_angle(pose['lhip'], pose['lknee'], pose['lankle'])
    right_knee_angle = get_angle(pose['rhip'], pose['rknee'], pose['rankle'])
    features['avg_knee_angle'] = (left_knee_angle + right_knee_angle) / 2
    features['knee_angle_diff'] = abs(left_knee_angle - right_knee_angle)  # Asymmetry
    
    # Stance width (how far apart are feet)
    features['stance_width'] = abs(pose['lankle']['x'] - pose['rankle']['x'])
    
    # Legs crossed detection (ankles/knees crossing over midline)
    left_ankle_x = pose['lankle']['x']
    right_ankle_x = pose['rankle']['x']
    # If left ankle is to the right of right ankle, legs are crossed
    features['legs_crossed'] = 1.0 if left_ankle_x > right_ankle_x else 0.0
    
    # One leg forward (depth difference approximated by y difference of ankles)
    features['leg_forward_diff'] = abs(pose['lankle']['y'] - pose['rankle']['y'])
    
    # Weight shift (is body weight on one leg? - hip center vs ankle center)
    hip_center_x = (pose['lhip']['x'] + pose['rhip']['x']) / 2
    ankle_center_x = (pose['lankle']['x'] + pose['rankle']['x']) / 2
    features['weight_shift'] = abs(hip_center_x - ankle_center_x)
    
    # === MASCULINE-ASSOCIATED FEATURES ===
    
    # Hands in pockets detection (wrists low, near thighs/below hips)
    # Wrist y-position relative to hip (positive = below hip)
    left_wrist_below_hip = pose['lwrist']['y'] - pose['lhip']['y']
    right_wrist_below_hip = pose['rwrist']['y'] - pose['rhip']['y']
    features['hands_low_score'] = max(0, left_wrist_below_hip) + max(0, right_wrist_below_hip)
    
    # Hands near body center (in pockets = close to hips horizontally but low)
    left_hand_near_center = abs(pose['lwrist']['x'] - pose['lhip']['x'])
    right_hand_near_center = abs(pose['rwrist']['x'] - pose['rhip']['x'])
    # Hands in pockets: low AND close to body horizontally
    in_pocket_left = max(0, left_wrist_below_hip) * (1.0 / (1.0 + left_hand_near_center))
    in_pocket_right = max(0, right_wrist_below_hip) * (1.0 / (1.0 + right_hand_near_center))
    features['hands_in_pockets_score'] = in_pocket_left + in_pocket_right
    
    # Hands behind back (wrists behind body midline - x close to 0 or negative when hands behind)
    # Also check wrists are near each other (behind back = wrists close together)
    wrist_dist = np.sqrt((pose['lwrist']['x'] - pose['rwrist']['x'])**2 + 
                         (pose['lwrist']['y'] - pose['rwrist']['y'])**2)
    wrists_close = 1.0 / (1.0 + wrist_dist)
    wrists_center_x = (pose['lwrist']['x'] + pose['rwrist']['x']) / 2
    wrists_behind = abs(wrists_center_x) < 0.3  # Close to body center horizontally
    wrists_mid_height = abs(pose['lwrist']['y'] - pose['midhip']['y']) < 0.5  # Around waist height
    features['hands_behind_back_score'] = wrists_close * (1.0 if wrists_behind and wrists_mid_height else 0.3)
    
    # Arms hanging relaxed (wrists low, arms relatively straight, spread apart)
    arms_straight = features['avg_elbow_angle'] > 150  # Elbow angle > 150 = relatively straight
    arms_down = features['max_arm_raised'] < 0  # Negative = hands below shoulders
    features['arms_relaxed_score'] = (1.0 if arms_straight else 0.5) * (1.0 if arms_down else 0.3)
    
    # Squared/level shoulders (low shoulder tilt)
    features['shoulders_squared'] = 1.0 / (1.0 + features['shoulder_tilt_abs'])
    
    # Power pose score (wide stance + squared shoulders + upright + symmetric)
    wide_stance = 1.0 if features['stance_width'] > 0.5 else features['stance_width'] * 2
    upright = 1.0 / (1.0 + features['body_lean_abs'] + features['head_tilt_abs'])
    symmetric = 1.0 / (1.0 + features['elbow_angle_diff'] + features['knee_angle_diff'])
    features['power_pose_score'] = wide_stance * features['shoulders_squared'] * upright * symmetric
    
    # Overall symmetry (lower = more symmetric, often masculine)
    features['overall_symmetry'] = 1.0 / (1.0 + features['elbow_angle_diff'] + features['knee_angle_diff'] + 
                                           features['shoulder_tilt_abs'] + features['hip_tilt_abs'])
    
    # Straight posture score (low tilts, low lean)
    features['straight_posture'] = 1.0 / (1.0 + features['head_tilt_abs'] + features['body_lean_abs'] + 
                                          features['shoulder_tilt_abs'] + features['hip_tilt_abs'])
    
    # === OVERALL POSE CHARACTERISTICS ===
    # Compactness (how "closed" is the pose - arms and legs close to body)
    features['compactness'] = 1.0 / (1.0 + features['arm_spread'] + features['stance_width'])
    
    # Asymmetry score (how asymmetric is the pose overall)
    features['asymmetry'] = (features['elbow_angle_diff'] + features['knee_angle_diff'] + 
                             features['leg_forward_diff']) / 3
    
    return features

def compute_cluster_prototype(poses):
    """
    Compute the prototype (centroid) pose for a cluster.
    Returns averaged joint positions.
    """
    if len(poses) == 0:
        return None
    
    # Average each joint position
    prototype = {}
    for joint in JOINTS:
        xs = [p[joint]['x'] for p in poses if joint in p]
        ys = [p[joint]['y'] for p in poses if joint in p]
        if xs and ys:
            prototype[joint] = {'x': np.mean(xs), 'y': np.mean(ys)}
    
    return prototype

def load_data(csv_path):
    """Load and preprocess pose data."""
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    print(f"Loaded {len(df)} poses")
    return df

def process_poses(df):
    """Normalize all poses and extract features."""
    normalized_poses = []
    features_list = []
    valid_indices = []
    
    print("Normalizing poses and extracting features...")
    for idx, row in df.iterrows():
        norm = normalize_pose(row)
        if norm is not None:
            feat = extract_pose_features(norm)
            if feat is not None:
                normalized_poses.append(norm)
                features_list.append(feat)
                valid_indices.append(idx)
    
    print(f"Successfully processed {len(normalized_poses)} poses")
    return normalized_poses, features_list, valid_indices

def cluster_poses(features_list, n_clusters=150):
    """
    Two-stage clustering:
    1. K-means for initial grouping
    2. Merge very similar clusters
    """
    # Convert to array
    feature_names = list(features_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in features_list])
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means clustering
    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    labels = kmeans.fit_predict(X_scaled)
    
    # Compute silhouette score for quality assessment
    if len(set(labels)) > 1:
        sil_score = silhouette_score(X_scaled, labels, sample_size=min(5000, len(labels)))
        print(f"Silhouette score: {sil_score:.3f}")
    
    return labels, feature_names

def analyze_clusters(labels, normalized_poses, features_list, valid_indices, df, n_examples=3, min_size=10):
    """Analyze each cluster and create cluster info with prototype and examples."""
    clusters_info = []
    
    unique_labels = sorted(set(labels))
    print(f"Analyzing {len(unique_labels)} clusters...")
    
    for cluster_id in unique_labels:
        # Get indices of poses in this cluster
        mask = labels == cluster_id
        cluster_indices = [i for i, m in enumerate(mask) if m]
        
        if len(cluster_indices) == 0:
            continue
        
        total = len(cluster_indices)
        
        # Filter out clusters that are too small
        if total < min_size:
            print(f"  Skipping cluster #{cluster_id}: size {total} < minimum {min_size}")
            continue
        
        # Get poses and features for this cluster
        cluster_poses = [normalized_poses[i] for i in cluster_indices]
        cluster_features = [features_list[i] for i in cluster_indices]
        cluster_orig_indices = [valid_indices[i] for i in cluster_indices]
        
        # Gender distribution
        genders = [df.iloc[idx]['gender'] for idx in cluster_orig_indices]
        gender_counts = pd.Series(genders).value_counts()
        
        male_pct = gender_counts.get('male', 0) / total * 100
        female_pct = gender_counts.get('female', 0) / total * 100
        nb_pct = gender_counts.get('non-binary', 0) / total * 100
        
        # Compute prototype (centroid pose)
        prototype = compute_cluster_prototype(cluster_poses)
        
        # Average features
        avg_features = {}
        for key in cluster_features[0].keys():
            vals = [f[key] for f in cluster_features]
            avg_features[key] = float(np.mean(vals))
        
        # Get diverse examples (spread across the cluster)
        step = max(1, len(cluster_indices) // n_examples)
        example_indices = cluster_indices[::step][:n_examples]
        examples = []
        for i in example_indices:
            pose = normalized_poses[i].copy()
            pose['gender'] = df.iloc[valid_indices[i]]['gender']
            examples.append(pose)
        
        # Determine character (based on gender distribution)
        if male_pct > 70:
            character = 'masculine'
        elif female_pct > 70:
            character = 'feminine'
        else:
            character = 'neutral'
        
        clusters_info.append({
            'id': int(cluster_id),
            'size': int(total),
            'malePercent': round(male_pct, 1),
            'femalePercent': round(female_pct, 1),
            'nonbinaryPercent': round(nb_pct, 1),
            'character': character,
            'prototype': prototype,
            'examples': examples,
            'metrics': {
                'headTilt': round(avg_features.get('head_tilt_abs', 0), 1),
                'bodyLean': round(avg_features.get('body_lean_abs', 0), 1),
                'contrapposto': round(avg_features.get('contrapposto', 0), 1),
                'stanceWidth': round(avg_features.get('stance_width', 0), 2),
                'legsCrossed': round(avg_features.get('legs_crossed', 0), 2),
                'handsOnHips': round(avg_features.get('hands_on_hips_score', 0), 2),
                'armsCrossed': round(avg_features.get('arms_crossed_score', 0), 2),
                'armsRaised': round(avg_features.get('max_arm_raised', 0), 2),
                'kneeAngle': round(avg_features.get('avg_knee_angle', 180), 1),
                'asymmetry': round(avg_features.get('asymmetry', 0), 2),
                # New masculine-associated metrics
                'handsInPockets': round(avg_features.get('hands_in_pockets_score', 0), 2),
                'handsBehindBack': round(avg_features.get('hands_behind_back_score', 0), 2),
                'armsRelaxed': round(avg_features.get('arms_relaxed_score', 0), 2),
                'powerPose': round(avg_features.get('power_pose_score', 0), 2),
                'straightPosture': round(avg_features.get('straight_posture', 0), 2),
                'symmetry': round(avg_features.get('overall_symmetry', 0), 2),
            }
        })
    
    print(f"Exported {len(clusters_info)} clusters after filtering by minimum size {min_size}")
    return clusters_info

def find_extreme_clusters(clusters_info, n=4):
    """Find most masculine and feminine clusters."""
    # Filter clusters with reasonable size
    sizeable = [c for c in clusters_info if c['size'] >= 5]
    
    sorted_male = sorted(sizeable, key=lambda x: x['malePercent'], reverse=True)
    sorted_female = sorted(sizeable, key=lambda x: x['femalePercent'], reverse=True)
    
    return sorted_male[:n], sorted_female[:n]

def export_data(clusters_info, most_male, most_female, output_path):
    """Export cluster data to JSON."""
    data = {
        'clusters': clusters_info,
        'mostMale': most_male,
        'mostFemale': most_female,
        'totalClusters': len(clusters_info)
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported to {output_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    csv_path = base_dir / "Unveiling_digital_mirrors.csv"
    output_path = base_dir / "frontend" / "data" / "pose_clusters.json"
    
    # Load data
    df = load_data(csv_path)
    
    # Process poses
    normalized_poses, features_list, valid_indices = process_poses(df)
    
    # Cluster
    labels, feature_names = cluster_poses(features_list, n_clusters=150)
    
    # Analyze
    clusters_info = analyze_clusters(labels, normalized_poses, features_list, 
                                      valid_indices, df, n_examples=3, min_size=10)
    
    # Find extremes
    most_male, most_female = find_extreme_clusters(clusters_info)
    
    print(f"\nMost Male Clusters:")
    for c in most_male:
        print(f"  #{c['id']}: {c['malePercent']:.0f}% male ({c['size']} poses)")
    
    print(f"\nMost Female Clusters:")
    for c in most_female:
        print(f"  #{c['id']}: {c['femalePercent']:.0f}% female ({c['size']} poses)")
    
    # Export
    export_data(clusters_info, most_male, most_female, output_path)
    
    print("\nDone!")
