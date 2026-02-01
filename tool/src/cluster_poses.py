"""
Advanced pose clustering with proper normalization and hierarchical refinement.
Creates meaningful pose categories with prototype (centroid) and examples.
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

def extract_pose_features(norm_pose):
    """
    Extract meaningful features from a normalized pose.
    Focus on angles and relative positions that define pose character.
    """
    if norm_pose is None:
        return None
    
    def get_angle(p1, p2, p3):
        """Angle at p2 formed by p1-p2-p3."""
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if mag1 == 0 or mag2 == 0:
            return 180.0
        cos_angle = np.clip(np.dot(v1, v2) / (mag1 * mag2), -1, 1)
        return np.degrees(np.arccos(cos_angle))
    
    def get_vector_angle(p1, p2):
        """Angle of vector from p1 to p2 relative to vertical."""
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        return np.degrees(np.arctan2(dx, -dy))  # -dy because Y increases downward
    
    features = {}
    
    # 1. Head tilt (nose relative to neck)
    features['head_tilt'] = get_vector_angle(norm_pose['neck'], norm_pose['nose'])
    
    # 2. Shoulder line angle
    features['shoulder_angle'] = get_vector_angle(norm_pose['lshoulder'], norm_pose['rshoulder'])
    
    # 3. Hip line angle
    features['hip_angle'] = get_vector_angle(norm_pose['lhip'], norm_pose['rhip'])
    
    # 4. Torso twist (difference between shoulder and hip angles)
    features['torso_twist'] = features['shoulder_angle'] - features['hip_angle']
    
    # 5. Left arm bend
    features['left_elbow_angle'] = get_angle(norm_pose['lshoulder'], norm_pose['lelbow'], norm_pose['lwrist'])
    
    # 6. Right arm bend
    features['right_elbow_angle'] = get_angle(norm_pose['rshoulder'], norm_pose['relbow'], norm_pose['rwrist'])
    
    # 7. Left arm raise (shoulder-elbow angle from vertical)
    features['left_arm_raise'] = get_vector_angle(norm_pose['lshoulder'], norm_pose['lelbow'])
    
    # 8. Right arm raise
    features['right_arm_raise'] = get_vector_angle(norm_pose['rshoulder'], norm_pose['relbow'])
    
    # 9. Left leg bend
    features['left_knee_angle'] = get_angle(norm_pose['lhip'], norm_pose['lknee'], norm_pose['lankle'])
    
    # 10. Right leg bend
    features['right_knee_angle'] = get_angle(norm_pose['rhip'], norm_pose['rknee'], norm_pose['rankle'])
    
    # 11. Left leg spread (hip-knee angle from vertical)
    features['left_leg_spread'] = get_vector_angle(norm_pose['lhip'], norm_pose['lknee'])
    
    # 12. Right leg spread
    features['right_leg_spread'] = get_vector_angle(norm_pose['rhip'], norm_pose['rknee'])
    
    # 13. Stance width (normalized)
    features['stance_width'] = abs(norm_pose['lankle']['x'] - norm_pose['rankle']['x'])
    
    # 14. Weight shift (center of feet relative to hips)
    ankle_center_x = (norm_pose['lankle']['x'] + norm_pose['rankle']['x']) / 2
    features['weight_shift'] = ankle_center_x
    
    # 15. Contrapposto score (hip vs shoulder angle difference)
    features['contrapposto'] = abs(features['shoulder_angle'] - features['hip_angle'])
    
    # 16. Arm symmetry (difference in arm angles)
    features['arm_asymmetry'] = abs(features['left_arm_raise'] - features['right_arm_raise'])
    
    # 17. Leg symmetry
    features['leg_asymmetry'] = abs(features['left_leg_spread'] - features['right_leg_spread'])
    
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
        if male_pct > 60:
            character = 'masculine'
        elif female_pct > 60:
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
                'headTilt': round(avg_features.get('head_tilt', 0), 1),
                'shoulderAngle': round(avg_features.get('shoulder_angle', 0), 1),
                'hipAngle': round(avg_features.get('hip_angle', 0), 1),
                'contrapposto': round(avg_features.get('contrapposto', 0), 1),
                'stanceWidth': round(avg_features.get('stance_width', 0), 2),
                'leftArmRaise': round(avg_features.get('left_arm_raise', 0), 1),
                'rightArmRaise': round(avg_features.get('right_arm_raise', 0), 1),
                'leftKneeAngle': round(avg_features.get('left_knee_angle', 180), 1),
                'rightKneeAngle': round(avg_features.get('right_knee_angle', 180), 1),
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
    base_dir = Path(__file__).parent.parent
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
