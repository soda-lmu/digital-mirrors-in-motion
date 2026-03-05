"""
Pose Clustering Pipeline (Tsolak & Kühne, 2025)
================================================
Implements the methodology from "Unveiling digital mirrors: Decoding gendered body
poses in Instagram imagery" (Computers in Human Behavior 163, 2025).

Paper methodology:
1. Normalize each pose: bounding box → align bottom-left to (0,0) → unit-length vector
2. K-means clustering on normalized coordinate vectors (Euclidean distance)
3. Prototype = arithmetic mean of observations in each cluster
4. Gender labels used only for statistical analysis after clustering (not during)

Keypoints: nose, neck, shoulders, elbows, wrists, midhip, hips, knees, ankles, big toes
(Paper excludes ears, eyes, small toes, heels; uses nose + big toes + ankles.)
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import json

# Null model from dataset (paper Section 3.2) — matches frontend pose-utils.js
NULL_FEMALE, NULL_MALE, NULL_NB = 0.5246, 0.4350, 0.0404
P_CRIT = 0.001 / 150  # Bonferroni: α=0.001, k=150 clusters

# Joint names (order must match frontend KEYPOINT_NAMES)
JOINTS = [
    'nose', 'neck', 'lshoulder', 'rshoulder', 'lelbow', 'relbow',
    'lwrist', 'rwrist', 'midhip', 'lhip', 'rhip', 'lknee', 'rknee',
    'lankle', 'rankle', 'bigtoe', 'rbigtoe'
]


def pose_row_to_vector(row):
    """
    Convert a pose row to normalized vector per Tsolak & Kühne (2025).
    
    Steps (from paper Section 3.1):
    1. Extract boundary (bounding box) for each image
    2. Align so bottom-left keypoint = (0,0) — in image coords, bottom-left = (x_min, y_max)
    3. Normalize vector to unit length (L2 norm = 1)
    
    Returns: 1D numpy array of length 2*len(JOINTS), or None if invalid.
    """
    coords = []
    for joint in JOINTS:
        x_col, y_col = f'x_{joint}', f'y_{joint}'
        if x_col not in row.index or y_col not in row.index:
            return None
        try:
            x = float(str(row[x_col]).replace(',', '.'))
            y = float(str(row[y_col]).replace(',', '.'))
        except (ValueError, TypeError):
            return None
        coords.append([x, y])
    
    arr = np.array(coords)
    x_vals = arr[:, 0]
    y_vals = arr[:, 1]
    
    # Bounding box; in image coords y increases downward, so bottom = max(y)
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    
    # Align bottom-left to (0,0): subtract (x_min, y_max)
    x_shifted = x_vals - x_min
    y_shifted = y_vals - y_max  # y_max is "bottom"
    
    # Flatten to vector [x1,y1,x2,y2,...]
    vec = np.concatenate([x_shifted, y_shifted])
    
    # Unit length
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return None
    vec = vec / norm
    
    return vec


def vector_to_pose_dict(vec):
    """Convert normalized vector back to pose dict {joint: {x, y}} for display."""
    n = len(JOINTS)
    x_part = vec[:n]
    y_part = vec[n:]
    pose = {}
    for i, joint in enumerate(JOINTS):
        pose[joint] = {'x': float(x_part[i]), 'y': float(y_part[i])}
    return pose


def compute_cluster_prototype(vectors):
    """
    Compute prototype as arithmetic mean of vectors (paper: "average across features").
    Returns pose dict for display (mean coords, not re-normalized to unit length).
    """
    if len(vectors) == 0:
        return None
    centroid = np.mean(vectors, axis=0)
    return vector_to_pose_dict(centroid)


def load_data(csv_path):
    """Load raw pose CSV (semicolon-separated, comma decimal)."""
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    print(f"Loaded {len(df)} poses")
    return df


def process_poses(df):
    """
    Normalize poses per paper: bounding box → bottom-left origin → unit length.
    Returns (vectors, valid_indices) for clustering. No feature extraction.
    """
    vectors = []
    valid_indices = []
    
    print("Normalizing poses (paper method: bounding box, bottom-left origin, unit length)...")
    for idx, row in df.iterrows():
        vec = pose_row_to_vector(row)
        if vec is not None:
            vectors.append(vec)
            valid_indices.append(idx)
    
    print(f"Successfully processed {len(vectors)} poses")
    return np.array(vectors), valid_indices


def cluster_poses(vectors, n_clusters=150):
    """
    K-means clustering on normalized pose vectors (paper Section 3.3).
    Euclidean distance, no standardization (vectors already unit length).
    """
    # Handle NaN/Inf
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Clustering into {n_clusters} groups (Euclidean distance)...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    labels = kmeans.fit_predict(vectors)
    
    if len(set(labels)) > 1:
        sil_score = silhouette_score(vectors, labels, sample_size=min(5000, len(labels)))
        print(f"Silhouette score: {sil_score:.3f}")
    
    return labels


def analyze_clusters(labels, vectors, valid_indices, df):
    """
    Analyze each cluster: prototype, gender distribution, examples.
    Paper keeps all clusters (no minimum size filter); significance via multinomial test.
    """
    clusters_info = []
    
    unique_labels = sorted(set(labels))
    print(f"Analyzing {len(unique_labels)} clusters...")
    
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_indices = [i for i, m in enumerate(mask) if m]
        
        if len(cluster_indices) == 0:
            continue
        
        total = len(cluster_indices)
        cluster_vectors = [vectors[i] for i in cluster_indices]
        cluster_orig_indices = [valid_indices[i] for i in cluster_indices]
        
        # Gender distribution (labels used only here, not during clustering)
        genders = [df.iloc[idx]['gender'] for idx in cluster_orig_indices]
        gender_counts = pd.Series(genders).value_counts()
        
        male_pct = gender_counts.get('male', 0) / total * 100
        female_pct = gender_counts.get('female', 0) / total * 100
        nb_pct = gender_counts.get('non-binary', 0) / total * 100
        
        # Prototype = arithmetic mean (paper)
        prototype = compute_cluster_prototype(cluster_vectors)
        
        # All poses in cluster (for drill-down view)
        all_poses = []
        for i in cluster_indices:
            pose = vector_to_pose_dict(vectors[i]).copy()
            pose['gender'] = df.iloc[valid_indices[i]]['gender']
            all_poses.append(pose)
        
        # Multinomial test (paper Section 3.2): chi-squared goodness-of-fit vs null
        observed = np.array([
            gender_counts.get('female', 0),
            gender_counts.get('male', 0),
            gender_counts.get('non-binary', 0)
        ])
        expected = total * np.array([NULL_FEMALE, NULL_MALE, NULL_NB])
        if expected.min() >= 1:  # chi-squared needs expected >= 1
            _, p_value = stats.chisquare(observed, expected)
            significant = bool(p_value < P_CRIT)
        else:
            p_value = float('nan')
            significant = False

        # Character: when significant, which gender is most over-represented vs null
        male_dev = male_pct / 100 - NULL_MALE
        female_dev = female_pct / 100 - NULL_FEMALE
        if significant and male_dev > female_dev and male_dev > 0:
            character = 'masculine'
        elif significant and female_dev > male_dev and female_dev > 0:
            character = 'feminine'
        else:
            character = 'neutral'

        clusters_info.append({
            'id': int(cluster_id),
            'size': int(total),
            'malePercent': round(male_pct, 1),
            'femalePercent': round(female_pct, 1),
            'nonbinaryPercent': round(nb_pct, 1),
            'significant': significant,
            'pValue': float(p_value) if (expected.min() >= 1 and not np.isnan(p_value)) else None,
            'character': character,
            'prototype': prototype,
            'poses': all_poses,
        })
    
    print(f"Exported {len(clusters_info)} clusters (paper: all clusters, no size filter)")
    return clusters_info


def export_data(clusters_info, output_path):
    """Export cluster data to JSON for the frontend."""
    data = {
        'clusters': clusters_info,
        'totalClusters': len(clusters_info)
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Exported to {output_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "Unveiling_digital_mirrors.csv"
    output_path = base_dir / "frontend" / "data" / "pose_clusters.json"
    
    df = load_data(csv_path)
    vectors, valid_indices = process_poses(df)
    
    labels = cluster_poses(vectors, n_clusters=150)
    
    clusters_info = analyze_clusters(labels, vectors, valid_indices, df)
    export_data(clusters_info, output_path)
    
    print("\nDone!")
