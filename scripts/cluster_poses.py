"""
Pose Clustering Pipeline
================================================
Implements the methodology from "Unveiling digital mirrors: Decoding gendered body
poses in Instagram imagery"

Paper methodology (Section 3: Data and Methods):
  1. The dataset contains only the 17 selected keypoints.
  2. Normalize each pose: bounding-box → align bottom-left to (0,0) → unit-length vector.
  3. K-means clustering (Euclidean distance) on normalized coordinate vectors.
  4. Prototype = arithmetic column mean of observations in each cluster.
  5. Gender labels are used ONLY for post-clustering statistical testing (multinomial test).
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path
import json

# --- Constants from the paper ---

# null-model proportions for the multinomial test (paper section 3.2)
# these reflect the overall gender distribution in the final analysis sample
# of 15,167 images: 52.46% women, 43.50% men, 4.04% non-binary
NULL_FEMALE, NULL_MALE, NULL_NB = 0.5246, 0.4350, 0.0404

# bonferroni-corrected significance threshold (paper section 3.2):
# α = 0.001 divided by k = 150 clusters
P_CRIT = 0.001/150

# The 17 keypoints present in the CSV. The original authors already excluded
# ears, eyes, small toes, and heels before publishing the dataset (Section 3.1).
# Order must match the frontend KEYPOINT_NAMES.
JOINTS = [
    'nose', 'neck', 'lshoulder', 'rshoulder', 'lelbow', 'relbow',
    'lwrist', 'rwrist', 'midhip', 'lhip', 'rhip', 'lknee', 'rknee',
    'lankle', 'rankle', 'bigtoe', 'rbigtoe'
]


def load_data(csv_path):
    """Load the pose CSV (semicolon-separated, comma as decimal mark)."""
    # dataset uses semicolons as separators and commas as decimal marks
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    print(f"Loaded {len(df)} poses")
    return df


def normalize_poses(df):
    """
    Normalize all poses into unit-length coordinate vectors (paper Section 3.1).

    Steps for each pose:
      1. Extract x/y coordinates for all JOINTS.
      2. Compute the bounding box to account for different scales (distance
         from camera, zoom level, image resolution).
      3. Shift so the bottom-left keypoint sits at (0, 0).
         In image coordinates y increases downward, so "bottom" = max(y).
      4. Flatten to a 1-D vector [x0, x1, …, y0, y1, …].
      5. Normalize to unit length (L2 norm = 1).

    The dataset is pre-filtered to full-body single-person images with
    average confidence > 0.75, so we know that every row has valid
    coordinates for all joints.

    Returns:
        np.ndarray of shape (n_poses, 2 * len(JOINTS))
    """
    n_joints = len(JOINTS)

    # Gather all x and y columns into (n_poses, n_joints) arrays.
    # pd.read_csv with decimal=',' already parsed commas into floats.
    x_cols = [f'x_{j}' for j in JOINTS]
    y_cols = [f'y_{j}' for j in JOINTS]
    xs = df[x_cols].values  # shape (n, n_joints)
    ys = df[y_cols].values

    # bounding-box alignment: shift so bottom-left = (0, 0).
    x_min = xs.min(axis=1, keepdims=True)
    y_max = ys.max(axis=1, keepdims=True)  # "bottom" in image coords
    xs = xs - x_min
    ys = ys - y_max

    # flatten each pose to [x0, x1, …, y0, y1, …]
    vecs = np.hstack([xs, ys])  # shape (n, 2 * n_joints)

    # unit-length normalization (L2)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    print(f"Normalized {len(vecs)} poses "
          f"(bounding-box → origin shift → unit length)")
    return vecs


def cluster_poses(vectors, n_clusters=150):
    """
    K-means clustering on normalized pose vectors (paper Section 3.3).

    The paper uses Euclidean distance with k = 150. The original authors
    determined k = 150 via a two-step procedure: hierarchical clustering
    (Ward's method) followed by qualitative examination of the resulting
    dendrogram. We adopt k = 150 directly from their finding rather than
    re-deriving it, since we are using the same dataset.

    random_state is set for reproducibility only, it controls the initial
    centroid positions. With n_init=20 (best of 20 random starts), results
    are robust.
    """
    print(f"Clustering {len(vectors)} poses into {n_clusters} groups…")
    kmeans = KMeans(n_clusters=n_clusters, random_state=34,
                    n_init=20, max_iter=500)
    labels = kmeans.fit_predict(vectors)

    sil = silhouette_score(vectors, labels,
                           sample_size=min(5000, len(labels)))
    print(f"Silhouette score: {sil:.3f}")
    return labels


def reindex_labels_by_female_pct(labels, df):
    """
    Helper function to reassign cluster IDs from 1 to N based on ascending 
    female percentage. This ensures Cluster 1 is the most "masculine" 
    and Cluster 150 is the most "feminine".
    """
    print("Re-indexing clusters based on female percentage...")
    temp = pd.DataFrame({'label': labels, 'is_female': df['gender'] == 'female'})
    pct_series = temp.groupby('label')['is_female'].mean().sort_values()
    mapping = {old_label: new_id for new_id, old_label in enumerate(pct_series.index, start=1)}
    
    return np.array([mapping[l] for l in labels])


def vector_to_pose_dict(vec):
    """Convert a normalized vector back to {joint: {x, y}} for display."""
    n = len(JOINTS)
    return {
        joint: {'x': float(vec[i]), 'y': float(vec[n + i])}
        for i, joint in enumerate(JOINTS)
    }


def analyze_clusters(labels, vectors, df):
    """
    Post-clustering analysis (paper Section 3.2).

    For each cluster:
      - Compute the prototype (arithmetic mean of member vectors).
      - Count the gender distribution.
      - Run a chi-squared goodness-of-fit test (approximation of the exact
        multinomial test) against the null model of equal proportions.
      - Label the cluster as feminine / masculine / neutral based on which
        gender is most over-represented when the test is significant.

    Gender labels are never used during clustering — only here, afterwards.
    """
    clusters_info = []
    unique_labels = sorted(set(labels))
    print(f"Analyzing {len(unique_labels)} clusters …")

    for cluster_id in unique_labels:
        mask = (labels == cluster_id)
        cluster_vecs = vectors[mask]
        total = len(cluster_vecs)

        # --- Prototype: arithmetic column mean (paper Section 3.2) ---
        centroid = cluster_vecs.mean(axis=0)
        prototype = vector_to_pose_dict(centroid)

        # --- Gender distribution ---
        genders = df.loc[mask, 'gender']
        counts = genders.value_counts()
        n_female = counts.get('female', 0)
        n_male   = counts.get('male', 0)
        n_nb     = counts.get('non-binary', 0)
        female_pct = n_female / total * 100
        male_pct   = n_male   / total * 100
        nb_pct     = n_nb     / total * 100

        # --- Statistical test (paper Section 3.2) ---
        # The original paper uses an exact multinomial test (R's "EMT" package).
        # We approximate this with scipy's chi-squared goodness-of-fit test,
        # which converges to the same result for the cluster sizes in this
        # dataset but is not identical for very small clusters.
        observed = np.array([n_female, n_male, n_nb])
        expected = total * np.array([NULL_FEMALE, NULL_MALE, NULL_NB])

        if expected.min() >= 1:
            _, p_value = stats.chisquare(observed, expected)
            significant = bool(p_value < P_CRIT)
        else:
            p_value = float('nan')
            significant = False

        # --- Cluster character ---
        # When the test is significant, label the cluster by whichever
        # gender is most over-represented relative to the null model.
        female_dev = female_pct / 100 - NULL_FEMALE
        male_dev   = male_pct   / 100 - NULL_MALE
        if significant and female_dev > male_dev and female_dev > 0:
            character = 'feminine'
        elif significant and male_dev > female_dev and male_dev > 0:
            character = 'masculine'
        else:
            character = 'neutral'

        # --- Individual poses for drill-down view ---
        all_poses = []
        for i in np.where(mask)[0]:
            pose = vector_to_pose_dict(vectors[i])
            pose['gender'] = df.iloc[i]['gender']
            all_poses.append(pose)

        clusters_info.append({
            'id': int(cluster_id),
            'size': int(total),
            'malePercent': round(male_pct, 1),
            'femalePercent': round(female_pct, 1),
            'nonbinaryPercent': round(nb_pct, 1),
            'significant': significant,
            'pValue': float(p_value) if not np.isnan(p_value) else None,
            'character': character,
            'prototype': prototype,
            'poses': all_poses,
        })

    print(f"Exported {len(clusters_info)} clusters")
    return clusters_info


def export_clusters_csv(clusters_info, output_path):
    """Export cluster data to CSV for the frontend."""
    import csv
    
    headers = [
        'cluster_id', 'character', 'malePercent', 'femalePercent', 
        'nonbinaryPercent', 'significant', 'pValue', 'size'
    ]
    for joint in JOINTS:
        headers.extend([f'x_{joint}', f'y_{joint}'])
        
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for c in clusters_info:
            row = {
                'cluster_id': c['id'],
                'character': c['character'],
                'malePercent': c['malePercent'],
                'femalePercent': c['femalePercent'],
                'nonbinaryPercent': c['nonbinaryPercent'],
                'significant': 1 if c['significant'] else 0,
                'pValue': c['pValue'] if c['pValue'] is not None else '',
                'size': c['size']
            }
            # Add joint coordinates
            for joint in JOINTS:
                if joint in c['prototype']:
                    row[f'x_{joint}'] = "{:.6f}".format(c['prototype'][joint]['x'])
                    row[f'y_{joint}'] = "{:.6f}".format(c['prototype'][joint]['y'])
                else:
                    row[f'x_{joint}'] = ''
                    row[f'y_{joint}'] = ''
            writer.writerow(row)
    print(f"Exported clusters to {output_path}")

def export_normalized_poses_csv(vectors, labels, df, output_path):
    """Export all normalized poses and their cluster assignments."""
    import csv
    
    headers = ['pose_id', 'gender', 'cluster_id']
    for joint in JOINTS:
        headers.extend([f'x_{joint}', f'y_{joint}'])
        
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        n_joints = len(JOINTS)
        for i, (vec, label) in enumerate(zip(vectors, labels)):
            row = {
                'pose_id': i,
                'gender': df.iloc[i]['gender'],
                'cluster_id': label
            }
            for j, joint in enumerate(JOINTS):
                row[f'x_{joint}'] = "{:.6f}".format(vec[j])
                row[f'y_{joint}'] = "{:.6f}".format(vec[n_joints + j])
            writer.writerow(row)
    print(f"Exported {len(vectors)} normalized poses to {output_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "data" / "Unveiling_digital_mirrors.csv"
    
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    clusters_csv_path = data_dir / "pose_clusters.csv"
    poses_csv_path = data_dir / "normalized_poses.csv"

    df = load_data(csv_path)
    vectors = normalize_poses(df)
    
    # 1. Cluster poses (returns arbitrary IDs from K-Means)
    labels = cluster_poses(vectors, n_clusters=150)
    
    # 2. Re-index labels (1 -> 150) based on ascending female percentage
    labels = reindex_labels_by_female_pct(labels, df)
    
    # 3. Analyze the new sorted labels (will automatically process 1 to 150 in order)
    clusters_info = analyze_clusters(labels, vectors, df)
    
    # 4. Export using the newly aligned data
    export_clusters_csv(clusters_info, clusters_csv_path)
    export_normalized_poses_csv(vectors, labels, df, poses_csv_path)

    # 5. Print statistical info about the produced dataset
    print("\n" + "="*48)
    print("DATASET STATISTICAL SUMMARY")
    print("="*48)
    
    sig_clusters = [c for c in clusters_info if c['significant']]
    fem_clusters = [c for c in clusters_info if c['character'] == 'feminine']
    masc_clusters = [c for c in clusters_info if c['character'] == 'masculine']
    neut_clusters = [c for c in clusters_info if c['character'] == 'neutral']
    sizes = [c['size'] for c in clusters_info]
    
    print(f"Total Normalized Poses: {len(vectors)}")
    print(f"Total Clusters:         {len(clusters_info)}")
    print(f"Cluster Size Range:     Min = {min(sizes)}, Max = {max(sizes)}, Avg = {sum(sizes)/len(sizes):.1f}, Med = {sorted(sizes)[len(sizes)//2]}\n")
    
    print(f"Significant Clusters:   {len(sig_clusters)} / {len(clusters_info)}")
    print(f"  ↳ Feminine Character:  {len(fem_clusters)}")
    print(f"  ↳ Masculine Character: {len(masc_clusters)}")
    print(f"  ↳ Neutral Character:   {len(neut_clusters)}")
    print("="*48)

    print("\nDone!")
