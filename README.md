# Unveiling Digital Mirrors

Web tool and dataset for analyzing body-pose patterns and gendered pose signatures from short videos.

```bash
# From repository root
python scripts/serve.py
# Open http://localhost:8000 in your browser
```

What this repo contains

- `frontend/` — UI (HTML, JS, CSS) and `frontend/data/pose_clusters.json` used by the interface
- `scripts/` — Python scripts for clustering and pose extraction (paper-aligned normalization in cluster_poses.py)

What the UI offers

- **Image mode**: single-image pose detection; match to closest cluster (Euclidean distance on normalized pose vector).
- **Video mode**: extract poses at 0.5s intervals; gallery shows each frame with skeleton overlay; match each to closest cluster; Export downloads all poses as JSON (time, frameIndex, normalizedKeypoints).
- **Patterns**: browse 150 pose clusters by gender category; click a cluster to see prototype and all member poses.

Dataset summary

- Total poses: 15,167
- Gender counts: Female 7,957 | Male 6,597 | Non-binary 613
- Keypoints: 17 (nose, neck, shoulders, elbows, wrists, midhip, hips, knees, ankles, big toes).

Clustering methodology (Tsolak & Kühne, 2025)
- Normalization: bounding box → bottom-left (0,0) → unit-length vector
- K-means on raw coordinate vectors (Euclidean distance)
- Prototype = arithmetic mean of cluster members
- Gender labels used only for analysis after clustering

Key configuration (matches frontend code)

- All 150 clusters kept; significance via multinomial test (p < 0.001/150, Bonferroni).
- Character: masculine/feminine when significant and over-represented vs null; else neutral.
- Video limits: `MAX_DURATION = 30s`, `FRAME_INTERVAL = 0.5s`.

Developer pipeline

1. Cluster poses (k=150) and export frontend JSON:

```bash
python scripts/cluster_poses.py
```

Key files

- `frontend/data/pose_clusters.json` — cluster prototypes and gender stats (from cluster_poses.py)
- `frontend/image.js`, `frontend/video.js` — pose detection (MediaPipe), normalization, cluster matching
- `frontend/pose-utils.js` — shared normalization, distance, and matching logic
- `frontend/patterns.html` — browse clusters by gender category
- `scripts/cluster_poses.py` — clustering pipeline (paper-aligned normalization)
- `scripts/mediapipe_pipeline.py` — optional Python MediaPipe extraction (browser uses client-side API)

Pose extraction (MediaPipe)

- Image and video analysis use MediaPipe for pose detection in the browser (client-side).
- Landmarks with visibility < 0.5 are filtered out for matching.
- Run `python scripts/serve.py` and open http://localhost:8000.

Notes

- Image/video: match pose to closest cluster by Euclidean distance. k=1 uses cluster character (multinomial test); k>1 blends top k clusters and classifies via deviation from null (paper-aligned).
- Video Export: JSON contains only `time`, `frameIndex`, and `normalizedKeypoints` (paper-normalized coordinates) for each extracted pose.

Citation

If you reuse the data or tool, please cite: Unveiling Digital Mirrors (2024).
