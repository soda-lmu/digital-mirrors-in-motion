# Unveiling Digital Mirrors

Web tool and dataset for analyzing body-pose patterns and gendered pose signatures from short videos.

```bash
# From repository root
python tool/serve.py
# Open http://localhost:8000 in your browser
```

What this repo contains

- `frontend/` — UI (HTML, JS, CSS) and `frontend/data/pose_clusters.json` used by the interface
- `tool/src/` — Python scripts for normalization, clustering, and pose extraction
- `data/` — raw and normalized CSVs (normalized_poses.csv from normalize.py)

What the UI offers

- Image mode: single-image pose detection; match to closest cluster (Euclidean distance on normalized pose vector).
- Video mode: extract frames at 0.5s intervals; match each frame to closest cluster.
- Patterns: browse 150 pose clusters, view prototypes and examples.

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

1. Normalize raw poses (optional; cluster_poses reads raw CSV directly):

```bash
python tool/src/normalize.py
```

2. Cluster poses (k=150) and export frontend JSON:

```bash
python tool/src/cluster_poses.py
```

Key files

- `frontend/data/pose_clusters.json` — cluster prototypes and gender stats (from cluster_poses.py)
- `frontend/image.js`, `frontend/video.js` — pose detection (MediaPipe), normalization, cluster matching
- `frontend/patterns.html` — browse clusters by gender category

Pose extraction (MediaPipe)

- Image and video analysis use MediaPipe for pose detection in the browser (client-side).
- Run `python tool/serve.py` and open http://localhost:8000. No Python dependencies needed for the web app.

Notes

- Image/video: match pose to closest cluster by Euclidean distance. k=1 uses cluster character (multinomial test); k>1 blends top k clusters and classifies via deviation from null (paper-aligned).

Citation

If you reuse the data or tool, please cite: Unveiling Digital Mirrors (2024).
