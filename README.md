# Unveiling Digital Mirrors

Web tool and dataset for analyzing body-pose patterns and gendered pose signatures from short videos.

```bash
# from repository root
python serve.py
# open http://localhost:8000
```

What this repo contains

- `frontend/` — UI (HTML, JS, CSS) and `frontend/data/` JSON used by the interface
- `src/` — Python scripts for normalization, clustering, and export
- `data/` — raw and normalized CSVs used for analysis
- `Prototypes/` — prototype images for exported clusters

What the UI offers

- Image mode: single-image pose detection; weighted KNN (K=5) to average nearby cluster gender distributions.
- Video mode: extract frames at 0.5s intervals and classify frames by the closest cluster (majority vote).
- Patterns: browse 150 pose clusters, view prototypes and examples.

Dataset summary

- Total poses: 15,167
- Gender counts: Female 7,957 | Male 6,597 | Non-binary 613
- Keypoints: 15 (OpenPose-style). Raw CSV coordinates are in pixels; normalization applied in `src/normalize.py`.

Key configuration (matches frontend code)

- `MIN_CLUSTER_SIZE = 10` — clusters smaller than this are ignored.
- Gender classification: difference-based threshold = 30% (if male − female ≥ 30% → `masculine`; if female − male ≥ 30% → `feminine`; otherwise `neutral`).
- Neutral window: ±15% (i.e., within 30% total difference).
- Video limits: `MAX_DURATION = 30s`, `FRAME_INTERVAL = 0.5s`.

# TODO: The logic above needs refactoring

Developer pipeline

1. Normalize raw poses:

```bash
python src/normalize.py
```

2. Cluster normalized poses (k=150) and export frontend JSON:

```bash
python src/cluster_poses.py
python src/export_data.py
```

Files to check

- `frontend/data/pose_clusters.json` — exported cluster prototypes and stats
- `frontend/image.js`, `frontend/video.js`, `frontend/patterns.html` — implement thresholds and matching logic

Notes

- Image matching uses inverse-distance weighting to combine the top-K clusters' gender percentages.
- Video uses single-cluster matching per frame and majority voting across frames.

Citation

If you reuse the data or tool, please cite: Unveiling Digital Mirrors (2024).
