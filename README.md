# Digital Mirrors in Motion

Web tool and dataset for analyzing body-pose patterns and gendered pose signatures from short videos.

## Video Explaining the Tool and Setup

Watch the full setup and a demonstration of the interface, including image and video analysis:

*(Click the image above to watch the video on YouTube)*

[![Digital Mirrors in Motion Demo](https://img.youtube.com/vi/sDeKJPJI14Y/0.jpg)](https://youtu.be/sDeKJPJI14Y)


## Quick Start

This project uses `uv` to ensure consistent execution.

1. **Setup**: 
   - Windows: Double-click `setup.bat`
   - Linux/Mac: Run `bash setup.sh`
2. **Start Server**: The setup scripts will start the server automatically. If you want to start it manually:
   ```bash
   uv run python scripts/server.py
   ```
3. **Open Browser**: Go to [http://localhost:8000/frontend/index.html](http://localhost:8000/frontend/index.html)

What this repo contains

- `data/` — Dataset CSVs, images, and videos used by the application
- `frontend/` — UI (HTML, JS, CSS)
- `scripts/` — Python scripts for clustering and pose extraction

What the UI offers

- **Image mode**: single-image pose detection; match to closest individual dataset pose using Euclidean distance, then categorize by its parent cluster.
- **Video mode**: extract poses at 0.5s intervals; gallery shows each frame with skeleton overlay; match each to closest individual dataset pose; timeline chart shows temporal dynamics; export per-frame data as CSV.
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

Key configuration

- All 150 clusters kept; significance via multinomial test (p < 0.001/150, Bonferroni).
- Character: masculine/feminine when significant and over-represented vs null; else neutral.
- Video limits: `MAX_DURATION = 30s`, `FRAME_INTERVAL = 0.5s`.


Key files

- `data/pose_clusters.csv` — cluster prototypes and gender stats (from cluster_poses.py)
- `frontend/image.js`, `frontend/video.js` — pose detection (MediaPipe), normalization, cluster matching
- `frontend/pose-utils.js` — shared normalization, distance, and matching logic
- `frontend/patterns.html` — browse clusters by gender category
- `scripts/server.py` — Local HTTP server to host the tool
- `scripts/cluster_poses.py` — Clustering pipeline (paper-aligned normalization)

Pose extraction (MediaPipe)

- Image and video analysis use MediaPipe for pose detection in the browser (client-side).
- Landmarks with visibility < 0.5 are filtered out for matching.
- Run `uv run python scripts/server.py` and open [http://localhost:8000/frontend/index.html](http://localhost:8000/frontend/index.html).

Notes

- Image/video: match pose to closest individual dataset pose by Euclidean distance. Take the parent cluster of the closest pose.
- Video Export: CSV contains `frame`, `time`, `cluster_id`, `character`, `male_pct`, `female_pct`, `distance`.
- Timeline chart: shows masculine/feminine/neutral classification per frame over time.

## AI Acknowledgement

Artificial intelligence tools were used for code implementation, debugging, and for grammar and style suggestions during drafting the associated report. All substantive content, analysis, interpretation, and conclusions are original, and all sources were verified. No AI tools were used to generate data or produce results.

## Citation

If you use this tool or codebase, please cite the original study and this repository:

```
@article{tsolak2025unveiling,
  title={Unveiling digital mirrors: Decoding gendered body poses in Instagram imagery},
  author={Tsolak, Dorian and Kühne, Simon},
  journal={Computers in Human Behavior},
  volume={163},
  pages={108464},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.chb.2024.108464}
}
```
