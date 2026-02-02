# Unveiling Digital Mirrors - Interactive Pose Analysis Tool

> **Extended from:** [Unveiling Digital Mirrors: Decoding Gendered Body Poses in Instagram Imagery](https://doi.org/10.1016/j.chb.2024.108464) published in [Computers in Human Behavior](https://www.sciencedirect.com/journal/computers-in-human-behavior).

This repository extends the original research with an **interactive web-based tool** for analyzing gendered body language in media imagery through real-time pose detection and gender pattern matching.

---

## 📋 Project Overview

This project provides:

1. **Backend Data Pipeline** - Processes pose data through normalization and clustering
2. **Pose Clustering** - Groups 15,167 poses into 150 clusters based on normalized keypoint positions
3. **Frontend Analysis Tools** - Three interactive modes for analyzing gender patterns in poses
4. **MediaPipe Integration** - Real-time pose detection from uploaded images or videos

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Modern web browser (Chrome, Firefox, Safari, Edge)
- No GPU required (uses CPU inference)

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn
```

### Running the Interactive Tool

```bash
# Start the web server
python serve.py
```

Then open **http://localhost:8000** in your browser.

---

## 📁 Project Structure

```
├── src/
│   ├── normalize.py          # Normalize poses to 0-1 scale, center on midhip
│   ├── geometry.py           # Calculate pose angles and features
│   ├── cluster_poses.py      # Cluster 150 groups from normalized data
│   ├── mediapipe_pipeline.py # Extract poses from images using MediaPipe
│   └── export_data.py        # Export cluster data to JSON for frontend
│
├── frontend/
│   ├── index.html            # Landing page with tool selection
│   ├── image.html            # Single image analysis
│   ├── video.html            # Video frame-by-frame analysis
│   ├── patterns.html         # Pose database explorer
│   ├── image.js              # Image analysis logic
│   ├── video.js              # Video analysis logic
│   ├── styles.css            # UI styling
│   └── data/
│       ├── pose_clusters.json      # 150 pose clusters + prototypes
│       ├── average_poses.json      # Average poses per gender
│       ├── brand_averages.json     # Gender signature metrics
│       └── histograms.json         # Distribution histograms
│
├── data/
│   ├── Unveiling_digital_mirrors.csv  # Original 15,167 labeled poses
│   ├── normalized_poses.csv           # Normalized coordinate data
│   └── pose_signatures.csv            # Angle & feature calculations
│
└── Prototypes/
    └── clusters/                      # 150 PNG prototype images
```

---

## 🎯 How It Works

### 1. Pose Normalization

**Problem**: Raw pose coordinates vary by person size, camera angle, and position.

**Solution**: Each pose is:
- **Centered** on the midhip (origin point)
- **Scaled** by torso length (neck-to-midhip distance)
- **Converted** to relative coordinates (0-1 range for standardization)

```python
# Normalized pose: all keypoints relative to torso
normalized = {
    'nose': {'x': -0.19, 'y': -1.35},
    'lshoulder': {'x': 0.35, 'y': -1.00},
    'midhip': {'x': 0.0, 'y': 0.0},  # origin
    ...
}
```

### 2. Pose Clustering (150 Groups)

**Method**: K-means clustering on normalized poses
- **Input**: 15,167 poses across 3 genders
- **Output**: 150 clusters with:
  - **Prototype**: Average pose (centroid)
  - **Examples**: 3 real poses from each cluster
  - **Statistics**: Male/Female/Non-binary distribution percentages
  - **Character**: Classification based on gender percentage (>70% = gendered, else neutral)

**Minimum Cluster Size**: 10 poses minimum (enforced during clustering in backend)

```json
{
  "id": 0,
  "size": 67,
  "malePercent": 70.1,
  "femalePercent": 19.4,
  "nonbinaryPercent": 10.4,
  "character": "masculine",
  "prototype": { "nose": {...}, "neck": {...}, ... },
  "examples": [ {...}, {...}, {...} ]
}
```

### 3. Pose Matching (Frontend) - Weighted KNN

When a user uploads an image/video, we use **Weighted K-Nearest Neighbors (K=5)** instead of single cluster matching to handle the fact that similar poses might be split across different clusters:

1. **Extract** keypoints using MediaPipe Pose
2. **Normalize** using same torso-based scaling
3. **Calculate distance** to all 150 cluster prototypes (Euclidean distance on normalized keypoints)
4. **Select top K=5 nearest clusters**
5. **Weight by inverse distance** (closer clusters count more)
6. **Average gender percentages** using the weights

**Distance Formula**:
```
distance = average(√[(x₁-x₂)² + (y₁-y₂)²]) for all keypoints
weight = 1 / (distance + 0.001)
```

**Weighted Classification**:
```javascript
weightedMale% = Σ(weight_i × malePercent_i) / Σ(weight_i)
weightedFemale% = Σ(weight_i × femalePercent_i) / Σ(weight_i)

if weightedMale >= 70% → masculine
if weightedFemale >= 70% → feminine
else → neutral
```

**Why Weighted KNN?**
- K-means clustering can split similar poses into different clusters by chance
- Two nearly identical poses might end up in clusters with different gender distributions
- By averaging the top 5 nearest clusters (weighted by distance), we get more stable classifications
- The closest cluster still has the most influence, but nearby clusters smooth out edge cases

---

## 🖼️ Three Analysis Modes

### Mode 1: Image Analysis (`image.html`)

Upload a single photo:
- **Instant pose detection** with skeleton overlay
- **Weighted KNN classification** using top 5 nearest clusters
- **Percentage breakdown** showing weighted average (Male/Female/Non-binary)
- **Media literacy insights** explaining what the pose may communicate

**Visualization**:
- Current pose with overlay
- 4 closest matching clusters with miniature skeletons
- Weighted gender distribution charts

### Mode 2: Video Analysis (`video.html`)

Upload a video up to 30 seconds:
- **Frame-by-frame extraction** at 0.5-second intervals
- **Weighted KNN per frame** - uses top 5 clusters (not just 1)
- **Overall gender prediction** calculated by counting classified frames
- **Playback with skeleton overlay** synchronized with original video

**Gender Classification:**

1. Extract frame at 0.5s intervals (e.g., 0-0.5s-1.0s-1.5s...)
2. For **each frame**:
   - Detect pose keypoints with MediaPipe
   - Normalize the pose (torso-based scaling)
   - Calculate distance to all 150 cluster prototypes
   - **Select the single closest match** (best cluster)
   - Classify as: masculine, feminine, or neutral based on cluster's character
3. **Count matched frames**:
   ```
   Total frames = N
   Masculine frames = X (frames matching masculine clusters)
   Feminine frames = Y (frames matching feminine clusters)
   Neutral frames = Z (frames matching neutral clusters)
   
   Classification: whichever count is highest
   If tied: show as neutral
   ```

**Example**: 20 frames extracted
- Frames 1-8 match masculine clusters → 8 masculine counts
- Frames 9-14 match feminine clusters → 6 feminine counts  
- Frames 15-20 match neutral clusters → 6 neutral counts
- **Result: 8/20 masculine, 6/20 feminine, 6/20 neutral → Classification: MASCULINE**

**Key Features**:
- Frame extraction rate: 0.5 seconds
- Maximum video duration: 30 seconds
- Shows extracted pose count
- Gallery thumbnail selection
- Keyboard shortcuts: ← → for pose navigation, Space for play/pause

**Output**:
```
Frames Analyzed: 20
♂ Masculine: 8 frames
♀ Feminine: 6 frames
⚖ Neutral: 6 frames
→ Classification: MASCULINE (majority vote)
```

### Mode 3: Pose Database (`patterns.html`)

Explore all 150 clusters:
- **Filter by gender category**: Masculine (>70% male), Feminine (>70% female), or Neutral (balanced)
- **Statistics panel**: Shows totals, averages, category counts
- **Interactive cluster view**: Click any cluster to see prototype + examples
- **Gender distribution info** for each cluster

**Filtering Logic**:
- **Masculine**: Only shows clusters where male ≥ 70%
- **Feminine**: Only shows clusters where female ≥ 70%
- **Neutral**: Only shows clusters where male/female difference ≤ 30%

---

## 📊 Original Dataset

The underlying research analyzed **15,167 poses** from Instagram images:

| Gender | Count | Percentage |
|--------|-------|-----------|
| Female | 7,957 | 52.46% |
| Male | 6,597 | 43.50% |
| Non-Binary | 613 | 4.04% |

**Keypoint Format** (OpenPose specification):
- `x_` prefix: horizontal pixel coordinate
- `y_` prefix: vertical pixel coordinate
- 15 keypoints per pose: nose, neck, shoulders, elbows, wrists, hips, knees, ankles

---

## 🔬 Research Foundation

This tool is based on media analysis research exploring how body language is gendered:

### Key Concepts
- **Gender coding**: How poses are associated with masculinity/femininity in media
- **Space-claiming**: Expansive (masculine) vs. contractive (feminine) postures
- **Body positioning**: Head tilt, hip angle, shoulder symmetry as gender markers
- **Statistical patterns**: The 150 clusters represent learned gender associations

### Cited Research
- Goffman, E. (1979) - *Gender Advertisements*
- Cuddy, A. - Power posing and expansive postures
- Kilbourne, J. - Media literacy and advertising analysis

---

## 🛠️ Backend Pipeline (For Developers)

### Step 1: Data Normalization

```bash
python src/normalize.py
```

**Input**: `Unveiling_digital_mirrors.csv` (raw OpenPose coordinates)

**Output**: `data/normalized_poses.csv` (standardized coordinates)

**Process**:
- Extracts 15 keypoints per pose
- Centers on midhip (0, 0)
- Scales by torso length
- Filters out invalid poses (missing keypoints)

### Step 2: Pose Clustering

```bash
python src/cluster_poses.py
```

**Input**: `data/normalized_poses.csv`

**Output**: `frontend/data/pose_clusters.json`

**Process**:
- K-means clustering: k=150 clusters
- **Filters clusters with <10 poses** (keeps only robust groups)
- Computes prototype (centroid) for each cluster
- Extracts 3 example poses from each cluster
- Calculates gender distribution percentages
- Logs which clusters were filtered out due to small size

**Result**: Only clusters with 10+ poses are exported to frontend

### Step 3: Generate Frontend Data

```bash
python src/export_data.py
```

**Output Files**:
- `pose_clusters.json` - Main cluster data (used by all tools)
- `average_poses.json` - Per-gender average poses
- `brand_averages.json` - Aggregated statistics by gender
- `histograms.json` - Distribution data for charts

---

## 📝 Configuration Constants

### Video Analysis (`frontend/video.js`)
```javascript
const MAX_DURATION = 30;       // Max video: 30 seconds
const FRAME_INTERVAL = 0.5;    // Extract pose every 0.5 seconds
const MIN_CLUSTER_SIZE = 10;   // Minimum cluster size
const GENDER_THRESHOLD = 60;   // >60% = gendered pose
```

### Image Analysis (`frontend/image.js`)
```javascript
const MIN_CLUSTER_SIZE = 10;   // Minimum cluster size
const GENDER_THRESHOLD = 70;   // >70% = gendered pose
```

### Pattern Database (`frontend/patterns.html`)
```javascript
const SIGNIFICANT_THRESHOLD = 70;  // Masculine/Feminine threshold
const NEUTRAL_THRESHOLD = 15;      // Within 30% diff = neutral
```

---

## 🎬 Supported Video Formats

- **MP4** (.mp4)
- **WebM** (.webm)
- **MOV** (.mov)
- **AVI** (.avi)

Maximum duration: **30 seconds**

---

## 🔍 Understanding the Results

### Gender Classification

1. **Masculine** (>70% male in dataset)
   - Tends to convey power, authority, confidence
   - Often wider stances, more open postures
   - Examples: relaxed sitting, standing with arms open

2. **Feminine** (>70% female in dataset)
   - Often associated with approachability, elegance
   - Tends to have more closed or tilted postures
   - Examples: head tilt, one-legged stance, crossed arms

3. **Neutral** (<70% in either direction)
   - Used roughly equally by all genders
   - Not strongly coded as masculine or feminine
   - Examples: standing straight, hands at sides, neutral expression

### Distance-Based Matching

The tool matches poses by calculating distance to all 150 cluster prototypes:
- **Closer match** = more similar pose
- **No confidence score** = only raw similarity ranking
- **Top 4 matches** shown for context

---

## 📝 Citation

If you use this tool or data in your research:

```bibtex
@article{unveiling-digital-mirrors-2024,
  title={Unveiling Digital Mirrors: Decoding Gendered Body Poses in Instagram Imagery},
  journal={Computers in Human Behavior},
  year={2024},
  doi={10.1016/j.chb.2024.108464}
}
```

---

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for real-time pose detection
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for keypoint specification
- Original research team for dataset and analysis
