"""
MediaPipe Pose Extraction
=========================
Extracts poses from video and images using MediaPipe Pose.
Output format matches cluster_poses.py (17 joints) for cluster matching.

Requires: pip install mediapipe opencv-python
"""

import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import mediapipe as mp
except ImportError:
    mp = None

# MediaPipe landmark indices → our joint names (matches cluster_poses.py JOINTS)
# MP: 0=Nose, 11=LShoulder, 12=RShoulder, 13=LElbow, 14=RElbow, 15=LWrist, 16=RWrist,
#     23=LHip, 24=RHip, 25=LKnee, 26=RKnee, 27=LAnkle, 28=RAnkle, 29=LFootIndex, 30=RFootIndex
MP_TO_OUR = {
    0: 'nose',
    11: 'lshoulder',
    12: 'rshoulder',
    13: 'lelbow',
    14: 'relbow',
    15: 'lwrist',
    16: 'rwrist',
    23: 'lhip',
    24: 'rhip',
    25: 'lknee',
    26: 'rknee',
    27: 'lankle',
    28: 'rankle',
    29: 'bigtoe',
    30: 'rbigtoe',
}

FRAME_INTERVAL = 0.5
SKIP_EDGE_SECONDS = 2
MIN_VISIBILITY = 0.5


def _mp_to_keypoints(landmarks, img_w: int, img_h: int) -> Optional[Dict[str, Dict[str, float]]]:
    """Convert MediaPipe landmarks to {joint: {x,y}} in [0,1]."""
    if not landmarks or len(landmarks) < 31:
        return None
    kp = {}
    for mp_idx, our_name in MP_TO_OUR.items():
        lm = landmarks[mp_idx]
        vis = getattr(lm, 'visibility', 1.0)
        if vis < MIN_VISIBILITY:
            continue
        kp[our_name] = {
            'x': lm.x,
            'y': lm.y
        }
    # Derived: neck = midpoint of shoulders, midhip = midpoint of hips
    if kp.get('lshoulder') and kp.get('rshoulder'):
        kp['neck'] = {
            'x': (kp['lshoulder']['x'] + kp['rshoulder']['x']) / 2,
            'y': (kp['lshoulder']['y'] + kp['rshoulder']['y']) / 2
        }
    if kp.get('lhip') and kp.get('rhip'):
        kp['midhip'] = {
            'x': (kp['lhip']['x'] + kp['rhip']['x']) / 2,
            'y': (kp['lhip']['y'] + kp['rhip']['y']) / 2
        }
    if 'nose' not in kp or len(kp) < 5:
        return None
    return kp


def extract_poses_from_video(
    video_path: Path,
    frame_interval: float = FRAME_INTERVAL,
    skip_edges: float = SKIP_EDGE_SECONDS,
) -> List[Dict[str, Any]]:
    """Extract poses from video at frame_interval. Returns [{time, keypoints}, ...]."""
    if not mp:
        raise ImportError("mediapipe not installed. Run: pip install mediapipe")
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    poses = []
    frame_idx = 0
    last_t = -999
    start_time = skip_edges
    end_time = max(start_time + 0.5, duration - skip_edges)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps if fps > 0 else frame_idx * frame_interval
        if t < start_time or t > end_time:
            frame_idx += 1
            continue
        if t - last_t < frame_interval - 0.01:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            kp = _mp_to_keypoints(results.pose_landmarks.landmark, video_w, video_h)
            if kp:
                poses.append({'time': t, 'keypoints': kp})
                last_t = t
        frame_idx += 1

    cap.release()
    pose.close()
    return poses


def extract_poses_from_image(image_path: Path) -> Optional[Dict[str, Dict[str, float]]]:
    """Extract pose from single image. Returns {joint: {x,y}} or None."""
    if not mp:
        raise ImportError("mediapipe not installed. Run: pip install mediapipe")
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    img_h, img_w = img.shape[:2]

    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    )
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    pose.close()

    if not results.pose_landmarks:
        return None
    return _mp_to_keypoints(results.pose_landmarks.landmark, img_w, img_h)
