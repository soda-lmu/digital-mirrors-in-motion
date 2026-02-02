/**
 * VIDEO POSE ANALYSIS
 * ====================
 * This script handles video upload, pose extraction using MediaPipe,
 * and gender expression analysis based on pose clusters from research data.
 * 
 * MAIN FLOW:
 * 1. User uploads video → handleVideoUpload()
 * 2. Video is processed frame-by-frame → processVideoInBackground()
 * 3. Each frame: extract pose with MediaPipe → detectPose()
 * 4. Match pose to database clusters → findMatchingClusters()
 * 5. Show results with playback → showResults()
 */

// ============================================================================
// GLOBAL STATE
// ============================================================================

let POSE_CLUSTERS = null;      // Database of pose clusters loaded from JSON
let mediapipePose = null;      // MediaPipe Pose instance
let mediapipeReady = false;    // Flag: is MediaPipe ready to use?
let pendingCallback = null;    // Callback for async pose detection

let extractedPoses = [];       // Array of poses extracted from video
let currentPoseIndex = 0;      // Currently selected pose in gallery
let isProcessing = false;      // Flag: is video being processed?
let isPlaying = false;         // Flag: is video playing?
let animationFrameId = null;   // For canceling playback animation
let isCancelled = false;       // Flag: was processing cancelled?

// ============================================================================
// CONFIGURATION
// ============================================================================

const MAX_DURATION = 30;       // Maximum video length in seconds
const FRAME_INTERVAL = 0.5;    // Extract pose every 0.5 seconds
const MIN_CLUSTER_SIZE = 10;   // Ignore clusters with fewer poses
const GENDER_DIFF_THRESHOLD = 30;   // >=30% difference = gendered pose, else neutral

// Skeleton bones: pairs of keypoints to connect with lines
const BONES = [
    ['nose', 'neck'], ['neck', 'rshoulder'], ['neck', 'lshoulder'],
    ['rshoulder', 'relbow'], ['lshoulder', 'lelbow'],
    ['relbow', 'rwrist'], ['lelbow', 'lwrist'],
    ['neck', 'midhip'], ['midhip', 'rhip'], ['midhip', 'lhip'],
    ['rhip', 'rknee'], ['lhip', 'lknee'],
    ['rknee', 'rankle'], ['lknee', 'lankle']
];

// All keypoint names we track
const KEYPOINT_NAMES = [
    'nose', 'neck', 'lshoulder', 'rshoulder', 'lelbow', 'relbow',
    'lwrist', 'rwrist', 'midhip', 'lhip', 'rhip', 'lknee', 'rknee', 'lankle', 'rankle'
];

// Map our keypoint names to MediaPipe landmark indices
const LANDMARK_MAP = {
    nose: 0, lshoulder: 11, rshoulder: 12, lelbow: 13, relbow: 14,
    lwrist: 15, rwrist: 16, lhip: 23, rhip: 24, lknee: 25, rknee: 26,
    lankle: 27, rankle: 28
};

// Shorthand for document.getElementById
const $ = id => document.getElementById(id);

// ============================================================================
// GENDER CLASSIFICATION
// ============================================================================

/**
 * Classify a cluster as masculine, feminine, or neutral
 * Uses 70% threshold: >70% male → masculine, >70% female → feminine
 */
/**
 * Classify a cluster as masculine, feminine, or neutral
 * Uses difference-based threshold: if difference between genders > 30%, categorize as that gender
 */
function getGenderClass(cluster) {
    if (!cluster) return 'neutral';
    const maleDiff = cluster.malePercent - cluster.femalePercent;
    const femaleDiff = cluster.femalePercent - cluster.malePercent;
    
    if (maleDiff >= 30) return 'masculine';     // >=30% more male
    if (femaleDiff >= 30) return 'feminine';    // >=30% more female
    return 'neutral';                            // Difference < 30%
}

/** Get color for gender class (blue=masc, pink=fem, green=neutral) */
function getGenderColor(cluster) {
    const cls = getGenderClass(cluster);
    if (cls === 'masculine') return '#89b4fa';
    if (cls === 'feminine') return '#f5c2e7';
    return '#a6e3a1';
}

// ============================================================================
// DATA LOADING
// ============================================================================

/** Load pose clusters from JSON file */
async function loadData() {
    try {
        const res = await fetch('data/pose_clusters.json');
        if (res.ok) {
            const data = await res.json();
            POSE_CLUSTERS = {
                ...data,
                clusters: data.clusters.filter(c => c.size >= MIN_CLUSTER_SIZE)
            };
            console.log('✓ Loaded', POSE_CLUSTERS.clusters.length, 'pose clusters');
        }
    } catch (err) {
        console.error('Error loading pose clusters:', err);
    }
}

// ============================================================================
// POSE NORMALIZATION & MATCHING
// ============================================================================

/**
 * Normalize keypoints relative to torso
 * Makes poses comparable regardless of person size/position
 */
function normalizeKeypoints(kp) {
    if (!kp?.midhip || !kp?.neck) return null;
    const torsoLength = Math.hypot(kp.neck.x - kp.midhip.x, kp.neck.y - kp.midhip.y);
    if (torsoLength === 0) return null;
    
    const normalized = {};
    for (const name of KEYPOINT_NAMES) {
        if (kp[name]) {
            normalized[name] = {
                x: (kp[name].x - kp.midhip.x) / torsoLength,
                y: (kp[name].y - kp.midhip.y) / torsoLength
            };
        }
    }
    return normalized;
}

/** Calculate distance between two poses (lower = more similar) */
function poseDistance(pose1, pose2) {
    if (!pose1 || !pose2) return Infinity;
    let totalDist = 0, count = 0;
    for (const name of KEYPOINT_NAMES) {
        if (pose1[name] && pose2[name]) {
            totalDist += Math.hypot(pose1[name].x - pose2[name].x, pose1[name].y - pose2[name].y);
            count++;
        }
    }
    return count > 0 ? totalDist / count : Infinity;
}

/** Find closest matching clusters for a pose with distances */
function findMatchingClusters(keypoints, count = 4) {
    if (!POSE_CLUSTERS?.clusters || !keypoints) return [];
    const normalizedPose = normalizeKeypoints(keypoints);
    if (!normalizedPose) return [];
    // Note: Don't canonicalize - prototypes in JSON are not canonicalized
    
    return POSE_CLUSTERS.clusters
        .filter(c => c.prototype)
        .map(cluster => ({ cluster, distance: poseDistance(normalizedPose, cluster.prototype) }))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, count)
        .map(s => s.cluster);
}

/**
 * Find the single closest matching cluster for a pose.
 * Returns the cluster's actual stats (not weighted averages).
 */
function findClosestCluster(keypoints) {
    if (!POSE_CLUSTERS?.clusters || !keypoints) return null;
    const normalizedPose = normalizeKeypoints(keypoints);
    if (!normalizedPose) return null;
    
    // Find closest cluster by distance
    let closest = null;
    let minDist = Infinity;
    
    for (const cluster of POSE_CLUSTERS.clusters) {
        if (!cluster.prototype) continue;
        const dist = poseDistance(normalizedPose, cluster.prototype);
        if (dist < minDist) {
            minDist = dist;
            closest = cluster;
        }
    }
    
    if (!closest) return null;
    
    // Classification based on the cluster's actual stats
    const maleDiff = closest.malePercent - closest.femalePercent;
    const femaleDiff = closest.femalePercent - closest.malePercent;
    let character;
    if (maleDiff >= GENDER_DIFF_THRESHOLD) character = 'masculine';
    else if (femaleDiff >= GENDER_DIFF_THRESHOLD) character = 'feminine';
    else character = 'neutral';
    
    return {
        cluster: closest,
        character,
        malePercent: closest.malePercent,
        femalePercent: closest.femalePercent,
        distance: minDist
    };
}

// ============================================================================
// COORDINATE CONVERSION
// ============================================================================

/** Convert MediaPipe landmarks (0-1) to canvas pixel coordinates */
function landmarksToCanvasKeypoints(landmarks, canvas, videoW, videoH) {
    const scale = Math.min(canvas.width / videoW, canvas.height / videoH);
    const drawW = videoW * scale;
    const drawH = videoH * scale;
    const offsetX = (canvas.width - drawW) / 2;
    const offsetY = (canvas.height - drawH) / 2;
    
    const kp = {};
    for (const name in LANDMARK_MAP) {
        const lm = landmarks[LANDMARK_MAP[name]];
        if (lm) {
            kp[name] = { x: lm.x * drawW + offsetX, y: lm.y * drawH + offsetY };
        }
    }
    // Calculate neck and midhip from shoulders and hips
    if (kp.lshoulder && kp.rshoulder) {
        kp.neck = { x: (kp.lshoulder.x + kp.rshoulder.x) / 2, y: (kp.lshoulder.y + kp.rshoulder.y) / 2 };
    }
    if (kp.lhip && kp.rhip) {
        kp.midhip = { x: (kp.lhip.x + kp.rhip.x) / 2, y: (kp.lhip.y + kp.rhip.y) / 2 };
    }
    return kp;
}

/** Convert landmarks to normalized keypoints for matching */
function landmarksToNormalizedKeypoints(landmarks) {
    const kp = {};
    for (const name in LANDMARK_MAP) {
        const lm = landmarks[LANDMARK_MAP[name]];
        if (lm) kp[name] = { x: lm.x, y: lm.y };
    }
    if (kp.lshoulder && kp.rshoulder) {
        kp.neck = { x: (kp.lshoulder.x + kp.rshoulder.x) / 2, y: (kp.lshoulder.y + kp.rshoulder.y) / 2 };
    }
    if (kp.lhip && kp.rhip) {
        kp.midhip = { x: (kp.lhip.x + kp.rhip.x) / 2, y: (kp.lhip.y + kp.rhip.y) / 2 };
    }
    return kp;
}

// ============================================================================
// DRAWING FUNCTIONS
// ============================================================================

/** Draw video frame on canvas (centered, aspect ratio preserved) */
function drawVideoFrame(ctx, video) {
    const canvas = ctx.canvas;
    const scale = Math.min(canvas.width / video.videoWidth, canvas.height / video.videoHeight);
    const w = video.videoWidth * scale;
    const h = video.videoHeight * scale;
    const x = (canvas.width - w) / 2;
    const y = (canvas.height - h) / 2;
    
    ctx.fillStyle = '#1e1e2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, x, y, w, h);
}

/** Draw skeleton overlay */
function drawSkeleton(ctx, kp) {
    if (!kp) return;
    
    // Draw bones (green lines)
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    for (const [start, end] of BONES) {
        if (kp[start] && kp[end]) {
            ctx.beginPath();
            ctx.moveTo(kp[start].x, kp[start].y);
            ctx.lineTo(kp[end].x, kp[end].y);
            ctx.stroke();
        }
    }
    
    // Draw joints (red circles)
    ctx.fillStyle = '#ff0000';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    for (const name in kp) {
        const pt = kp[name];
        if (!pt?.x) continue;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    }
}

/** Draw mini skeleton for thumbnails */
function drawMiniSkeleton(canvas, pose, color = '#f9e2af') {
    if (!pose || !canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const scale = 20, offsetX = canvas.width / 2, offsetY = canvas.height / 2;
    const kp = {};
    for (const name in pose) {
        if (pose[name]?.x !== undefined) {
            kp[name] = { x: pose[name].x * scale + offsetX, y: pose[name].y * scale + offsetY };
        }
    }
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    for (const [start, end] of BONES) {
        if (kp[start] && kp[end]) {
            ctx.beginPath();
            ctx.moveTo(kp[start].x, kp[start].y);
            ctx.lineTo(kp[end].x, kp[end].y);
            ctx.stroke();
        }
    }
    
    ctx.fillStyle = color;
    for (const name in kp) {
        ctx.beginPath();
        ctx.arc(kp[name].x, kp[name].y, 2, 0, Math.PI * 2);
        ctx.fill();
    }
}

/** Format seconds as MM:SS */
function formatTime(s) {
    return `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, '0')}`;
}

// ============================================================================
// UI UPDATE FUNCTIONS
// ============================================================================

/** Update "Current Pose" panel - shows the closest cluster's actual stats */
function updatePatternPanel(keypoints) {
    const container = $('patternContent');
    if (!keypoints) {
        container.innerHTML = '<p class="placeholder-text">Upload a video to begin</p>';
        return;
    }
    
    // Find the single closest matching cluster
    const match = findClosestCluster(keypoints);
    if (!match) {
        container.innerHTML = '<p class="placeholder-text">No pattern match</p>';
        return;
    }
    
    const c = match.cluster;
    const color = match.character === 'masculine' ? '#89b4fa' : match.character === 'feminine' ? '#f5c2e7' : '#a6e3a1';
    
    container.innerHTML = `
        <div class="pattern-display compact">
            <div class="pattern-circle small" style="background: conic-gradient(#89b4fa ${c.malePercent}%, #f5c2e7 0%);">
            </div>
            <div class="pattern-info">
                <span class="cluster-ref" style="color:${color};">Cluster #${c.id}</span>
                <span style="font-size:0.8rem; color:#89b4fa;">${c.malePercent.toFixed(0)}% M</span>
                <span style="font-size:0.8rem; color:#f5c2e7;">${c.femalePercent.toFixed(0)}% F</span>
            </div>
        </div>
    `;
}

/** Update "Matching Categories" panel */
function updateMatchingClusters(keypoints) {
    const container = $('matchingClusters');
    if (!keypoints) {
        container.innerHTML = '<p class="placeholder-text">Pose matches will appear here</p>';
        return;
    }
    
    const matches = findMatchingClusters(keypoints, 4);
    if (matches.length === 0) {
        container.innerHTML = '<p class="placeholder-text">No matches</p>';
        return;
    }
    
    container.innerHTML = '<div class="clusters-grid compact">' + matches.map((c, i) => {
        const color = getGenderColor(c);
        return `
            <div class="cluster-card mini" style="border-color:${color};">
                <canvas id="cluster${i}" width="60" height="70"></canvas>
                <span class="cluster-id" style="color:${color};">#${c.id}</span>
                <span style="font-size:0.65rem; color:#6c7086;">${c.malePercent.toFixed(0)}%M</span>
            </div>
        `;
    }).join('') + '</div>';
    
    setTimeout(() => {
        matches.forEach((c, i) => {
            const canvas = $(`cluster${i}`);
            if (canvas && c.prototype) drawMiniSkeleton(canvas, c.prototype, getGenderColor(c));
        });
    }, 10);
}

/** Show summary of detected poses - no final verdict, just breakdown */
function showPrediction() {
    if (extractedPoses.length === 0) return;
    
    let mascCount = 0, femCount = 0, neutCount = 0;
    let totalMale = 0, totalFemale = 0;
    
    extractedPoses.forEach(p => {
        // Use stored character (based on cluster's actual stats)
        const gc = p.character || 'neutral';
        if (gc === 'masculine') mascCount++;
        else if (gc === 'feminine') femCount++;
        else neutCount++;
        totalMale += p.malePercent || 0;
        totalFemale += p.femalePercent || 0;
    });
    
    const total = extractedPoses.length;
    const avgMale = total > 0 ? (totalMale / total).toFixed(1) : 0;
    const avgFemale = total > 0 ? (totalFemale / total).toFixed(1) : 0;
    
    // Show summary card with pose breakdown (no final verdict)
    $('summaryCard').style.display = 'block';
    $('summaryContent').innerHTML = `
        <div class="summary-header">
            <h3 style="margin:0; color:#cdd6f4;">Detected Poses: ${total}</h3>
            <p style="font-size:0.8rem; color:#6c7086; margin:4px 0 12px 0;">Extracted every 0.5 seconds</p>
        </div>
        <div class="summary-breakdown">
            <div class="breakdown-item" style="border-left: 3px solid #89b4fa; padding-left: 12px; margin-bottom: 10px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #89b4fa;">${mascCount}</div>
                <div style="font-size: 0.85rem; color: #6c7086;">Masculine-associated poses</div>
                <div style="font-size: 0.75rem; color: #585b70;">(≥30% more male in dataset)</div>
            </div>
            <div class="breakdown-item" style="border-left: 3px solid #f5c2e7; padding-left: 12px; margin-bottom: 10px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #f5c2e7;">${femCount}</div>
                <div style="font-size: 0.85rem; color: #6c7086;">Feminine-associated poses</div>
                <div style="font-size: 0.75rem; color: #585b70;">(≥30% more female in dataset)</div>
            </div>
            <div class="breakdown-item" style="border-left: 3px solid #a6e3a1; padding-left: 12px; margin-bottom: 10px;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #a6e3a1;">${neutCount}</div>
                <div style="font-size: 0.85rem; color: #6c7086;">Neutral poses</div>
                <div style="font-size: 0.75rem; color: #585b70;">(<30% difference)</div>
            </div>
        </div>
        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #313244;">
            <p style="font-size: 0.8rem; color: #6c7086; margin: 0;">Average across all poses: <span style="color:#89b4fa;">${avgMale}% male</span> / <span style="color:#f5c2e7;">${avgFemale}% female</span></p>
        </div>
    `;
    
    // Hide the prediction card (no final verdict)
    $('predictionCard').style.display = 'none';
}

/** Render extracted poses gallery */
function renderPosesGallery() {
    const container = $('posesContainer');
    
    container.innerHTML = extractedPoses.map((p, i) => {
        // Use stored character and malePercent from the closest cluster
        const color = p.character === 'masculine' ? '#89b4fa' : p.character === 'feminine' ? '#f5c2e7' : '#a6e3a1';
        const malePercent = p.malePercent?.toFixed(0) || '?';
        return `
            <div class="pose-thumb ${i === currentPoseIndex ? 'active' : ''}" data-index="${i}">
                <canvas id="pose${i}" width="60" height="50"></canvas>
                <span class="pose-time">${formatTime(p.time)}</span>
                <span class="pose-cluster" style="color:${color}">${p.cluster ? '#' + p.cluster.id + ' (' + malePercent + '%M)' : '-'}</span>
            </div>
        `;
    }).join('');
    
    setTimeout(() => {
        extractedPoses.forEach((p, i) => {
            const canvas = $(`pose${i}`);
            const color = p.character === 'masculine' ? '#89b4fa' : p.character === 'feminine' ? '#f5c2e7' : '#a6e3a1';
            if (canvas && p.normalizedKeypoints) drawMiniSkeleton(canvas, p.normalizedKeypoints, color);
        });
    }, 10);
    
    container.querySelectorAll('.pose-thumb').forEach(el => {
        el.onclick = () => selectPose(parseInt(el.dataset.index));
    });
    
    $('poseCounter').textContent = `${currentPoseIndex + 1} / ${extractedPoses.length}`;
}

/** Select a pose from gallery */
function selectPose(index) {
    if (index < 0 || index >= extractedPoses.length) return;
    currentPoseIndex = index;
    
    document.querySelectorAll('.pose-thumb').forEach((el, i) => el.classList.toggle('active', i === index));
    $('poseCounter').textContent = `${index + 1} / ${extractedPoses.length}`;
    
    const pose = extractedPoses[index];
    const video = $('videoElement');
    const canvas = $('playbackCanvas');
    const ctx = canvas.getContext('2d');
    
    video.currentTime = pose.time;
    video.onseeked = () => {
        drawVideoFrame(ctx, video);
        if (pose.landmarks) {
            drawSkeleton(ctx, landmarksToCanvasKeypoints(pose.landmarks, canvas, video.videoWidth, video.videoHeight));
        }
        video.onseeked = null;
    };
    
    if (pose.landmarks) {
        const kp = landmarksToNormalizedKeypoints(pose.landmarks);
        updatePatternPanel(kp);
        updateMatchingClusters(kp);
    }
    
    const thumb = document.querySelector(`.pose-thumb[data-index="${index}"]`);
    if (thumb) thumb.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
}

// ============================================================================
// MEDIAPIPE SETUP
// ============================================================================

function handlePoseResults(results) {
    if (pendingCallback) {
        const cb = pendingCallback;
        pendingCallback = null;
        cb(results);
    }
}

/** Initialize MediaPipe Pose model */
async function initMediaPipe() {
    return new Promise((resolve) => {
        console.log('Initializing MediaPipe...');
        
        mediapipePose = new Pose({
            locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`
        });
        mediapipePose.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        mediapipePose.onResults(handlePoseResults);
        
        // Warm up with test image
        const testCanvas = document.createElement('canvas');
        testCanvas.width = 100;
        testCanvas.height = 100;
        testCanvas.getContext('2d').fillRect(0, 0, 100, 100);
        
        pendingCallback = () => { mediapipeReady = true; console.log('✓ MediaPipe ready'); resolve(); };
        mediapipePose.send({ image: testCanvas }).catch(() => { mediapipeReady = true; resolve(); });
        setTimeout(() => { if (!mediapipeReady) { mediapipeReady = true; resolve(); } }, 8000);
    });
}

/** Detect pose in video frame */
async function detectPose(videoElement) {
    if (isCancelled || !videoElement.videoWidth || !videoElement.videoHeight) return null;
    
    return new Promise((resolve) => {
        const timeout = setTimeout(() => { pendingCallback = null; resolve(null); }, 5000);
        
        // Draw frame to canvas (MediaPipe needs canvas, not video)
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        canvas.getContext('2d').drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        pendingCallback = (results) => {
            clearTimeout(timeout);
            resolve(isCancelled ? null : results);
        };
        
        mediapipePose.send({ image: canvas }).catch(() => { clearTimeout(timeout); pendingCallback = null; resolve(null); });
    });
}

// ============================================================================
// VIDEO PROCESSING
// ============================================================================

/** Process video frame-by-frame */
async function processVideoInBackground(video) {
    isProcessing = true;
    isCancelled = false;
    extractedPoses = [];
    
    const canvas = $('processingCanvas');
    const ctx = canvas.getContext('2d');
    
    if (video.readyState < 2) {
        await new Promise(resolve => { video.onloadeddata = resolve; setTimeout(resolve, 2000); });
    }
    
    // Skip first and last second of video to avoid skeleton detection issues
    const startTime = 1.0;
    const endTime = Math.max(startTime + 0.5, video.duration - 1.0);
    const totalFrames = Math.floor((endTime - startTime) / FRAME_INTERVAL) + 1;
    console.log(`Processing: ${video.duration.toFixed(1)}s, analyzing ${startTime}s to ${endTime.toFixed(1)}s (${totalFrames} frames)`);
    $('processingText').textContent = 'Extracting poses...';
    
    for (let i = 0; i < totalFrames && isProcessing && !isCancelled; i++) {
        const time = startTime + (i * FRAME_INTERVAL);
        
        // Seek to frame
        await new Promise(resolve => {
            if (isCancelled) { resolve(); return; }
            const onSeeked = () => { video.removeEventListener('seeked', onSeeked); setTimeout(resolve, 100); };
            video.addEventListener('seeked', onSeeked);
            video.currentTime = time;
            setTimeout(() => { video.removeEventListener('seeked', onSeeked); resolve(); }, 500);
        });
        
        if (isCancelled) break;
        drawVideoFrame(ctx, video);
        
        const result = await detectPose(video);
        if (isCancelled) break;
        
        if (result?.poseLandmarks) {
            const landmarks = result.poseLandmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z, visibility: lm.visibility }));
            const kp = landmarksToNormalizedKeypoints(landmarks);
            
            // Find the single closest matching cluster
            const match = findClosestCluster(kp);
            
            // Debug: log each match
            if (match) {
                console.log(`Frame ${i}: Cluster #${match.cluster.id}, ${match.malePercent.toFixed(1)}%M - ${match.femalePercent.toFixed(1)}%F = ${(match.malePercent - match.femalePercent).toFixed(1)}% diff → ${match.character}`);
            }
            
            extractedPoses.push({
                time, frameIndex: i, landmarks,
                normalizedKeypoints: normalizeKeypoints(kp),
                cluster: match?.cluster || null,
                character: match?.character || 'neutral',
                malePercent: match?.malePercent || 0,
                femalePercent: match?.femalePercent || 0
            });
            
            drawSkeleton(ctx, landmarksToCanvasKeypoints(landmarks, canvas, video.videoWidth, video.videoHeight));
        }
        
        const progress = ((i + 1) / totalFrames) * 100;
        $('progressFill').style.width = `${progress}%`;
        $('progressText').textContent = `${progress.toFixed(0)}%`;
        $('processingText').textContent = `Extracting poses... (${extractedPoses.length} found)`;
    }
    
    isProcessing = false;
    
    // Debug: log summary
    const mascCount = extractedPoses.filter(p => p.character === 'masculine').length;
    const femCount = extractedPoses.filter(p => p.character === 'feminine').length;
    const neutCount = extractedPoses.filter(p => p.character === 'neutral').length;
    console.log(`Processing complete: ${extractedPoses.length} poses (${mascCount} masc, ${femCount} fem, ${neutCount} neutral)`);
    
    return !isCancelled && extractedPoses.length > 0;
}

// ============================================================================
// VIDEO PLAYBACK
// ============================================================================

function startPlayback() {
    const video = $('videoElement');
    const canvas = $('playbackCanvas');
    const ctx = canvas.getContext('2d');
    
    if (!video.src || extractedPoses.length === 0) return;
    
    video.currentTime = 0;
    video.onseeked = () => {
        video.onseeked = null;
        isPlaying = true;
        $('playBtn').textContent = '⏸ Pause';
        video.play();
        
        function render() {
            if (!isPlaying || video.paused || video.ended) { stopPlayback(); return; }
            
            drawVideoFrame(ctx, video);
            
            // Find closest pose to current time
            let closestPose = null, closestDiff = Infinity;
            for (const pose of extractedPoses) {
                const diff = Math.abs(pose.time - video.currentTime);
                if (diff < closestDiff) { closestDiff = diff; closestPose = pose; }
            }
            
            if (closestPose && closestDiff < FRAME_INTERVAL && closestPose.landmarks) {
                drawSkeleton(ctx, landmarksToCanvasKeypoints(closestPose.landmarks, canvas, video.videoWidth, video.videoHeight));
            }
            
            $('playbackFill').style.width = `${(video.currentTime / video.duration) * 100}%`;
            $('timeDisplay').textContent = `${formatTime(video.currentTime)} / ${formatTime(video.duration)}`;
            animationFrameId = requestAnimationFrame(render);
        }
        render();
    };
}

function stopPlayback() {
    isPlaying = false;
    $('videoElement').pause();
    if (animationFrameId) { cancelAnimationFrame(animationFrameId); animationFrameId = null; }
    $('playBtn').textContent = '▶ Play with Skeleton';
}

function togglePlayback() {
    if (isPlaying) stopPlayback(); else startPlayback();
}

// ============================================================================
// UPLOAD & UI FLOW
// ============================================================================

async function handleVideoUpload(file) {
    // Clear all previous state FIRST
    isCancelled = false;
    isProcessing = false;
    isPlaying = false;
    pendingCallback = null;
    extractedPoses = [];       // Clear previous poses
    currentPoseIndex = 0;      // Reset pose index
    
    // Cancel any ongoing playback
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    
    // Show processing UI
    $('uploadPlaceholder').style.display = 'none';
    $('processingSection').style.display = 'block';
    $('resultsSection').style.display = 'none';
    $('predictionCard').style.display = 'none';
    $('summaryCard').style.display = 'none';
    $('clearBtn').style.display = 'inline-block';
    
    $('progressFill').style.width = '0%';
    $('progressText').textContent = '0%';
    $('processingText').textContent = 'Loading video...';
    
    // Clean up previous video
    const video = $('videoElement');
    if (video.src?.startsWith('blob:')) {
        URL.revokeObjectURL(video.src);
    }
    video.src = URL.createObjectURL(file);
    
    try {
        await new Promise((resolve, reject) => {
            video.onloadedmetadata = resolve;
            video.onerror = () => reject(new Error('Failed to load'));
            setTimeout(() => reject(new Error('Timeout')), 10000);
        });
        await new Promise(resolve => {
            if (video.readyState >= 2) resolve();
            else { video.onloadeddata = resolve; setTimeout(resolve, 2000); }
        });
    } catch (err) {
        alert('Failed to load video. Please try another file.');
        resetToUpload();
        return;
    }
    
    if (video.duration > MAX_DURATION) {
        alert(`Video too long (${video.duration.toFixed(0)}s). Max ${MAX_DURATION}s allowed.`);
        resetToUpload();
        return;
    }
    
    if (isCancelled) return;
    
    $('processingText').textContent = 'Initializing...';
    if (!mediapipeReady) await initMediaPipe();
    if (isCancelled) return;
    
    const success = await processVideoInBackground(video);
    if (isCancelled) return;
    
    if (!success) {
        alert('No poses detected. Try a video with a visible person.');
        resetToUpload();
        return;
    }
    showResults();
}

function showResults() {
    $('uploadPlaceholder').style.display = 'none';
    $('processingSection').style.display = 'none';
    $('resultsSection').style.display = 'block';
    $('clearBtn').style.display = 'inline-block';
    
    const video = $('videoElement');
    const canvas = $('playbackCanvas');
    const ctx = canvas.getContext('2d');
    
    video.currentTime = 0;
    video.onseeked = () => {
        drawVideoFrame(ctx, video);
        if (extractedPoses[0]?.landmarks) {
            drawSkeleton(ctx, landmarksToCanvasKeypoints(extractedPoses[0].landmarks, canvas, video.videoWidth, video.videoHeight));
        }
        video.onseeked = null;
    };
    
    $('timeDisplay').textContent = `0:00 / ${formatTime(video.duration)}`;
    currentPoseIndex = 0;
    renderPosesGallery();
    showPrediction();
    
    if (extractedPoses[0]?.landmarks) {
        const kp = landmarksToNormalizedKeypoints(extractedPoses[0].landmarks);
        updatePatternPanel(kp);
        updateMatchingClusters(kp);
    }
}

function resetToUpload() {
    isCancelled = true;
    isProcessing = false;
    isPlaying = false;
    pendingCallback = null;
    extractedPoses = [];
    currentPoseIndex = 0;
    
    if (animationFrameId) { cancelAnimationFrame(animationFrameId); animationFrameId = null; }
    
    const video = $('videoElement');
    video.pause();
    video.onseeked = video.onloadeddata = video.onloadedmetadata = null;
    if (video.src?.startsWith('blob:')) URL.revokeObjectURL(video.src);
    video.src = '';
    video.load();
    
    $('uploadPlaceholder').style.display = 'flex';
    $('processingSection').style.display = 'none';
    $('resultsSection').style.display = 'none';
    $('predictionCard').style.display = 'none';
    $('summaryCard').style.display = 'none';
    $('clearBtn').style.display = 'none';
    $('videoInput').value = '';
    
    updatePatternPanel(null);
    updateMatchingClusters(null);
    
    setTimeout(() => { isCancelled = false; }, 100);
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    await loadData();
    await initMediaPipe();
    
    // Button handlers
    $('uploadBtn').onclick = () => $('videoInput').click();
    $('videoInput').onchange = (e) => { if (e.target.files[0]) handleVideoUpload(e.target.files[0]); };
    $('cancelBtn').onclick = () => { isProcessing = false; resetToUpload(); };
    $('clearBtn').onclick = resetToUpload;
    $('playBtn').onclick = togglePlayback;
    $('prevPoseBtn').onclick = () => selectPose(currentPoseIndex - 1);
    $('nextPoseBtn').onclick = () => selectPose(currentPoseIndex + 1);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (extractedPoses.length === 0) return;
        if (e.key === 'ArrowLeft') { e.preventDefault(); selectPose(currentPoseIndex - 1); }
        if (e.key === 'ArrowRight') { e.preventDefault(); selectPose(currentPoseIndex + 1); }
        if (e.key === ' ') { e.preventDefault(); togglePlayback(); }
    });
    
    // Click progress bar to seek
    $('playbackProgressBar').onclick = (e) => {
        const video = $('videoElement');
        if (!video.duration) return;
        const rect = e.currentTarget.getBoundingClientRect();
        video.currentTime = ((e.clientX - rect.left) / rect.width) * video.duration;
    };
    
    $('videoElement').onended = stopPlayback;
});
