/**
 * VIDEO POSE ANALYSIS
 * ===================
 * Core logic for the video analysis page.
 *
 * Flow:
 *   1. User uploads a video (max 30s).
 *   2. We seek to each 0.5s mark, run MediaPipe Pose, store landmarks.
 *   3. Each frame's landmarks are normalized and matched to the nearest
 *      of 150 pose clusters (same pipeline as cluster_poses.py).
 *   4. Results are shown as a gallery, playback with skeleton overlay,
 *      summary counts (masculine/feminine/neutral), and a timeline chart.
 *
 * The first/last 2 seconds of the video are skipped to exclude
 * intro/outro frames that rarely contain representative body poses.
 */

import {
    BONES,
    findClosestPoses,
    getTopMatchPoses,
    getGenderColor,
    drawMiniSkeleton,
    getHowItWorksHTML,
    normalizeKeypoints,
    parsePoseClustersCSV,
    parseNormalizedPosesCSV
} from './pose-utils.js';

import { drawTimeline, exportCSV } from './timeline.js';

// -----------------------------------------------------------------------------
// GLOBAL STATE
// -----------------------------------------------------------------------------

// --- Global state ---
// These track the current analysis session: loaded clusters, MediaPipe
// model state, extracted poses, playback state, and UI selections.

let POSE_CLUSTERS = null;              // 150 clusters from pose_clusters.csv
let NORMALIZED_POSES = null;           // Loaded from normalized_poses.csv
let mediapipePose = null;              // MediaPipe Pose model instance
let mediapipeReady = false;            // True once the model is warmed up
let pendingCallback = null;            // Callback for async MediaPipe results

let extractedPoses = [];               // Array of detected poses from video
let currentPoseIndex = 0;              // Currently selected pose in gallery
let isProcessing = false;              // True while extracting frames
let isPlaying = false;                 // True during video playback
let animationFrameId = null;           // rAF handle for playback loop
let isCancelled = false;               // User pressed cancel
let lastPlaybackLandmarks = null;      // Most recent live-detected landmarks
let playbackDetectionPending = false;  // Waiting for MediaPipe result
let lastPlaybackDetectionTime = 0;     // Timestamp of last live detection
const PLAYBACK_DETECTION_INTERVAL = 150; // ms between live detections


const MAX_DURATION = 30;
const FRAME_INTERVAL = 0.5;

// MediaPipe landmark index → our 17-joint format.
// Indices 31/32 = left/right foot index (closest to OpenPose big toe).
// Neck and midhip are derived as midpoints (not in MediaPipe's 33 landmarks).
const LANDMARK_MAP = {
    nose: 0, lshoulder: 11, rshoulder: 12, lelbow: 13, relbow: 14,
    lwrist: 15, rwrist: 16, lhip: 23, rhip: 24, lknee: 25, rknee: 26,
    lankle: 27, rankle: 28, bigtoe: 31, rbigtoe: 32
};

const $ = id => document.getElementById(id);

// -----------------------------------------------------------------------------
// DATA LOADING
// -----------------------------------------------------------------------------

async function loadData() {
    try {
        const res = await fetch('/data/pose_clusters.csv');
        if (res.ok) {
            const text = await res.text();
            POSE_CLUSTERS = { clusters: parsePoseClustersCSV(text) };
        }
        const resPoses = await fetch('/data/normalized_poses.csv');
        if (resPoses.ok) {
            const text = await resPoses.text();
            NORMALIZED_POSES = parseNormalizedPosesCSV(text);
        }
    } catch (err) {
        console.error('Error loading pose data:', err);
    }
}

/**
 * Get the cluster match for a pose object (used by timeline and gallery).
 * Converts raw MediaPipe landmarks → normalized keypoints → closest cluster.
 */
function getMatchForPose(pose) {
    if (!pose?.landmarks) return null;
    const kp = landmarksToNormalizedKeypoints(pose.landmarks);
    return findClosestPoses(NORMALIZED_POSES, POSE_CLUSTERS?.clusters, kp);
}

// -----------------------------------------------------------------------------
// COORDINATE CONVERSION
// -----------------------------------------------------------------------------

/**
 * Convert raw MediaPipe landmarks to canvas pixel coordinates.
 * This is for DRAWING the skeleton overlay on the video canvas.
 * Also derives neck (midpoint of shoulders) and midhip (midpoint of hips)
 * since MediaPipe doesn't output these directly.
 */
function landmarksToCanvasKeypoints(landmarks, canvas, videoW, videoH) {
    const scale = Math.min(canvas.width / videoW, canvas.height / videoH);
    const drawW = videoW * scale;
    const drawH = videoH * scale;
    const offsetX = (canvas.width - drawW) / 2;
    const offsetY = (canvas.height - drawH) / 2;

    const kp = {};
    for (const name in LANDMARK_MAP) {
        const lm = landmarks[LANDMARK_MAP[name]];
        if (lm && (lm.visibility ?? 1) >= MIN_VISIBILITY) {
            kp[name] = { x: lm.x * drawW + offsetX, y: lm.y * drawH + offsetY };
        }
    }
    // Derive neck and midhip (not in MediaPipe's 33 landmarks)
    if (kp.lshoulder && kp.rshoulder) {
        kp.neck = { x: (kp.lshoulder.x + kp.rshoulder.x) / 2, y: (kp.lshoulder.y + kp.rshoulder.y) / 2 };
    }
    if (kp.lhip && kp.rhip) {
        kp.midhip = { x: (kp.lhip.x + kp.rhip.x) / 2, y: (kp.lhip.y + kp.rhip.y) / 2 };
    }
    return kp;
}

const MIN_VISIBILITY = 0.5;

/**
 * Convert raw MediaPipe landmarks to normalized [0,1] keypoints.
 * This is for CLUSTER MATCHING (not drawing). Uses MediaPipe's native
 * normalized coordinates (0-1 range), no pixel conversion needed.
 */
function landmarksToNormalizedKeypoints(landmarks) {
    const kp = {};
    for (const name in LANDMARK_MAP) {
        const lm = landmarks[LANDMARK_MAP[name]];
        if (lm) {
            const vis = lm.visibility ?? 1;
            if (vis >= MIN_VISIBILITY) kp[name] = { x: lm.x, y: lm.y };
        }
    }
    // Derive neck and midhip
    if (kp.lshoulder && kp.rshoulder) {
        kp.neck = { x: (kp.lshoulder.x + kp.rshoulder.x) / 2, y: (kp.lshoulder.y + kp.rshoulder.y) / 2 };
    }
    if (kp.lhip && kp.rhip) {
        kp.midhip = { x: (kp.lhip.x + kp.rhip.x) / 2, y: (kp.lhip.y + kp.rhip.y) / 2 };
    }
    return kp;
}

// -----------------------------------------------------------------------------
// DRAWING
// -----------------------------------------------------------------------------

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

function drawSkeleton(ctx, kp) {
    if (!kp) return;
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

function formatTime(s) {
    return `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, '0')}`;
}

// -----------------------------------------------------------------------------
// UI UPDATES
// -----------------------------------------------------------------------------

function updatePatternPanel(keypoints) {
    const container = $('patternContent');
    if (!container) return;
    if (!keypoints || !NORMALIZED_POSES || !POSE_CLUSTERS?.clusters) {
        container.innerHTML = '<p class="placeholder-text">Analyzing pose...</p>';
        return;
    }
    const clusters = POSE_CLUSTERS?.clusters;
    const match = findClosestPoses(NORMALIZED_POSES, clusters, keypoints);
    const topMatches = getTopMatchPoses(NORMALIZED_POSES, clusters, keypoints);
    if (!match) {
        container.innerHTML = '<p class="placeholder-text">No pose match</p>';
        return;
    }

    const primaryCluster = topMatches[0].cluster;
    const primaryPose = topMatches[0].pose;
    const label = match.character === 'masculine' ? 'Masculine-associated' : match.character === 'feminine' ? 'Feminine-associated' : 'Neutral';
    const malePct = match.malePercent.toFixed(0);
    const femalePct = match.femalePercent.toFixed(0);

    container.innerHTML = `
        <div class="match-sections">
            <div class="match-section">
                <h4 class="match-section-title">Best Match Prediction</h4>
                <div class="result-label">${label}</div>
                <div class="result-bars">
                    <div class="result-bar"><span>Male</span> <span>${malePct}%</span></div>
                    <div class="bar-track"><div class="bar-segment male" style="width:${malePct}%"></div><div class="bar-segment female" style="width:${femalePct}%"></div></div>
                    <div class="result-bar"><span>Female</span> <span>${femalePct}%</span></div>
                </div>
            </div>
        </div>
        <div class="matched-categories">
            <h4>Matched Pose Instance:</h4>
            <div class="match-grid" style="display: flex; gap: 1rem; align-items: stretch;">
                <div class="match-card" style="flex: 1; min-width: 0; background-color: var(--surface0); display: flex; flex-direction: column; align-items: center; padding: 0.5rem; border-radius: 8px;">
                    <canvas id="matchCanvasProto" width="80" height="100"></canvas>
                    <div class="match-info" style="text-align: center; margin-top: 0.5rem;">
                        <span style="display: block; font-weight: 600; font-size: 1.1em;">#${primaryCluster.id} (Prototype)</span>
                        <span class="match-stats" style="display: block;">${primaryCluster.malePercent.toFixed(0)}%M / ${primaryCluster.femalePercent.toFixed(0)}%F</span>
                    </div>
                </div>
                <div class="match-card" style="flex: 1; min-width: 0; background-color: var(--surface0); display: flex; flex-direction: column; align-items: center; padding: 0.5rem; border-radius: 8px;">
                    <canvas id="matchCanvasPose" width="80" height="100"></canvas>
                    <div class="match-info" style="text-align: center; margin-top: 0.5rem;">
                        <span style="display: block; font-weight: 600; font-size: 1.1em;">Dataset Pose</span>
                        <span class="match-stats" style="display: block;">Labeled: ${primaryPose.gender}</span>
                    </div>
                </div>
            </div>
        </div>
    `;

    setTimeout(() => {
        const canvasProto = $('matchCanvasProto');
        const canvasPose = $('matchCanvasPose');
        if (canvasProto && primaryCluster.prototype) {
            drawMiniSkeleton(canvasProto, primaryCluster.prototype, getGenderColor(primaryCluster));
        }
        if (canvasPose && primaryPose.keypoints) {
            drawMiniSkeleton(canvasPose, primaryPose.keypoints, getGenderColor(primaryCluster));
        }
    }, 10);
}

function updateMatchingClusters() {
    const container = $('matchingClusters');
    if (!container) return;
    container.innerHTML = getHowItWorksHTML();
}



function showPrediction() {
    const card = $('summaryCard');
    const content = $('summaryContent');
    if (extractedPoses.length === 0 || !card || !content) return;

    let mascCount = 0, femCount = 0, neutCount = 0;

    extractedPoses.forEach(p => {
        const m = getMatchForPose(p);
        const gc = m?.character || 'neutral';
        if (gc === 'masculine') mascCount++;
        else if (gc === 'feminine') femCount++;
        else neutCount++;
    });

    const total = extractedPoses.length;

    card.style.display = 'block';
    content.innerHTML = `
        <div class="summary-header">
            <h3 style="margin:0; color:#cdd6f4;">Detected Poses: ${total}</h3>
            <p style="font-size:0.8rem; color:#6c7086; margin:4px 0 12px 0;">Extracted every 0.5 seconds</p>
        </div>
        <div class="summary-breakdown">
            <div class="breakdown-item" style="border-left: 3px solid #89b4fa; padding-left: 12px; margin-bottom: 8px;">
                <div style="font-size: 1.25rem; font-weight: bold; color: #89b4fa;">${mascCount}</div>
                <div style="font-size: 0.8rem; color: #6c7086;">Masculine-associated</div>
            </div>
            <div class="breakdown-item" style="border-left: 3px solid #f5c2e7; padding-left: 12px; margin-bottom: 8px;">
                <div style="font-size: 1.25rem; font-weight: bold; color: #f5c2e7;">${femCount}</div>
                <div style="font-size: 0.8rem; color: #6c7086;">Feminine-associated</div>
            </div>
            <div class="breakdown-item" style="border-left: 3px solid #a6e3a1; padding-left: 12px; margin-bottom: 8px;">
                <div style="font-size: 1.25rem; font-weight: bold; color: #a6e3a1;">${neutCount}</div>
                <div style="font-size: 0.8rem; color: #6c7086;">Neutral</div>
            </div>
        </div>
    `;
}

function renderPosesGallery() {
    const container = $('posesContainer');
    if (!container) return;

    container.innerHTML = extractedPoses.map((p, i) => {
        const m = getMatchForPose(p);
        const gc = m?.character || 'neutral';
        const label = gc === 'masculine' ? 'Masc' : gc === 'feminine' ? 'Fem' : 'Neut';
        const color = getGenderColor({ character: gc });
        const frameSrc = p.frameDataUrl || '';
        return `
            <div class="pose-thumb ${i === currentPoseIndex ? 'active' : ''}" data-index="${i}">
                <div class="pose-thumb-img">
                    ${frameSrc ? `<img src="${frameSrc}" alt="Frame ${i}" width="100" height="100" />` : `<canvas id="pose${i}" width="100" height="100"></canvas>`}
                </div>
                <span class="pose-time">${formatTime(p.time)}</span>
                <span class="pose-label" style="color:${color}">${label}</span>
            </div>
        `;
    }).join('');

    setTimeout(() => {
        extractedPoses.forEach((p, i) => {
            if (!p.frameDataUrl && p.normalizedKeypoints) {
                const canvas = container.querySelector(`#pose${i}`);
                const m = getMatchForPose(p);
                const gc = m?.character || 'neutral';
                const color = getGenderColor({ character: gc });
                if (canvas) drawMiniSkeleton(canvas, p.normalizedKeypoints, color);
            }
        });
    }, 10);

    container.querySelectorAll('.pose-thumb').forEach(el => {
        el.onclick = () => selectPose(parseInt(el.dataset.index));
    });

    $('poseCounter').textContent = `${currentPoseIndex + 1} / ${extractedPoses.length}`;
}

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
        video.onseeked = null;
        drawVideoFrame(ctx, video);
        if (pose.landmarks) {
            drawSkeleton(ctx, landmarksToCanvasKeypoints(pose.landmarks, canvas, video.videoWidth, video.videoHeight));
        }
        updatePlaybackUI();
    };

    if (pose.landmarks) {
        const kp = landmarksToNormalizedKeypoints(pose.landmarks);
        updatePatternPanel(kp);
    }

    // Update timeline highlight
    drawTimeline($('timelineCanvas'), extractedPoses, getMatchForPose, index);

    const thumb = document.querySelector(`.pose-thumb[data-index="${index}"]`);
    if (thumb) thumb.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
}

// -----------------------------------------------------------------------------
// MEDIAPIPE
// -----------------------------------------------------------------------------

function handlePoseResults(results) {
    if (pendingCallback) {
        const cb = pendingCallback;
        pendingCallback = null;
        cb(results);
    }
}

async function initMediaPipe() {
    return new Promise((resolve) => {
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
        const testCanvas = document.createElement('canvas');
        testCanvas.width = 100;
        testCanvas.height = 100;
        testCanvas.getContext('2d').fillRect(0, 0, 100, 100);
        pendingCallback = () => { mediapipeReady = true; resolve(); };
        mediapipePose.send({ image: testCanvas }).catch(() => { mediapipeReady = true; resolve(); });
        setTimeout(() => { if (!mediapipeReady) { mediapipeReady = true; resolve(); } }, 8000);
    });
}

async function detectPose(videoElement) {
    if (isCancelled || !videoElement.videoWidth || !videoElement.videoHeight) return null;

    return new Promise((resolve) => {
        const timeout = setTimeout(() => { pendingCallback = null; resolve(null); }, 5000);
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

// -----------------------------------------------------------------------------
// VIDEO PROCESSING
// -----------------------------------------------------------------------------

async function processVideoInBackground(video) {
    isProcessing = true;
    isCancelled = false;
    extractedPoses = [];

    const canvas = $('processingCanvas');
    const ctx = canvas.getContext('2d');

    if (video.readyState < 2) {
        await new Promise(resolve => { video.onloadeddata = resolve; setTimeout(resolve, 2000); });
    }

    const startTime = 0;
    const endTime = video.duration;
    const totalFrames = Math.floor((endTime - startTime) / FRAME_INTERVAL) + 1;
    $('processingText').textContent = 'Extracting poses...';

    for (let i = 0; i < totalFrames && isProcessing && !isCancelled; i++) {
        const time = startTime + (i * FRAME_INTERVAL);

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

            const match = findClosestPoses(NORMALIZED_POSES, POSE_CLUSTERS?.clusters, kp);
            drawSkeleton(ctx, landmarksToCanvasKeypoints(landmarks, canvas, video.videoWidth, video.videoHeight));
            const frameDataUrl = canvas.toDataURL('image/png');

            extractedPoses.push({
                time, frameIndex: i, landmarks,
                normalizedKeypoints: normalizeKeypoints(kp),
                cluster: match?.cluster || null,
                character: match ? match.character : 'neutral',
                malePercent: match?.malePercent || 0,
                femalePercent: match?.femalePercent || 0,
                frameDataUrl,
                match
            });
        }

        const progress = ((i + 1) / totalFrames) * 100;
        $('progressFill').style.width = `${progress}%`;
        $('progressText').textContent = `${progress.toFixed(0)}%`;
        $('processingText').textContent = `Extracting poses... (${extractedPoses.length} found)`;
    }

    isProcessing = false;
    return !isCancelled && extractedPoses.length > 0;
}

// -----------------------------------------------------------------------------
// VIDEO PLAYBACK
// -----------------------------------------------------------------------------

function updatePlaybackUI() {
    const video = $('videoElement');
    if (!video?.duration) return;
    $('playbackFill').style.width = `${(video.currentTime / video.duration) * 100}%`;
    $('timeDisplay').textContent = `${formatTime(video.currentTime)} / ${formatTime(video.duration)}`;
}

function getLandmarksForCurrentFrame() {
    if (lastPlaybackLandmarks) return lastPlaybackLandmarks;
    if (extractedPoses.length === 0) return null;
    const video = $('videoElement');
    let closest = null, best = Infinity;
    for (const p of extractedPoses) {
        const d = Math.abs(p.time - video.currentTime);
        if (d < best) { best = d; closest = p; }
    }
    return closest?.landmarks || null;
}

function drawCurrentFrameWithSkeleton() {
    const video = $('videoElement');
    const canvas = $('playbackCanvas');
    const ctx = canvas.getContext('2d');
    if (!video.src || !canvas) return;
    drawVideoFrame(ctx, video);
    const landmarks = getLandmarksForCurrentFrame();
    if (landmarks) {
        drawSkeleton(ctx, landmarksToCanvasKeypoints(landmarks, canvas, video.videoWidth, video.videoHeight));
    }
}

function startPlayback() {
    const video = $('videoElement');
    const canvas = $('playbackCanvas');
    const ctx = canvas.getContext('2d');

    if (!video.src || extractedPoses.length === 0) return;

    lastPlaybackLandmarks = null;
    lastPlaybackDetectionTime = 0;

    const doPlay = () => {
        isPlaying = true;
        $('playBtn').textContent = '⏸ Pause';
        video.play();
        function render() {
            if (!isPlaying || video.paused || video.ended) { stopPlayback(); return; }
            drawVideoFrame(ctx, video);
            const landmarks = getLandmarksForCurrentFrame();
            if (landmarks) {
                drawSkeleton(ctx, landmarksToCanvasKeypoints(landmarks, canvas, video.videoWidth, video.videoHeight));
            }
            updatePlaybackUI();
            const now = performance.now();
            if (!playbackDetectionPending && (now - lastPlaybackDetectionTime) >= PLAYBACK_DETECTION_INTERVAL && video.videoWidth) {
                playbackDetectionPending = true;
                lastPlaybackDetectionTime = now;
                const frame = document.createElement('canvas');
                frame.width = video.videoWidth;
                frame.height = video.videoHeight;
                frame.getContext('2d').drawImage(video, 0, 0);
                pendingCallback = (results) => {
                    playbackDetectionPending = false;
                    if (results?.poseLandmarks) {
                        lastPlaybackLandmarks = results.poseLandmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z, visibility: lm.visibility }));
                    }
                };
                mediapipePose.send({ image: frame }).catch(() => { playbackDetectionPending = false; pendingCallback = null; });
            }
            animationFrameId = requestAnimationFrame(render);
        }
        render();
    };

    if (video.ended) {
        video.currentTime = 0;
        video.onseeked = () => { video.onseeked = null; doPlay(); };
    } else {
        doPlay();
    }
}

function stopPlayback() {
    isPlaying = false;
    lastPlaybackLandmarks = null;
    playbackDetectionPending = false;
    $('videoElement').pause();
    if (animationFrameId) { cancelAnimationFrame(animationFrameId); animationFrameId = null; }
    $('playBtn').textContent = '▶ Play';
    const video = $('videoElement');
    let closestIdx = 0, best = Infinity;
    for (let i = 0; i < extractedPoses.length; i++) {
        const d = Math.abs(extractedPoses[i].time - video.currentTime);
        if (d < best) { best = d; closestIdx = i; }
    }
    currentPoseIndex = closestIdx;
    document.querySelectorAll('.pose-thumb').forEach((el, i) => el.classList.toggle('active', i === closestIdx));
    $('poseCounter').textContent = `${closestIdx + 1} / ${extractedPoses.length}`;
    const pose = extractedPoses[closestIdx];
    if (pose?.landmarks) {
        const kp = landmarksToNormalizedKeypoints(pose.landmarks);
        updatePatternPanel(kp);
    }
    const thumb = document.querySelector(`.pose-thumb[data-index="${closestIdx}"]`);
    if (thumb) thumb.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    drawCurrentFrameWithSkeleton();
}

function togglePlayback() {
    if (isPlaying) stopPlayback(); else startPlayback();
}

// -----------------------------------------------------------------------------
// UPLOAD & FLOW
// -----------------------------------------------------------------------------

async function handleVideoUpload(file) {
    isCancelled = false;
    isProcessing = false;
    isPlaying = false;
    pendingCallback = null;
    lastPlaybackLandmarks = null;
    playbackDetectionPending = false;
    extractedPoses = [];
    currentPoseIndex = 0;

    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    $('uploadPlaceholder').style.display = 'none';
    $('processingSection').style.display = 'block';
    $('resultsSection').style.display = 'none';
    $('summaryCard').style.display = 'none';
    $('timelineCard').style.display = 'none';
    $('clearBtn').style.display = 'inline-block';

    $('progressFill').style.width = '0%';
    $('progressText').textContent = '0%';
    $('processingText').textContent = 'Loading video...';

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

    $('processingText').textContent = 'Loading model...';
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

    const startTime = extractedPoses[0]?.time ?? 0;
    video.currentTime = startTime;
    video.onseeked = () => {
        video.onseeked = null;
        drawVideoFrame(ctx, video);
        if (extractedPoses[0]?.landmarks) {
            drawSkeleton(ctx, landmarksToCanvasKeypoints(extractedPoses[0].landmarks, canvas, video.videoWidth, video.videoHeight));
        }
        updatePlaybackUI();
    };
    updatePlaybackUI();
    currentPoseIndex = 0;
    renderPosesGallery();
    showPrediction();

    if (extractedPoses[0]?.landmarks) {
        const kp = landmarksToNormalizedKeypoints(extractedPoses[0].landmarks);
        updatePatternPanel(kp);
    }
    updateMatchingClusters();

    // Timeline chart
    $('timelineCard').style.display = 'block';
    drawTimeline($('timelineCanvas'), extractedPoses, getMatchForPose, 0);
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
    $('summaryCard').style.display = 'none';
    $('timelineCard').style.display = 'none';
    $('clearBtn').style.display = 'none';
    $('videoInput').value = '';

    updatePatternPanel(null);
    updateMatchingClusters();

    setTimeout(() => { isCancelled = false; }, 100);
}

// -----------------------------------------------------------------------------
// INIT
// -----------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', async () => {
    await loadData();
    await initMediaPipe();
    updateMatchingClusters();

    $('uploadBtn').onclick = () => $('videoInput').click();
    $('videoInput').onchange = (e) => { if (e.target.files[0]) handleVideoUpload(e.target.files[0]); };
    
    // Simple Example Selection
    (async () => {
        try {
            const res = await fetch('/api/list?type=videos');
            const files = await res.json();
            if (files?.length) {
                const select = $('exampleSelect');
                select.style.display = 'inline-block';
                files.forEach(f => {
                    const opt = document.createElement('option');
                    opt.value = opt.textContent = f;
                    select.appendChild(opt);
                });
                select.onchange = async () => {
                    if (!select.value) return;
                    const res = await fetch(`/data/videos/${select.value}`);
                    const blob = await res.blob();
                    handleVideoUpload(new File([blob], select.value, { type: blob.type }));
                    select.value = "";
                };
            }
        } catch (e) { console.error('Examples failed', e); }
    })();

    $('cancelBtn').onclick = () => { isProcessing = false; resetToUpload(); };
    $('clearBtn').onclick = resetToUpload;
    $('playBtn').onclick = togglePlayback;
    $('prevPoseBtn').onclick = () => selectPose(currentPoseIndex - 1);
    $('nextPoseBtn').onclick = () => selectPose(currentPoseIndex + 1);
    if ($('exportCsvBtn')) $('exportCsvBtn').onclick = () => exportCSV(extractedPoses, getMatchForPose);

    document.addEventListener('keydown', (e) => {
        if (extractedPoses.length === 0) return;
        if (e.key === 'ArrowLeft') { e.preventDefault(); selectPose(currentPoseIndex - 1); }
        if (e.key === 'ArrowRight') { e.preventDefault(); selectPose(currentPoseIndex + 1); }
        if (e.key === ' ') { e.preventDefault(); togglePlayback(); }
    });

    $('playbackProgressBar').onclick = (e) => {
        const video = $('videoElement');
        if (!video.duration) return;
        const rect = e.currentTarget.getBoundingClientRect();
        video.currentTime = ((e.clientX - rect.left) / rect.width) * video.duration;
        video.onseeked = () => {
            video.onseeked = null;
            updatePlaybackUI();
            if (!isPlaying) drawCurrentFrameWithSkeleton();
        };
    };

    $('videoElement').onended = stopPlayback;
});
