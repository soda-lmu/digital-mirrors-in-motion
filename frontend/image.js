/**
 * IMAGE POSE ANALYSIS
 * ===================
 * Single-image pose detection via MediaPipe; match to 150 pose clusters from
 * Tsolak & Kühne (2025). Shows gender-associated pattern (masculine/feminine/neutral).
 */

import {
    BONES,
    findClosestPoses,
    getTopMatchPoses,
    getGenderColor,
    drawMiniSkeleton,
    getHowItWorksHTML,
    parsePoseClustersCSV,
    parseNormalizedPosesCSV
} from './pose-utils.js';

// -----------------------------------------------------------------------------
// GLOBAL STATE
// -----------------------------------------------------------------------------

let POSE_CLUSTERS = null;      // Loaded from pose_clusters.csv
let NORMALIZED_POSES = null;   // Loaded from normalized_poses.csv
let mediapipePose = null;      // MediaPipe Pose instance
let mediapipeReady = false;
let pendingCallback = null;   // One-shot callback for async pose detection

const $ = id => document.getElementById(id);

// -----------------------------------------------------------------------------
// DATA LOADING
// -----------------------------------------------------------------------------

/** Load 150 pose clusters from CSV and normalized individual poses */
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

// -----------------------------------------------------------------------------
// DRAWING
// -----------------------------------------------------------------------------

/** Draw image + skeleton overlay on canvas (green bones, red joints) */
function drawSkeleton(ctx, kp, img) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    if (img) {
        const scale = Math.min(ctx.canvas.width / img.width, ctx.canvas.height / img.height);
        const w = img.width * scale, h = img.height * scale;
        ctx.drawImage(img, (ctx.canvas.width - w) / 2, (ctx.canvas.height - h) / 2, w, h);
    }
    if (!kp) return;
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    for (const [start, end] of BONES) {
        if (kp[start] && kp[end]) {
            ctx.beginPath();
            ctx.moveTo(kp[start].x, kp[start].y);
            ctx.lineTo(kp[end].x, kp[end].y);
            ctx.stroke();
        }
    }
    for (const name in kp) {
        const pt = kp[name];
        if (!pt?.x) continue;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 7, 0, Math.PI * 2);
        ctx.fillStyle = '#ff0000';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
}

// -----------------------------------------------------------------------------
// UI UPDATES
// -----------------------------------------------------------------------------

/** Render pattern panel: best match result and matched pose instance */
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

/** Show "How it works" static content */
function updateMatchingClusters() {
    const container = $('matchingClusters');
    if (!container) return;
    container.innerHTML = getHowItWorksHTML();
}

// -----------------------------------------------------------------------------
// MEDIAPIPE
// -----------------------------------------------------------------------------

/** Route MediaPipe results to the current pending callback */
function handlePoseResults(results) {
    if (pendingCallback) {
        const cb = pendingCallback;
        pendingCallback = null;
        cb(results);
    }
}

/** MediaPipe landmarks → canvas pixel coords; derive neck and midhip from shoulders/hips */
function convertLandmarks(landmarks, canvasW, canvasH, imgW, imgH) {
    const scale = Math.min(canvasW / imgW, canvasH / imgH);
    const w = imgW * scale, h = imgH * scale;
    const offsetX = (canvasW - w) / 2, offsetY = (canvasH - h) / 2;
    // MediaPipe landmark index → our 17-joint format.
    // Note: indices 29/30 are "left/right foot index" (closest to OpenPose big toe).
    // MediaPipe does NOT output neck or midhip — we derive them below.
    const map = {
        nose: 0, lshoulder: 11, rshoulder: 12, lelbow: 13, relbow: 14,
        lwrist: 15, rwrist: 16, lhip: 23, rhip: 24, lknee: 25, rknee: 26,
        lankle: 27, rankle: 28, bigtoe: 31, rbigtoe: 32
    };
    const MIN_VISIBILITY = 0.5;
    const kp = {};
    for (const name in map) {
        const lm = landmarks[map[name]];
        if (lm && (lm.visibility ?? 1) >= MIN_VISIBILITY) kp[name] = { x: lm.x * w + offsetX, y: lm.y * h + offsetY };
    }
    if (kp.lshoulder && kp.rshoulder) {
        kp.neck = { x: (kp.lshoulder.x + kp.rshoulder.x) / 2, y: (kp.lshoulder.y + kp.rshoulder.y) / 2 };
    }
    if (kp.lhip && kp.rhip) {
        kp.midhip = { x: (kp.lhip.x + kp.rhip.x) / 2, y: (kp.lhip.y + kp.rhip.y) / 2 };
    }
    return kp;
}

/** Initialize MediaPipe Pose (warm-up with test image) */
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

// -----------------------------------------------------------------------------
// IMAGE PROCESSING
// -----------------------------------------------------------------------------

/** Detect pose in image, draw skeleton, update pattern panel */
async function processImage(img) {
    pendingCallback = null;
    if (!mediapipeReady) await initMediaPipe();
    return new Promise((resolve) => {
        const timeout = setTimeout(() => { pendingCallback = null; resolve(null); }, 10000);
        pendingCallback = (results) => {
            clearTimeout(timeout);
            if (!results.poseLandmarks) { resolve(null); return; }
            const canvas = $('previewCanvas');
            const kp = convertLandmarks(results.poseLandmarks, canvas.width, canvas.height, img.width, img.height);
            drawSkeleton(canvas.getContext('2d'), kp, img);
            updatePatternPanel(kp);
            updateMatchingClusters();
            resolve(kp);
        };
        mediapipePose.send({ image: img }).catch(() => {
            clearTimeout(timeout);
            pendingCallback = null;
            resolve(null);
        });
    });
}

// -----------------------------------------------------------------------------
// INIT
// -----------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', async () => {
    await loadData();
    await initMediaPipe();
    updateMatchingClusters();
    $('uploadBtn').onclick = () => $('imageInput').click();
    async function handleImageUpload(file) {
        if (!file) return;
        $('uploadPlaceholder').style.display = 'none';
        $('previewArea').style.display = 'flex';
        $('clearBtn').style.display = 'inline-block';
        const img = new Image();
        img.onload = async () => {
            const canvas = $('previewCanvas');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
            const w = img.width * scale, h = img.height * scale;
            ctx.drawImage(img, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
            await processImage(img);
        };
        img.src = URL.createObjectURL(file);
    }

    $('imageInput').onchange = (e) => handleImageUpload(e.target.files[0]);

    // Simple Example Selection
    (async () => {
        try {
            const res = await fetch('/api/list?type=images');
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
                    const filename = select.value;
                    if (!filename) return;
                    const res = await fetch(`/data/images/${filename}`);
                    const blob = await res.blob();
                    await handleImageUpload(new File([blob], filename, { type: blob.type || 'image/png' }));
                    select.value = "";
                };
            }
        } catch (e) { console.error('Examples failed', e); }
    })();

    $('clearBtn').onclick = () => {
        pendingCallback = null;
        const canvas = $('previewCanvas');
        if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
        $('uploadPlaceholder').style.display = 'flex';
        $('previewArea').style.display = 'none';
        $('clearBtn').style.display = 'none';
        $('imageInput').value = '';
        updatePatternPanel(null);
        updateMatchingClusters();
    };
});
