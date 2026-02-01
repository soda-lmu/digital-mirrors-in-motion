/**
 * IMAGE POSE ANALYSIS
 * ====================
 * This script handles image upload, pose detection using MediaPipe,
 * and gender expression analysis based on pose clusters from research data.
 * 
 * MAIN FLOW:
 * 1. User uploads image → onchange handler
 * 2. Detect pose with MediaPipe → processImage()
 * 3. Match pose to database clusters → findMatchingClusters()
 * 4. Display results in analysis panels
 */

// ============================================================================
// GLOBAL STATE
// ============================================================================

let POSE_CLUSTERS = null;      // Database of pose clusters loaded from JSON
let mediapipePose = null;      // MediaPipe Pose instance
let mediapipeReady = false;    // Flag: is MediaPipe ready to use?
let pendingCallback = null;    // Callback for async pose detection

// ============================================================================
// CONFIGURATION
// ============================================================================

const MIN_CLUSTER_SIZE = 10;   // Ignore clusters with fewer poses
const GENDER_THRESHOLD = 60;   // >60% = gendered pose, else neutral

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

// Shorthand for document.getElementById
const $ = id => document.getElementById(id);

// ============================================================================
// GENDER CLASSIFICATION
// ============================================================================

/**
 * Classify a cluster as masculine, feminine, or neutral
 * Uses 60% threshold: >60% male → masculine, >60% female → feminine
 */
function getGenderClass(cluster) {
    if (cluster.malePercent >= GENDER_THRESHOLD) return 'masculine';
    if (cluster.femalePercent >= GENDER_THRESHOLD) return 'feminine';
    return 'neutral';
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

/** Find closest matching clusters for a pose */
function findMatchingClusters(keypoints, count = 4) {
    if (!POSE_CLUSTERS?.clusters || !keypoints) return [];
    const normalizedPose = normalizeKeypoints(keypoints);
    if (!normalizedPose) return [];
    
    return POSE_CLUSTERS.clusters
        .filter(c => c.prototype)
        .map(cluster => ({ cluster, distance: poseDistance(normalizedPose, cluster.prototype) }))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, count)
        .map(s => s.cluster);
}

// ============================================================================
// DRAWING FUNCTIONS
// ============================================================================

/** Draw skeleton overlay on image */
function drawSkeleton(ctx, kp, img) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    // Draw image (centered, aspect ratio preserved)
    if (img) {
        const scale = Math.min(ctx.canvas.width / img.width, ctx.canvas.height / img.height);
        const w = img.width * scale, h = img.height * scale;
        ctx.drawImage(img, (ctx.canvas.width - w) / 2, (ctx.canvas.height - h) / 2, w, h);
    }
    
    if (!kp) return;
    
    // Draw bones (green lines)
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
    
    // Draw joints (red circles)
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

/** Draw mini skeleton for thumbnails */
function drawMiniSkeleton(canvas, pose, color = '#f9e2af') {
    if (!pose || !canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const scale = 25, offsetX = canvas.width / 2, offsetY = canvas.height / 2;
    const kp = {};
    for (const name in pose) {
        if (pose[name]?.x !== undefined) {
            kp[name] = { x: pose[name].x * scale + offsetX, y: pose[name].y * scale + offsetY };
        }
    }
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
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

/** Update status bar */
function setStatus(text, type = 'info') {
    const colors = { info: '#6c7086', success: '#a6e3a1', warning: '#f9e2af', error: '#f38ba8' };
    $('statusBar').innerHTML = `<span style="color: ${colors[type]};">${text}</span>`;
}

// ============================================================================
// UI UPDATE FUNCTIONS
// ============================================================================

/** Update "Pose Pattern" panel */
function updatePatternPanel(keypoints) {
    const container = $('patternContent');
    if (!keypoints) {
        container.innerHTML = '<p class="placeholder-text">Upload an image to see pattern analysis</p>';
        return;
    }
    
    const matches = findMatchingClusters(keypoints, 1);
    if (matches.length === 0) {
        container.innerHTML = '<p class="placeholder-text">No matching patterns found</p>';
        return;
    }
    
    const c = matches[0];
    const genderClass = getGenderClass(c);
    const color = getGenderColor(c);
    
    let label, description;
    if (genderClass === 'masculine') {
        label = 'Masculine';
        description = `This pose is predominantly used by males (${c.malePercent.toFixed(0)}%)`;
    } else if (genderClass === 'feminine') {
        label = 'Feminine';
        description = `This pose is predominantly used by females (${c.femalePercent.toFixed(0)}%)`;
    } else {
        label = 'Neutral / Mixed';
        description = 'This pose is used similarly by both genders';
    }
    
    container.innerHTML = `
        <div class="pattern-display">
            <div class="pattern-circle" style="background: conic-gradient(#89b4fa ${c.malePercent}%, #f5c2e7 0%);">
                <span class="pattern-label" style="color: ${color};">${label}</span>
            </div>
            <div class="pattern-bars">
                <div class="pattern-bar-item">
                    <span style="color: #89b4fa;">♂ Male</span>
                    <div class="pattern-bar"><div class="pattern-fill male" style="width: ${c.malePercent}%"></div></div>
                    <span>${c.malePercent.toFixed(0)}%</span>
                </div>
                <div class="pattern-bar-item">
                    <span style="color: #f5c2e7;">♀ Female</span>
                    <div class="pattern-bar"><div class="pattern-fill female" style="width: ${c.femalePercent}%"></div></div>
                    <span>${c.femalePercent.toFixed(0)}%</span>
                </div>
            </div>
            <p class="pattern-description">${description}</p>
            <p class="pattern-cluster-info">Closest match: Cluster #${c.id} (${c.size} similar poses)</p>
        </div>
    `;
}

/** Update "Matching Categories" panel */
function updateMatchingClusters(keypoints) {
    const container = $('matchingClusters');
    if (!keypoints) {
        container.innerHTML = '<p class="placeholder-text">Upload an image to find matching poses</p>';
        return;
    }
    
    const matches = findMatchingClusters(keypoints, 6);
    if (matches.length === 0) {
        container.innerHTML = '<p class="placeholder-text">No matching clusters found</p>';
        return;
    }
    
    container.innerHTML = '<div class="clusters-grid">' + matches.map((c, i) => {
        const color = getGenderColor(c);
        const symbol = getGenderClass(c) === 'masculine' ? '♂' : getGenderClass(c) === 'feminine' ? '♀' : '⚖';
        return `
            <div class="cluster-card" style="border-color:${color};">
                <canvas id="cluster${i}" width="80" height="100"></canvas>
                <div class="cluster-info">
                    <span class="cluster-id" style="color:${color};">${symbol} #${c.id}</span>
                    <span class="cluster-stats">${c.malePercent.toFixed(0)}%M / ${c.femalePercent.toFixed(0)}%F</span>
                </div>
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

/** Update "Media Literacy Analysis" panel */
function updateLiteracyPanel(keypoints) {
    const container = $('literacyOutput');
    if (!keypoints) {
        container.innerHTML = '<p class="placeholder-text">Upload an image to see analysis...</p>';
        return;
    }
    
    const matches = findMatchingClusters(keypoints, 1);
    if (matches.length === 0) {
        container.innerHTML = '<p class="placeholder-text">No matching patterns for analysis</p>';
        return;
    }
    
    const c = matches[0];
    const genderClass = getGenderClass(c);
    const genderDiff = Math.abs(c.malePercent - c.femalePercent);
    
    let typeLabel, typeClass, insight;
    
    if (genderClass === 'masculine') {
        typeLabel = 'MASCULINE CODED';
        typeClass = 'dominant';
        insight = `<p class="insight"><strong>💡 Insight:</strong> This pose is predominantly used by men in media (${c.malePercent.toFixed(0)}% male). Poses like this often convey power, confidence, or authority.</p>`;
    } else if (genderClass === 'feminine') {
        typeLabel = 'FEMININE CODED';
        typeClass = 'submissive';
        insight = `<p class="insight"><strong>💡 Insight:</strong> This pose is predominantly used by women in media (${c.femalePercent.toFixed(0)}% female). Such poses are often associated with grace, approachability, or elegance.</p>`;
    } else {
        typeLabel = 'GENDER NEUTRAL';
        typeClass = 'neutral';
        if (genderDiff < 10) {
            insight = `<p class="insight"><strong>💡 Insight:</strong> This is a truly neutral pose, used almost equally by both genders in media (${genderDiff.toFixed(0)}% difference).</p>`;
        } else {
            const leaning = c.malePercent > c.femalePercent ? 'masculine' : 'feminine';
            insight = `<p class="insight"><strong>💡 Insight:</strong> This pose leans slightly ${leaning} but doesn't strongly favor either gender (below ${GENDER_THRESHOLD}% threshold).</p>`;
        }
    }
    
    container.innerHTML = `
        <div class="pose-type ${typeClass}">${typeLabel}</div>
        <p>Gender distribution: <strong style="color:#89b4fa;">${c.malePercent.toFixed(0)}%</strong> male, <strong style="color:#f5c2e7;">${c.femalePercent.toFixed(0)}%</strong> female</p>
        <p><strong>Category:</strong> Cluster #${c.id} (${c.size} similar poses in dataset)</p>
        ${insight}
    `;
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

/** Convert MediaPipe landmarks to canvas keypoints */
function convertLandmarks(landmarks, canvasW, canvasH, imgW, imgH) {
    const scale = Math.min(canvasW / imgW, canvasH / imgH);
    const w = imgW * scale, h = imgH * scale;
    const offsetX = (canvasW - w) / 2, offsetY = (canvasH - h) / 2;
    
    // Map our keypoint names to MediaPipe indices
    const map = {
        nose: 0, lshoulder: 11, rshoulder: 12, lelbow: 13, relbow: 14,
        lwrist: 15, rwrist: 16, lhip: 23, rhip: 24, lknee: 25, rknee: 26,
        lankle: 27, rankle: 28
    };
    
    const kp = {};
    for (const name in map) {
        const lm = landmarks[map[name]];
        if (lm) kp[name] = { x: lm.x * w + offsetX, y: lm.y * h + offsetY };
    }
    
    // Calculate derived points
    if (kp.lshoulder && kp.rshoulder) {
        kp.neck = { x: (kp.lshoulder.x + kp.rshoulder.x) / 2, y: (kp.lshoulder.y + kp.rshoulder.y) / 2 };
    }
    if (kp.lhip && kp.rhip) {
        kp.midhip = { x: (kp.lhip.x + kp.rhip.x) / 2, y: (kp.lhip.y + kp.rhip.y) / 2 };
    }
    return kp;
}

/** Initialize MediaPipe Pose model */
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
        
        // Warm up with test image
        const testCanvas = document.createElement('canvas');
        testCanvas.width = 100;
        testCanvas.height = 100;
        testCanvas.getContext('2d').fillRect(0, 0, 100, 100);
        
        pendingCallback = () => { mediapipeReady = true; resolve(); };
        mediapipePose.send({ image: testCanvas }).catch(() => { mediapipeReady = true; resolve(); });
        setTimeout(() => { if (!mediapipeReady) { mediapipeReady = true; resolve(); } }, 8000);
    });
}

// ============================================================================
// IMAGE PROCESSING
// ============================================================================

/** Process uploaded image and detect pose */
async function processImage(img) {
    pendingCallback = null;
    
    if (!mediapipeReady) {
        setStatus('⏳ Initializing...', 'warning');
        await initMediaPipe();
    }
    setStatus('⏳ Analyzing pose...', 'warning');
    
    return new Promise((resolve) => {
        const timeout = setTimeout(() => {
            pendingCallback = null;
            setStatus('⚠️ Detection timed out.', 'warning');
            resolve(null);
        }, 10000);
        
        pendingCallback = (results) => {
            clearTimeout(timeout);
            if (!results.poseLandmarks) {
                setStatus('⚠️ No pose detected.', 'warning');
                resolve(null);
                return;
            }
            
            const canvas = $('previewCanvas');
            const kp = convertLandmarks(results.poseLandmarks, canvas.width, canvas.height, img.width, img.height);
            
            drawSkeleton(canvas.getContext('2d'), kp, img);
            updatePatternPanel(kp);
            updateMatchingClusters(kp);
            updateLiteracyPanel(kp);
            
            const matches = findMatchingClusters(kp, 1);
            if (matches.length > 0) {
                const gc = getGenderClass(matches[0]);
                const label = gc === 'neutral' ? 'Neutral' : gc.charAt(0).toUpperCase() + gc.slice(1);
                setStatus(`✓ ${label} pose detected (Cluster #${matches[0].id})`, 'success');
            } else {
                setStatus('✓ Pose detected!', 'success');
            }
            resolve(kp);
        };
        
        mediapipePose.send({ image: img }).catch(() => {
            clearTimeout(timeout);
            pendingCallback = null;
            setStatus('⚠️ Detection failed.', 'error');
            resolve(null);
        });
    });
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    await loadData();
    setStatus('⏳ Loading model...', 'warning');
    await initMediaPipe();
    setStatus('Ready! Upload an image.', 'info');
    
    // Upload button
    $('uploadBtn').onclick = () => $('imageInput').click();
    
    // Handle image selection
    $('imageInput').onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        $('uploadPlaceholder').style.display = 'none';
        $('previewArea').style.display = 'flex';
        $('clearBtn').style.display = 'inline-block';
        setStatus('⏳ Loading image...', 'warning');
        
        const img = new Image();
        img.onload = async () => {
            const canvas = $('previewCanvas');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw image centered
            const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
            const w = img.width * scale, h = img.height * scale;
            ctx.drawImage(img, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
            
            await processImage(img);
        };
        img.src = URL.createObjectURL(file);
    };
    
    // Clear button
    $('clearBtn').onclick = () => {
        pendingCallback = null;
        $('previewCanvas').getContext('2d').clearRect(0, 0, 500, 600);
        $('uploadPlaceholder').style.display = 'flex';
        $('previewArea').style.display = 'none';
        $('clearBtn').style.display = 'none';
        $('imageInput').value = '';
        updatePatternPanel(null);
        updateMatchingClusters(null);
        updateLiteracyPanel(null);
        setStatus('Ready! Upload an image.', 'info');
    };
});
