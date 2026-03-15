/**
 * Shared Pose Analysis Utilities
 * ==============================
 * Core library used by image.js and video.js.
 *
 * This file implements the client-side version of the same normalization
 * and matching pipeline used in cluster_poses.py (Python backend).
 *
 * Pipeline: Raw keypoints → Normalize (bounding box → origin → unit vector)
 *           → Euclidean distance to 150 cluster prototypes → Nearest match
 *
 * Based on: Tsolak & Kühne (2025), "Unveiling digital mirrors: Decoding
 * gendered body poses in Instagram imagery", CHB 163.
 */

/**
 * Skeleton bones for stick-figure drawing.
 * Each pair [start, end] is a line segment connecting two joints.
 * The 17-joint set matches the paper (Section 3.1): nose, neck, shoulders,
 * elbows, wrists, midhip, hips, knees, ankles, and big toes.
 */
export const BONES = [
    ['nose', 'neck'], ['neck', 'rshoulder'], ['neck', 'lshoulder'],
    ['rshoulder', 'relbow'], ['lshoulder', 'lelbow'],
    ['relbow', 'rwrist'], ['lelbow', 'lwrist'],
    ['neck', 'midhip'], ['midhip', 'rhip'], ['midhip', 'lhip'],
    ['rhip', 'rknee'], ['lhip', 'lknee'],
    ['rknee', 'rankle'], ['lknee', 'lankle'],
    ['rankle', 'rbigtoe'], ['lankle', 'bigtoe']
];

/**
 * Joint names in fixed order — MUST match cluster_poses.py JOINTS.
 * The order matters because we flatten (x, y) pairs into a 1-D vector:
 * [x_nose, x_neck, ..., y_nose, y_neck, ...]
 * If the order differs between frontend and backend, distance calculations
 * will be meaningless.
 */
export const KEYPOINT_NAMES = [
    'nose', 'neck', 'lshoulder', 'rshoulder', 'lelbow', 'relbow',
    'lwrist', 'rwrist', 'midhip', 'lhip', 'rhip', 'lknee', 'rknee',
    'lankle', 'rankle', 'bigtoe', 'rbigtoe'
];

/**
 * Normalize keypoints per paper Section 3.1.
 *
 * This is the JavaScript equivalent of normalize_poses() in cluster_poses.py.
 * Both MUST produce identical output for the same input, otherwise cluster
 * matching will be inaccurate.
 *
 * Steps:
 *   1. Collect x/y values for all detected joints (in KEYPOINT_NAMES order).
 *   2. Bounding-box shift: bottom-left = (x_min, y_max) becomes (0, 0).
 *      (In image coordinates y increases downward, so "bottom" = max y.)
 *   3. Flatten to a 1-D vector [x'_0, y'_0, x'_1, y'_1, ...].
 *   4. Normalize to unit length (L2 norm = 1).
 *
 * @param {Object} kp - {joint_name: {x, y}} for detected joints
 * @returns {Object|null} Normalized {joint_name: {x, y}} or null if invalid
 */
export function normalizeKeypoints(kp) {
    if (!kp || Object.keys(kp).length === 0) return null;

    // Step 1: collect raw coordinates
    const xs = [], ys = [];
    for (const name of KEYPOINT_NAMES) {
        if (kp[name]?.x !== undefined && kp[name]?.y !== undefined) {
            xs.push(kp[name].x);
            ys.push(kp[name].y);
        }
    }
    if (xs.length === 0) return null;

    // Step 2: bounding-box shift (bottom-left → origin)
    const xMin = Math.min(...xs), yMax = Math.max(...ys);

    // Step 3: flatten to vector [x0, x1, ..., y0, y1, ...]
    // Must match cluster_poses.py normalize_poses() logic.
    // We include all 17 joints in the vector; missing ones are 0 (at origin).
    const vecX = [], vecY = [];
    for (const name of KEYPOINT_NAMES) {
        if (kp[name]) {
            vecX.push(kp[name].x - xMin);
            vecY.push(kp[name].y - yMax);
        } else {
            vecX.push(0);
            vecY.push(0);
        }
    }
    const vec = [...vecX, ...vecY];

    // Step 4: unit-length normalization
    const norm = Math.hypot(...vec);
    if (norm < 1e-10) return null;

    const normalized = {};
    for (let i = 0; i < KEYPOINT_NAMES.length; i++) {
        const name = KEYPOINT_NAMES[i];
        if (kp[name]) {
            normalized[name] = {
                x: vec[i] / norm,
                y: vec[i + KEYPOINT_NAMES.length] / norm
            };
        }
    }
    return normalized;
}

/** Per-joint distances for debug/trace. Returns [{joint, distance}, ...] sorted by distance desc. */
export function getPerJointDistances(pose1, pose2) {
    if (!pose1 || !pose2) return [];
    const norm1 = normalizeKeypoints(pose1);
    const norm2 = normalizeKeypoints(pose2);
    if (!norm1 || !norm2) return [];
    const joints = [];
    for (const name of KEYPOINT_NAMES) {
        if (norm1[name] && norm2[name]) {
            const dx = norm1[name].x - norm2[name].x;
            const dy = norm1[name].y - norm2[name].y;
            joints.push({ joint: name, distance: Math.sqrt(dx * dx + dy * dy) });
        }
    }
    return joints.sort((a, b) => b.distance - a.distance);
}

/**
 * Euclidean distance between two normalized poses.
 * This is the core metric used for cluster matching (paper Section 3.3).
 * Sums squared differences over all joints present in BOTH poses.
 */
export function poseDistance(pose1, pose2) {
    if (!pose1 || !pose2) return Infinity;
    let sumSq = 0, count = 0;
    for (const name of KEYPOINT_NAMES) {
        if (pose1[name] && pose2[name]) {
            const dx = pose1[name].x - pose2[name].x;
            const dy = pose1[name].y - pose2[name].y;
            sumSq += dx * dx + dy * dy;
            count++;
        }
    }
    return count > 0 ? Math.sqrt(sumSq) : Infinity;
}

/** Return the best matching pose by distance, with similarity score (0–100) */
export function getTopMatchPoses(poses, clusters, keypoints) {
    if (!poses || !clusters || !keypoints) return [];
    const normalizedPose = normalizeKeypoints(keypoints);
    if (!normalizedPose) return [];

    let bestPose = null, bestDist = Infinity;
    for (let i = 0; i < poses.length; i++) {
        const distance = poseDistance(normalizedPose, poses[i].keypoints);
        if (distance < bestDist) {
            bestDist = distance;
            bestPose = poses[i];
        }
    }
    if (!bestPose) return [];

    const cluster = clusters.find(c => c.id === bestPose.cluster_id);
    return [{
        pose: bestPose,
        cluster,
        distance: bestDist,
        similarity: Math.round(100 / (1 + bestDist))
    }];
}


/**
 * Find the best-matching individual pose for a detected pose.
 * Matches against the entire dataset of individual poses and returns
 * the single closest pose and its parent cluster.
 *
 * @param {Array} poses      - The 15,000+ individual normalized poses
 * @param {Array} clusters   - The 150 cluster objects
 * @param {Object} keypoints - Raw keypoints {joint: {x, y}} (will be normalized)
 * @returns {Object|null}    - {matchedPoses, cluster, character, malePercent, femalePercent, distance}
 */
export function findClosestPoses(poses, clusters, keypoints) {
    if (!poses || !clusters || !keypoints) return null;
    const normalizedPose = normalizeKeypoints(keypoints);
    if (!normalizedPose) return null;

    let bestPose = null, bestDist = Infinity;
    for (let i = 0; i < poses.length; i++) {
        const distance = poseDistance(normalizedPose, poses[i].keypoints);
        if (distance < bestDist) {
            bestDist = distance;
            bestPose = poses[i];
        }
    }
    if (!bestPose) return null;

    const c = clusters.find(cl => cl.id === bestPose.cluster_id);
    return {
        matchedPoses: [bestPose],
        cluster: c,
        character: c.character || 'neutral',
        malePercent: c.malePercent,
        femalePercent: c.femalePercent,
        distance: bestDist
    };
}

/** Map cluster character to a display color (Catppuccin palette) */
export function getGenderColor(cluster) {
    const cls = cluster?.character || 'neutral';
    if (cls === 'masculine') return '#89b4fa';  // blue
    if (cls === 'feminine') return '#f5c2e7';   // pink
    return '#a6e3a1';                           // green
}

/**
 * Draw a mini stick-figure skeleton on a canvas element.
 * Used for cluster prototype thumbnails in the gallery/match cards.
 * Auto-scales and centers the pose within the canvas bounding box.
 */
export function drawMiniSkeleton(canvas, pose, color = '#f9e2af') {
    if (!pose || !canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Compute bounding box of all joints to scale the skeleton to fit
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const name in pose) {
        if (name === 'gender') continue; // skip metadata field
        if (pose[name]?.x !== undefined) {
            xMin = Math.min(xMin, pose[name].x);
            xMax = Math.max(xMax, pose[name].x);
            yMin = Math.min(yMin, pose[name].y);
            yMax = Math.max(yMax, pose[name].y);
        }
    }
    const rangeX = xMax - xMin || 1, rangeY = yMax - yMin || 1;
    const padding = 8;
    const scale = Math.min(
        (canvas.width - padding * 2) / rangeX,
        (canvas.height - padding * 2) / rangeY
    );
    const offsetX = (canvas.width - (xMin + xMax) * scale) / 2;
    const offsetY = (canvas.height - (yMin + yMax) * scale) / 2;

    // Transform joint coordinates to canvas pixels
    const kp = {};
    for (const name in pose) {
        if (name === 'gender') continue;
        if (pose[name] && typeof pose[name].x === 'number') {
            kp[name] = { x: pose[name].x * scale + offsetX, y: pose[name].y * scale + offsetY };
        }
    }

    // Draw bones (line segments)
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

    // Draw joints (small circles)
    ctx.fillStyle = color;
    for (const name in kp) {
        ctx.beginPath();
        ctx.arc(kp[name].x, kp[name].y, 2, 0, Math.PI * 2);
        ctx.fill();
    }
}

/**
 * Parse the pose clusters CSV data.
 * Reconstructs the cluster objects with prototype {joint: {x, y}} structure.
 */
export function parsePoseClustersCSV(csvText) {
    const lines = csvText.trim().replace(/\r/g, '').split('\n');
    const [headerLine, ...rows] = lines;
    const headers = headerLine.split(',');

    return rows.map(line => {
        const values = line.split(',');
        const row = Object.fromEntries(headers.map((h, i) => [h, values[i]]));

        const cluster = {
            id: +row.cluster_id,
            character: row.character,
            malePercent: +row.malePercent,
            femalePercent: +row.femalePercent,
            nonbinaryPercent: +row.nonbinaryPercent,
            significant: row.significant === '1',
            pValue: +row.pValue,
            size: +row.size,
            prototype: {}
        };

        KEYPOINT_NAMES.forEach(name => {
            const x = +row[`x_${name}`], y = +row[`y_${name}`];
            if (!isNaN(x)) cluster.prototype[name] = { x, y };
        });

        return cluster;
    });
}

/**
 * Parse the normalized poses CSV data.
 */
export function parseNormalizedPosesCSV(csvText) {
    const lines = csvText.trim().replace(/\r/g, '').split('\n');
    const [headerLine, ...rows] = lines;
    const headers = headerLine.split(',');

    return rows.map(line => {
        const values = line.split(',');
        const row = Object.fromEntries(headers.map((h, i) => [h, values[i]]));

        const pose = {
            id: +row.pose_id,
            gender: row.gender,
            cluster_id: +row.cluster_id,
            keypoints: {}
        };

        KEYPOINT_NAMES.forEach(name => {
            const x = +row[`x_${name}`], y = +row[`y_${name}`];
            if (!isNaN(x)) pose.keypoints[name] = { x, y };
        });

        return pose;
    });
}

/** Static HTML explaining the matching pipeline */
export function getHowItWorksHTML() {
    return `
        <div class="how-it-works">
            <h4>How pose matching works</h4>
            <ol>
                <li><strong>Normalize</strong> — Your pose is scaled to a standard size (bounding box, unit length).</li>
                <li><strong>Compare</strong> — We measure Euclidean distance to each of the 15,000+ individual poses in the dataset (17 joints).</li>
                <li><strong>Match</strong> — The detected pose is assigned to the parent cluster of the single nearest dataset pose.</li>
            </ol>
        </div>
    `;
}
