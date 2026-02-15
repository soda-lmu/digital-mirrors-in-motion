/**
 * Shared pose analysis utilities (Tsolak & Kühne, 2025)
 * =====================================================
 * Used by image.js and video.js for normalization, cluster matching, and display.
 * Methodology: "Unveiling digital mirrors: Decoding gendered body poses in Instagram imagery"
 * (Computers in Human Behavior 163, 2025).
 */

/** Skeleton bones for drawing — 17 joints per paper (nose, neck, limbs, ankles, big toes) */
export const BONES = [
    ['nose', 'neck'], ['neck', 'rshoulder'], ['neck', 'lshoulder'],
    ['rshoulder', 'relbow'], ['lshoulder', 'lelbow'],
    ['relbow', 'rwrist'], ['lelbow', 'lwrist'],
    ['neck', 'midhip'], ['midhip', 'rhip'], ['midhip', 'lhip'],
    ['rhip', 'rknee'], ['lhip', 'lknee'],
    ['rknee', 'rankle'], ['lknee', 'lankle'],
    ['rankle', 'rbigtoe'], ['lankle', 'bigtoe']
];

/** Joint names in fixed order — must match cluster_poses.py JOINTS */
export const KEYPOINT_NAMES = [
    'nose', 'neck', 'lshoulder', 'rshoulder', 'lelbow', 'relbow',
    'lwrist', 'rwrist', 'midhip', 'lhip', 'rhip', 'lknee', 'rknee',
    'lankle', 'rankle', 'bigtoe', 'rbigtoe'
];

/**
 * Normalize keypoints per paper Section 3.1.
 * Steps: (1) Bounding box, (2) Align bottom-left to (0,0), (3) Unit-length vector.
 * Returns {joint: {x,y}} or null if invalid.
 */
export function normalizeKeypoints(kp) {
    if (!kp || Object.keys(kp).length === 0) return null;
    const xs = [], ys = [];
    for (const name of KEYPOINT_NAMES) {
        if (kp[name]?.x !== undefined && kp[name]?.y !== undefined) {
            xs.push(kp[name].x);
            ys.push(kp[name].y);
        }
    }
    if (xs.length === 0) return null;
    const xMin = Math.min(...xs), yMax = Math.max(...ys);
    const vec = [];
    for (const name of KEYPOINT_NAMES) {
        if (kp[name]) {
            vec.push(kp[name].x - xMin, kp[name].y - yMax);
        }
    }
    const norm = Math.hypot(...vec);
    if (norm < 1e-10) return null;
    const normalized = {};
    let i = 0;
    for (const name of KEYPOINT_NAMES) {
        if (kp[name]) {
            normalized[name] = { x: vec[i] / norm, y: vec[i + 1] / norm };
            i += 2;
        }
    }
    return normalized;
}

/** Euclidean distance between two normalized poses (sum of squared differences over joints) */
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

/** Return top k clusters by distance, with similarity score (0–100) */
export function getTopMatches(clusters, keypoints, k = 5) {
    if (!clusters || !keypoints) return [];
    const normalizedPose = normalizeKeypoints(keypoints);
    if (!normalizedPose) return [];
    return clusters
        .filter(c => c.prototype)
        .map(cluster => ({ cluster, distance: poseDistance(normalizedPose, cluster.prototype) }))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, k)
        .map(({ cluster, distance }) => ({
            cluster,
            distance,
            similarity: Math.round(100 / (1 + distance))
        }));
}

/** Null model from dataset (paper Section 3.2) — matches cluster_poses.py */
const NULL_MALE = 0.4350;
const NULL_FEMALE = 0.5246;

/**
 * Find best-matching cluster for a pose.
 * k=1: use closest cluster's character (from multinomial test).
 * k>1: blend top k clusters by inverse-distance weight; classify via deviation from null
 *      (masculine if male_dev > female_dev and male_dev > 0; feminine analogously).
 */
export function findClosestCluster(clusters, keypoints, k = 5) {
    if (!clusters || !keypoints) return null;
    const normalizedPose = normalizeKeypoints(keypoints);
    if (!normalizedPose) return null;

    const withDist = clusters
        .filter(c => c.prototype)
        .map(cluster => ({ cluster, distance: poseDistance(normalizedPose, cluster.prototype) }))
        .sort((a, b) => a.distance - b.distance)
        .slice(0, Math.max(1, k));

    if (withDist.length === 0) return null;

    if (k === 1) {
        const c = withDist[0].cluster;
        return {
            cluster: c,
            character: c.character || 'neutral',
            malePercent: c.malePercent,
            femalePercent: c.femalePercent,
            distance: withDist[0].distance
        };
    }

    const weights = withDist.map(({ distance }) => 1 / (1 + distance));
    const sumW = weights.reduce((a, b) => a + b, 0);
    const normWeights = weights.map(w => w / sumW);
    let malePercent = 0, femalePercent = 0;
    for (let i = 0; i < withDist.length; i++) {
        malePercent += normWeights[i] * withDist[i].cluster.malePercent;
        femalePercent += normWeights[i] * withDist[i].cluster.femalePercent;
    }
    // Paper-aligned: deviation from null, compare which gender is more over-represented
    const maleDev = malePercent / 100 - NULL_MALE;
    const femaleDev = femalePercent / 100 - NULL_FEMALE;
    let character = 'neutral';
    if (maleDev > femaleDev && maleDev > 0) character = 'masculine';
    else if (femaleDev > maleDev && femaleDev > 0) character = 'feminine';

    return {
        cluster: withDist[0].cluster,
        character,
        malePercent,
        femalePercent,
        distance: withDist[0].distance
    };
}

/** Return hex color for character: masculine=blue, feminine=pink, neutral=green */
export function getGenderColor(cluster) {
    const cls = cluster?.character || 'neutral';
    if (cls === 'masculine') return '#89b4fa';
    if (cls === 'feminine') return '#f5c2e7';
    return '#a6e3a1';
}

/** Draw a mini skeleton on canvas, scaled to fit; color by character */
export function drawMiniSkeleton(canvas, pose, color = '#f9e2af') {
    if (!pose || !canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const name in pose) {
        if (name === 'gender') continue;
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

    const kp = {};
    for (const name in pose) {
        if (name === 'gender') continue;
        if (pose[name] && typeof pose[name].x === 'number') {
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

/** Static HTML explaining the matching pipeline */
export function getHowItWorksHTML() {
    return `
        <div class="how-it-works">
            <h4>How pose matching works</h4>
            <ol>
                <li><strong>Normalize</strong> — Your pose is scaled to a standard size (bounding box, unit length).</li>
                <li><strong>Compare</strong> — We measure Euclidean distance to each of 150 pose categories (17 joints).</li>
                <li><strong>Blend</strong> — Top k categories weighted by similarity. Closer = more influence.</li>
            </ol>
        </div>
    `;
}
