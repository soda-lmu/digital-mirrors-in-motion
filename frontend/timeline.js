/**
 * TIMELINE CHART
 * ==============
 * Draws a pose-over-time chart showing how gendered classification
 * shifts across extracted video frames. Each frame is a colored dot
 * on the corresponding band (masculine/feminine/neutral).
 *
 * Usage:
 *   import { drawTimeline } from './timeline.js';
 *   drawTimeline(canvas, extractedPoses, matchFn);
 */

// Colors matching pose-utils.js getGenderColor
const COLORS = {
    masculine: '#89b4fa',
    feminine: '#f5c2e7',
    neutral: '#a6e3a1'
};

const BG_COLORS = {
    masculine: 'rgba(137, 180, 250, 0.08)',
    feminine: 'rgba(245, 194, 231, 0.08)',
    neutral: 'rgba(166, 227, 161, 0.08)'
};

const BAND_LABELS = ['Masculine', 'Neutral', 'Feminine'];
const BAND_KEYS = ['masculine', 'neutral', 'feminine'];

/**
 * Draw the pose-over-time timeline chart.
 *
 * @param {HTMLCanvasElement} canvas  - Target canvas element
 * @param {Array} poses              - Array of extracted pose objects (from video.js)
 * @param {Function} getMatchFn      - Function(pose) => { character, malePercent, femalePercent }
 * @param {number} [activeIndex=-1]  - Currently selected pose index (highlighted)
 */
export function drawTimeline(canvas, poses, getMatchFn, activeIndex = -1) {
    if (!canvas || !poses || poses.length === 0) return;

    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const dpr = window.devicePixelRatio || 1;

    // High-DPI support
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    ctx.scale(dpr, dpr);
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;

    ctx.clearRect(0, 0, w, h);

    // Layout
    const MARGIN = { top: 10, right: 15, bottom: 30, left: 70 };
    const plotW = w - MARGIN.left - MARGIN.right;
    const plotH = h - MARGIN.top - MARGIN.bottom;
    const bandH = plotH / 3;

    // Background bands
    BAND_KEYS.forEach((key, i) => {
        const y = MARGIN.top + i * bandH;
        ctx.fillStyle = BG_COLORS[key];
        ctx.fillRect(MARGIN.left, y, plotW, bandH);

        // Band label
        ctx.fillStyle = COLORS[key];
        ctx.font = '11px "Segoe UI", system-ui, sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(BAND_LABELS[i], MARGIN.left - 8, y + bandH / 2);
    });

    // Separator lines
    ctx.strokeStyle = 'rgba(108, 112, 134, 0.2)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 3; i++) {
        const y = MARGIN.top + i * bandH;
        ctx.beginPath();
        ctx.moveTo(MARGIN.left, y);
        ctx.lineTo(MARGIN.left + plotW, y);
        ctx.stroke();
    }

    // Border
    ctx.strokeStyle = 'rgba(108, 112, 134, 0.3)';
    ctx.strokeRect(MARGIN.left, MARGIN.top, plotW, plotH);

    // Time range
    const minTime = poses[0].time;
    const maxTime = poses[poses.length - 1].time;
    const timeRange = maxTime - minTime || 1;

    // X-axis time labels
    ctx.fillStyle = '#6c7086';
    ctx.font = '10px "Segoe UI", system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    const numTicks = Math.min(poses.length, 10);
    const tickStep = Math.max(1, Math.floor(poses.length / numTicks));
    for (let i = 0; i < poses.length; i += tickStep) {
        const x = MARGIN.left + ((poses[i].time - minTime) / timeRange) * plotW;
        const label = formatTimeShort(poses[i].time);
        ctx.fillText(label, x, MARGIN.top + plotH + 6);

        // Tick mark
        ctx.strokeStyle = 'rgba(108, 112, 134, 0.2)';
        ctx.beginPath();
        ctx.moveTo(x, MARGIN.top + plotH);
        ctx.lineTo(x, MARGIN.top + plotH + 4);
        ctx.stroke();
    }

    // X-axis label
    ctx.fillStyle = '#6c7086';
    ctx.font = '10px "Segoe UI", system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Time (s)', MARGIN.left + plotW / 2, MARGIN.top + plotH + 20);

    // Plot dots
    const dotRadius = Math.max(3, Math.min(6, plotW / poses.length * 0.3));

    poses.forEach((pose, idx) => {
        const match = getMatchFn(pose);
        const character = match?.character || 'neutral';
        const bandIdx = BAND_KEYS.indexOf(character);
        if (bandIdx === -1) return;

        const x = MARGIN.left + ((pose.time - minTime) / timeRange) * plotW;
        const y = MARGIN.top + bandIdx * bandH + bandH / 2;

        // Connecting line to previous
        if (idx > 0) {
            const prevMatch = getMatchFn(poses[idx - 1]);
            const prevChar = prevMatch?.character || 'neutral';
            const prevBandIdx = BAND_KEYS.indexOf(prevChar);
            const prevX = MARGIN.left + ((poses[idx - 1].time - minTime) / timeRange) * plotW;
            const prevY = MARGIN.top + prevBandIdx * bandH + bandH / 2;

            ctx.strokeStyle = 'rgba(108, 112, 134, 0.15)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(prevX, prevY);
            ctx.lineTo(x, y);
            ctx.stroke();
        }

        // Dot
        ctx.beginPath();
        ctx.arc(x, y, idx === activeIndex ? dotRadius + 2 : dotRadius, 0, Math.PI * 2);
        ctx.fillStyle = COLORS[character];
        if (idx === activeIndex) {
            ctx.shadowColor = COLORS[character];
            ctx.shadowBlur = 8;
        }
        ctx.fill();
        ctx.shadowBlur = 0;

        // Active ring
        if (idx === activeIndex) {
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
}

function formatTimeShort(s) {
    return `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, '0')}`;
}

/**
 * Export extracted poses as CSV for paper analysis.
 *
 * @param {Array} poses       - Array of extracted pose objects
 * @param {Function} getMatchFn - Function(pose) => match result
 */
export function exportCSV(poses, getMatchFn) {
    if (!poses || poses.length === 0) return;

    const header = 'frame,time,cluster_id,character,male_pct,female_pct,distance';
    const rows = poses.map(p => {
        const m = getMatchFn(p);
        return [
            p.frameIndex,
            p.time.toFixed(2),
            m?.cluster?.id ?? '',
            m?.character || 'neutral',
            (m?.malePercent ?? 0).toFixed(1),
            (m?.femalePercent ?? 0).toFixed(1),
            (m?.distance ?? 0).toFixed(4)
        ].join(',');
    });

    const csv = [header, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `pose_analysis_${new Date().toISOString().slice(0, 19).replace(/[:-]/g, '')}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
}
