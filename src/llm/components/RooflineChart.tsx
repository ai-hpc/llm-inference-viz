import React from 'react';
import { Regime } from './TransformerStages';

// A compact roofline chart: performance (log) vs arithmetic intensity (log, FLOP/byte).
// The roof is a memory-bandwidth diagonal that bends into a flat compute ceiling at the
// "ridge" point. Left of the ridge = memory-bound; right = compute-bound. A marker shows
// where the current stage/regime sits.

const AI_MIN = 0.1;
const AI_MAX = 1000;
const RIDGE = 200;       // illustrative ridge AI (FLOP/byte) for a modern GPU at FP16
const PERF_MIN = 0.001;  // bottom of the log perf axis (fraction of peak)

// plot area
const xL = 38, xR = 270, yT = 12, yB = 116;

function log10(v: number) { return Math.log(v) / Math.LN10; }

function xPix(ai: number) {
    let t = (log10(ai) - log10(AI_MIN)) / (log10(AI_MAX) - log10(AI_MIN));
    return xL + t * (xR - xL);
}
function perfOf(ai: number) { return Math.min(1, ai / RIDGE); }
function yPix(perf: number) {
    let t = (log10(Math.max(perf, PERF_MIN)) - log10(PERF_MIN)) / (0 - log10(PERF_MIN));
    return yB - t * (yB - yT);
}

export const RooflineChart: React.FC<{ regime: Regime; ai?: number; label?: string }> = ({ regime, ai, label }) => {
    // default marker per regime when nothing is hovered
    let markAi = ai ?? (regime === 'decode' ? 2 : 150);
    let markX = xPix(markAi);
    let markY = yPix(perfOf(markAi));
    let isMem = markAi < RIDGE;

    let ridgeX = xPix(RIDGE);

    // roofline polyline: diagonal up to the ridge, then flat
    let p0 = `${xPix(AI_MIN)},${yPix(perfOf(AI_MIN))}`;
    let p1 = `${ridgeX},${yPix(1)}`;
    let p2 = `${xPix(AI_MAX)},${yPix(1)}`;

    return <svg viewBox="0 0 280 140" className="w-full" style={{ fontSize: 8 }}>
        {/* region shading */}
        <rect x={xL} y={yT} width={ridgeX - xL} height={yB - yT} fill="#dbeafe" />
        <rect x={ridgeX} y={yT} width={xR - ridgeX} height={yB - yT} fill="#ffedd5" />

        {/* axes */}
        <line x1={xL} y1={yT} x2={xL} y2={yB} stroke="#94a3b8" strokeWidth={0.7} />
        <line x1={xL} y1={yB} x2={xR} y2={yB} stroke="#94a3b8" strokeWidth={0.7} />

        {/* roofline */}
        <polyline points={`${p0} ${p1} ${p2}`} fill="none" stroke="#475569" strokeWidth={1.4} />

        {/* ridge marker */}
        <line x1={ridgeX} y1={yT} x2={ridgeX} y2={yB} stroke="#9ca3af" strokeWidth={0.6} strokeDasharray="2 2" />

        {/* region labels */}
        <text x={(xL + ridgeX) / 2} y={yT + 9} textAnchor="middle" fill="#2563eb" fontWeight="bold">memory-bound</text>
        <text x={(ridgeX + xR) / 2} y={yT + 9} textAnchor="middle" fill="#ea580c" fontWeight="bold">compute-bound</text>

        {/* stage / regime marker */}
        <circle cx={markX} cy={markY} r={4} fill={isMem ? '#2563eb' : '#ea580c'} stroke="#fff" strokeWidth={1.2} />
        <text x={markX} y={markY - 6} textAnchor="middle" fill="#0f172a" fontWeight="bold">
            {label ?? regime}
        </text>

        {/* axis labels */}
        <text x={(xL + xR) / 2} y={yB + 14} textAnchor="middle" fill="#64748b">arithmetic intensity (FLOP / byte) →</text>
        <text x={xL - 2} y={yT - 3} textAnchor="start" fill="#64748b">↑ throughput (% of peak)</text>
    </svg>;
};
