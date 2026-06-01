import React from 'react';
import { Regime, H200_RIDGE_AI } from './TransformerStages';

// H200 SXM roofline chart.
//   Peak BF16 : ~989 TFLOPS
//   HBM3e BW  : 3.35 TB/s
//   Ridge     : 989e12 / 3.35e12 ≈ 295 FLOP/byte
//
// Left of ridge  = memory-bound (throughput = BW × AI)
// Right of ridge = compute-bound (throughput = peak TFLOPS)
// A stage marker moves when the user hovers a stage, showing where it sits.

const AI_MIN  = 0.1;
const AI_MAX  = 2000;
const RIDGE   = H200_RIDGE_AI; // 295
const PERF_MIN = 0.001;

const xL = 42, xR = 270, yT = 12, yB = 116;

function log10(v: number) { return Math.log(v) / Math.LN10; }

function xPix(ai: number) {
    let t = (log10(Math.max(ai, AI_MIN)) - log10(AI_MIN)) / (log10(AI_MAX) - log10(AI_MIN));
    return xL + t * (xR - xL);
}
function perfOf(ai: number) { return Math.min(1, ai / RIDGE); }
function yPix(perf: number) {
    let t = (log10(Math.max(perf, PERF_MIN)) - log10(PERF_MIN)) / (0 - log10(PERF_MIN));
    return yB - t * (yB - yT);
}

export const RooflineChart: React.FC<{ regime: Regime; ai?: number; label?: string; comm?: boolean }> = ({ regime, ai, label, comm }) => {
    let markAi  = ai ?? (regime === 'decode' ? 1.0 : 480);
    let markX   = xPix(markAi);
    let markY   = yPix(perfOf(markAi));
    let isMem   = markAi < RIDGE;
    let ridgeX  = xPix(RIDGE);

    let markerColor = comm ? '#a855f7' : isMem ? '#2563eb' : '#ea580c';

    let p0 = `${xPix(AI_MIN)},${yPix(perfOf(AI_MIN))}`;
    let p1 = `${ridgeX},${yPix(1)}`;
    let p2 = `${xPix(AI_MAX)},${yPix(1)}`;

    // x-axis tick labels (FLOP/B)
    const ticks = [0.1, 1, 10, 100, 1000];

    return <svg viewBox="0 0 280 150" className="w-full" style={{ fontSize: 7.5 }}>
        {/* region shading */}
        <rect x={xL} y={yT} width={ridgeX - xL} height={yB - yT} fill="#dbeafe" fillOpacity={0.6} />
        <rect x={ridgeX} y={yT} width={xR - ridgeX} height={yB - yT} fill="#ffedd5" fillOpacity={0.6} />

        {/* axes */}
        <line x1={xL} y1={yT} x2={xL}  y2={yB}  stroke="#94a3b8" strokeWidth={0.8} />
        <line x1={xL} y1={yB} x2={xR}  y2={yB}  stroke="#94a3b8" strokeWidth={0.8} />

        {/* roofline */}
        <polyline points={`${p0} ${p1} ${p2}`} fill="none" stroke="#334155" strokeWidth={1.6} />

        {/* ridge marker */}
        <line x1={ridgeX} y1={yT} x2={ridgeX} y2={yB} stroke="#64748b" strokeWidth={0.7} strokeDasharray="2 2" />
        <text x={ridgeX} y={yT - 1} textAnchor="middle" fontSize={6.5} fill="#475569" fontWeight="bold">
            ~{RIDGE} FLOP/B
        </text>
        <text x={ridgeX} y={yT + 6} textAnchor="middle" fontSize={6} fill="#64748b">
            H200 ridge
        </text>

        {/* region labels */}
        <text x={(xL + ridgeX) / 2} y={yT + 10} textAnchor="middle" fill="#1d4ed8" fontWeight="bold" fontSize={7.5}>memory-bound</text>
        <text x={(ridgeX + xR) / 2} y={yT + 10} textAnchor="middle" fill="#c2410c" fontWeight="bold" fontSize={7.5}>compute-bound</text>

        {/* x-axis ticks */}
        {ticks.map(v => {
            let x = xPix(v);
            return <g key={v}>
                <line x1={x} y1={yB} x2={x} y2={yB + 2} stroke="#94a3b8" strokeWidth={0.6} />
                <text x={x} y={yB + 9} textAnchor="middle" fill="#64748b">{v >= 1 ? v.toString() : v.toFixed(1)}</text>
            </g>;
        })}

        {/* stage marker */}
        <circle cx={markX} cy={markY} r={4.5} fill={markerColor} stroke="#fff" strokeWidth={1.5} />
        <text x={markX} y={markY - 7} textAnchor="middle" fill="#0f172a" fontWeight="bold" fontSize={7}>
            {label ?? regime}
        </text>

        {/* axis labels */}
        <text x={(xL + xR) / 2} y={yB + 20} textAnchor="middle" fill="#64748b" fontSize={7}>
            arithmetic intensity (FLOP / byte) →
        </text>
        <text x={xL - 2} y={yT - 4} textAnchor="start" fill="#64748b" fontSize={7}>
            ↑ throughput (% of H200 peak)
        </text>
    </svg>;
};
