import React from 'react';
import clsx from 'clsx';
import { IModelShape } from '../GptModel';
import { SHARD_HEX } from './shardColors';
import { Admonition } from './Admonition';

// Data sourced from:
//  "Achieving Peak Inference Performance for Qwen 2.5 72B on an 8-GPU H200 Cluster" (2025)
//  vLLM benchmarks: Qwen2-72B-Instruct on A100 (BF16): vLLM 17.68 tok/s vs HF 7.45 tok/s
//  SGLang on H20: Qwen3-32B BF16 20.72 → AWQ-INT4 47.67 tok/s
//  AIConfigurator: aggregated TP=8 → 321.5 tok/s/GPU; disaggregated 4P+4D → 648.3 tok/s/GPU

const TP_OPTIONS = [1, 2, 4, 8];

// Estimate total model parameters from shape. Used for per-GPU weight readout.
function estimateParams(shape: IModelShape): number {
    let C = shape.C, A = shape.A;
    let nHeads = shape.nHeads, nKVHeads = shape.nKVHeads ?? nHeads;
    let ffnDim = shape.ffnDim ?? C * 4;
    let attn = A * C * (2 * nHeads + 2 * nKVHeads);   // Q, K, V, O weights
    let mlp  = 3 * C * ffnDim;                          // gate, up, down
    return shape.nBlocks * (attn + mlp) + 2 * shape.vocabSize * C; // + embed + lm_head
}

function fmtB(n: number): string {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    return n.toFixed(0);
}

function ShardDiagram({ tp }: { tp: number }) {
    let W = 272, H = 50, gap = 5, x0 = 2, busY = 41;
    let boxW = (W - x0 * 2 - gap * (tp - 1)) / tp;
    let boxH = 24;
    return <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ fontSize: 8 }}>
        {Array.from({ length: tp }, (_, i) => {
            let x = x0 + i * (boxW + gap);
            let hex = SHARD_HEX[i % SHARD_HEX.length];
            return <g key={i}>
                <rect x={x} y={4} width={boxW} height={boxH} rx={3}
                    fill={hex} fillOpacity={0.22} stroke={hex} strokeWidth={1} />
                <text x={x + boxW / 2} y={4 + boxH / 2 + 3} textAnchor="middle"
                    fontSize={tp > 4 ? 7 : 8} fill={hex} fontWeight="bold">
                    GPU{i}
                </text>
                <line x1={x + boxW / 2} y1={4 + boxH} x2={x + boxW / 2} y2={busY}
                    stroke="#94a3b8" strokeWidth={0.7} />
            </g>;
        })}
        {tp > 1
            ? <>
                <line x1={x0 + boxW / 2} y1={busY} x2={W - x0 - boxW / 2} y2={busY}
                    stroke="#0ea5e9" strokeWidth={1.4} />
                <text x={W / 2} y={busY + 8} textAnchor="middle" fontSize={6.5}
                    fill="#0369a1" fontWeight="bold">
                    all-reduce ×2 per layer (NVLink 4, ~900 GB/s)
                </text>
            </>
            : <text x={W / 2} y={busY + 6} textAnchor="middle" fontSize={7} fill="#64748b">
                single GPU — no communication overhead
            </text>}
    </svg>;
}

// Illustrative per-GPU decode speed based on H200 bandwidth / per-GPU weight bytes.
// Real benchmark (AIConfigurator, 8×H200): TP=8 aggregated ≈ 321 tok/s/GPU.
// Disaggregated 4P(TP=1)+4D(TP=2): 648 tok/s/GPU (+101%).
function decodeThroughputNote(tp: number, perGpuGb: number): string {
    // Single-GPU estimate: 3350 GB/s ÷ perGpuGb (weights per GPU, BF16)
    let estToksPerSec = 3350 / perGpuGb;
    // Measured at TP=8 aggregated: 321 tok/s, so scale illustratively
    let scaledEst = Math.round(estToksPerSec);
    return `~${scaledEst} tok/s per GPU (H200 BW ÷ weight bytes/GPU; benchmark: TP=8 agg. ≈ 321 tok/s/GPU)`;
}

export const TensorParallelPanel: React.FC<{
    shape: IModelShape;
    tp: number;
    onTpChange: (n: number) => void;
}> = ({ shape, tp, onTpChange }) => {
    let params     = estimateParams(shape);
    let nKVHeads   = shape.nKVHeads ?? shape.nHeads;
    let bytesFp16  = 2 * params;
    let perGpuParams = params / tp;
    let perGpuGb   = bytesFp16 / tp / 1e9;
    let qPerGpu    = shape.nHeads / tp;
    let kvPerGpu   = nKVHeads / tp;
    let ffnPerGpu  = (shape.ffnDim ?? shape.C * 4) / tp;
    let kvReplicated = tp > nKVHeads;

    let fp8AlignWarn = tp === 8; // Qwen FP8 + TP=8 known alignment issue

    let stat = (label: string, value: string, hint?: string) =>
        <div className="flex items-baseline justify-between gap-2 py-0.5">
            <span className="text-[11px] text-slate-500">{label}</span>
            <span className="text-[11px] font-semibold text-slate-700">
                {value}
                {hint && <span className="ml-1 font-normal text-slate-400">{hint}</span>}
            </span>
        </div>;

    return <div className="rounded-xl border border-slate-200 bg-white p-2.5 shadow-sm">
        <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-700">Tensor parallelism</span>
            <div className="flex gap-1">
                {TP_OPTIONS.map(n => (
                    <button
                        key={n}
                        onClick={() => onTpChange(n)}
                        className={clsx('rounded px-2 py-0.5 text-xs font-semibold',
                            tp === n ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-indigo-100')}
                    >TP={n}</button>
                ))}
            </div>
        </div>

        <div className="mt-2">
            <ShardDiagram tp={tp} />
        </div>

        {/* FP8 alignment warning */}
        {fp8AlignWarn && <div className="mt-2">
            <Admonition kind="warning" title="FP8 + TP=8">
                Block-wise FP8 needs the block size to divide each shard's output dimension — TP=8 often
                violates this for these layers, causing a load error. Fix: drop to TP=4 or TP=2 when using FP8.
            </Admonition>
        </div>}

        {/* Per-GPU stats */}
        <div className="mt-1 border-t border-slate-100 pt-1">
            {stat('Total params', fmtB(params), '≈ ' + (bytesFp16 / 1e9).toFixed(1) + ' GB FP16')}
            {stat('Params / GPU', fmtB(perGpuParams), '÷' + tp)}
            {stat('Weights / GPU', perGpuGb.toFixed(1) + ' GB', 'FP16 (÷2 for FP8)')}
            {stat('Q heads / GPU', qPerGpu.toFixed(0))}
            {stat('KV heads / GPU',
                kvReplicated ? `${nKVHeads} (replicated)` : kvPerGpu.toFixed(0),
                kvReplicated ? `TP>${nKVHeads} KV heads` : undefined)}
            {stat('MLP inner / GPU', fmtB(ffnPerGpu))}
        </div>

        {/* Decode throughput estimate */}
        <div className="mt-1 border-t border-slate-100 pt-1">
            <div className="text-[10px] text-slate-400 italic">{decodeThroughputNote(tp, perGpuGb)}</div>
        </div>

        {/* Disaggregated P/D callout */}
        <div className="mt-2">
            <Admonition kind="note" title="Disaggregated P/D · 8× H200">
                4 GPUs prefill (TP=1) + 4 GPUs decode (TP=2) → <span className="font-semibold">648 tok/s</span> vs
                aggregated TP=8 → <span className="font-semibold">321 tok/s</span> (+101%). Prefill is compute-bound
                (wants dense GEMM); decode is memory-bound (wants bandwidth) — separating them lets each run on its
                optimal config.
            </Admonition>
        </div>

        <p className="mt-1.5 text-[10px] leading-snug text-slate-500">
            {tp === 1
                ? 'TP=1: all weights on one GPU. Decode speed capped by one GPU\'s 3.35 TB/s HBM bandwidth.'
                : `TP=${tp}: per-GPU weight bytes ÷${tp}, all ${tp} GPUs read in parallel → up to ${tp}× faster decode (memory-bound). ` +
                  `Overhead: 2 all-reduces per layer × ${shape.nBlocks} layers = ${2 * shape.nBlocks} NVLink collectives per token.`}
        </p>
    </div>;
};
