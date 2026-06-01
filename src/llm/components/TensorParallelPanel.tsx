import React, { useState } from 'react';
import clsx from 'clsx';
import { IModelShape } from '../GptModel';

// Tensor-parallelism (TP) explorer. TP shards each weight matrix across N GPUs:
// attention heads split column-wise, the MLP split column-then-row, with an all-reduce
// after the attention output projection and after the MLP down projection (2 per layer).
// Per-GPU weights and KV cache shrink by N; aggregate memory bandwidth grows by N
// (so memory-bound decode gets ~N× faster, minus the all-reduce overhead).

const TP_OPTIONS = [1, 2, 4, 8];

function estimateParams(shape: IModelShape): number {
    let C = shape.C;
    let A = shape.A;
    let nHeads = shape.nHeads;
    let nKVHeads = shape.nKVHeads ?? nHeads;
    let ffnDim = shape.ffnDim ?? C * 4;
    // per layer: attention (Q + O over nHeads, K + V over nKVHeads) + SwiGLU MLP (gate/up/down)
    let attn = A * C * (2 * nHeads + 2 * nKVHeads);
    let mlp = 3 * C * ffnDim;
    let perLayer = attn + mlp;
    let embed = shape.vocabSize * C;
    let lmHead = shape.vocabSize * C; // assume untied embeddings
    return shape.nBlocks * perLayer + embed + lmHead;
}

function fmtB(n: number): string {
    if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    return n.toFixed(0);
}

function ShardDiagram({ tp }: { tp: number }) {
    let W = 272, H = 46;
    let gap = 6, x0 = 2, busY = 38;
    let boxW = (W - x0 * 2 - gap * (tp - 1)) / tp;
    let boxH = 24;
    let boxes = [];
    for (let i = 0; i < tp; i++) {
        let x = x0 + i * (boxW + gap);
        boxes.push(
            <g key={i}>
                <rect x={x} y={4} width={boxW} height={boxH} rx={3} fill="#e0e7ff" stroke="#6366f1" strokeWidth={0.8} />
                <text x={x + boxW / 2} y={4 + boxH / 2 + 3} textAnchor="middle" fontSize={tp > 4 ? 7 : 8} fill="#3730a3" fontWeight="bold">
                    GPU{i}
                </text>
                <line x1={x + boxW / 2} y1={4 + boxH} x2={x + boxW / 2} y2={busY} stroke="#94a3b8" strokeWidth={0.7} />
            </g>,
        );
    }
    return <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ fontSize: 8 }}>
        {boxes}
        {tp > 1
            ? <>
                <line x1={x0 + boxW / 2} y1={busY} x2={W - x0 - boxW / 2} y2={busY} stroke="#0ea5e9" strokeWidth={1.4} />
                <text x={W / 2} y={busY + 7} textAnchor="middle" fontSize={7} fill="#0369a1" fontWeight="bold">
                    all-reduce ×2 / layer (NVLink)
                </text>
            </>
            : <text x={W / 2} y={busY + 5} textAnchor="middle" fontSize={7} fill="#64748b">single GPU — no communication</text>}
    </svg>;
}

export const TensorParallelPanel: React.FC<{ shape: IModelShape }> = ({ shape }) => {
    let [tp, setTp] = useState(1);

    let params = estimateParams(shape);
    let nKVHeads = shape.nKVHeads ?? shape.nHeads;
    let bytesFp16 = 2 * params;                 // weights only
    let perGpuParams = params / tp;
    let perGpuBytes = bytesFp16 / tp;
    let qPerGpu = shape.nHeads / tp;
    let kvPerGpu = nKVHeads / tp;
    let ffnPerGpu = (shape.ffnDim ?? shape.C * 4) / tp;
    let kvReplicated = tp > nKVHeads;

    let stat = (label: string, value: string, hint?: string) =>
        <div className="flex items-baseline justify-between gap-2 py-0.5">
            <span className="text-[11px] text-slate-500">{label}</span>
            <span className="text-[11px] font-semibold text-slate-700">{value}{hint && <span className="ml-1 font-normal text-slate-400">{hint}</span>}</span>
        </div>;

    return <div className="rounded-md border border-slate-200 bg-white p-2">
        <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-700">Tensor parallelism</span>
            <div className="flex gap-1">
                {TP_OPTIONS.map(n => (
                    <button
                        key={n}
                        onClick={() => setTp(n)}
                        className={clsx('rounded px-2 py-0.5 text-xs font-semibold',
                            tp === n ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-indigo-100')}
                    >TP={n}</button>
                ))}
            </div>
        </div>

        <div className="mt-2">
            <ShardDiagram tp={tp} />
        </div>

        <div className="mt-1 border-t border-slate-100 pt-1">
            {stat('Total params', fmtB(params), '(FP16 ' + fmtB(bytesFp16) + 'B)')}
            {stat('Params / GPU', fmtB(perGpuParams), '÷' + tp)}
            {stat('Weights / GPU', (perGpuBytes / 1e9).toFixed(1) + ' GB', 'FP16')}
            {stat('Q heads / GPU', qPerGpu.toString())}
            {stat('KV heads / GPU', kvReplicated ? `${nKVHeads} (replicated)` : kvPerGpu.toString(),
                kvReplicated ? `TP>${nKVHeads} KV heads` : undefined)}
            {stat('MLP width / GPU', fmtB(ffnPerGpu))}
        </div>

        <p className="mt-1.5 text-[10px] leading-snug text-slate-500">
            {tp === 1
                ? 'TP=1: the whole model lives on one GPU. Decode speed is capped by that GPU’s memory bandwidth.'
                : `TP=${tp}: each GPU holds 1/${tp} of the weights and KV cache, and all ${tp} read their shard in parallel → up to ~${tp}× faster (memory-bound) decode. Cost: ${tp > nKVHeads ? 'KV heads must replicate (GQA has only ' + nKVHeads + '), and ' : ''}2 all-reduces per layer, whose overhead grows with TP and dominates at very small batch.`}
        </p>
    </div>;
};
