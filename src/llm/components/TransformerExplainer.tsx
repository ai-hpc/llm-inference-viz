import React, { useState } from 'react';
import clsx from 'clsx';
import { useProgramState } from '../Sidebar';
import { TRANSFORMER_STAGES, Regime, Bound, effectivePoint, effectiveBound, fmtAi } from './TransformerStages';
import { RooflineChart } from './RooflineChart';
import { TensorParallelPanel } from './TensorParallelPanel';

// Left-hand 2D explanation of the dense decoder-only transformer forward pass.
// Hovering a stage highlights the matching blocks in the 3D model on the right
// (via state.display.hoveredStage, consumed by applyStageHighlight in Program.ts),
// expands the detailed description, and shows where the stage sits on the roofline.
// A Prefill/Decode toggle and the TP selector drive the compute-vs-memory readouts.

function IntensityBar({ c }: { c: number }) {
    let pct = Math.round(c * 100);
    return <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-slate-200" title={`${pct}% compute · ${100 - pct}% memory/comm`}>
        <div style={{ width: `${pct}%` }} className="bg-orange-400" />
        <div style={{ width: `${100 - pct}%` }} className="bg-blue-400" />
    </div>;
}

function BoundBadge({ label }: { label: Bound }) {
    let color = label === 'Compute-bound' ? 'bg-orange-100 text-orange-700'
        : label === 'Memory-bound' ? 'bg-blue-100 text-blue-700'
            : label === 'Comm-bound' ? 'bg-purple-100 text-purple-700'
                : 'bg-slate-200 text-slate-700';
    return <span className={clsx('rounded px-1.5 py-0.5 text-[10px] font-semibold', color)}>{label}</span>;
}

export const TransformerExplainer: React.FC = () => {
    let progState = useProgramState();
    let [hovered, setHovered] = useState<string | null>(null);
    let [regime, setRegime] = useState<Regime>('decode');
    let [tp, setTpState] = useState(1);

    function setTp(n: number) {
        setTpState(n);
        progState.display.tp = n; // drives the 3D weight sharding (applyTensorParallelShards)
        progState.markDirty();
    }

    function enter(key: string) {
        setHovered(key);
        progState.display.hoveredStage = key;
        progState.markDirty();
    }

    function leave() {
        setHovered(null);
        progState.display.hoveredStage = null;
        progState.markDirty();
    }

    let currentPreset = progState.modelSet[progState.currentModelIdx];
    let modelName = currentPreset?.name ?? 'this model';
    let hoveredStage = TRANSFORMER_STAGES.find(s => s.key === hovered) ?? null;
    let hoveredPt = hoveredStage ? effectivePoint(hoveredStage, regime, tp) : null;
    let markAi    = hoveredPt?.ai;
    let markComm  = hoveredPt?.comm ?? false;
    let markLabel = hoveredStage ? hoveredStage.title.split('·')[0].trim() : regime;

    return <div className="h-full overflow-y-auto bg-slate-50 px-4 py-4 text-slate-800">
        <h1 className="text-lg font-bold leading-tight">Dense decoder-only transformer</h1>
        <p className="mt-1 text-xs text-slate-500">
            How <span className="font-semibold text-slate-700">{modelName}</span> turns tokens into the
            next token. Hover a stage to highlight it in the 3D model and read the detail.
        </p>

        {/* regime toggle */}
        <div className="mt-3 flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-600">Regime:</span>
            {(['decode', 'prefill'] as Regime[]).map(r => (
                <button
                    key={r}
                    onClick={() => setRegime(r)}
                    className={clsx('rounded px-2 py-0.5 text-xs font-semibold capitalize',
                        regime === r ? 'bg-slate-800 text-white' : 'bg-white text-slate-600 hover:bg-slate-200')}
                >{r}</button>
            ))}
        </div>

        {/* roofline visualization */}
        <div className="mt-2 rounded-md border border-slate-200 bg-white p-2">
            <RooflineChart regime={regime} ai={markAi} label={markLabel} comm={markComm} />
            <p className="mt-1 text-[10px] leading-snug text-slate-500">
                {regime === 'decode'
                    ? 'Decode (1 token/pass) sits far left: it streams every weight from memory for a few FLOPs → memory-bound. Tokens/sec ≈ memory bandwidth ÷ model bytes.'
                    : 'Prefill (many tokens at once) sits right: the big matmuls reuse weights across tokens → compute-bound, limited by peak FLOPs.'}
            </p>
        </div>

        {/* tensor parallelism */}
        {currentPreset && <div className="mt-2">
            <TensorParallelPanel shape={currentPreset.shape} tp={tp} onTpChange={setTp} />
        </div>}

        {tp > 1 && <p className="mt-2 text-[10px] leading-snug text-slate-500">
            <span className="font-semibold text-slate-600">TP={tp} & the roofline:</span> per GPU, TP divides
            both FLOPs and weight bytes, so most stages keep the same arithmetic intensity. The exception is the
            two <span className="text-purple-700 font-semibold">all-reduce</span> stages (6 &amp; 7): their per-GPU
            compute shrinks but communication doesn’t, so they slide toward <span className="text-purple-700 font-semibold">comm-bound</span> as TP grows.
        </p>}

        {/* stages */}
        <div className="mt-3 flex flex-col gap-1.5">
            {TRANSFORMER_STAGES.map(stage => {
                let active = hovered === stage.key;
                let pt = effectivePoint(stage, regime, tp);
                let bound = effectiveBound(stage, regime, tp);
                return <div
                    key={stage.key}
                    onMouseEnter={() => enter(stage.key)}
                    onMouseLeave={leave}
                    className={clsx(
                        'rounded-md border p-2 transition-colors',
                        active ? 'border-blue-400 bg-blue-100' : 'border-slate-200 bg-white hover:bg-blue-50',
                    )}
                >
                    <div className="flex items-center justify-between gap-2">
                        <div className="text-sm font-semibold">{stage.title}</div>
                        <BoundBadge label={bound} />
                    </div>
                    <div className="text-xs text-slate-500">{stage.summary}</div>
                    <div className="mt-1.5 flex items-center gap-2">
                        <IntensityBar c={pt.compute} />
                        <span className="whitespace-nowrap text-[10px] text-slate-400">~{fmtAi(pt.ai)} FLOP/B</span>
                    </div>
                    {active && <>
                        <div className="mt-2 text-xs leading-relaxed text-slate-700">{stage.detail}</div>
                        <div className="mt-1.5 text-[11px] leading-snug text-slate-500">
                            <span className="font-semibold capitalize text-slate-600">{regime}:</span> {stage.why}
                        </div>
                        {pt.comm && <div className="mt-1 text-[11px] leading-snug text-purple-600">
                            TP={tp}: 1 all-reduce here per layer. Per-GPU compute ÷{tp}, but the collective doesn’t shrink → effective AI ~{fmtAi(pt.ai)} FLOP/B.
                        </div>}
                    </>}
                </div>;
            })}
        </div>

        {/* legend */}
        <div className="mt-3 flex items-center gap-3 text-[10px] text-slate-500">
            <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-sm bg-orange-400" /> compute</span>
            <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-sm bg-blue-400" /> memory / comm</span>
        </div>

        <p className="mt-3 text-[10px] leading-relaxed text-slate-400">
            Decode generates one token per forward pass: sample from stage 9, append it, and run the
            whole stack again. The KV cache (stage 5) is what makes that incremental instead of O(n²) each step.
        </p>

        <div className="mt-4 border-t border-slate-200 pt-2 text-[10px] text-slate-400">
            Companion to{' '}
            <a
                href="https://github.com/ai-hpc/ai-hardware-engineer-roadmap/tree/main/Phase%205%20-%20Advanced%20Topics%20and%20Specialization/7.%20ML%20Systems%20Engineering/AI%20Inference%20Engineer%202026"
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium text-blue-600 hover:underline"
            >
                AI Inference Engineer 2026
            </a>
            . Numbers calibrated to NVIDIA H200 (3.35 TB/s HBM3e, ~295 FLOP/B ridge).
        </div>
    </div>;
};
