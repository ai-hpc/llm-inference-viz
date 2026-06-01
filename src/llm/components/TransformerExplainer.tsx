import React, { useState } from 'react';
import clsx from 'clsx';
import { useProgramState } from '../Sidebar';
import { TRANSFORMER_STAGES, Regime, Bound, effectivePoint, effectiveBound, fmtAi } from './TransformerStages';
import { RooflineChart } from './RooflineChart';
import { TensorParallelPanel } from './TensorParallelPanel';
import { Admonition } from './Admonition';

// Left-hand 2D explanation of the dense decoder-only transformer forward pass.
// Hovering a stage highlights the matching blocks in the 3D model on the right
// (via state.display.hoveredStage, consumed by applyStageHighlight in Program.ts),
// expands the detailed description, and shows where the stage sits on the roofline.
// A Prefill/Decode toggle and the TP selector drive the compute-vs-memory readouts.

function SectionLabel({ children }: { children: React.ReactNode }) {
    return <div className="mb-1.5 mt-4 text-[10px] font-bold uppercase tracking-wider text-slate-400">{children}</div>;
}

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
            : label === 'Comm-bound' ? 'bg-violet-100 text-violet-700'
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
    let markAi = hoveredPt?.ai;
    let markComm = hoveredPt?.comm ?? false;
    let markLabel = hoveredStage ? hoveredStage.title.split('·')[0].trim() : regime;

    const courseUrl = 'https://github.com/ai-hpc/ai-hardware-engineer-roadmap/tree/main/Phase%205%20-%20Advanced%20Topics%20and%20Specialization/7.%20ML%20Systems%20Engineering/AI%20Inference%20Engineer%202026';

    return <div className="h-full overflow-y-auto bg-gradient-to-b from-white to-slate-50 px-4 py-4 text-slate-800">
        {/* header */}
        <div className="flex items-center gap-2">
            <span className="h-5 w-1 rounded-full bg-gradient-to-b from-sky-500 to-violet-500" />
            <h1 className="text-base font-bold tracking-tight text-slate-900">Dense decoder-only transformer</h1>
        </div>
        <p className="mt-1 pl-3 text-xs leading-relaxed text-slate-500">
            How <span className="font-semibold text-slate-700">{modelName}</span> turns tokens into the next token.
            Hover a stage to highlight it in the 3D model and read the detail.
        </p>

        {/* regime toggle — segmented control */}
        <div className="mt-3 flex items-center gap-2">
            <span className="text-[11px] font-semibold text-slate-500">Regime</span>
            <div className="inline-flex rounded-lg bg-slate-200/80 p-0.5">
                {(['decode', 'prefill'] as Regime[]).map(r => (
                    <button
                        key={r}
                        onClick={() => setRegime(r)}
                        className={clsx('rounded-md px-2.5 py-1 text-xs font-semibold capitalize transition',
                            regime === r ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-700')}
                    >{r}</button>
                ))}
            </div>
        </div>

        {/* roofline */}
        <SectionLabel>Roofline · NVIDIA H200</SectionLabel>
        <div className="rounded-xl border border-slate-200 bg-white p-2.5 shadow-sm">
            <RooflineChart regime={regime} ai={markAi} label={markLabel} comm={markComm} />
        </div>
        <div className="mt-2">
            <Admonition kind="note" title={regime === 'decode' ? 'Why decode is memory-bound' : 'Why prefill is compute-bound'}>
                {regime === 'decode'
                    ? 'Decode (1 token/pass) sits far left: it streams every weight from HBM for a few FLOPs. Tokens/sec ≈ memory bandwidth ÷ model bytes.'
                    : 'Prefill (many tokens at once) sits right: the big matmuls reuse each weight across the whole batch → limited by peak FLOPs, not bandwidth.'}
            </Admonition>
        </div>

        {/* tensor parallelism */}
        <SectionLabel>Tensor parallelism</SectionLabel>
        {currentPreset && <TensorParallelPanel shape={currentPreset.shape} tp={tp} onTpChange={setTp} />}
        {tp > 1 && <div className="mt-2">
            <Admonition kind="key" title={`TP=${tp} & the roofline`}>
                Per GPU, TP divides <em>both</em> FLOPs and weight bytes, so most stages keep the same arithmetic
                intensity. The exception is the two <span className="font-semibold">all-reduce</span> stages (6 &amp; 7):
                their per-GPU compute shrinks but the collective doesn’t, so they slide toward
                <span className="font-semibold"> comm-bound</span> as TP grows.
            </Admonition>
        </div>}

        {/* forward pass */}
        <SectionLabel>Forward pass · {regime}</SectionLabel>
        <div className="flex flex-col gap-1.5">
            {TRANSFORMER_STAGES.map(stage => {
                let active = hovered === stage.key;
                let pt = effectivePoint(stage, regime, tp);
                let bound = effectiveBound(stage, regime, tp);
                return <div
                    key={stage.key}
                    onMouseEnter={() => enter(stage.key)}
                    onMouseLeave={leave}
                    className={clsx(
                        'rounded-lg border p-2 transition-all',
                        active ? 'border-sky-400 bg-sky-50 shadow-sm ring-1 ring-sky-200' : 'border-slate-200 bg-white hover:border-sky-300 hover:bg-sky-50/40',
                    )}
                >
                    <div className="flex items-center justify-between gap-2">
                        <div className="text-sm font-semibold text-slate-800">{stage.title}</div>
                        <BoundBadge label={bound} />
                    </div>
                    <div className="text-xs text-slate-500">{stage.summary}</div>
                    <div className="mt-1.5 flex items-center gap-2">
                        <IntensityBar c={pt.compute} />
                        <span className="whitespace-nowrap text-[10px] tabular-nums text-slate-400">~{fmtAi(pt.ai)} FLOP/B</span>
                    </div>
                    {active && <>
                        <div className="mt-2 text-xs leading-relaxed text-slate-700">{stage.detail}</div>
                        <div className="mt-1.5 text-[11px] leading-snug text-slate-500">
                            <span className="font-semibold capitalize text-slate-600">{regime}:</span> {stage.why}
                        </div>
                        {pt.comm && <div className="mt-1 text-[11px] leading-snug text-violet-600">
                            TP={tp}: 1 all-reduce here per layer. Per-GPU compute ÷{tp}, but the collective doesn’t shrink → effective AI ~{fmtAi(pt.ai)} FLOP/B.
                        </div>}
                    </>}
                </div>;
            })}
        </div>

        {/* legend */}
        <div className="mt-2.5 flex items-center gap-3 text-[10px] text-slate-500">
            <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-sm bg-orange-400" /> compute</span>
            <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-sm bg-blue-400" /> memory / comm</span>
        </div>

        {/* key idea + a real-world tip */}
        <div className="mt-3 flex flex-col gap-2">
            <Admonition kind="key" title="The decode loop">
                One token per forward pass: sample from stage 9, append it, run the whole stack again.
                The KV cache (stage 5) is what makes that incremental instead of O(n²) every step.
            </Admonition>
            <Admonition kind="tip" title="Generation length">
                Post-trained models can generate coherent output far <em>longer</em> than their training length
                (e.g. 16,384 → 32,768 tokens), especially for tasks with clear rules like code. Evaluate output
                quality at several lengths before fixing a max generation length.
                <span className="mt-1 block text-[10px] text-emerald-700/70">— Qwen docs, “Key Concepts”</span>
            </Admonition>
        </div>

        {/* further reading */}
        <div className="mt-5 border-t border-slate-200 pt-3">
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Further reading</div>
            <ul className="mt-1.5 space-y-1 text-[11px] text-slate-600">
                <li>↳ <a href={courseUrl} target="_blank" rel="noopener noreferrer" className="font-medium text-sky-600 hover:underline">AI Inference Engineer 2026</a> — the companion course</li>
                <li>↳ <a href="https://qwen.readthedocs.io/en/latest/getting_started/concepts.html" target="_blank" rel="noopener noreferrer" className="font-medium text-sky-600 hover:underline">Qwen docs — Key Concepts</a> — context &amp; generation length</li>
                <li>↳ <a href="https://magazine.sebastianraschka.com/p/qwen3-from-scratch" target="_blank" rel="noopener noreferrer" className="font-medium text-sky-600 hover:underline">Qwen3 From Scratch</a> — S. Raschka <span className="text-slate-400">(paid)</span></li>
            </ul>
            <p className="mt-2 text-[10px] leading-relaxed text-slate-400">
                Numbers calibrated to NVIDIA H200 (3.35 TB/s HBM3e, ~295 FLOP/B ridge). Illustrative teaching figures, not a profiler.
            </p>
        </div>
    </div>;
};
