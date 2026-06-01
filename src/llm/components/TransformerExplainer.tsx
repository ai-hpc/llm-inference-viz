import React, { useState } from 'react';
import clsx from 'clsx';
import { useProgramState } from '../Sidebar';
import { TRANSFORMER_STAGES } from './TransformerStages';

// Left-hand 2D explanation of the dense decoder-only transformer forward pass.
// Hovering a stage highlights the matching blocks in the 3D model on the right
// (via state.display.hoveredStage, consumed by applyStageHighlight in Program.ts)
// and expands the detailed description.
export const TransformerExplainer: React.FC = () => {
    let progState = useProgramState();
    let [hovered, setHovered] = useState<string | null>(null);

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

    let modelName = progState.modelSet[progState.currentModelIdx]?.name ?? 'this model';

    return <div className="h-full overflow-y-auto bg-slate-50 px-4 py-4 text-slate-800">
        <h1 className="text-lg font-bold leading-tight">Dense decoder-only transformer</h1>
        <p className="mt-1 text-xs text-slate-500">
            How <span className="font-semibold text-slate-700">{modelName}</span> turns tokens into the
            next token, in execution order. Hover a stage to highlight it in the 3D model and read the detail.
        </p>

        <div className="mt-3 flex flex-col gap-1.5">
            {TRANSFORMER_STAGES.map(stage => {
                let active = hovered === stage.key;
                return <div
                    key={stage.key}
                    onMouseEnter={() => enter(stage.key)}
                    onMouseLeave={leave}
                    className={clsx(
                        'rounded-md border p-2 transition-colors',
                        active ? 'border-blue-400 bg-blue-100' : 'border-slate-200 bg-white hover:bg-blue-50',
                    )}
                >
                    <div className="text-sm font-semibold">{stage.title}</div>
                    <div className="text-xs text-slate-500">{stage.summary}</div>
                    {active && <div className="mt-2 text-xs leading-relaxed text-slate-700">{stage.detail}</div>}
                </div>;
            })}
        </div>

        <p className="mt-4 text-[10px] leading-relaxed text-slate-400">
            Decode generates one token per forward pass: sample from stage 9, append it, and run the
            whole stack again. The KV cache (stage 5) is what makes that incremental instead of O(n²) each step.
        </p>
    </div>;
};
