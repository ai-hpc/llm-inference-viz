import React from 'react';
import { useProgramState } from '../Sidebar';
import clsx from 'clsx';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faExpand } from '@fortawesome/free-solid-svg-icons';

export const ModelSelectorToolbar: React.FC<{
}> = () => {
    let progState = useProgramState();

    // Switch the displayed model: swap the shape (the render loop regenerates the layout
    // from state.shape) and fly the camera to that model's framing.
    function selectModel(idx: number) {
        let preset = progState.modelSet[idx];
        if (!preset) {
            return;
        }
        progState.currentModelIdx = idx;
        progState.shape = preset.shape;
        progState.mainExample.name = preset.name;
        progState.mainExample.shape = preset.shape;
        progState.mainExample.camera = preset.camera;
        progState.camera.desiredCamera = preset.camera;
        progState.markDirty();
    }

    // Reset the camera to the current model's default framing.
    function onExpandClick() {
        let preset = progState.modelSet[progState.currentModelIdx];
        progState.camera.desiredCamera = preset.camera;
        progState.markDirty();
    }

    return <div className='absolute top-3 left-3 flex flex-col items-start gap-2'>
        {/* model selector — frosted-glass segmented pills */}
        <div className='inline-flex flex-wrap items-center gap-1 rounded-xl border border-white/50 bg-white/70 p-1 shadow-lg ring-1 ring-black/5 backdrop-blur-md'>
            {progState.modelSet.map((preset, idx) => {
                let isActive = progState.currentModelIdx === idx;
                return <button
                    key={preset.name}
                    onClick={() => selectModel(idx)}
                    className={clsx('rounded-lg px-3 py-1.5 text-xs font-semibold transition-all',
                        isActive
                            ? 'bg-gradient-to-b from-sky-500 to-violet-500 text-white shadow-sm'
                            : 'text-slate-600 hover:bg-white hover:text-slate-900')}
                >
                    {preset.name}
                </button>;
            })}
        </div>
        {/* reset-view button */}
        <button
            onClick={onExpandClick}
            title="Reset camera to frame the model"
            className='inline-flex items-center gap-1.5 rounded-lg border border-white/50 bg-white/70 px-2.5 py-1.5 text-xs font-medium text-slate-600 shadow-lg ring-1 ring-black/5 backdrop-blur-md transition-all hover:bg-white hover:text-slate-900'
        >
            <FontAwesomeIcon icon={faExpand} />
            <span>Reset view</span>
        </button>
    </div>;

};
