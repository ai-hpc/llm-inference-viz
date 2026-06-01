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

    return <div className='absolute top-0 left-0 flex flex-col'>
        <div className='mt-2 ml-2 flex flex-row flex-wrap items-center'>
            {progState.modelSet.map((preset, idx) => {
                let isActive = progState.currentModelIdx === idx;
                return <div
                    key={preset.name}
                    className={clsx('m-2 p-2 rounded shadow cursor-pointer hover:bg-blue-300', isActive ? 'bg-blue-200 font-bold' : 'bg-white')}
                    onClick={() => selectModel(idx)}
                >
                    {preset.name}
                </div>;
            })}
        </div>
        <div className='ml-2 flex flex-row'>
            <div className={clsx('m-2 p-2 bg-white min-w-[2rem] flex justify-center rounded shadow cursor-pointer hover:bg-blue-300')} onClick={onExpandClick} title="Reset camera">
                <FontAwesomeIcon icon={faExpand} />
            </div>
        </div>
    </div>;

};
