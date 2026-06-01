import React from 'react';
import { LayerView } from '@/src/llm/LayerView';

export const metadata = {
  title: 'Qwen 2.5 72B Visualization',
  description: 'A 3D structural visualization of the Qwen 2.5 72B inference network (GQA, SwiGLU, RMSNorm, RoPE).',
};

import { Header } from '@/src/homepage/Header';

export default function Page() {
    return <>
        <Header title="Qwen 2.5 72B Visualization" />
        <LayerView />
        <div id="portal-container"></div>
    </>;
}
