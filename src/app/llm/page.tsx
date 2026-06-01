import React from 'react';
import { LayerView } from '@/src/llm/LayerView';

export const metadata = {
  title: 'Qwen & Llama Inference Visualization',
  description: 'A 3D structural visualization of modern LLM inference networks — Qwen 2.5 7B/72B and Llama 3.3 70B (GQA, SwiGLU, RMSNorm, RoPE).',
};

import { Header } from '@/src/homepage/Header';

export default function Page() {
    return <>
        <Header title="Qwen & Llama Inference Visualization" />
        <LayerView />
        <div id="portal-container"></div>
    </>;
}
