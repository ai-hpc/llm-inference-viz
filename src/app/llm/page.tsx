import React from 'react';
import { LayerView } from '@/src/llm/LayerView';

export const metadata = {
    title: 'LLM Inference Visualizer — Qwen & Llama',
    description: 'Interactive 3D visualization of dense decoder-only LLM inference (Qwen 2.5 7B/72B, Llama 3.3 70B): the forward pass, the roofline (memory- vs compute-bound), and tensor parallelism.',
};

const REPO_URL = 'https://github.com/ai-hpc/llm-inference-viz';
const COURSE_URL = 'https://github.com/ai-hpc/ai-hardware-engineer-roadmap/tree/main/Phase%205%20-%20Advanced%20Topics%20and%20Specialization/7.%20ML%20Systems%20Engineering/AI%20Inference%20Engineer%202026';

export default function Page() {
    return <>
        <header className="flex h-12 flex-shrink-0 items-center justify-between border-b border-slate-800/80 bg-gradient-to-r from-slate-950 via-slate-900 to-slate-900 px-4 text-slate-100">
            <div className="flex items-center gap-2.5">
                <span className="h-5 w-1.5 rounded-full bg-gradient-to-b from-sky-400 to-violet-500" />
                <span className="text-base font-semibold tracking-tight">LLM Inference Visualizer</span>
                <span className="hidden border-l border-slate-700 pl-2.5 text-xs text-slate-400 sm:inline">
                    Qwen 2.5 · Llama 3.3 — forward pass, roofline &amp; tensor parallelism
                </span>
            </div>
            <nav className="flex items-center gap-1.5 text-sm">
                <a href={COURSE_URL} target="_blank" rel="noopener noreferrer"
                    className="rounded-md px-2.5 py-1 text-xs font-medium text-slate-300 transition-colors hover:bg-white/10 hover:text-white">
                    AI Inference Engineer 2026 ↗
                </a>
                <a href={REPO_URL} target="_blank" rel="noopener noreferrer"
                    className="flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium text-slate-300 transition-colors hover:bg-white/10 hover:text-white">
                    <svg viewBox="0 0 16 16" width="15" height="15" fill="currentColor" aria-hidden="true">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
                    </svg>
                    GitHub
                </a>
            </nav>
        </header>
        <LayerView />
        <div id="portal-container"></div>
    </>;
}
