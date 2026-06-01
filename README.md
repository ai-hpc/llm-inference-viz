<div align="center">

# 🧠 LLM Inference Visualizer

### An interactive 3D model of how a modern dense LLM *actually runs* inference

**[Qwen 2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) · [Qwen 2.5 72B](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) · [Llama 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)**

Forward pass · roofline (memory- vs compute-bound) · tensor parallelism

![Next.js](https://img.shields.io/badge/Next.js-13-black?logo=next.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5-blue?logo=typescript)
![WebGL2](https://img.shields.io/badge/WebGL2-3D-red)
![Fork](https://img.shields.io/badge/fork%20of-bbycroft%2Fllm--viz-orange?logo=github)

*Companion to the [**AI Inference Engineer 2026**](https://github.com/ai-hpc/ai-hardware-engineer-roadmap/tree/main/Phase%205%20-%20Advanced%20Topics%20and%20Specialization/7.%20ML%20Systems%20Engineering/AI%20Inference%20Engineer%202026) course.*

</div>

---

## What is this?

A 3D, walk-through visualization of a **dense decoder-only transformer** running inference — the architecture behind Qwen 2.5 and Llama 3.3. The right side renders the model's geometry to scale; the left side explains the forward pass and the **performance physics** that decide how fast (and how expensive) inference is.

It answers the questions an inference engineer actually asks:

- **Where does the time go?** Every stage is labelled **memory-bound** or **compute-bound** on an NVIDIA H200 roofline (~295 FLOP/byte ridge).
- **Why is decode slow?** Generating one token streams *all ~144 GB of weights* from HBM — see it land on the far-left, memory-bound side of the roofline.
- **What does tensor parallelism buy?** Slice the model across **TP = 1 / 2 / 4 / 8** GPUs and watch the weights split into colour-coded shards, with per-GPU memory, KV-head replication, and all-reduce cost computed live.

## ✨ Features

| | |
|---|---|
| 🧬 **Three real models** | Switch between Qwen 2.5 7B, Qwen 2.5 72B, and Llama 3.3 70B — true-scale structure (layers, GQA heads, SwiGLU width, vocab). |
| 🔍 **Synced explainer** | Hover a stage on the left → the matching blocks glow in the 3D model on the right. |
| 📈 **Roofline chart** | H200-calibrated. Toggle **Prefill / Decode** and watch every stage move between memory- and compute-bound. |
| 🔀 **Tensor parallelism** | TP=1/2/4/8 re-shards the 3D weights into per-GPU colour bands; panel shows params/GPU, KV replication, all-reduce count, and the FP8+TP=8 alignment gotcha. |
| 🌐 **Disaggregated P/D** | Real benchmark callout: 4-prefill + 4-decode = 648 tok/s vs aggregated TP=8 = 321 tok/s. |
| 🧩 **Modern ops** | GQA, RoPE (θ=1e6), RMSNorm, SwiGLU — the actual Qwen/Llama operator set, not GPT-2. |

## 🚀 Getting started

**Requirements:** Node.js ≥ 18 and a WebGL2-capable browser (recent Chrome / Firefox / Edge).

```bash
# 1. Clone
git clone https://github.com/ai-hpc/llm-inference-viz.git
cd llm-inference-viz

# 2. Install
npm install

# 3. Run the dev server
npm run dev
```

Then open **http://localhost:3002/llm**.

### Production build

```bash
npm run build
npm start -- -p 3002        # serves on http://localhost:3002/llm
```

### How to use it

1. Pick a model with the **top-left buttons** (default: Qwen 2.5 7B).
2. **Drag** to orbit, **scroll** to zoom, **WASD / arrows** to pan; the **⤢ expand** button re-frames.
3. In the left panel, **hover any stage** to highlight it in 3D and read the detail.
4. Flip the **Prefill / Decode** toggle and the **TP = 1/2/4/8** selector to see the roofline and sharding change.

## 🎓 Part of a course

This visualizer is a teaching companion to **[AI Inference Engineer 2026](https://github.com/ai-hpc/ai-hardware-engineer-roadmap/tree/main/Phase%205%20-%20Advanced%20Topics%20and%20Specialization/7.%20ML%20Systems%20Engineering/AI%20Inference%20Engineer%202026)** — part of the open [AI Hardware Engineer Roadmap](https://ai-hpc.github.io/ai-hardware-engineer-roadmap/). It maps most directly to:

- **Part 1 · Lecture 03** — Roofline, bandwidth, and the memory hierarchy
- **Part 2 · Lecture 01** — Anatomy of a 70B-class dense model (Llama 3.3 70B vs Qwen 2.5 72B)
- **Part 2 · Lecture 04** — Single-node multi-GPU serving: tensor parallelism on 8× H100/H200

## 🛠️ Tech & structure

Next.js 13 + React + TypeScript, with a hand-written WebGL2 renderer.

```
src/llm/
├─ Program.ts                  # state, model presets, render loop, stage-highlight & TP sharding
├─ GptModelLayout.ts           # 3D geometry of the transformer (GQA / SwiGLU / RMSNorm / RoPE)
├─ render/blockRender.ts       # WebGL2 block renderer
└─ components/
   ├─ TransformerExplainer.tsx  # left panel (forward pass + roofline + TP)
   ├─ TransformerStages.ts      # per-stage data, H200-calibrated arithmetic intensity
   ├─ RooflineChart.tsx         # SVG roofline
   ├─ TensorParallelPanel.tsx   # TP=1/2/4/8 explorer
   └─ shardColors.ts            # shared per-GPU palette
```

Numbers (arithmetic intensity, weight sizes, throughput) are calibrated to **NVIDIA H200 SXM** (989 TFLOPS BF16, 3.35 TB/s HBM3e) and to published vLLM / SGLang / AIConfigurator benchmarks. They are illustrative teaching figures, not a profiler.

## 🙏 Attribution & license

Forked from and built on **[Brendan Bycroft's `llm-viz`](https://github.com/bbycroft/llm-viz)** — the original 3D LLM visualization — re-targeted from GPT-2 to the modern Qwen / Llama dense-decoder architecture, with the roofline and tensor-parallelism layers added. **Huge thanks to Brendan** for the renderer and the original idea.

> ⚠️ **Licensing note.** The upstream `llm-viz` repository does **not** publish an explicit
> open-source license, so its code is — by default — *all rights reserved by the original
> author*. This fork exists **for educational purposes** as a companion to the course, with
> full attribution. The **new code added in this fork** (the Qwen/Llama retargeting, the
> roofline charts, and the tensor-parallelism layers — see [`NOTICE.md`](NOTICE.md)) is
> offered freely for educational use, but it does **not** and **cannot** relicense the
> upstream work. **If you intend to redistribute, host publicly long-term, or use this
> commercially, please obtain permission from the upstream author first.** See
> [`NOTICE.md`](NOTICE.md) for details.
