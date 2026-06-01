# Qwen LLM Visualization (fork)

A fork of Brendan Bycroft's [llm-viz](https://github.com/bbycroft/llm-viz) — the 3D
interactive "LLM Visualization" — customized to render **Qwen 2.5–class** models
(GQA + RoPE + RMSNorm + SwiGLU) instead of only the GPT-2 architecture.

Built for the **AI Inference Engineer 2026** course (Phase 5 → ML Systems Engineering,
Stage 2 — Transformer Execution Internals / Part 2 — Dense at Hopper).

---

## What this fork adds

Upstream renders the GPT-2 block exactly: learned position embeddings, LayerNorm,
full multi-head attention, single GELU MLP. Qwen uses none of those. This fork adds the
modern-architecture operators, gated behind `shape.arch === 'qwen'` so **every existing
GPT-2 / GPT-3 view is byte-for-byte unchanged**:

| Qwen op | Change | Where |
|---------|--------|-------|
| **GQA** (64 Q / 8 KV) | `nKVHeads` on the shape; shared query heads no longer draw their own K/V weight/vector/bias columns | `GptModelLayout.ts` (head loop) |
| **SwiGLU** FFN | FFN width driven by `ffnDim` (not hard-coded `4·C`); blocks relabelled Gate+Up / SiLU⊙ / Down; bias-free | `GptModelLayout.ts` (MLP) |
| **RMSNorm** | β (bias) term hidden, μ aggregate relabelled, "Layer Norm" → "RMS Norm" | `GptModelLayout.ts` (`createLn`) |
| **RoPE** | learned position-embedding matrix hidden and annotated | `GptModelLayout.ts` (posEmbed) |
| Model presets | `Qwen 2.5 72B` (true scale) + `Qwen (nano)` (down-scaled twin) | `Program.ts`, `ModelSelectorToolbar.tsx` |
| Model card | architecture caption: `GQA 64Q/8KV · SwiGLU · RMSNorm · RoPE` | `components/ModelCard.ts` |

The fork is also **trimmed to the LLM viz only** — upstream bundled a RISC-V CPU
simulator and a fluid sim (and a homepage) that broke the production build via a
build-time `fetch`. Those routes were removed and `/` now redirects to `/llm`.

See [`CUSTOMIZATIONS.md`](CUSTOMIZATIONS.md) for the full change rationale and the
remaining work.

## Three switchable models

Top-left buttons switch between three structural views (default **Qwen 2.5 7B**):

| Model | hidden | layers | heads (Q/KV) | SwiGLU ffn | vocab |
|-------|--------|--------|--------------|------------|-------|
| **Qwen 2.5 7B** (default) | 3584 | 28 | 28 / 4 | 18944 | 152064 |
| **Qwen 2.5 72B** | 8192 | 80 | 64 / 8 | 29568 | 152064 |
| **Llama 3.3 70B** | 8192 | 80 | 64 / 8 | 28672 | 128256 |

All three share the modern-arch operator set (GQA + RoPE + RMSNorm + SwiGLU), so the same
`arch: 'qwen'` geometry drives all of them; the model-set lives in `Program.ts`
(`initProgramState`). Like upstream's GPT-3 view these render *structure at scale*, not
every weight cell (70B+ cells can't be drawn in a browser), so per-cube text labels are
suppressed for models with > 12 blocks — the **model card** carries the per-model summary
(`GQA …Q/…KV · SwiGLU · RMSNorm · RoPE` + param count).

The GPT-2/GPT-3 presets and the nano-gpt guided walkthrough were removed. The **expand**
button (top-left) resets the camera to the current model's framing; mouse drag / scroll
and WASD-or-arrow keys navigate.

### Left explainer panel

A 2D **"Dense decoder-only transformer"** panel (left, resizable) walks the forward pass
in execution order: token embedding → RMSNorm → GQA Q/K/V → RoPE → causal softmax →
attention output+residual → SwiGLU MLP → ×N layers → final norm/logits/sample. **Hovering
a stage** expands its detailed explanation *and* glows the matching blocks in the 3D model
on the right (kept in sync via `state.display.hoveredStage` → `applyStageHighlight` in
`Program.ts`). Stage content lives in `components/TransformerStages.ts`; the panel is
`components/TransformerExplainer.tsx`.

### Compute- vs memory-bound (roofline)

The panel has a **Prefill / Decode** toggle and a compact **roofline chart**
(`components/RooflineChart.tsx`): performance vs arithmetic intensity, with the
memory-bandwidth diagonal bending into the compute ceiling at the ridge point (left =
memory-bound, right = compute-bound). Each stage shows a **compute/memory bar** + a
**bound badge** + an approximate FLOP/byte for the selected regime, and hovering a stage
drops its marker onto the roofline. The teaching point falls straight out: at **decode**
almost every stage is memory-bound (stream all weights for one token), while at **prefill**
the matmuls become compute-bound. Intensity numbers are illustrative, not measured.

> Note on framing: each model has a hand-set camera. They were tuned without a GPU to
> view them, so the 7B framing in particular may need a scroll/drag to sit perfectly —
> easy to adjust in the `modelSet` cameras in `Program.ts`.

## Run locally

```bash
npm install
npm run dev      # http://localhost:3002/llm
```

Production build / serve:

```bash
npm run build
npm start -- -p 3007   # http://localhost:3007/llm
```

## Embedding in the course (mkdocs)

The viz is a client-rendered Next.js app. To embed in the AI Inference Engineer 2026
Stage 2 page, host this app (any static/Node host) and iframe it:

```html
<iframe src="https://<your-host>/llm" width="100%" height="720"
        style="border:0" loading="lazy" title="Qwen 2.5 inference visualization"></iframe>
```

## ⚠️ Status & honest limits

- **Builds, typechecks, and serves** (`npm run build` is green; `/llm` returns 200).
- **3D output is not visually verified here.** It was developed in a headless
  environment with no GPU/browser, so the geometry changes are verified by types and
  build only — **open it in a browser to confirm the 3D layout reads correctly** and to
  tune block spacing / camera for the Qwen presets.
- **Structural, not numeric, for Qwen.** Upstream runs a real WebGL forward pass only
  for the tiny nano-gpt weights. The Qwen presets reuse the structural renderer; a
  numerically-running Qwen micro-model (GQA / RoPE / SwiGLU / RMSNorm in GLSL) is the
  documented follow-on in `CUSTOMIZATIONS.md`.
- **Walkthrough narration** (the GPT-2 guided tour) was **removed** in this build, not
  ported. If you later want a Qwen guided tour, the narration would need rewriting for
  RMSNorm / GQA+RoPE / SwiGLU (see `CUSTOMIZATIONS.md`).

## Attribution & license

Forked from **Brendan Bycroft's llm-viz** (https://github.com/bbycroft/llm-viz).
The upstream repository did **not** include an explicit license file at the commit this
was forked from. **Confirm licensing / permission with the upstream author before
publishing or distributing this fork** (including embedding on a public course site).
