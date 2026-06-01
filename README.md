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

## Single model: Qwen 2.5 72B

This build renders **only Qwen 2.5 72B** — true-scale structural view: 80 layers, hidden
8192, 64 Q / 8 KV heads, head-dim 128, SwiGLU ffn 29568, vocab 152064. Like upstream's
GPT-3 view, this renders *structure at scale*, not every weight cell (72B cells cannot be
drawn in a browser).

The GPT-2/GPT-3 presets, the down-scaled "Qwen (nano)" twin, the model-selector buttons,
and the nano-gpt guided **walkthrough sidebar** ("chapter overview") were all removed so
the view is focused on the one model. Two camera controls remain top-left:
**expand** (reset framing) and **magnify** (auto-frame the model) — plus mouse drag /
scroll and WASD-or-arrow keys to navigate.

> Note: the down-scaled Qwen twin and the modern-arch operator code still live in the
> source (gated on `arch === 'qwen'`); only the *presets/UI* were removed. Re-add an
> example entry in `Program.ts` to bring a detailed per-cell Qwen view back.

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
