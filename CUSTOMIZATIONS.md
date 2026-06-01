# Customizations — GPT-2 viz → Qwen 2.5

This document records exactly what was changed from upstream `bbycroft/llm-viz`, why,
and what remains. All architectural changes are gated on `shape.arch === 'qwen'`; with
that flag absent the code path is identical to upstream (GPT-2/GPT-3 presets unaffected).

## Shape / config

`src/llm/GptModel.ts` — `IModelShape` gained optional fields:

- `arch?: 'gpt' | 'qwen'` — master switch for the modern-arch operators.
- `nKVHeads?` — GQA key/value head count (defaults to `nHeads` → full MHA).
- `ffnDim?` — FFN inner dim (defaults to `4·C` → GPT-2 MLP).
- `normEps?`, `ropeTheta?` — display-only metadata.

`src/llm/GptModelLayout.ts` — `genGptModelLayout` derives:
`isQwen`, `nKVHeads`, `kvGroup = nHeads / nKVHeads`, `ffnDim`.

## Presets

`src/llm/Program.ts`:

- `qwen72bShape` — true-scale Qwen 2.5 72B (C 8192, 64/8 heads, A 128, 80 layers,
  ffn 29568, vocab 152064). Renders in large-model structural mode.
- `qwenNanoShape` — down-scaled twin (C 64, 8/2 heads, A 8, 4 layers, ffn 176, vocab 6).
  Small enough that the per-cell detail renders, so GQA/SwiGLU/RMSNorm are visible.
- Two new entries appended to `examples[]` (indices 3 and 4).

`src/llm/components/ModelSelectorToolbar.tsx` — `makeButton(3)` and `makeButton(4)`.

## Geometry (`GptModelLayout.ts`)

- **RoPE** — the learned `Position Embed` weight matrix is `hidden` for Qwen and
  relabelled to note RoPE is applied inside attention.
- **RMSNorm** — in `createLn`, the β bias block is `hidden`, the μ/σ aggregate is
  relabelled `RMS Agg: σ`, and the output block is `RMS Norm`.
- **GQA** — in the per-head loop, `isKVRep = !isQwen || (i % kvGroup === 0)`. Non-rep
  query heads no longer push their K/V weight/vector/bias cubes into the render list, so
  only `nKVHeads` K/V columns appear while all `nHeads` Q columns remain. (The block
  objects still exist so dependency wiring stays valid; they are simply not drawn.)
- **SwiGLU** — MLP widths use `ffnDim` instead of `C·4`; the up-projection block is
  `Gate + Up Weights (SwiGLU)`, the activation is `SiLU(Gate) ⊙ Up`, the down-projection
  is `Down Weights (SwiGLU)`, and both MLP biases are `hidden` (Qwen FFN is bias-free).

## Model card (`components/ModelCard.ts`)

When `arch === 'qwen'`, a caption is drawn under `n_params`:
`GQA {nHeads}Q/{nKVHeads}KV · SwiGLU (ffn {ffnDim}) · RMSNorm · RoPE`.

## Build trimming

Removed `src/app/cpu` and `src/app/fluid-sim` routes (unrelated upstream projects whose
build-time `fetch('/riscv/examples/...')` broke `next build`). `src/app/page.tsx` now
redirects `/` → `/llm`. The `src/cpu`, `src/fluidsim`, `src/homepage` source remain only
where still imported (e.g. `homepage/Header` is used by the LLM page).

---

## Remaining work (follow-ons)

1. **Visual pass in a browser.** The geometry was written headless (no GPU). Open
   `/llm`, switch to the Qwen presets, and tune block spacing / camera framing — the GQA
   skip leaves gaps where shared-head K/V columns used to be; that spacing may want
   tightening so the 8 KV columns read clearly against the 64 Q columns.
2. **Explicit gate/up split.** Right now the SwiGLU input is one `ffnDim`-wide block
   relabelled "Gate + Up". For full fidelity, draw gate and up as two separate
   `ffnDim`-wide matrices feeding an element-wise `SiLU(gate)⊙up` block.
3. **Numeric Qwen micro-model.** Upstream runs a real WebGL forward pass for nano-gpt
   only. To make `Qwen (nano)` actually *compute*, add GLSL phases for RMSNorm (no mean),
   RoPE on Q/K, GQA attention (broadcast KV across groups), and SwiGLU — in
   `GptModel.ts` `runModel` + the render phases. This is the deepest piece.
4. **Walkthrough narration.** `walkthrough/Walkthrough03_LayerNorm` → RMSNorm,
   `04_SelfAttention` → GQA + RoPE, `07_Mlp` → SwiGLU, and drop the learned-position
   narration in `02_Embedding`. Currently the guided tour is GPT-2-specific and runs on
   the nano-gpt main example; the Qwen presets are free-explore.
5. **RoPE annotation geometry.** Optionally draw a small rotation glyph on the Q/K
   vector blocks for Qwen to make "rotary" legible rather than only labelled.
