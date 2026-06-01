# NOTICE — attribution & licensing

## Upstream origin

This project is a **fork of [`bbycroft/llm-viz`](https://github.com/bbycroft/llm-viz)** by
**Brendan Bycroft** — the original interactive 3D LLM visualization. The WebGL renderer,
the block-layout engine, the camera/interaction system, and the overall design are his work.

At the time of forking, the upstream repository **did not include an explicit open-source
license**. Under default copyright law that means the upstream code is **all rights reserved
by the original author**. This fork is published **for educational purposes** as a companion
to the *AI Inference Engineer 2026* course, with prominent attribution.

**This fork does not — and legally cannot — relicense the upstream code.** If you wish to
redistribute, host this publicly on an ongoing basis, fork it further, or use it
commercially, please **contact Brendan Bycroft for permission** regarding the upstream
portions first.

## What this fork adds

The following work was added on top of the upstream renderer and is original to this fork:

- Re-targeting the model from GPT-2 to the modern **dense decoder-only architecture**
  (GQA, RoPE, RMSNorm, SwiGLU, bias-free FFN) used by Qwen 2.5 and Llama 3.3.
- Three switchable model presets (Qwen 2.5 7B, Qwen 2.5 72B, Llama 3.3 70B).
- The left **dense decoder-only forward-pass explainer**, synced to the 3D model on hover.
- The **roofline** visualization (memory- vs compute-bound), calibrated to NVIDIA H200.
- The **tensor-parallelism** explorer (TP = 1/2/4/8): 3D weight sharding, per-GPU stats,
  all-reduce / communication modeling, and the disaggregated prefill/decode callout.

The fork authors offer **these additions** freely for educational and non-commercial use.
This permission applies only to the newly-added code listed above, not to the upstream work.

## Course

Built as a companion to **AI Inference Engineer 2026**, part of the open
[AI Hardware Engineer Roadmap](https://github.com/ai-hpc/ai-hardware-engineer-roadmap).
