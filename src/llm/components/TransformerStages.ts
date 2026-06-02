// The forward pass of a dense, decoder-only transformer (Qwen 2.5 / Llama 3.3 family),
// in execution order.
//
// Roofline calibration — H200 SXM (BF16):
//   Peak throughput : ~989 TFLOPS (BF16)
//   HBM3e bandwidth: 3.35 TB/s
//   Ridge point     : 989e12 / 3.35e12 ≈ 295 FLOP/B
//
// Decode (batch=1): every major op is a GEMV — stream the full weight matrix for ONE
//   vector. AI ≈ 1–2 FLOP/B for BF16, well below the ridge → purely memory-bound.
//   Tokens/sec ≈ HBM bandwidth ÷ model bytes (3.35 TB/s ÷ ~144 GB ≈ 23 tok/s single GPU).
//
// Prefill (512 tokens typical): GEMMs reuse each weight across all tokens in the batch.
//   AI grows linearly with batch/sequence length → crosses the ridge at ~300+ tokens
//   and becomes compute-bound. Dominant cost is then peak TFLOPS, not bandwidth.
//
// Source for numbers: "Achieving Peak Inference Performance for Qwen 2.5 72B on an
// 8-GPU H200 Cluster" (2025) + vLLM/SGLang benchmark data cited therein.

export type Regime = 'prefill' | 'decode';

export interface IRooflinePoint {
    compute: number; // 0 (memory-bound) .. 1 (compute-bound), visual guide
    ai: number;      // arithmetic intensity, FLOP/byte — H200 ridge ≈ 295
}

export interface ITransformerStage {
    key: string;
    title: string;
    summary: string;
    detail: string;
    match: string[];                 // cube-name substrings to highlight in 3D
    prefill: IRooflinePoint;
    decode: IRooflinePoint;
    why: string;
    comm?: boolean;                  // ends in an all-reduce under tensor parallelism
}

export type Bound = 'Compute-bound' | 'Memory-bound' | 'Mixed' | 'Comm-bound';

export function boundLabel(c: number): Bound {
    if (c >= 0.70) return 'Compute-bound';
    if (c <= 0.25) return 'Memory-bound';
    return 'Mixed';
}

// H200 ridge point used by the roofline chart and by effectivePoint.
export const H200_RIDGE_AI = 295; // FLOP/byte

// Per-GPU point adjusted for tensor parallelism.
// TP divides both FLOPs and weight bytes equally → per-GPU AI is essentially
// unchanged for column-parallel stages. The exception: the two row-parallel
// sub-layers (attention output, MLP down) end in an all-reduce whose cost is
// proportional to (TP-1)/TP and does NOT shrink with TP → they become comm-bound
// at high TP. Each all-reduce over NVLink costs on the order of tens of µs (highly
// dependent on GPU count, message size, and NCCL impl); across 160 collectives per
// forward pass (2 per layer × 80) that can become a meaningful fraction of decode
// latency at TP=8. Treat any specific µs/ms figure as illustrative, not a constant.
export function effectivePoint(stage: ITransformerStage, regime: Regime, tp: number): { compute: number; ai: number; comm: boolean } {
    let base = stage[regime];
    if (stage.comm && tp > 1) {
        // all-reduce overhead grows with (TP-1); effective AI falls toward 0
        let penalty = 1 + 0.7 * (tp - 1);
        return { compute: base.compute / penalty, ai: base.ai / penalty, comm: true };
    }
    return { compute: base.compute, ai: base.ai, comm: false };
}

export function effectiveBound(stage: ITransformerStage, regime: Regime, tp: number): Bound {
    let p = effectivePoint(stage, regime, tp);
    if (p.comm) return 'Comm-bound';
    return boundLabel(p.compute);
}

export function fmtAi(ai: number): string {
    if (ai >= 100) return Math.round(ai).toString();
    if (ai >= 10)  return ai.toFixed(0);
    return ai.toFixed(1);
}

export const TRANSFORMER_STAGES: ITransformerStage[] = [
    {
        key: 'embed',
        title: '1 · Token embedding',
        summary: 'Token ids → d_model vectors.',
        detail:
            'Each input token id indexes a row of the embedding table (vocab × d_model). ' +
            'For Qwen 2.5 72B: 152,064 × 8,192 × 2 B = ~2.5 GB table — a memory gather, not a matmul. ' +
            'Almost no arithmetic per byte read. No learned positional embedding — ' +
            'position is injected inside attention via RoPE.',
        match: ['Token Embed', 'Input Embed', 'Tokens'],
        // Decode: 1 row read (~16 KB for BF16 d=8192). Prefill: L rows but still no reuse.
        prefill: { compute: 0.08, ai: 0.5 },
        decode: { compute: 0.04, ai: 0.3 },
        why: 'Table lookup — 0 MAC per byte. Both regimes are memory-bound.',
    },
    {
        key: 'rmsnorm',
        title: '2 · RMSNorm (pre-norm)',
        summary: 'Normalise the residual before each sub-layer.',
        detail:
            'RMSNorm computes σ = rms(x), then output = γ ⊙ (x / σ). ' +
            'No mean subtraction, no bias (β). The γ weight is tiny (d_model scalars = 16 KB). ' +
            'Applied TWICE per block (pre-attention and pre-MLP) — 160 calls over 80 layers. ' +
            'It is elementwise and accesses each activation once, so it is always bandwidth-bound.',
        match: ['RMS Norm', 'γ'],
        prefill: { compute: 0.10, ai: 3 },
        decode: { compute: 0.05, ai: 0.5 },
        why: 'Elementwise over d_model per token. Tiny AI in both regimes — pure memory bandwidth.',
    },
    {
        key: 'qkv',
        title: '3 · Q / K / V projection (GQA)',
        summary: 'Project to queries, keys, values — grouped-query attention.',
        detail:
            'Three linear projections: Q (8192×8192), K (8192×1024), V (8192×1024) for Qwen 2.5 72B. ' +
            'GQA uses 64 Q heads / 8 KV heads: the K and V projections are 8× smaller than Q, ' +
            'reducing the KV cache ~8× vs full MHA. In BF16 the combined weight is ' +
            '~167 MB per layer (×80 layers = ~13 GB just for QKV). ' +
            'Decode (batch=1): GEMV — all 167 MB must be streamed from HBM3e for 1 output vector. ' +
            'Prefill (512 tokens): GEMM — each weight byte is reused 512×, AI≈512 FLOP/B → compute-bound.',
        match: ['Q Weights', 'K Weights', 'V Weights', 'Q vectors', 'K vectors', 'V vectors', 'QKV'],
        prefill: { compute: 0.90, ai: 480 },
        decode: { compute: 0.12, ai: 1.0 },
        why:
            'Decode: GEMV — 167 MB weight streamed for 1 output token (AI ≈ 1 FLOP/B, far below H200 ridge of 295). ' +
            'Prefill: GEMM — same weights reused across every token → AI scales with seq_len → compute-bound above ~300 tokens.',
    },
    {
        key: 'rope',
        title: '4 · RoPE (rotary position)',
        summary: 'Rotate Q and K by a position-dependent angle.',
        detail:
            'Rotary Position Embedding rotates each pair of Q/K elements by θ = pos × base^(-2i/d), ' +
            'where base = 1,000,000 for Qwen 2.5 (raised from 10,000 to support 128K context). ' +
            'The rotation is applied as x\' = x cos(θ) − x_rot sin(θ). ' +
            'Only Q and K are rotated — V is untouched. ' +
            'Relative attention (score depends on angle difference) means the model sees position ' +
            'without any learned table, and extrapolates to context lengths beyond training.',
        match: ['Q vectors', 'K vectors'],
        prefill: { compute: 0.10, ai: 3 },
        decode: { compute: 0.04, ai: 0.3 },
        why: 'Elementwise sin/cos rotation — almost no reuse, always memory-bound. Negligible vs QKV or MLP.',
    },
    {
        key: 'attn',
        title: '5 · Scores + causal softmax',
        summary: 'QKᵀ/√d, mask the future, softmax.',
        detail:
            'At each decode step: score_i = Q · K_i / √128 for every past token i (context N) — O(N) KV reads per token. ' +
            'KV cache size with GQA: 2 (K+V) × 8 KV heads × 128 head_dim × 2 B (BF16) = ~4 KB per token per layer. ' +
            'At N=128K that is ~512 MB per layer, so ~40 GB across all 80 layers — large enough that ' +
            'PagedAttention (vLLM) is essential: it stores the KV cache in fixed-size blocks (pages), ' +
            'eliminating fragmentation rather than shrinking it. Prefill: QKᵀ is an O(N²) GEMM, ' +
            'becoming compute-bound at long sequences. FlashAttention avoids materialising the full N×N score matrix.',
        match: ['Attention Matrix', 'Attn Matrix Softmax'],
        prefill: { compute: 0.68, ai: 55 },
        decode: { compute: 0.08, ai: 0.8 },
        why:
            'Decode: reads full KV cache for each token — scales linearly with context, always bandwidth-bound. ' +
            'Prefill: O(N²) QKᵀ GEMM — compute-bound for long sequences.',
    },
    {
        key: 'attnout',
        title: '6 · Attention output + residual',
        summary: 'Weighted sum of V, output projection, add back.',
        detail:
            'Each head computes a weighted sum of its V vectors; heads are concatenated (d_model = 8192). ' +
            'An output projection W_O (8192×8192, ~134 MB BF16 per layer) mixes the heads. ' +
            'The result is ADDED to the residual stream — skip connection — so all the information ' +
            'from before attention is preserved. ' +
            'W_O is a row-parallel layer in tensor parallelism: each GPU computes its shard, ' +
            'then all GPUs all-reduce before the result is valid.',
        match: ['V Output', 'Attention Output', 'Projection Weights', 'Projection Bias', 'Attention Residual'],
        prefill: { compute: 0.90, ai: 480 },
        decode: { compute: 0.12, ai: 1.0 },
        why:
            'Decode: GEMV, ~134 MB weight stream per token (AI ≈ 1 FLOP/B) → memory-bandwidth bound, not compute-bound. ' +
            'Row-parallel under TP → ends in an NVLink all-reduce; at TP=8 that collective (a few tens of µs, ' +
            'hardware-dependent) becomes a meaningful fraction of per-layer decode latency.',
        comm: true,
    },
    {
        key: 'mlp',
        title: '7 · SwiGLU MLP + residual',
        summary: 'Gate ⊙ Up → Down, then add back.',
        detail:
            'After a second RMSNorm: gate_proj (8192→29568) and up_proj (8192→29568) run in parallel; ' +
            'output = SiLU(gate) ⊙ up; then down_proj (29568→8192) projects back. All bias-free. ' +
            'Total MLP weight per layer: 3 × 8192 × 29568 × 2 B ≈ 1.45 GB; across 80 layers = ~116 GB. ' +
            'This is the largest single weight component — more than half the model. ' +
            'Decode: all three projections are GEMVs. At H200 bandwidth (3.35 TB/s): ' +
            '1.45 GB ÷ 3.35 TB/s ≈ 0.43 ms per layer → 34 ms for 80 layers per token. ' +
            'Prefill: the GEMMs become compute-bound above ~300 tokens, yielding the highest FLOP utilisation in the model.',
        match: ['Gate + Up', 'SiLU', 'Down Weights', 'MLP'],
        prefill: { compute: 0.95, ai: 780 },
        decode: { compute: 0.13, ai: 1.2 },
        why:
            'Decode: dominant weight volume — ~1.45 GB per layer streamed for 1 token (AI ≈ 1.2 FLOP/B). ' +
            'The down projection is row-parallel → ends in an all-reduce per layer. ' +
            'Prefill: largest GEMMs in the model, saturates H200 compute (AI ≫ 295).',
        comm: true,
    },
    {
        key: 'layers',
        title: '8 · Repeat × N layers',
        summary: 'The block above stacks N times (28 for 7B, 80 for 70B-class).',
        detail:
            'Stages 2–7 repeat 80 times for Qwen 2.5 72B. At decode the dominant cost is ' +
            'reading ~144 GB of weights from HBM3e per generated token. ' +
            'On a single H200 at 3.35 TB/s: 144 GB ÷ 3.35 TB/s ≈ 43 ms/token → 23 tok/s peak. ' +
            'TP=8 halves weight bytes per GPU (18 GB) and all 8 read in parallel → up to 8× faster, ' +
            'limited in practice by all-reduce overhead. Benchmark: disaggregated P/D topology ' +
            '(4 GPUs prefill TP=1 + 4 GPUs decode TP=2) achieved 648 tok/s vs 321 tok/s aggregated TP=8.',
        match: ['Residual'],
        prefill: { compute: 0.90, ai: 700 },
        decode: { compute: 0.12, ai: 1.0 },
        why:
            'Decode: total weight bytes per token ≈ 144 GB (BF16). AI ≈ 1 FLOP/B. ' +
            'TP distributes the load: TP=8 on H200 → 8× the effective bandwidth. ' +
            '2 all-reduces per layer × 80 layers = 160 collectives per token.',
    },
    {
        key: 'lmhead',
        title: '9 · Final norm → logits → sample',
        summary: 'Project to vocabulary, softmax, pick next token.',
        detail:
            'A final RMSNorm then the LM head: a (d_model × vocab_size) = (8192 × 152,064) matrix, ' +
            '~2.5 GB in BF16. At decode only the LAST position produces logits. ' +
            'Softmax over 152K logits → multinomial or greedy sample → 1 token appended to sequence. ' +
            'The entire forward pass then restarts for the next token. ' +
            'With prefix caching (vLLM, SGLang), repeated prompts skip prefill entirely — ' +
            'serving cost collapses to just the KV cache lookup and decode phases.',
        match: ['LM Head', 'Logits'],
        prefill: { compute: 0.88, ai: 480 },
        decode: { compute: 0.11, ai: 1.0 },
        why:
            'Decode: ~2.5 GB weight stream, one token (AI ≈ 1 FLOP/B). ' +
            'Prefill: GEMM only over the LAST position for the next-token prediction (still a large matmul).',
    },
];
