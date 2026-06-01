// The forward pass of a dense, decoder-only transformer (the Qwen 2.5 / Llama 3.3 family),
// in execution order. Each stage maps to one or more named blocks in the 3D layout
// (`match`) so the left explainer panel can light up the matching geometry on hover.
//
// Each stage also carries a roofline character per regime: `compute` is a 0..1 "compute
// intensity" (0 = pure memory-bound, 1 = pure compute-bound) used to draw the per-stage
// bar, and `ai` is an order-of-magnitude arithmetic intensity (FLOP/byte) used to place
// the stage on the roofline chart. Numbers are illustrative, not measured.

export type Regime = 'prefill' | 'decode';

export interface IRooflinePoint {
    compute: number; // 0 (memory-bound) .. 1 (compute-bound)
    ai: number;      // arithmetic intensity, FLOP/byte (log-scale marker)
}

export interface ITransformerStage {
    key: string;
    title: string;
    summary: string;
    detail: string;
    match: string[];                 // cube-name substrings (case-insensitive) to highlight
    prefill: IRooflinePoint;
    decode: IRooflinePoint;
    why: string;                     // one line: why this is compute- or memory-bound
    comm?: boolean;                  // has an all-reduce under tensor parallelism (row-parallel sub-layer)
}

export type Bound = 'Compute-bound' | 'Memory-bound' | 'Mixed' | 'Comm-bound';

export function boundLabel(c: number): Bound {
    if (c >= 0.66) return 'Compute-bound';
    if (c <= 0.34) return 'Memory-bound';
    return 'Mixed';
}

// Per-GPU roofline point adjusted for tensor parallelism. TP divides BOTH the FLOPs and the
// weight bytes per GPU, so per-GPU arithmetic intensity is ~unchanged for the matmuls.
// The exception is the two row-parallel stages (attention output, MLP down): each ends in an
// all-reduce whose cost grows with TP while per-GPU compute shrinks — so their effective AI
// falls and they become communication-bound at high TP.
export function effectivePoint(stage: ITransformerStage, regime: Regime, tp: number): { compute: number; ai: number; comm: boolean } {
    let base = stage[regime];
    if (stage.comm && tp > 1) {
        let commFactor = 1 + 0.6 * (tp - 1); // illustrative all-reduce growth with TP
        return { compute: base.compute / commFactor, ai: base.ai / commFactor, comm: true };
    }
    return { compute: base.compute, ai: base.ai, comm: false };
}

export function effectiveBound(stage: ITransformerStage, regime: Regime, tp: number): Bound {
    let p = effectivePoint(stage, regime, tp);
    if (p.comm) return 'Comm-bound';
    return boundLabel(p.compute);
}

export function fmtAi(ai: number): string {
    return ai >= 10 ? Math.round(ai).toString() : ai.toFixed(ai < 1 ? 2 : 1);
}

export const TRANSFORMER_STAGES: ITransformerStage[] = [
    {
        key: 'embed',
        title: '1 · Token embedding',
        summary: 'Token ids → d_model vectors.',
        detail: 'Each input token id indexes a row of the embedding matrix, producing a ' +
            'vector of width d_model (the residual stream). There is NO learned positional ' +
            'embedding — modern dense decoders inject position later via RoPE, inside attention.',
        match: ['Token Embed', 'Input Embed', 'Tokens'],
        prefill: { compute: 0.08, ai: 0.3 },
        decode: { compute: 0.05, ai: 0.25 },
        why: 'A memory gather — one row read per token, almost no arithmetic.',
    },
    {
        key: 'rmsnorm',
        title: '2 · RMSNorm (pre-norm)',
        summary: 'Normalise the residual before each sub-layer.',
        detail: 'RMSNorm divides each vector by its root-mean-square (no mean subtraction, ' +
            'no bias term) and rescales by a learned γ. It is cheaper and more stable than ' +
            'LayerNorm and is applied pre-attention and pre-MLP (pre-norm residual design).',
        match: ['RMS Norm', 'γ'],
        prefill: { compute: 0.12, ai: 0.5 },
        decode: { compute: 0.06, ai: 0.3 },
        why: 'Elementwise over the activation — bandwidth-bound in both regimes.',
    },
    {
        key: 'qkv',
        title: '3 · Q / K / V projection (GQA)',
        summary: 'Project to queries, keys, values — with grouped-query attention.',
        detail: 'The normed residual is projected to Q, K and V. With Grouped-Query ' +
            'Attention there are FEWER key/value heads than query heads (e.g. 64 Q / 8 KV), ' +
            'so several query heads share one KV head. That shrinks the KV cache ~8× — the ' +
            'main memory bottleneck during decode.',
        match: ['Q Weights', 'K Weights', 'V Weights', 'Q vectors', 'K vectors', 'V vectors', 'QKV'],
        prefill: { compute: 0.9, ai: 120 },
        decode: { compute: 0.15, ai: 2 },
        why: 'Prefill: a GEMM that reuses each weight across all tokens (compute-bound). ' +
            'Decode: a GEMV — stream the whole weight matrix to multiply by one vector (memory-bound).',
    },
    {
        key: 'rope',
        title: '4 · RoPE (rotary position)',
        summary: 'Rotate Q and K by a position-dependent angle.',
        detail: 'Rotary Position Embedding rotates each Q and K vector by an angle ' +
            'proportional to its position. Because attention scores depend on the angle ' +
            'difference, the model sees RELATIVE positions. No learned table, and it ' +
            'extrapolates to long context — applied inside attention, not at the embedding.',
        match: ['Q vectors', 'K vectors'],
        prefill: { compute: 0.1, ai: 0.4 },
        decode: { compute: 0.05, ai: 0.3 },
        why: 'A cheap elementwise rotation — memory-bound, negligible FLOPs.',
    },
    {
        key: 'attn',
        title: '5 · Scores + causal softmax',
        summary: 'QKᵀ/√d, mask the future, softmax.',
        detail: 'Each query dots every key (scaled by 1/√head_dim). A causal mask zeroes ' +
            'the upper triangle so a position can only attend to itself and the past — the ' +
            'defining property of DECODE. Softmax turns scores into weights. At each decode ' +
            'step this reads the whole KV cache, which is why decode is bandwidth-bound.',
        match: ['Attention Matrix', 'Attn Matrix Softmax'],
        prefill: { compute: 0.8, ai: 60 },
        decode: { compute: 0.1, ai: 1.2 },
        why: 'Prefill: O(N²) QKᵀ GEMM is compute-heavy. Decode: read the entire KV cache ' +
            'for a few FLOPs per element — pure bandwidth, and it grows with context length.',
    },
    {
        key: 'attnout',
        title: '6 · Attention output + residual',
        summary: 'Weighted sum of V, output projection, add back.',
        detail: 'The softmax weights produce a weighted sum of the value vectors per head; ' +
            'heads are concatenated and mixed by the output projection. The result is ADDED ' +
            'to the residual stream (the skip connection) rather than replacing it.',
        match: ['V Output', 'Attention Output', 'Projection Weights', 'Projection Bias', 'Attention Residual'],
        prefill: { compute: 0.9, ai: 120 },
        decode: { compute: 0.15, ai: 2 },
        why: 'Another projection GEMM/GEMV — compute-bound batched (prefill), memory-bound at batch=1 (decode). ' +
            'Row-parallel under TP, so it ends in an all-reduce.',
        comm: true,
    },
    {
        key: 'mlp',
        title: '7 · SwiGLU MLP + residual',
        summary: 'Gate ⊙ Up → Down, then add back.',
        detail: 'After a second RMSNorm, the feed-forward block uses SwiGLU: two parallel ' +
            'projections up to the inner dim (gate and up), combined as SiLU(gate) ⊙ up, then ' +
            'a down projection back to d_model. It is bias-free and ~3.5× wider than d_model. ' +
            'It holds the majority of the model’s weights.',
        match: ['Gate + Up', 'SiLU', 'Down Weights', 'MLP'],
        prefill: { compute: 0.95, ai: 200 },
        decode: { compute: 0.12, ai: 2 },
        why: 'The biggest matmuls and most of the weights. Prefill: the dominant FLOPs ' +
            '(compute-bound). Decode: the dominant weight bytes streamed from HBM (memory-bound). ' +
            'The down projection is row-parallel, so it ends in an all-reduce.',
        comm: true,
    },
    {
        key: 'layers',
        title: '8 · Repeat × N layers',
        summary: 'The block above stacks N times.',
        detail: 'Stages 2–7 form one transformer block. It repeats N times (28 layers for ' +
            'Qwen 2.5 7B, 80 for the 70B-class models), each with its own weights, the ' +
            'residual stream threading straight through all of them.',
        match: ['Residual'],
        prefill: { compute: 0.85, ai: 150 },
        decode: { compute: 0.1, ai: 2 },
        why: 'Per token, decode must read EVERY layer’s weights once — total weight bytes ' +
            'set the decode speed (memory bandwidth bound).',
    },
    {
        key: 'lmhead',
        title: '9 · Final norm → logits → sample',
        summary: 'Project to vocabulary, softmax, pick next token.',
        detail: 'A final RMSNorm, then the LM head projects the last position’s vector to one ' +
            'logit per vocabulary token; softmax gives a probability distribution. Decode ' +
            'samples ONE token from it, appends it to the sequence, and the whole pass runs ' +
            'again for the next token — one token per forward pass.',
        match: ['LM Head', 'Logits'],
        prefill: { compute: 0.85, ai: 100 },
        decode: { compute: 0.1, ai: 1.5 },
        why: 'A large (vocab × d_model) matrix. At decode it is one more big weight read — memory-bound.',
    },
];
