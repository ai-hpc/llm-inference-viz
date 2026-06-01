// The forward pass of a dense, decoder-only transformer (the Qwen 2.5 / Llama 3.3 family),
// in execution order. Each stage maps to one or more named blocks in the 3D layout
// (`match`) so the left explainer panel can light up the matching geometry on hover.

export interface ITransformerStage {
    key: string;
    title: string;
    summary: string;   // always shown
    detail: string;    // shown on hover
    match: string[];   // cube-name substrings (case-insensitive) to highlight on the right
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
    },
    {
        key: 'rmsnorm',
        title: '2 · RMSNorm (pre-norm)',
        summary: 'Normalise the residual before each sub-layer.',
        detail: 'RMSNorm divides each vector by its root-mean-square (no mean subtraction, ' +
            'no bias term) and rescales by a learned γ. It is cheaper and more stable than ' +
            'LayerNorm and is applied pre-attention and pre-MLP (pre-norm residual design).',
        match: ['RMS Norm', 'γ'],
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
    },
    {
        key: 'attnout',
        title: '6 · Attention output + residual',
        summary: 'Weighted sum of V, output projection, add back.',
        detail: 'The softmax weights produce a weighted sum of the value vectors per head; ' +
            'heads are concatenated and mixed by the output projection. The result is ADDED ' +
            'to the residual stream (the skip connection) rather than replacing it.',
        match: ['V Output', 'Attention Output', 'Projection Weights', 'Projection Bias', 'Attention Residual'],
    },
    {
        key: 'mlp',
        title: '7 · SwiGLU MLP + residual',
        summary: 'Gate ⊙ Up → Down, then add back.',
        detail: 'After a second RMSNorm, the feed-forward block uses SwiGLU: two parallel ' +
            'projections up to the inner dim (gate and up), combined as SiLU(gate) ⊙ up, then ' +
            'a down projection back to d_model. It is bias-free and ~3.5× wider than d_model. ' +
            'The output is added to the residual stream.',
        match: ['Gate + Up', 'SiLU', 'Down Weights', 'MLP'],
    },
    {
        key: 'layers',
        title: '8 · Repeat × N layers',
        summary: 'The block above stacks N times.',
        detail: 'Stages 2–7 form one transformer block. It repeats N times (28 layers for ' +
            'Qwen 2.5 7B, 80 for the 70B-class models), each with its own weights, the ' +
            'residual stream threading straight through all of them.',
        match: ['Residual'],
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
    },
];
