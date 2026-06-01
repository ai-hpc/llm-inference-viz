// Distinct per-GPU shard colors, shared by the 2D tensor-parallel panel (hex) and the
// 3D model sharding (converted to Vec4 in Program.ts). Index by GPU id modulo length.
export const SHARD_HEX = [
    '#6366f1', // indigo
    '#14b8a6', // teal
    '#f59e0b', // amber
    '#f43f5e', // rose
    '#22c55e', // green
    '#a855f7', // purple
    '#fb923c', // orange
    '#38bdf8', // sky
];
