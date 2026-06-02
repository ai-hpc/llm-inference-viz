import { camScaleToScreen } from "../Camera";
import { IProgramState } from "../Program";
import { drawText, IFontOpts, measureText } from "../render/fontRender";
import { RenderPhase } from "../render/sharedRender";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { drawRoundedRect } from "./DataFlow";

export function drawBlockInfo(state: IProgramState) {

    // Per-cube text labels are only legible for small models. A large structural model
    // (e.g. Qwen 2.5 72B: 80 blocks → tens of thousands of cubes) would write a sea of
    // overlapping labels — unreadable, slow, and well past the font buffer. Skip them;
    // the model card carries the summary instead.
    if (state.layout.shape.nBlocks > 12) {
        return;
    }

    for (let blk of state.layout.cubes) {

        let blkTopMid = new Vec3(blk.x + blk.dx / 2, blk.y, blk.z + blk.dz / 2);

        let scale = camScaleToScreen(state, blkTopMid);

        scale = Math.min(scale, 1.45);
        // have a max scale

        let textColor = new Vec4(1, 1, 1, 1).mul(blk.opacity);
        let bgColor = new Vec4(0, 0, 0, 1).mul(blk.opacity);

        if (blk.opacity === 0 || !blk.name) {
            continue;
        }

        // draw text, centered on top of the block
        let text = blk.name;
        let mtx = Mat4f.fromTranslation(blkTopMid);
        let textOpts: IFontOpts = { color: textColor, size: scale * 2.5, mtx };
        let textW = measureText(state.render.modelFontBuf, text, textOpts);

        let pad = 0.4;
        state.render.sharedRender.activePhase = RenderPhase.Opaque;
        drawRoundedRect(state.render, new Vec3(-textW / 2 - pad, -textOpts.size - pad * 2, 0), new Vec3(textW / 2 + pad, 0, 0), bgColor, mtx, scale * 0.4);

        state.render.sharedRender.activePhase = RenderPhase.Overlay;
        drawText(state.render.modelFontBuf, text, -textW / 2, -textOpts.size - pad, textOpts);
    }
}

// Annotate the rendered token window as a *slice* of the real context, e.g. a 1,024-token
// window standing in for a 32K context — drawn with braces + an ellipsis rather than
// rendering 32,768 cells (which the geometry can't do).
export function drawContextWindowLabel(state: IProgramState) {
    let blk = state.layout.residual0;
    if (!blk) {
        return;
    }
    let T = state.shape.T;
    let ctx = state.display.ctxLen ?? state.shape.ctxLen ?? T;
    if (ctx <= T) {
        return; // window already covers the whole context — nothing to express
    }

    let ctxFmt = ctx >= 1024 ? `${Math.round(ctx / 1024)}K` : `${ctx}`;
    let shownFmt = T.toLocaleString();

    // Centered label floating above the input-embed block (the token axis runs along x).
    let mid = new Vec3(blk.x + blk.dx / 2, blk.y, blk.z + blk.dz / 2);
    let scale = Math.min(camScaleToScreen(state, mid), 1.6);

    let label = `{  ${ctxFmt}-token context  …  ${shownFmt} shown  }`;
    let mtx = Mat4f.fromTranslation(mid);
    let opts: IFontOpts = { color: new Vec4(0.13, 0.16, 0.28, 1), size: scale * 3.2, mtx };
    let w = measureText(state.render.modelFontBuf, label, opts);
    let pad = scale * 1.0;

    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    drawRoundedRect(state.render,
        new Vec3(-w / 2 - pad, -opts.size - pad * 3, 0),
        new Vec3(w / 2 + pad, -pad * 1.5, 0),
        new Vec4(1, 1, 1, 0.86), mtx, scale * 0.6);
    state.render.sharedRender.activePhase = RenderPhase.Overlay;
    drawText(state.render.modelFontBuf, label, -w / 2, -opts.size - pad * 2, opts);

    // Trailing "…" just past the end of the shown tokens, implying the window continues.
    let tail = new Vec3(blk.x + blk.dx + blk.dx * 0.04, blk.y + blk.dy / 2, blk.z + blk.dz / 2);
    let tailScale = Math.min(camScaleToScreen(state, tail), 2.0);
    let tailMtx = Mat4f.fromTranslation(tail);
    let tailOpts: IFontOpts = { color: new Vec4(0.45, 0.5, 0.62, 1), size: tailScale * 5, mtx: tailMtx };
    state.render.sharedRender.activePhase = RenderPhase.Overlay;
    drawText(state.render.modelFontBuf, '…', 0, tailOpts.size / 2, tailOpts);
}
