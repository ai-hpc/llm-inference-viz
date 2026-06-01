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
