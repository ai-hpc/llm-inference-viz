import { genModelViewMatrices, ICamera, ICameraPos, updateCamera } from "./Camera";
import { drawAllArrows } from "./components/Arrow";
import { drawBlockLabels } from "./components/SectionLabels";
import { drawModelCard } from "./components/ModelCard";
import { IGptModelLink, IGpuGptModel, IModelShape } from "./GptModel";
import { genGptModelLayout, IBlkDef, IGptModelLayout } from "./GptModelLayout";
import { drawText, IFontAtlasData, IFontOpts, measureText } from "./render/fontRender";
import { initRender, IRenderState, IRenderView, renderModel, resetRenderBuffers } from "./render/modelRender";
import { beginQueryAndGetPrevMs, endQuery } from "./render/queryManager";
import { SavedState } from "./SavedState";
import { isNotNil } from "@/src/utils/data";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { initWalkthrough, runWalkthrough } from "./walkthrough/Walkthrough";
import { IColorMix } from "./Annotations";
import { Mat4f } from "@/src/utils/matrix";
import { runMouseHitTesting } from "./Interaction";
import { RenderPhase } from "./render/sharedRender";
import { drawBlockInfo } from "./components/BlockInfo";
import { NativeFunctions } from "./NativeBindings";
import { IWasmGptModel, stepWasmModel, syncWasmDataWithJsAndGpu } from "./GptModelWasm";
import { IMovementInfo, manageMovement } from "./components/MovementControls";
import { IBlockRender } from "./render/blockRender";
import { ILayout } from "../utils/layout";
import { DimStyle } from "./walkthrough/WalkthroughTools";
import { Subscriptions } from "../utils/hooks";

export interface IProgramState {
    native: NativeFunctions | null;
    wasmGptModel: IWasmGptModel | null;
    stepModel: boolean;
    mouse: IMouseState;
    render: IRenderState;
    inWalkthrough: boolean;
    walkthrough: ReturnType<typeof initWalkthrough>;
    camera: ICamera;
    htmlSubs: Subscriptions;
    layout: IGptModelLayout;
    mainExample: IModelExample;
    examples: IModelExample[];
    currExampleId: number;
    modelSet: IModelPreset[];   // switchable models (Qwen 7B / Qwen 72B / Llama 70B)
    currentModelIdx: number;    // index into modelSet of the currently displayed model
    shape: IModelShape;
    gptGpuModel: IGpuGptModel | null;
    jsGptModel: IGptModelLink | null;
    movement: IMovementInfo;
    display: IDisplayState;
    pageLayout: ILayout;
    markDirty: () => void;
}

export interface IModelPreset {
    name: string;
    shape: IModelShape;
    camera: ICameraPos;
}

export interface IModelExample {
    name: string;
    shape: IModelShape;
    enabled: boolean;
    layout?: IGptModelLayout;
    blockRender: IBlockRender;
    offset: Vec3;
    modelCardOffset: Vec3;
    camera?: ICameraPos;
}

export interface IMouseState {
    mousePos: Vec3;
}

export interface IDisplayState {
    tokenColors: IColorMix | null;
    tokenIdxColors: IColorMix | null;
    tokenOutputColors: IColorMix | null;
    tokenIdxModelOpacity?: number[];
    topOutputOpacity?: number;
    lines: string[];
    hoverTarget: IHoverTarget | null;
    blkIdxHover: number[] | null;
    dimHover: DimStyle | null;
}

export interface IHoverTarget {
    subCube: IBlkDef;
    mainCube: IBlkDef;
    mainIdx: Vec3;
}

export function initProgramState(canvasEl: HTMLCanvasElement, fontAtlasData: IFontAtlasData): IProgramState {

    let render = initRender(canvasEl, fontAtlasData);
    let walkthrough = initWalkthrough();

    let prevState = SavedState.state;

    function makeCamera(center: Vec3, angle: Vec3): ICameraPos {
        return { center, angle };
    }

    // ---- Switchable models. All three share the modern-arch operator set
    // (RMSNorm + RoPE + GQA + SwiGLU), so arch: 'qwen' drives the geometry for all. ----

    // Qwen 2.5 7B Instruct (DEFAULT). hidden 3584, 28 Q / 4 KV heads, head dim 128,
    // 28 layers, SwiGLU ffn 18944, vocab 152064.
    let qwen7bShape: IModelShape = {
        B: 1, T: 1024, C: 3584, nHeads: 28, nKVHeads: 4, A: 128,
        nBlocks: 28, ffnDim: 18944, vocabSize: 152064,
        arch: 'qwen', normEps: 1e-6, ropeTheta: 1000000,
    };

    // Qwen 2.5 72B Instruct. hidden 8192, 64 Q / 8 KV heads, head dim 128,
    // 80 layers, SwiGLU ffn 29568, vocab 152064.
    let qwen72bShape: IModelShape = {
        B: 1, T: 1024, C: 8192, nHeads: 64, nKVHeads: 8, A: 128,
        nBlocks: 80, ffnDim: 29568, vocabSize: 152064,
        arch: 'qwen', normEps: 1e-6, ropeTheta: 1000000,
    };

    // Llama 3.3 70B Instruct. hidden 8192, 64 Q / 8 KV heads, head dim 128,
    // 80 layers, SwiGLU ffn 28672, vocab 128256. (No QKV bias, unlike Qwen 2.5.)
    let llama70bShape: IModelShape = {
        B: 1, T: 1024, C: 8192, nHeads: 64, nKVHeads: 8, A: 128,
        nBlocks: 80, ffnDim: 28672, vocabSize: 128256,
        arch: 'qwen', normEps: 1e-5, ropeTheta: 500000,
    };

    let modelSet: IModelPreset[] = [
        { name: 'Qwen 2.5 7B',   shape: qwen7bShape,   camera: makeCamera(new Vec3(-21000, 0, -150000), new Vec3(238.959, 10.501, 5200)) },
        { name: 'Qwen 2.5 72B',  shape: qwen72bShape,  camera: makeCamera(new Vec3(-62322.0, 0, -485242.286), new Vec3(238.959, 10.501, 12583.939)) },
        { name: 'Llama 3.3 70B', shape: llama70bShape, camera: makeCamera(new Vec3(-62322.0, 0, -485242.286), new Vec3(238.959, 10.501, 12583.939)) },
    ];
    let currentModelIdx = 0; // default: Qwen 2.5 7B

    // Live camera starts framed on the default model (Qwen 2.5 7B).
    let camera: ICamera = {
        angle: new Vec3(238.959, 10.501, 5200),
        center: new Vec3(-21000, 0, -150000),
        transition: {},
        modelMtx: new Mat4f(),
        viewMtx: new Mat4f(),
        lookAtMtx: new Mat4f(),
        camPos: new Vec3(),
        camPosModel: new Vec3(),
    }

    return {
        native: null,
        wasmGptModel: null,
        render: render!,
        inWalkthrough: false, // structural views only; the nano-gpt guided walkthrough is disabled
        walkthrough,
        camera,
        shape: modelSet[currentModelIdx].shape,
        layout: genGptModelLayout(modelSet[currentModelIdx].shape),
        currExampleId: -1,
        modelSet,
        currentModelIdx,
        mainExample: {
            name: modelSet[currentModelIdx].name,
            enabled: true,
            shape: modelSet[currentModelIdx].shape,
            offset: new Vec3(),
            modelCardOffset: new Vec3(),
            blockRender: null!,
            camera: modelSet[currentModelIdx].camera,
        },
        examples: [],
        gptGpuModel: null,
        jsGptModel: null,
        stepModel: false,
        markDirty: () => { },
        htmlSubs: new Subscriptions(),
        mouse: {
            mousePos: new Vec3(),
        },
        movement: {
            action: null,
            actionHover: null,
            target: [0, 0],
            depth: 1,
            cameraLerp: null,
         },
        display: {
            tokenColors: null,
            tokenIdxColors: null,
            tokenOutputColors: null,
            lines: [],
            hoverTarget: null,
            dimHover: null,
            blkIdxHover: null,
        },
        pageLayout: {
            height: 0,
            width: 0,
            isDesktop: true,
            isPhone: true,
        }
    };
}

export function runProgram(view: IRenderView, state: IProgramState) {
    let timer0 = performance.now();

    if (!state.render) {
        return;
    }

    resetRenderBuffers(state.render);
    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    state.display.lines = [];
    state.display.hoverTarget = null;
    state.display.tokenColors = null;
    state.display.tokenIdxColors = null;

    if (state.wasmGptModel && state.jsGptModel) {
        syncWasmDataWithJsAndGpu(state.wasmGptModel, state.jsGptModel);
    }

    if (state.stepModel && state.wasmGptModel && state.jsGptModel) {
        state.stepModel = false;
        stepWasmModel(state.wasmGptModel, state.jsGptModel);
    }

    // generate the base model. Qwen-only build renders structurally (no live weights).
    state.layout = genGptModelLayout(state.shape, null);

    // @TODO: handle different models in the same scene.
    // Maybe need to copy a lot of different things like the entire render state per model?
    for (let example of state.examples) {
        if (example.enabled && !example.layout) {
            let layout = genGptModelLayout(example.shape, null, example.offset);
            example.layout = layout;
        }
    }

    genModelViewMatrices(state, state.layout!);

    let queryRes = beginQueryAndGetPrevMs(state.render.queryManager, 'render');
    if (isNotNil(queryRes)) {
        state.render.lastGpuMs = queryRes;
    }

    state.render.renderTiming = false; // state.pageLayout.isDesktop;

    // will modify layout; view; render a few things.
    if (state.inWalkthrough) {
        runWalkthrough(state, view);
    }

    updateCamera(state, view);

    drawBlockInfo(state);
    // these will get modified by the walkthrough (stored where?)
    drawAllArrows(state.render, state.layout);

    drawModelCard(state, state.layout, state.modelSet[state.currentModelIdx].name, new Vec3());
    // drawTokens(state.render, state.layout, state.display);

    for (let example of state.examples) {
        if (example.enabled && example.layout) {
            drawModelCard(state, example.layout, example.name, example.offset.add(example.modelCardOffset));
        }
    }

    // manageMovement(state, view);
    runMouseHitTesting(state);
    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    drawBlockLabels(state.render, state.layout);

    let lineNo = 1;
    let tw = state.render.size.x;
    state.render.sharedRender.activePhase = RenderPhase.Overlay2D;
    for (let line of state.display.lines) {
        let opts: IFontOpts = { color: new Vec4(), size: 14 };
        let w = measureText(state.render.modelFontBuf, line, opts);
        drawText(state.render.modelFontBuf, line, tw - w - 4, lineNo * opts.size * 1.3 + 4, opts)
        lineNo++;
    }

    // render everything; i.e. here's where we actually do gl draw calls
    // up until now, we've just been putting data in cpu-side buffers
    renderModel(state);

    endQuery(state.render.queryManager, 'render');
    state.render.gl.flush();

    state.render.lastJsMs = performance.now() - timer0;
}
