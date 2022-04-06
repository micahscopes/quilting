import { cellsTransformer } from "../src/transformer";
import { PicoGL } from "picogl";
import { setStructUniforms } from "../src/util";
import Algebra from "ganja.js";
import { patchDrawCall } from "../src/patch";

const canvas = document.createElement("canvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
document.body.appendChild(canvas);
const app = PicoGL.createApp(canvas).clearColor(0.0, 0.0, 0.0, 1.0);
app.clear();

  const defaultWeights = () => [
    [ 0, 0.5, 1, 0.2, ],
    [ 0, 0.1, 1, 0.2, ],
    [ 0, 0.5, 0.1, 0.4, ],
  ];
  // const patchesUniforms = bunnyPolys(20).map(([p0, p1, p2]) => ({
  //   p0: new Float32Array(p0),
  //   p1: new Float32Array(p1),
  //   p2: new Float32Array(p2),
  //   ...defaultWeights(),
  // }));

const meshPolys = bunnyPolys(20)

const {
  drawer: transformer,
  getTransformedWeights,
  getTransformedPoints,
  getLODs,
  numCells,
  transformedPointsBuffer,
  transformedWeightsBuffer
} = cellsTransformer(
  app,
  meshPolys,
  meshPolys.map(() => defaultWeights())
);

app.enable(PicoGL.RASTERIZER_DISCARD);

// const CGA = Algebra(4, 1);
// const ni = CGA.Vector(0, 0, 0, 1, 1);
// const no = CGA.Vector(0, 0, 0, -0.5, 0.5);
// const up = (x) => CGA.Add(no, x).Add(CGA.Scalar(0.5).Mul(x).Mul(x).Mul(ni));
// window.mv = up(CGA.Vector(1, 2));

setStructUniforms(transformer, "transformation", { scalar: 1, e1: 1 }).draw();

// console.log('numCells', numCells, 'mesh cells', meshPolys.length)
// const gl = transformer.gl;
// let tmpBuffer = new Float32Array(9 * numCells);
// getTransformedPoints(tmpBuffer);
// console.log(tmpBuffer);

// tmpBuffer = new Float32Array(12 * numCells);
// getTransformedWeights(tmpBuffer);
// console.log(tmpBuffer);

import { sample } from "lodash-es";
// import { sub, normalize } from "@thi.ng/vectors";

const randomElement = (array) =>
  array[Math.floor(Math.random() * array.length)];

import bunnyPolys from "./simple-bunny-polys";
import { makeCamera, setupCamera } from "./camera";
import { requestAnimationFrame } from "@luma.gl/api";

document.addEventListener("DOMContentLoaded", async function () {
  const camera = makeCamera();
  setupCamera(canvas, camera);
  const { seafoamImage, matcapImages } = await import("./load-textures");
  // const N = 5;
  // const patch = Patch(regl, 64, {type: TRI});

  const matcap = app.createTexture2D(randomElement(matcapImages));
  const matcaps = matcapImages.map((data) => app.createTexture2D(data));
  const texture = app.createTexture2D(seafoamImage);


  // const drawCalls = patchesUniforms.map((p) =>
  //   Object.entries(p)
  //     .reduce((drawCall, [k, v]) => {
  //       return drawCall.uniform(k, v);
  //     },
  //     patchDrawCall(app, 128, transformedPointsBuffer, transformedWeightsBuffer))
  //     .texture("texture", texture)
  //     .texture("matcapTexture", matcap)
  // );
  const drawCall = patchDrawCall(app, 32, transformedPointsBuffer, transformedWeightsBuffer)
      .texture("colorTexture", texture)
      .texture("matcapTexture", matcap)

  app.disable(PicoGL.RASTERIZER_DISCARD);
  app.enable(PicoGL.DEPTH_TEST);
  // app.enable(PicoGL.BLEND);
  // app.colorMask(true, true, true, true);
  // app.depthMask(true);

  // console.log(drawCalls);
  // window.drawCalls = drawCalls;

  const raf = () => {
    app.clear();
    camera.tick();
    drawCall
        .uniform("projection", camera.state.projection)
        .uniform("eye", camera.state.eye)
        .uniform("view", camera.state.view)
        .draw()
    // drawCalls.forEach((d) =>
    //   d
    //     .uniform("projection", camera.state.projection)
    //     .uniform("eye", camera.state.eye)
    //     .uniform("view", camera.state.view)
    //     .draw()
    // );
    // console.log('drawing', camera)
    requestAnimationFrame(raf);
  };

  requestAnimationFrame(raf);
});
