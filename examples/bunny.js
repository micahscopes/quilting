import { cellsTransformer } from "../src/transformer";
import { PicoGL } from "picogl";
import { setStructUniforms } from "../src/util";
import Algebra from "ganja.js";
import { patchDrawCall } from "../src/patch";

const canvas = document.createElement("canvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
document.body.appendChild(canvas);
const app = PicoGL.createApp(canvas)
  .clearColor(0.0, 0.0, 0.0, 1.0)
  // .enable(PicoGL.BLEND)
  .blendFunc(PicoGL.ONE, PicoGL.ONE_MINUS_SRC_ALPHA);
app.clear();

// const defaultWeights = () => [
//   [0, 0.5, 1, 0.2],
//   [0, 0.1, 1, 0.2],
//   [0, 0.5, 0.1, 0.4],
// ];
const defaultWeights = () => [
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
];
// const patchesUniforms = bunnyPolys(20).map(([p0, p1, p2]) => ({
//   p0: new Float32Array(p0),
//   p1: new Float32Array(p1),
//   p2: new Float32Array(p2),
//   ...defaultWeights(),
// }));

// const meshPolys = refineBunny(bunny, {});
const meshPolys = prepareMesh(simpleBunny(200));

const {
  drawer: transformer,
  getTransformedWeights,
  getTransformedPoints,
  getLODs,
  numCells,
  transformedPointsBuffer,
  transformedWeightsBuffer,
} = cellsTransformer(
  app,
  meshPolys,
  meshPolys.map(() => defaultWeights())
);

app.enable(PicoGL.RASTERIZER_DISCARD);

setStructUniforms(transformer, "transformation", {
  scalar: 1,
  e1: 1,
  e2: 1,
  e3: 1,
}).draw();

import { sample } from "lodash-es";

const randomElement = (array) =>
  array[Math.floor(Math.random() * array.length)];

import bunnyPolys, { prepareMesh, simpleBunny } from "./polys";
import { makeCamera, setupCamera } from "./camera";
import { requestAnimationFrame } from "@luma.gl/api";
import { arrayToCga3StructProps } from "../src/struct-interface";
import { randomUnit } from "../src/patch-old";

document.addEventListener("DOMContentLoaded", async function () {
  const camera = makeCamera();
  setupCamera(canvas, camera);
  const { swooshImage, seafoamImage, matcapImages } = await import(
    "./load-textures"
  );
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
  const t = {
    // transformation: arrayToCga3StructProps(randomUnit(2, 32)),
    transformation: arrayToCga3StructProps([
      0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1,
    ]),
    lod: 32,
  };

  setInterval(() => {
    t.transformation = arrayToCga3StructProps(randomUnit(1, 32));
  }, 10*1000);

  setInterval(() => {
    t.lod = sample([4, 64, 256]);
  }, 0.5*1000);

  const raf = () => {
    const drawCall = patchDrawCall(
      app,
      t.lod || 32,
      transformedPointsBuffer,
      transformedWeightsBuffer
    )
      .texture("colorTexture", texture)
      .texture("matcapTexture", matcap);
    camera.tick();
    app.enable(PicoGL.RASTERIZER_DISCARD);
    app.disable(PicoGL.DEPTH_TEST);
    setStructUniforms(transformer, "transformation", t.transformation)
      .uniform("angle", Math.PI)
      // .uniform("angle", Date.now() / 1000 - 1649250660482 / 1000)
      .draw();
    app.disable(PicoGL.RASTERIZER_DISCARD);
    app.enable(PicoGL.DEPTH_TEST);
    app.clear();
    drawCall
      // .drawRanges(...(new Array(50)).fill([0,90*4,1]))
      .uniform("projection", camera.state.projection)
      .uniform("eye", camera.state.eye)
      .uniform("view", camera.state.view)
      .draw();
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
