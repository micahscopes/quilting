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

const {
  drawer: transformer,
  getTransformedWeights,
  getTransformedPoints,
  getLODs,
  numCells,
} = cellsTransformer(
  app,
  [
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3],
  ],
  [
    [4, 4, 4, 4],
    [5, 5, 5, 5],
    [6, 6, 6, 6],
  ]
);

app.enable(PicoGL.RASTERIZER_DISCARD);

const CGA = Algebra(4, 1);
const ni = CGA.Vector(0, 0, 0, 1, 1);
const no = CGA.Vector(0, 0, 0, -0.5, 0.5);
const up = (x) => CGA.Add(no, x).Add(CGA.Scalar(0.5).Mul(x).Mul(x).Mul(ni));
window.mv = up(CGA.Vector(1, 2));

setStructUniforms(transformer, "transformation", { scalar: 2, e1: 32 }).draw();

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

  const defaultWeights = () => ({
    w0: new Float32Array([1, 0, 0, 0, ]),
    w1: new Float32Array([1, 0, 0, 0, ]),
    w2: new Float32Array([1, 0, 0, 0, ]),
  });
  const patchesUniforms = bunnyPolys.map(([p0, p1, p2]) => ({
    p0: new Float32Array(p0),
    p1: new Float32Array(p1),
    p2: new Float32Array(p2),
    ...defaultWeights(),
  }));

  const drawCalls = patchesUniforms.map((p) =>
    Object.entries(p)
      .reduce((drawCall, [k, v]) => {
        return drawCall.uniform(k, v);
      },
      patchDrawCall(app, 4))
      .texture("texture", texture)
      .texture("matcapTexture", matcap)
  );

  app.disable(PicoGL.RASTERIZER_DISCARD);
  app.clear();
  // console.log("camera", camera);
  console.log(drawCalls);
  window.drawCalls = drawCalls;

  const raf = () => {
    app.clear();
    camera.tick();
    drawCalls.forEach((d) =>
      d
        .uniform("projection", camera.state.projection)
        .uniform("eye", camera.state.eye)
        .uniform("view", camera.state.view)
        .draw()
    );
    // console.log('drawing', camera)
    requestAnimationFrame(raf);
  };

  requestAnimationFrame(raf);
});
