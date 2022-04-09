import { cellsTransformer } from "../src/transformer";
import { PicoGL } from "picogl";
import { setStructUniforms } from "../src/util";
import Algebra from "ganja.js";
import { patchDrawCall } from "../src/patch";
import createCube from "primitive-cube";

const canvas = document.createElement("canvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
document.body.appendChild(canvas);
const app = PicoGL.createApp(canvas)
  .clearColor(0.0, 0.0, 0.0, 1.0)
  // .enable(PicoGL.BLEND)
  .blendFunc(PicoGL.ONE, PicoGL.ONE_MINUS_SRC_ALPHA);
app.clear();

// Create a Clifford Algebra with 4,1 metric for 3D CGA.
const cga = Algebra(4, 1, () => {
  const Element = this;
  // We start by defining a null basis, and upcasting for points
  var ni = 1e4 + 1e5,
    no = 0.5e5 - 0.5e4;
  var up = (x) => no + x + 0.5 * x * x * ni;

  // Next we'll define 4 points
  var p1 = up(0.5e1),
      p2 = up(0 + 1e2),
      p3 = up(0 - 1e2),
      p4 = up(0 + 1e1 + 1e2);

  // The outer product can be used to construct the sphere through
  // any four points.
  var s = () => p1 ^ p2 ^ p3 ^ p4;

  // The outer product between any three points and infinity is a plane.
  var p = () => p1 ^ p2 ^ p3 ^ ni;

  // Graph the items.
  const graph = this.graph(
    [
      0x00ff0000,
      p1,
      "p1",
      p2,
      "p2",
      p3,
      "p3",
      p4,
      "p4", // points
      0xe0008800,
      p,
      "p", // plane
      0xe00000ff,
      s,
      "s", // sphere
    ],
    { conformal: true, gl: true, grid: true, alpha: true }
  );
  graph.setAttribute("id", "graph");
  window.graph = graph;
  document.body.appendChild(graph);
  return this;
});

import H from "quaternion";

const edgeCentroid = (x, y) => x.add(y).div(2);
const centroid = (...pts) =>
  pts.reduce((x, y) => x.add(y), new H(0)).div(pts.length);
const inv = (x) => new H(1).div(x);
const edgeInf = (x, y) => inv(centroid(x, y));
const patchWeights = (p1, p2, p3) => {
  // console.log(p1)
  p1 = new H(...p1);
  p2 = new H(...p2);
  p3 = new H(...p3);
  let weights = [
    // p1.sub(centroid(p1, p2)),
    // p2.sub(centroid(p2, p3)),
    // p3.sub(centroid(p3, p1)),
    // p1.sub(centroid(p1, p2, p3)),
    // p2.sub(centroid(p2, p3, p1)),
    // p3.sub(centroid(p3, p1, p2)),
    p1.sub(inv(centroid(p1, p2, p3))),
    p2.sub(inv(centroid(p2, p3, p1))),
    p3.sub(inv(centroid(p3, p1, p2))),
    // p1.sub(inv(centroid(p1, p2))),
    // p2.sub(inv(centroid(p2, p3))),
    // p3.sub(inv(centroid(p3, p1))),
  ];
  // weights = [
  //   new H(1,),
  //   new H(1,),
  //   new H(1,),
  // ]
  // weights = weights.map(inv);
  weights = weights.map(x => x.inverse())
  // weights = weights.map((x) => x.div(weights[0]));
  // weights = weights.map(x => x.normalize())

  // console.log(p1.sub(edgeInf(p1,p2)));
  return weights.map((x) => x.toVector());
}; //.map(normalize(4));
console.log(patchWeights([1, 1, 1, 1], [10, 10, 10, 10], [32, 32, 32, 32]));

// const meshPolys = refineBunny(bunny, {});
// const mesh = refinedBunny(0.2);
// const mesh = simpleBunny(200);
const divs = 4;
const mesh = createCube(1, 1, 1, divs);
console.log("number of cells", mesh.cells.length);
const meshPolys = prepareMesh(mesh);
const meshWeights = meshPolys.map((points) => patchWeights(...points))
console.log(meshPolys, meshWeights)
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
  meshWeights
);

app.enable(PicoGL.RASTERIZER_DISCARD);

setStructUniforms(transformer, "transformation", {
  scalar: 1,
  e1: 1,
  e2: 1,
  e3: 1,
}).draw();

import { sample, sum } from "lodash-es";

const randomElement = (array) =>
  array[Math.floor(Math.random() * array.length)];

import bunnyPolys, { prepareMesh, refinedBunny, simpleBunny } from "./polys";
import { makeCamera, setupCamera } from "./camera";
import { requestAnimationFrame } from "@luma.gl/api";
import { arrayToCga3StructProps } from "../src/struct-interface";
import { randomUnit } from "../src/patch-old";

document.addEventListener("DOMContentLoaded", async function () {
  const camera = makeCamera();
  window.cam = camera;
  setupCamera(canvas, camera);
  const { swooshImage, seafoamImage, matcapImages } = await import(
    "./load-textures"
  );
  // const N = 5;
  // const patch = Patch(regl, 64, {type: TRI});

  // const matcap = app.createTexture2D(randomElement(matcapImages));
  const matcap = app.createTexture2D(matcapImages[4]);
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
    lod: 50,
  };

  setInterval(() => {
    t.transformation = arrayToCga3StructProps(randomUnit(1, 32));
  }, 10 * 1000);

  // setInterval(() => {
  //   t.lod = sample([1, 2, 3]);
  // }, 0.5 * 1000);

  const drawCall = patchDrawCall(
    app,
    t.lod || 32,
    transformedPointsBuffer,
    transformedWeightsBuffer
  )
    .texture("colorTexture", texture)
    .texture("matcapTexture", matcap);

  const raf = () => {
    const sph = arrayToCga3StructProps(graph.value[13]());
    // console.log(sph)
    camera.tick();
    app.enable(PicoGL.RASTERIZER_DISCARD);
    app.disable(PicoGL.DEPTH_TEST);
    setStructUniforms(transformer, "transformation", sph)
      // .uniform("angle", Math.PI)
      // .uniform("angle", Date.now() / 1000 - 1649250660482 / 1000)
      .draw();
    app.disable(PicoGL.RASTERIZER_DISCARD);
    app.enable(PicoGL.DEPTH_TEST);
    app.clear();
    drawCall
      // .drawRanges(...(new Array(50)).fill([0,90*4,1]))
      .uniform("projection", new Float32Array(graph.options.proj))
      .uniform("eye", camera.state.eye)
      .uniform("view", new Float32Array(graph.options.mv))
      .draw();

    requestAnimationFrame(raf);
    // const gl = transformer.gl;
    // let tmpBuffer = new Float32Array(9);
    // getTransformedWeights(tmpBuffer);
    // console.log(tmpBuffer);
  };

  requestAnimationFrame(raf);
});
