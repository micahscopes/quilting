import { cellsTransformer } from "../src/transformation";
import { PicoGL } from "picogl";
import { fill } from "lodash-es";
import { setStructUniforms } from "../src/util";

const canvas = document.createElement("canvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
document.body.appendChild(canvas);
const app = PicoGL.createApp(canvas).clearColor(0.0, 0.0, 0.0, 1.0);
app.clear();

const {
  drawer,
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

setStructUniforms(drawer, "transformation", { scalar: 2, e1: 32 })
  .uniform("scale", 1)
  .draw();

console.log(drawer.setTransformation);
// console.log(drawCall.appState)

// var arrBuffer = new ArrayBuffer((9+12) * Float32Array.BYTES_PER_ELEMENT);
const gl = drawer.gl;
let tmpBuffer = new Float32Array(9 * numCells);
getTransformedPoints(tmpBuffer);
console.log(tmpBuffer);

tmpBuffer = new Float32Array(12 * numCells);
getTransformedWeights(tmpBuffer);
console.log(tmpBuffer);

tmpBuffer = new Float32Array(numCells);
getLODs(tmpBuffer);
console.log(tmpBuffer);
