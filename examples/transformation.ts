import { cellsTransformer } from "../src/transformer";
import { PicoGL } from "picogl";
import { setStructUniforms } from "../src/util";
import Algebra from 'ganja.js';

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
    [0, 1, 1, 1],
    [0, 2, 2, 2],
    [0, 3, 3, 3],
  ],
  [
    [4, 4, 4, 4],
    [5, 5, 5, 5],
    [6, 6, 6, 6],
  ]
);

app.enable(PicoGL.RASTERIZER_DISCARD)

const CGA = Algebra(4,1)
const ni = CGA.Vector(0,0,0,1,1)
const no = CGA.Vector(0,0,0,-0.5,0.5)
const up = (x) => CGA.Add(no, x).Add(CGA.Scalar(.5).Mul(x).Mul(x).Mul(ni))
window.mv = up(CGA.Vector(1,2))

setStructUniforms(transformer, "transformation", { scalar: 2, e1: 32 })
  .draw();

const gl = transformer.gl;
let tmpBuffer = new Float32Array(4 * numCells);
getTransformedPoints(tmpBuffer);
console.log(tmpBuffer);

tmpBuffer = new Float32Array(12 * numCells);
getTransformedWeights(tmpBuffer);
console.log(tmpBuffer);

tmpBuffer = new Float32Array(numCells);
getLODs(tmpBuffer);
console.log(tmpBuffer);
