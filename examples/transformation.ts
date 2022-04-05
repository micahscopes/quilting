import { cellTransformer } from "../src/transformation";
import { PicoGL } from "picogl";
import type { VertexBuffer } from "picogl";

const canvas = document.createElement("canvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
document.body.appendChild(canvas);
const app = PicoGL.createApp(canvas).clearColor(0.0, 0.0, 0.0, 1.0);
app.clear();

let pointIn = app.createVertexBuffer(
  PicoGL.FLOAT,
  3,
  new Float32Array([1, 2, 3])
);
let pointOut = app.createVertexBuffer(PicoGL.FLOAT, 3, 3);

const vertexArray = app.createVertexArray().vertexAttributeBuffer(0, pointIn);

var transformFeedback = app
  .createTransformFeedback()
  .feedbackBuffer(0, pointOut);

const transformer = cellTransformer(app);

var drawCall = app
  .createDrawCall(transformer, vertexArray)
  .primitive(PicoGL.POINTS)
  .transformFeedback(transformFeedback);

app.enable(PicoGL.RASTERIZER_DISCARD);
app.clear();

drawCall.uniform("scale", 40).draw();
drawCall.draw();

// output
const gl = drawCall.gl;
const outputBuffer = new Float32Array(3)
gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, pointOut.buffer);
gl.getBufferSubData(gl.TRANSFORM_FEEDBACK_BUFFER, 0, outputBuffer)
console.log(outputBuffer)

