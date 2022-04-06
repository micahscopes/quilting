// generate a PicoGL program that:
// - lifts cell points/weights into conformal model
// - applies a transformation defined by params passed in via uniform
// - computes cell LOD levels
// - writes transformed points/weights as vectors into output buffer

import { flatten } from "lodash-es";
import type { App } from "picogl";
import { PicoGL } from "picogl";
import { bufferGetter, glsl } from "./util";
import transformationGL from "../gl/transformation.glsl";

const cellTransformerProgram = (app: App, numSides = 3) =>
  app.createProgram(
    transformationGL,
    glsl`
#version 300 es
precision highp float;
void main() {}
`,
    {
      transformFeedbackVaryings: [
        "transformedPoints",
        "transformedWeights",
        "lod",
      ],
    }
  );

export const cellsTransformer = (
  app: App,
  points: number[][] | Float32Array,
  weights: number[][] | Float32Array
) => {
  const numCells = points.length / 3;
  points = new Float32Array(flatten(points));
  weights = new Float32Array(flatten(weights));

  console.log(points, weights);

  let pointsIn = app.createMatrixBuffer(PicoGL.FLOAT_MAT3, points);
  let weightsIn = app.createMatrixBuffer(PicoGL.FLOAT_MAT4x3, weights);

  let transformedPointsBuffer = app.createMatrixBuffer(
    PicoGL.FLOAT_MAT3,
    numCells * 9
  );
  let transformedWeightsBuffer = app.createMatrixBuffer(
    PicoGL.FLOAT_MAT3x4,
    numCells * 12
  );
  let lodBuffer = app.createVertexBuffer(PicoGL.FLOAT, 4, 2);

  const vertexArray = app
    .createVertexArray()
    .vertexAttributeBuffer(0, pointsIn)
    .vertexAttributeBuffer(1, pointsIn)
    .vertexAttributeBuffer(2, pointsIn)
    .vertexAttributeBuffer(3, weightsIn)
    .vertexAttributeBuffer(4, weightsIn)
    .vertexAttributeBuffer(5, weightsIn);

  const transformFeedback = app
    .createTransformFeedback()
    .feedbackBuffer(0, transformedPointsBuffer)
    .feedbackBuffer(1, transformedWeightsBuffer)
    .feedbackBuffer(2, lodBuffer);

  const transformer = cellTransformerProgram(app);
  const getTransformedPoints = bufferGetter(
    app.gl,
    transformedPointsBuffer.buffer
  );
  const getTransformedWeights = bufferGetter(
    app.gl,
    transformedWeightsBuffer.buffer
  );
  const getLODs = bufferGetter(app.gl, lodBuffer.buffer);

  const drawer = app
      .createDrawCall(transformer, vertexArray)
      .primitive(PicoGL.POINTS)
      .transformFeedback(transformFeedback)


  return {
    drawer,
    transformedPointsBuffer,
    getTransformedPoints,
    transformedWeightsBuffer,
    getTransformedWeights,
    lodBuffer,
    getLODs,
    numCells,
  };
};