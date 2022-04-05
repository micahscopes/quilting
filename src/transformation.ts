// generate a PicoGL program that:
// - lifts cell points/weights into conformal model
// - applies a transformation defined by params passed in via uniform
// - computes cell LOD levels
// - writes transformed points/weights as vectors into output buffer

import type { App } from "picogl";
import { PicoGL } from "picogl";
import { glsl } from "./util";

export const cellTransformer = (app: App, numSides = 3) =>
  app.createProgram(
    glsl`
#version 300 es
layout(location=0) in vec3 pointA;
layout(location=1) in vec3 pointB;
layout(location=2) in vec3 pointC;
// layout(location=3) in vec4 weights;

uniform float scale;

out vec3 transformedPointA;
out vec3 transformedPointB;
out vec3 transformedPointC;
// out vec4 transformedWeights;

void main(){
  transformedPointA = scale*pointA;
  transformedPointB = scale*pointB;
  transformedPointC = scale*pointC;
  // transformedPoints[0] = vec3(4.0);
  // transformedWeights = scale*weights;
}

`,
    glsl`
#version 300 es
precision highp float;

out vec4 fragColor;
void main() {
    fragColor = vec4(0,0,0,1.0);
}
`,
    {
      transformFeedbackVaryings: [
        "transformedPointA",
        // "transformedPointB",
        // "transformedPointC",
        // "transformedWeights",
      ],
      // transformFeedbackMode: PicoGL.SEPARATE_ATTRIBS,
    }
  );
