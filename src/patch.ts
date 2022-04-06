import { flatten, reverse } from "lodash-es";
import moize from "moize";
import { App, PicoGL } from "picogl";
import patchGL from "../gl/patch.glsl";
import triangleVertexGL from "../gl/triangle.vs.glsl";
import { tesselationMesh } from "./tessellation";
import { glsl } from "./util";
const vs = triangleVertexGL;
const fs = glsl`
      #version 300 es
      precision highp float;
      #pragma glslify: matcap = require('matcap');
      uniform sampler2D matcapTexture;
      uniform sampler2D colorTexture;
      uniform vec3 eye;
      in vec3 n;
      in vec2 uv;
      out vec4 fragColor;
      void main () {
        vec4 color = texture(colorTexture, uv);
        vec2 mat_uv = matcap(normalize(eye), vec3(color.r*n.x, color.g*n.y, color.b*n.z));

        fragColor = vec4(texture(
          matcapTexture, mat_uv
        ).rgb, color.r);

        // gl_FragColor = texture2D(texture, uv);
        // gl_FragColor = vec4(0,1,1,1);
      }`;

const patchProgram = moize((app: App) => app.createProgram(vs, fs), {
  maxAge: Infinity,
  maxSize: Infinity,
});

const meshVertexArray =
// moize.infinite(
  (mesh: any, app: App) => {
  const positionBuffer = app.createVertexBuffer(
    PicoGL.FLOAT,
    3,
    new Float32Array(flatten(flatten(mesh.cellPositions)))
    // new Float32Array(flatten(mesh.positions))
  );
  const indexBuffer = app.createIndexBuffer(
    PicoGL.UNSIGNED_SHORT,
    new Uint32Array(flatten(mesh.cells))
  );
  const normalBuffer = app.createVertexBuffer(
    PicoGL.FLOAT,
    3,
    // new Float32Array(flatten(flatten(mesh.cellNormals)))
    new Float32Array(flatten(mesh.normals))
  );
  const vertexArray = app
    .createVertexArray()
    .vertexAttributeBuffer(0, positionBuffer);
  // .vertexAttributeBuffer(0, positionBuffer)
  // .vertexAttributeBuffer(1, normalBuffer);

  return vertexArray;
}
// );

export const patchDrawCall =
  // moize(
  (
    app: App,
    sideLODs: [number, number, number] | number,
    pointsBuffer: any,
    weightsBuffer: any
  ) => {
    const mesh = tesselationMesh(sideLODs);
    const vertexArray = meshVertexArray(mesh, app)
      .instanceAttributeBuffer(1, pointsBuffer)
      // .instanceAttributeBuffer(2, pointsBuffer, {type: PicoGL.FLOAT, stride: 3*3*4, offset: 3*4})
      // .instanceAttributeBuffer(2, pointsBuffer) //, {type: PicoGL.FLOAT_MAT3, size: 1, stride: 3*3*4, offset: 0})
      // .instanceAttributeBuffer(3, pointsBuffer) //, {type: PicoGL.FLOAT_MAT3, size: 1, stride: 3*3*4, offset: 0})
      .instanceAttributeBuffer(4, weightsBuffer)//, {type: PicoGL.FLOAT, size: 4, stride: 4*3*4, offset: 0})
      // .instanceAttributeBuffer(1, pointsBuffer , {type: PicoGL.FLOAT, size: 3, stride: 3*3*4, offset: 0})
      // .instanceAttributeBuffer(2, pointsBuffer , {type: PicoGL.FLOAT, size: 3, stride: 3*3*4, offset: 3*4})
      // .instanceAttributeBuffer(3, pointsBuffer , {type: PicoGL.FLOAT, size: 3, stride: 3*3*4, offset: 6*4})
      // .instanceAttributeBuffer(4, weightsBuffer, {type: PicoGL.FLOAT, size: 4, stride: 4*3*4, offset: 0})
      // .instanceAttributeBuffer(5, weightsBuffer, {type: PicoGL.FLOAT, size: 4, stride: 4*3*4, offset: 4*4})
      // .instanceAttributeBuffer(6, weightsBuffer, {type: PicoGL.FLOAT, size: 4, stride: 4*3*4, offset: 8*4})

    return app
      .createDrawCall(patchProgram(app), vertexArray)
      .uniform("eye", new Float32Array([0, 0, 1]));
  };
// );
