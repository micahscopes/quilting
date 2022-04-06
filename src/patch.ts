import { flatten } from "lodash-es";
import moize from "moize";
import { App, PicoGL } from "picogl";
import patchGL from "../gl/patch.glsl";
import triangleVertexGL from "../gl/triangle.vs.glsl";
import { tesselationMesh } from "./tessellation";
import { glsl } from "./util";
const vs = triangleVertexGL;
const fs = glsl`
      precision highp float;
      #pragma glslify: matcap = require('matcap');
      uniform sampler2D matcapTexture;
      uniform sampler2D texture;
      uniform vec3 eye;
      varying vec3 n;
      varying vec2 uv;
      void main () {
        vec4 color = texture2D(texture, uv);
        vec2 mat_uv = matcap(normalize(eye), vec3(color.r*n.x, color.g*n.y, color.b*n.z));

        gl_FragColor = vec4(texture2D(
          matcapTexture, mat_uv
        ).rgb, color.r);

        gl_FragColor = texture2D(texture, uv);
        gl_FragColor = vec4(0,1,1,1);
      }`;

const patchProgram = moize((app: App) => app.createProgram(vs, fs), {
  maxAge: Infinity,
  maxSize: Infinity,
});

const meshVertexArray = moize.infinite((mesh: any, app: App) => {
  const positionBuffer = app.createVertexBuffer(
    PicoGL.FLOAT,
    3,
    new Float32Array(flatten(flatten(mesh.positions)))
  );
  const normalBuffer = app.createVertexBuffer(
    PicoGL.FLOAT,
    3,
    new Float32Array(flatten(mesh.normals))
  );
  const vertexArray = app
    .createVertexArray()
    .vertexAttributeBuffer(0, positionBuffer)
    .vertexAttributeBuffer(1, normalBuffer);

  return vertexArray;
});

export const patchDrawCall =
  // moize(
  (app: App, sideLODs: [number, number, number] | number) => {
    const mesh = tesselationMesh(sideLODs);
    const vertexArray = meshVertexArray(mesh, app);

    return app
      .createDrawCall(patchProgram(app), vertexArray)
      .uniform("eye", new Float32Array([0, 0, 1]));
  };
// );
