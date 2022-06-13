import { constant, flatten, repeat, sample, times } from "lodash-es";
import PicoGL from "picogl";
import { makeTessellationAtlas } from "../src/tessellation";

import "./util";

import { glsl } from "../src/util";
import { permutationIndices3 } from "../src/permutator";
import { FLOAT_MAT2x3 } from "./gltf/src/gltf/gl-const";

const atlas = makeTessellationAtlas([0,1,2,3].map((x) => 2 ** (2*x)));
// console.log("meshes", atlas);
console.log(permutationIndices3);

const vs = glsl`
    #version 300 es
    layout(location=0) in vec3 baryCo;
    layout(location=1) in ivec3 I;
    layout(location=2) in mat3x2 corners;

    out vec3 vColor; 

    void main() {
        vec3 bary;
        int i = gl_VertexID % 3;
        bary.x = float(i == 0);
        bary.y = float(i == 1);
        bary.z = float(i == 2);
    
        vColor = bary;
        vec2 p = vec2(
          baryCo[I[0]]*corners[0][0] + baryCo[I[1]]*corners[1][0] + baryCo[I[2]]*corners[2][0],
          baryCo[I[0]]*corners[0][1] + baryCo[I[1]]*corners[1][1] + baryCo[I[2]]*corners[2][1]
        );
        gl_Position = vec4(p.xy*4.0 - 1.0, 0.0, 1.0);
    }
`;
const fs = glsl`
    #version 300 es
    precision highp float;

    in vec3 vColor;

    out vec4 fragColor;
    void main() {
        // fragColor = vec4(1,1,0,1);
        fragColor = vec4(vColor,1);
    }
`;

document.addEventListener("DOMContentLoaded", async function () {
  const canvas = document.createElement("canvas");
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  document.body.appendChild(canvas);
  const app = PicoGL.createApp(canvas).clearColor(0.0, 0.0, 0.0, 1.0);
  const timer = app.createTimer();
  window.utils.addTimerElement();

  window.onresize = function () {
    app.resize(window.innerWidth, window.innerHeight);
  };

  let positions = app.createVertexBuffer(
    PicoGL.UNSIGNED_SHORT,
    3,
    new Uint16Array(flatten(atlas.combinedMesh.positions))
  );

  const numPatches = 10000;
  const patchesPerMeshlet = numPatches/50;

  let corners = app.createMatrixBuffer(
    PicoGL.FLOAT_MAT3x2,
    new Float32Array(
      flatten(
        new Array(numPatches)
          .fill(null)
          .map(Math.random)
          .map((x) => new Array(6).fill(null).map(() => x + Math.random() / 10))
      )
    )
  );

  let permutations = app.createVertexBuffer(PicoGL.BYTE, 3, 3 * numPatches);

  // COMBINE VERTEX BUFFERS INTO VERTEX AR
  let triangleArray = app
    .createVertexArray()
    .vertexAttributeBuffer(0, positions, { normalized: true })
    .instanceAttributeBuffer(1, permutations)
    .instanceAttributeBuffer(2, corners);

  app.gl.disable(app.gl.CULL_FACE);

  const program = (await app.createPrograms([vs, fs]))[0];
  // CREATE DRAW CALL FROM PROGRAM AND VERTEX ARRAY
  let drawCall = app.createDrawCall(program, triangleArray);


  const draw = () => {
    if (timer.ready()) {
      utils.updateTimerElement(timer.cpuTime, timer.gpuTime);
    }

    timer.start();
    const meshlets = new Array(numPatches/patchesPerMeshlet)
      .fill(null)
      .map(() => sample(Object.values(atlas.lookup)));

    // console.log(meshlets)
    permutations.data(
      new Int8Array(flatten(flatten(times(patchesPerMeshlet, constant(meshlets.map((m) => m.permutation))))))
    );
    //   console.log(atlas.combinedMesh)

    // DRAW
    app.clear();
    drawCall.drawRanges(
      ...meshlets.map((m, i) => [m.baseIndex, m.count, patchesPerMeshlet, i*patchesPerMeshlet])
    );
    drawCall.draw();

      // let offset = meshlet.baseIndex;
      // let numElements = meshlet.count;
      // let numInstances = 0;
      // let baseInstance = 1;
      // let baseElement = undefined;

      // drawCall.drawRanges([
      //     offset,
      //     numElements,
      //     numInstances,
      //     baseInstance,
      //     baseElement
      // ]);

    timer.end();
    requestAnimationFrame(draw);
  };
  
  draw();

  //   window.glcheck_renderDone = true;

  // CLEANUP
  //   program.delete();
  //   positions.delete();
  //   colors.delete();
  //   triangleArray.delete();
});
