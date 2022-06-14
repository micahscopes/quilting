import { constant, flatten, repeat, sample, times } from "lodash-es";
import PicoGL from "picogl";
import { makeTessellationAtlas } from "../src/tessellation";
import "./pico-util";

import { glsl } from "../src/util";
import { permutationIndices3 } from "../src/permutator";
import randomMesh from "./random-mesh";

const atlas = makeTessellationAtlas([0, 1, 7].map((x) => 2 ** x));
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
        gl_Position = vec4(p.xy*2.0, 0.0, 1.0);
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

import mda from "mda";

import { tap, runEffects, merge, map } from "@most/core";
import { newDefaultScheduler } from "@most/scheduler";
import positionInElement from "./position-in-element";
import { mousemove, touchmove } from "@most/dom-event";
import { positionInCanvas } from "./position-in-element";
import { flow } from "lodash-es";
import knn from 'rbush-knn'

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
11
  const numPatches = 200;
  const patchesPerMeshlet = numPatches / 1;

  const mesh = randomMesh(numPatches);

  window.M = mesh.mda;
  window.mda = mda;

  const mouseInCanvas$ = flow(
    map(({x,y}) => knn(mesh.rbush, x, y, 2)),
    map(faces => faces.map(({faceIndex}) => faceIndex))
  )(
    positionInCanvas(merge(mousemove(canvas), touchmove(canvas)))
  );

  runEffects(tap(console.log, mouseInCanvas$), newDefaultScheduler());

  console.log(mesh);
  let corners = app.createMatrixBuffer(
    PicoGL.FLOAT_MAT3x2,
    new Float32Array(
      flatten(flatten(mesh.cellPositions).map(([x, y]) => [x, y]))
    )
    // new Float32Array(
    //   flatten(
    //     new Array(numPatches)
    //       .fill(null)
    //       .map(() => [Math.random(), Math.random()])
    //       .map(([x,y]) => new Array(6).fill(null).flatMap(() => [x + (Math.random() - 0.5) / 10, y + (Math.random() - 0.5) / 10]))
    //   )
    // )
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

  let meshlets = new Array(numPatches / patchesPerMeshlet)
    .fill(null)
    .map(() => sample(Object.values(atlas.lookup)));

  permutations.data(
    new Int8Array(
      flatten(
        flatten(
          times(patchesPerMeshlet, constant(meshlets.map((m) => m.permutation)))
        )
      )
    )
  );

  console.log(meshlets);

  const draw = () => {
    if (timer.ready()) {
      utils.updateTimerElement(timer.cpuTime, timer.gpuTime);
    }

    timer.start();
    meshlets = new Array(numPatches / patchesPerMeshlet)
      .fill(null)
      .map(() => sample(Object.values(atlas.lookup)));

    // console.log(meshlets)
    permutations.data(
      new Int8Array(
        flatten(
          flatten(
            times(
              patchesPerMeshlet,
              constant(meshlets.map((m) => m.permutation))
            )
          )
        )
      )
    );
    //   console.log(atlas.combinedMesh)

    // DRAW
    app.clear();
    drawCall.drawRanges(
      ...meshlets.map((m, i) => [
        m.baseIndex,
        m.count,
        patchesPerMeshlet,
        i * patchesPerMeshlet,
      ])
    );
    drawCall.draw();

    timer.end();
    requestAnimationFrame(draw);
  };

  draw();
});
