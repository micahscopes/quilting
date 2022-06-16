import {
  add,
  constant,
  flatten,
  repeat,
  sample,
  sum,
  times,
  uniq,
  zipWith,
} from "lodash-es";
import PicoGL from "picogl";
import { makeTessellationAtlas } from "../src/tessellation";
import "./pico-util";

import { glsl } from "../src/util";
import { permutationIndices3 } from "../src/permutator";
import randomMesh from "./random-mesh";

const closeVerts = 1;
const farVerts = closeVerts * 5;
const numVerts = 70;
const debugText = true;

const [low, mid, high] = [0, 2, 4];
const lods = [0,1,2,3,4,5,6];
const lodLevels = uniq([...lods, ...[low, mid, high]]).map((x) => 2 ** x);
const exampleLodLookup = (i) => `[${2 ** i},${2 ** i},${2 ** i}]`;
const lowLodKey = exampleLodLookup(low);
const midLodKey = exampleLodLookup(mid);
const highLodKey = exampleLodLookup(high);

const atlas = makeTessellationAtlas(lodLevels);
console.log("meshlet atlas", atlas);
console.log(permutationIndices3);

const vs = glsl`
    #version 300 es
    layout(location=0) in vec3 patchBary;
    layout(location=1) in ivec3 J;
    layout(location=2) in mat3x2 corners;

    out vec3 vColor; 
    out vec3 cornerColor;

    void main() {
        vec3 subpatchBary;
        ivec3 I = J.xyz;
        int i = gl_VertexID % 3;
        subpatchBary.x = float(i == 0 || i == 1);
        subpatchBary.y = float(i == 1 || i == 2);
        subpatchBary.z = float(i == 2 || i == 0);
        vColor = subpatchBary;
        cornerColor = vec3(patchBary[I[0]], patchBary[I[1]], patchBary[I[2]]) * 2.0;
        // cornerColor = patchBary*2.0;

        vec2 p = vec2(
          patchBary[I[0]]*corners[0][0] + patchBary[I[1]]*corners[1][0] + patchBary[I[2]]*corners[2][0],
          patchBary[I[0]]*corners[0][1] + patchBary[I[1]]*corners[1][1] + patchBary[I[2]]*corners[2][1]
        );
        // vColor = patchBary ? patchBary * 2.0;
        // vColor = patchBary;
        gl_Position = vec4(p.xy*4.0, 0.0, 1.0);
    }
`;
const fs = glsl`
    #version 300 es
    precision highp float;

    in vec3 vColor;
    in vec3 cornerColor;

    out vec4 fragColor;
    void main() {
        // fragColor = vec4(1,1,0,1);
        // fragColor = vec4(vColor,1);
        
        float threshold = 0.95;
        bool c1 = cornerColor.x > threshold;
        bool c2 = cornerColor.y > threshold;
        bool c3 = cornerColor.z > threshold;
        
        vec3 color = vec3(
          c1 ? 1.0 : !(c2 || c3) ? vColor.x : 0.0, //vColor.r,
          c2 ? 1.0 : !(c1 || c3) ? vColor.y : 0.0, //vColor.g,
          c3 ? 1.0 : !(c1 || c2) ? vColor.z : 0.0 //vColor.b
        );
        
        fragColor = vec4(color, 1.0);
    }
`;

import mda from "mda";

import { tap, runEffects, merge, map } from "@most/core";
import { newDefaultScheduler } from "@most/scheduler";
import positionInElement from "./position-in-element";
import { mousemove, touchmove } from "@most/dom-event";
import { positionInCanvas } from "./position-in-element";
import { flow } from "lodash-es";
import knn from "rbush-knn";
import humanFormat from "human-format"

document.addEventListener("DOMContentLoaded", async function () {
  const canvas = document.createElement("canvas");
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  document.body.appendChild(canvas);

  const canvas2d = document.createElement("canvas");
  canvas2d.style.pointerEvents = "none";
  canvas2d.width = window.innerWidth;
  canvas2d.height = window.innerHeight;
  document.body.appendChild(canvas2d);
  const ctx2D = canvas2d.getContext("2d");
  ctx2D!.font = "12px Helvetica";

  const app = PicoGL.createApp(canvas).clearColor(0.0, 0.0, 0.0, 1.0);
  
  app.gl.enable(app.gl.CULL_FACE);
  app.gl.cullFace(app.gl.BACK);


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
  11;
  const patchesPerMeshlet = numVerts / 1;

  const mesh = randomMesh(numVerts);
  // mesh.mda.faces.forEach(face => face.fallbackLod = sample([0,0,1,1,2]))

  window.M = mesh.mda;
  window.mda = mda;

  const mouseInCanvas$ = flow(
    map(({ x, y }) => [
      knn(mesh.rbush, x, y, closeVerts),
      knn(mesh.rbush, x, y, farVerts),
    ]),
    map(([closer, further]) => [
      closer.flatMap(({ index }) => {
        try {
          return mda
            .VertexFaces(mesh.mda.vertices[index])
            .map((face) => face.index);
        } catch (e) {
          return [];
        }
      }),
      further.flatMap(({ index }) => {
        try {
          return mda
            .VertexFaces(mesh.mda.vertices[index])
            .map((face) => face.index);
        } catch (e) {
          return [];
        }
      }),
    ])
  )(positionInCanvas(merge(mousemove(canvas), touchmove(canvas))));
  const scheduler = newDefaultScheduler();

  // runEffects(tap(console.log, mouseInCanvas$), scheduler);

  console.log(mesh);
  let corners = app.createMatrixBuffer(
    PicoGL.FLOAT_MAT3x2,
    new Float32Array(
      flatten(flatten(mesh.cellPositions).map(([x, y]) => [x, y]))
    )
  );

  let permutations = app.createVertexBuffer(
    PicoGL.BYTE,
    3,
    mesh.mda.faces.length * 3
  );

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

  let meshlets = mesh.mda.faces.map(() => atlas.lookup[lowLodKey]);
  console.log("meshlets", meshlets);

  permutations.data(
    new Int8Array(flatten(meshlets.flatMap((m) => m.permutation)))
  );

  runEffects(
    tap(([closerFaceIDs, furtherFaceIds]) => {
      mesh.mda.faces.forEach((face) => {
        if (closerFaceIDs.includes(face.index)) {
          face.lod = high;
          // face.meshlet = atlas.lookup[highLodKey];
        } else if (furtherFaceIds.includes(face.index)) {
          face.lod = mid;
          // face.meshlet = atlas.lookup[midLodKey];
        } else {
          face.lod = low;
          // face.meshlet = atlas.lookup[lowLodKey];
        }
      });
      mesh.mda.faces.forEach((face) => {
        const lodKey = JSON.stringify(
          mda
            .FaceHalfEdges(face)
            .map((he) =>
              // Math.max(face.lod || (face.fallbackLod || low), he.flipHalfEdge?.face.lod || (face.fallbackLod || low))
              Math.max(face.lod || face.fallbackLod || low, he.flipHalfEdge?.face.lod || he.flipHalfEdge?.face.fallbackLod || low)
            )
            .map((i) => 2 ** i)
        );
        face.lodKey = lodKey;
        face.meshlet = atlas.lookup[lodKey];
      });
      meshlets = mesh.mda.faces.map(({ meshlet }) => meshlet);
      permutations.data(
        new Int8Array(flatten(meshlets.flatMap((m) => m.permutation)))
      );
      // console.log(meshlets.map((m) => m.permutation));
      ctx2D?.clearRect(
        -ctx2D.canvas.width / 2,
        -ctx2D.canvas.height / 2,
        ctx2D.canvas.width,
        ctx2D.canvas.height
      );

      mesh.mda.faces.forEach((face) => {
        const faceMidpoint = zipWith(
          ...mda.FaceVertices(face).map((v) => mesh.mda.positions[v.index]),
          (x, y, z) => x + y + z
        )
          .map((x) => x / 3)
          .slice(0, 2);
        // const faceMidpoint = mesh.mda.positions[face.halfEdge.vertex.index].slice(0,2);

        // ctx2D?.fillText(1, faceMidpoint[0]*ctx2D.canvas.width, faceMidpoint[1]*ctx2D.canvas.height);
        if (debugText && JSON.stringify(face.meshlet.edgePermutation) === "[2,0,1]") {
          ctx2D?.fillText(
            // face.lod,
            `${[face.meshlet.lod || "0"]} :: ${face.meshlet.edgePermutation} :: ${face.meshlet.permutation}`,
            // face.lodKey,
            // `${face.meshlet.lod} @ ${face.meshlet.permutation}`,
            faceMidpoint[0] * ctx2D.canvas.width - 40,
            -faceMidpoint[1] * ctx2D.canvas.height
          );
        }
      });
      const triangles = `
      <p>
      triangles: 
        ${humanFormat(sum(mesh.mda.faces.map(({meshlet}) => meshlet.count))/3)}
      </p><p>
      patches:
        ${mesh.mda.faces.length} 
      </p>
      `
      // console.log(triangles)
      document.querySelector('#stats')!.innerHTML = triangles 
    }, mouseInCanvas$),
    scheduler
  );

  ctx2D?.translate(ctx2D.canvas.width / 2, ctx2D.canvas.height / 2);
  const draw = () => {
    if (timer.ready()) {
      utils.updateTimerElement(timer.cpuTime, timer.gpuTime);
    }

    timer.start();

    // DRAW
    app.clear();
    drawCall.drawRanges(
      ...meshlets.map((m, i) => [
        m.baseIndex,
        m.count,
        //
        1,
        i,
        // patchesPerMeshlet,
        // i * patchesPerMeshlet,
      ])
    );
    drawCall.draw();

    timer.end();
    requestAnimationFrame(draw);
  };

  draw();
});
