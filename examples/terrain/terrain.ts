import { flatten, isEqual, sample, sum, uniq, zipWith } from "lodash-es";
import PicoGL from "picogl";
import { loadTessellationAtlas } from "../../src/load-tessellation-atlas";
import "../pico-util";

// @ts-ignore
import vs from "./terrain.vs.glsl";
// @ts-ignore
import fs from "./terrain.fs.glsl";

import randomMesh from "../random-mesh";
import "@polymer/paper-spinner/paper-spinner-lite.js";
import { scaleLinear } from 'd3-scale'

// @ts-ignore
import createCamera from "inertial-turntable-camera";

const numVerts = 400;
// const debugText = false;


const lodFn = (x) => 2 ** (2*x);
const possibleLods = [0,1,2,3];
const lodLevels = uniq(possibleLods).map(lodFn);
const low = Math.min(...possibleLods);
const high = Math.max(...possibleLods);

// @ts-ignore
import mda from "mda";

import {
  tap,
  runEffects,
  periodic,
} from "@most/core";
import { newDefaultScheduler } from "@most/scheduler";

// @ts-ignore
import humanFormat from "human-format";
import setupCamera from "../setup-camera";
import { whileTabFocus } from "../while-tab-focus";
const clamper = (min, max) => num => Math.min(Math.max(num, min), max)

document.addEventListener("DOMContentLoaded", async function () {
  // generate the tiling tessellation mesh atlas
  const atlas = await loadTessellationAtlas(lodLevels);

  // setup canvas
  const canvas = document.createElement("canvas");
  document.body.appendChild(canvas);
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  // setup webgl2 context
  const app = PicoGL.createApp(canvas)
    .clearColor(0.0, 0.0, 0.0, 1.0)
    .enable(PicoGL.DEPTH_TEST)
    .depthFunc(PicoGL.LEQUAL)
    .enable(PicoGL.CULL_FACE);

  app.gl.enable(app.gl.CULL_FACE);
  app.gl.cullFace(app.gl.BACK);

  window.onresize = function () {
    app.resize(window.innerWidth, window.innerHeight);
    camera.aspectRatio = window.innerWidth / window.innerHeight;
  };

  // setup camera
  const camera = createCamera({
    aspectRatio: window.innerWidth / window.innerHeight,
    phi: Math.PI / 48,
    theta: Math.PI / 4,
    distance: 1,
    fovY: Math.PI / 2,
  });

  setupCamera(canvas, camera);

  // debug info canvas
  // const canvas2d = document.createElement("canvas");
  // canvas2d.style.pointerEvents = "none";
  // canvas2d.width = window.innerWidth;
  // canvas2d.height = window.innerHeight;
  // document.body.appendChild(canvas2d);
  // const ctx2D = canvas2d.getContext("2d");
  // ctx2D!.font = "8px Helvetica";
  // ctx2D?.translate(ctx2D.canvas.width / 2, ctx2D.canvas.height / 2);


  // stats
  const timer = app.createTimer();
  // @ts-ignore
  window.utils.addTimerElement();


  // initialize buffers and mesh
  let positions = app.createVertexBuffer(
    PicoGL.UNSIGNED_SHORT,
    3,
    new Uint16Array(flatten(atlas.combinedMesh.positions))
  );

  const mesh = randomMesh(numVerts);
  mesh.mda.faces.forEach((face) => (face.fallbackLod = low));

  window.M = mesh.mda;
  window.mda = mda;

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

  let lods = app.createVertexBuffer(PicoGL.FLOAT, 3, mesh.mda.faces.length * 3);

  // COMBINE VERTEX BUFFERS INTO VERTEX AR
  let triangleArray = app
    .createVertexArray()
    .vertexAttributeBuffer(0, positions, { normalized: true })
    .instanceAttributeBuffer(1, permutations)
    .instanceAttributeBuffer(2, corners)
    .instanceAttributeBuffer(5, lods);

  // @ts-ignore
  window.triangleArray = triangleArray;

  app.gl.disable(app.gl.CULL_FACE);

  const program = (await app.createPrograms([vs, fs]))[0];
  // CREATE DRAW CALL FROM PROGRAM AND VERTEX ARRAY
  let drawCall = app.createDrawCall(program, triangleArray);

  let meshlets: any[] = []; // = mesh.mda.faces.map(() => atlas.lookup[lowLodKey]);

  permutations.data(
    new Int8Array(flatten(meshlets.flatMap((m) => m.permutation)))
  );

  const scale = scaleLinear([0, 1], [high, low])
  const clamp = clamper(low, high)
  const computeLod = x => clamp(2 * Math.round(scale(x)))

  // setup LOD update pipeline (async from draw loop)
  const scheduler = newDefaultScheduler();


  // helper function for updating debug info / stats
  const updateDebugInfo = () => {
    // if (debugText) {
    //   ctx2D?.clearRect(
    //     -ctx2D.canvas.width / 2,
    //     -ctx2D.canvas.height / 2,
    //     ctx2D.canvas.width,
    //     ctx2D.canvas.height
    //   );

    //   mesh.mda.faces.forEach((face) => {
    //     const faceMidpoint = zipWith(
    //       ...mda.FaceVertices(face).map((v) => mesh.mda.positions[v.index]),
    //       (x, y, z) => x + y + z
    //     )
    //       .map((x) => x / 3)
    //       .slice(0, 2);
    //     ctx2D?.fillText(
    //       face.lod || face.fallbackLod,
    //       faceMidpoint[0] * ctx2D.canvas.width - 40,
    //       -faceMidpoint[1] * ctx2D.canvas.height
    //     );
    //   });
    // }
    const triangles = `
      <p>
      triangles: 
        ${humanFormat(
      sum(mesh.mda.faces.map(({ meshlet }) => meshlet.count)) / 3
    )}
      </p><p>
      patches:
        ${mesh.mda.faces.length} 
      </p>
      `;
    // console.log(triangles)
    document.querySelector("#stats")!.innerHTML = triangles;
  }

  runEffects(
    // update LOD levels
    // (currently this is happening on the CPU but it could happen on the GPU at some point too)
    tap(() => {
      // get cell-wise LOD levels for each cell
      mesh.mda.faces.forEach((face) => {
        try {
          face.lod =
            computeLod(sum(
              mda
                .FaceVertices(face)
                .map(
                  v => {
                    const [x, y] = mesh.mda.positions[v.index]
                    return (Math.sqrt((x - 0.5 * Math.sin(Date.now() / 1000)) ** 2 +
                      (y - 0.5 * Math.cos(Date.now() / 1000)) ** 2))
                  }
                )
            ) / 3.0);
        } catch (e) {
          face.lod = low
        }
      });
      // choose edge-wise LOD levels for each cell by looking at its neighboring cells' cell-wise LODs
      mesh.mda.faces.forEach((face) => {
        const lodKey = JSON.stringify(
          mda
            .FaceHalfEdges(face)
            .map((he) =>
              // between this cell's cell-wise LOD and its neighbor's cell-wise LOD, choose the largest for the shared edge-wise LOD
              Math.max(
                face.lod || face.fallbackLod || low,
                he.flipHalfEdge?.face.lod ||
                he.flipHalfEdge?.face.fallbackLod ||
                low
              )
            )
            .map(lodFn)
        );
        // lodKey is a string e.g. `[32,16,2]` for looking up that specific tessellation in the atlas 
        face.lodKey = lodKey;

        face.vertexLods = mda.FaceVertices(face).map((v) => v.lod || low);
        face.meshlet = atlas.lookup[lodKey];
      });

      // compute vertex-wise LODs as the minimum LOD of cell-wise LODs surrounding a vertex
      // (kind of arbitrary... it's really just to be able to roughly visualize LOD levels based on info
      //  available to the vertex shader)
      mesh.mda.vertices.forEach((v) => {
        try {
          v.lod = Math.min(...mda.VertexFaces(v).map((face) => face.lod || 0));
        } catch (e) {
          v.lod = 0;
        }
      });

      // "meshlets" is a fun nickname for group of specific tessellations to be instance drawn each frame
      meshlets = mesh.mda.faces.map(({ meshlet }) => meshlet);

      // ensure that for each instance its corresponding tessellation geometry is flipped/rotated correctly
      permutations.data(
        new Int8Array(flatten(meshlets.flatMap((m) => m.permutation)))
      );
      // update the vertex-wise LOD buffer
      lods.data(
        new Float32Array(
          mesh.mda.faces.flatMap((f) =>
            f.vertexLods?.map((x) => (x - low) / (high - low))
          )
        )
      );

      // draw debug info
      updateDebugInfo()

    }, whileTabFocus(periodic(100))),
    scheduler
  );

  // helper function to update instance draw ranges
  let ranges: number[][] = [];
  const updateDrawRanges = () => {
    // Consolidate draw ranges so that adjacent patches with the same tessellation geometry 
    // share a draw range whenever possible.  This reduces the number of rebinds needed when
    // `WEBGL_multi_draw_instanced_base_vertex_base_instance` is unavailable.
    let nextRanges: number[][] = [];
    for (let i = 0; i < meshlets.length; i++) {
      const mNext = meshlets[i];
      const prev = nextRanges.at(-1);
      if (isEqual(mNext.lod, prev?.at(-1))) {
        prev[2] += 1;
      } else {
        nextRanges.push([mNext.baseIndex, mNext.count, 1, i, mNext.lod]);
      }
    }
    ranges = nextRanges.map(
      ([baseIndex, count, numInstances, baseInstance]: number[]) => [
        baseIndex,
        count,
        numInstances,
        baseInstance,
      ]
    );
  }

  // draw loop
  const draw = () => {
    if (timer.ready()) {
      utils.updateTimerElement(timer.cpuTime, timer.gpuTime);
    }

    timer.start();
    updateDrawRanges();

    camera.tick({
      near: camera.params.distance * 0.001,
      far: camera.params.distance * 2 + 200,
    });

    drawCall.uniform("projection", camera.state.projection);
    drawCall.uniform("view", camera.state.view);

    drawCall.drawRanges(...ranges);

    app.clear();
    drawCall.draw();

    timer.end();
    requestAnimationFrame(draw);
  };

  draw();
});
