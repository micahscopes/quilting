import { flatten, sample } from "lodash-es";
import PicoGL from "picogl";
import { makeTessellationAtlas } from "../src/tessellation";

import { glsl } from "../src/util";

const atlas = makeTessellationAtlas([1,4,8].map((x) => 2 ** x));
console.log("meshes", atlas);

const vs = glsl`
    #version 300 es
            
    layout(location=0) in vec3 position;
    layout(location=1) in vec3 color;

    out vec3 vColor; 
    void main() {
        vec3 bary;
        int i = gl_VertexID % 3;
        bary.x = float(i == 0);
        bary.y = float(i == 1);
        bary.z = float(i == 2);
    
        vColor = bary;
        gl_Position = vec4(2.0*position-1.0, 1.0);
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
  document.body.appendChild(canvas);
  const app = PicoGL.createApp(canvas).clearColor(0.0, 0.0, 0.0, 1.0);

  window.onresize = function () {
    app.resize(window.innerWidth, window.innerHeight);
  };

  //   let positions = app.createVertexBuffer(
  //     PicoGL.FLOAT,
  //     3,
  //     new Float32Array(atlas.combinedMesh.positions)
  //   );
  //   let cells = app.createIndexBuffer(
  //     PicoGL.UNSIGNED_SHORT,
  //     new Float32Array(atlas.combinedMesh.cells)
  //   );

  let positions = app.createVertexBuffer(
    PicoGL.FLOAT,
    3,
    new Float32Array(flatten(atlas.combinedMesh.positions))
  );

  let cells = app.createIndexBuffer(
    PicoGL.UNSIGNED_SHORT,
    new Uint16Array(flatten(atlas.combinedMesh.cells))
  );

  let offsets = app.createVertexBuffer(
    PicoGL.FLOAT,
    2,
    new Float32Array([-0.4, -0.4, 0.4, -0.4, -0.4, 0.4, 0.4, 0.4])
  );

  // COMBINE VERTEX BUFFERS INTO VERTEX ARRAY
  let triangleArray = app
    .createVertexArray()
    .instanceAttributeBuffer(1, offsets)
    .vertexAttributeBuffer(0, positions)
    // .indexBuffer(cells);
    // 
  app.gl.disable(app.gl.CULL_FACE)

  const program = (await app.createPrograms([vs, fs]))[0];
  // CREATE DRAW CALL FROM PROGRAM AND VERTEX ARRAY
  let drawCall = app.createDrawCall(program, triangleArray);


  setInterval(() => {
      const meshlet = sample(Object.values(atlas.lookup))
    //   console.log(meshlet);
    //   console.log(atlas.combinedMesh)

      // DRAW
      app.clear();
      let offset = meshlet.baseIndex;
      let numElements = meshlet.count;
      let numInstances = 0;
      let baseInstance = undefined;
      let baseElement = undefined;
    //   baseElement = meshlet.baseIndex;
    //   const baseElement = meshlet.baseIndex;
    //   const baseElement = atlas.combinedMesh.cells.at(-3*3)[0]+;
    //   const baseElement = meshlet.baseIndex-2;
    //   const baseElement = 200;
      
    //   console.log(baseElement, numElements)

      drawCall.drawRanges([
          offset,
          numElements,
          numInstances,
          baseInstance,
          baseElement
      ]);
      drawCall.draw();
  }, 200)

  //   window.glcheck_renderDone = true;

  // CLEANUP
  //   program.delete();
  //   positions.delete();
  //   colors.delete();
  //   triangleArray.delete();
});
