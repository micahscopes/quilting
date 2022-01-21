import normals from "angle-normals";
import { fquad, uvGrid, Alg } from "../src";

import Delaunator from "delaunator";
import { chunk } from "lodash-es";
import wireframe from "glsl-solid-wireframe";

const grid = Delaunator.from(uvGrid(32, 32));

const D = 10;
const r = (x) => x + Math.random()*20;

const patchGenerator = () => {
  const guidePoints = [
    [Alg.Vector(-r(D), -r(D), 0), Alg.Vector(-r(D), r(D), 0)],
    [Alg.Vector(r(D), -r(D), 0), Alg.Vector(r(D), D, 0)],
  ];

  const weights = [
    [Alg.Vector(1, 0, -1), Alg.Vector(1, -1)],
    [Alg.Vector(0, 0, -1), Alg.Vector(0, 1, -1)],
  ];

  return fquad(guidePoints, weights);
};

let patch = {};
patch.cells = chunk(grid.triangles, 3);

const updatePatch = () => {
  let quad = patchGenerator()
  const points = chunk(grid.coords, 2).map((uv) => quad(...uv));
  patch.positions = points.map((x) => x.Vector.slice(0, 3));
  patch.normals = normals(patch.cells, patch.positions);
}

updatePatch()
// setInterval(updatePatch, 1000/90)

// console.log(patch)
export default draw = (regl) =>
  regl({
    vert: `
      precision mediump float;
      attribute vec3 normals;
      attribute vec3 positions;
      uniform mat4 projection, view;
      varying vec3 n;
      void main () {
        n = normals;
        gl_Position = projection * view * vec4(positions, 1);
      }`,
    frag: `
      precision mediump float;
      varying vec3 n;
      void main () {
        gl_FragColor = vec4(0.5 + 0.5 * n, 1);
      }`,
    attributes: {
      positions: () => patch.positions,
      normals: () => patch.normals,

      // return {
      //   positions: patch.positions,
      //   normals: patch.normals,
      //   cells: patch.cells
      // }
    },
    // attributes: patch,
    elements: patch.cells,
    count: patch.cells.length * 3,
  });
