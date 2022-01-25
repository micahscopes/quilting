import normals from "angle-normals";
import { fquad, uvGrid, Alg } from "../src";

import Delaunator from "delaunator";
import { chunk } from "lodash-es";
import wireframe from "glsl-solid-wireframe";

import glLib from "../gpu/lib.glsl";

let gl = function (s) {
  return `precision mediump float;
  ${glLib}\n\n${s[0]}`;
};

console.log('triangulating grid...')
const grid = Delaunator.from(uvGrid(1024,1024));
console.log(grid)
const D = 10;
const r = (x) => x + Math.random() * 20;

const patchGenerator = () => {
  return fquad(guidePoints, weights);
};

let patch = {};
patch.cells = chunk(grid.triangles, 3);

const updatePatch = () => {
  // let quad = patchGenerator();
  // const points = chunk(grid.coords, 2).map((uv) => quad(...uv));
  patch.positions = chunk(grid.coords, 2).map(([u, v]) => [u, v, 0]);
  console.log(patch.positions);
  patch.normals = normals(patch.cells, patch.positions);
};

updatePatch();
// setInterval(updatePatch, 1000/90)

console.log('cells', patch.cells.length*3)
console.log('positions', patch.positions.length)

window.el = Alg.Vector(-r(D), -r(D)+50, 0)

// console.log(patch)
export default draw = (regl) =>
  regl({
    vert: gl`
      precision mediump float;

      attribute vec3 normal;
      attribute vec3 position;

      uniform float p00[32], p01[32], p10[32], p11[32];
      uniform float w00[32], w01[32], w10[32], w11[32];
      uniform mat4 projection, view;
      uniform float time;

      varying vec3 n;
      varying vec2 uv;

      void main () {

      CGA3 animatedWeight01 = fromArray(w01);
      animatedWeight01.e1 = animatedWeight01.e1 + 30.0*sin(time);
      animatedWeight01.e2 = animatedWeight01.e2 + 20.0*cos(time);

      CGA3 animatedWeight11 = fromArray(w11);
      animatedWeight11.e2 = animatedWeight11.e2 + 10.0*sin(time/3.);
      animatedWeight11.e3 = animatedWeight11.e3 + 15.0*cos(time/3.);
      CGA3 p =
          bilinearQuad(
            fromArray(p00),
            fromArray(p01),
            fromArray(p10),
            fromArray(p11),
            fromArray(w00),
            animatedWeight01,
            fromArray(w10),
            animatedWeight11,
            position.x,
            position.y
        );
        n = normal;
        uv = vec2(position.x, position.y);
        gl_Position = projection * view * vec4(vecFromPoint(p), 1.0);
      }`,
    frag: gl`
      precision mediump float;
      uniform float time;
      varying vec3 n;
      varying vec2 uv;
      void main () {
        vec3 light = vec3(0.5,0.9,0.5);
        light = normalize(light);

        gl_FragColor = vec4(uv, 0.25+0.5*abs(cos(time/2.0)), 1.0);
        // gl_FragColor = vec4(1, 0, 0, 1);
      }`,
    attributes: {
      position: () => patch.positions,
      normal: () => patch.normals,
    },
    uniforms: {
      // time(){ return Date.now() },
      p00: Alg.Vector(-r(D), -r(D), 0),
      p01: Alg.Vector(-r(D), r(D), 0),
      p10: Alg.Vector(r(D), -r(D), 0),
      p11: Alg.Vector(r(D), D, 0),
      w00: Alg.Vector(1, 0, -1),
      w01: Alg.Vector(1, -1),
      w10: Alg.Vector(0, 0, -1),
      w11: Alg.Vector(0, 1, -1),
      time: ({tick}) => {
        // console.log(Date.now())
        return tick/50
      }
      // time: 10
    },
    elements: patch.cells,
    count: patch.cells.length * 3,
  });
