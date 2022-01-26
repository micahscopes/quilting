import normals from "angle-normals";
import { fquad, uvGrid, Alg } from "../src";

import Delaunator from "delaunator";
import { chunk } from "lodash-es";

import glLib from "../gpu/lib.glsl";

let gl = function (s) {
  return `precision mediump float;
  ${glLib}\n\n${s[0]}`;
};

console.log('triangulating grid...')
const grid = Delaunator.from(uvGrid(256,256));
console.log(grid)
const D = 10;
const r = (x) => x + Math.random() * 20;

const patchGenerator = () => {
  return fquad(guidePoints, weights);
};

let patch = {};
patch.cells = chunk(grid.triangles, 3);

const updatePatch = () => {
  patch.positions = chunk(grid.coords, 2).map(([u, v]) => [u, v, 0]);
  console.log(patch.positions);
  patch.normals = normals(patch.cells, patch.positions);
};

updatePatch();

console.log('cells', patch.cells.length*3)
console.log('positions', patch.positions.length)

window.el = Alg.Vector(-r(D), -r(D)+50, 0)

// console.log(patch)
export default draw = (regl) =>
  regl({
    vert: gl`
      precision highp float;

      attribute vec3 normal;
      attribute vec3 position;

      uniform vec3
        p0, p1, p2, p3,
        w0, w1, w2, w3;
      uniform mat4 projection, view;
      uniform float time;

      varying vec3 n;
      varying vec2 uv;

      void main () {

      CGA3 p =
          bilinearQuad(
            p0,
            p1,
            p2,
            p3,
            w0,
            w1,
            w2,
            w3,
            position.x,
            position.y
        );
        n = normal;
        uv = vec2(position.x, position.y);
        gl_Position = projection * view * vec4(p.e1, p.e2, p.e3, 1.0);
      }`,
    frag: gl`
      precision highp float;
      uniform float time;
      varying vec3 n;
      varying vec2 uv;
      void main () {
        vec3 light = vec3(0.5,0.9,0.5);
        light = normalize(light);

        // gl_FragColor = vec4(uv, 0.25+0.5*abs(cos(time/2.0)), 1.0);
        gl_FragColor = vec4(0, pow(uv.x, 2.0), pow(uv.y, 2.0), 1);
      }`,
    attributes: {
      position: () => patch.positions,
      normal: () => patch.normals,
    },
    uniforms: {
      // time(){ return Date.now() },
      p0: [0, 0, 0],
      p1: [D, 0, 0],
      p2: [0, D, 0],
      p3: [D, D, 0],
      w0: [0, 0, 1],
      w1: [1, 0, 0],
      w2: [1, 0, 0],
      w3: [0, 1, 0],
      // w00: [1, 0, -1],
      // w0: ({tick}) => [Math.sin(tick/150), 1, 0],
      // w10: ({tick}) => [0, 0, 2*Math.sin(tick/150)],
      // w11: ({tick}) => [0, 2*3.1415926*Math.sin(tick/50), -1],
      time: ({tick}) => {
        // console.log(Date.now())
        return tick/50
      }
      // time: 10
    },
    elements: patch.cells,
    count: patch.cells.length * 3,
  });
