import normals from "angle-normals";
import { fquad, uvGrid, Alg } from "../src";

import Delaunator from "delaunator";
import { chunk, add, zip, unzipWith } from "lodash-es";

import glLib from "../gpu/lib.glsl";

let gl = function (s) {
  return `precision mediump float;
  ${glLib}\n\n${s[0]}`;
};

const normalize = x => {
    const norm = x.reduce((a,b)=> a+b, 0)
    return x.map(n => n/norm);
}
function addVector(a,b){
  return a.map((e,i) => e + b[i]);
}

export default draw = (regl,D=10, resolution = 64, defaultOffset = [0, 0, 0], randomness=1, squashFactor=5) => {
  const grid = Delaunator.from(uvGrid(resolution));
  const r = (x) => x + Math.random() * 20;
  const randomUnit = (D=1) =>
    normalize([Math.random(), Math.random(), Math.random()]).map(x => x*D)

  const period = Math.random()/50

  let patch = {};
  patch.cells = chunk(grid.triangles, 3);
  patch.positions = chunk(grid.coords, 2).map(([u, v]) => [u, v, 0]);
  patch.normals = normals(patch.cells, patch.positions);

  const D_SQUASHED = D*squashFactor;
  const p0 = addVector(normalize([0,0,randomness*Math.random()/5]).map(x => x*D_SQUASHED), randomUnit(randomness*D_SQUASHED/10));
  const p1 = addVector(normalize([1,0,randomness*Math.random()/5]).map(x => x*D_SQUASHED), randomUnit(randomness*D_SQUASHED/10));
  const p2 = addVector(normalize([0,1,randomness*Math.random()/5]).map(x => x*D_SQUASHED), randomUnit(randomness*D_SQUASHED/10));
  const p3 = addVector(normalize([1,1,randomness*Math.random()/5]).map(x => x*D_SQUASHED), randomUnit(randomness*D_SQUASHED/10));

  return regl({
    vert: gl`
      precision highp float;

      attribute vec3 normal;
      attribute vec3 position;

      uniform vec3
        p0, p1, p2, p3,
        w0, w1, w2, w3;
      uniform mat4 projection, view;
      uniform vec3 offset;

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
        gl_Position = projection * view * (vec4(offset, 0.0) + vec4(p.e1, p.e2, p.e3, 1.0));
      }`,
    frag: gl`
      precision highp float;
      uniform float redAmount;
      varying vec3 n;
      varying vec2 uv;
      void main () {
        vec3 light = vec3(0.5,0.9,0.5);
        light = normalize(light);

        gl_FragColor = 1.5*vec4(redAmount, uv.x, uv.y, 0.75);
      }`,
    attributes: {
      position: () => patch.positions,
      normal: () => patch.normals,
    },
    depth: {
      enable: false
    },

    blend: {
      enable: true,
      func: {
        // srcRGB: 'one minus dst alpha',
        srcRGB: 'one minus dst alpha',
        srcAlpha: 1,
        dstRGB: 1,
        dstAlpha: 1
      },
      equation: {
        rgb: 'subtract',
        alpha: 'subtract'
      },
      color: [0, 0, 0, 0]
    },
    uniforms: {
      p0: ({tick}) => p0.map(x => x*Math.cos(tick*period/4)),
      p1: ({tick}) => p1.map(x => x*Math.sin(tick*period/3)),
      p2: ({tick}) => p2.map(x => x*Math.sin(tick*period/5)),
      p3: ({tick}) => p3.map(x => x*Math.cos(tick*period/7)),
      w0: randomUnit(D),
      w1: randomUnit(D),
      w2: randomUnit(D),
      w3: randomUnit(D),
      offset: (context, props) =>
        props?.offset ? props.offset.map((x) => x*D) : defaultOffset.map(x => x*D),
      redAmount: ({tick}) => 1.0+Math.cos(tick*period)
    },
    elements: patch.cells,
    count: patch.cells.length * 3,
  });
};
