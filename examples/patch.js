import normals from "angle-normals";
import { fquad, uvGrid, Alg } from "../src";
import Delaunator from "delaunator";
import { chunk } from "lodash-es";

import glLib from "../gpu/lib.glsl";
import moize from "moize";

const tessellation = moize.infinite((resolution) =>
  Delaunator.from(uvGrid(resolution))
);

const preparePatch = moize(
  (grid) => {
    console.log('preparing patch')
    let patch = {};
    patch.cells = chunk(grid.triangles, 3);
    patch.positions = chunk(grid.coords, 2).map(([u, v]) => [u, v, 0]);
    patch.normals = normals(patch.cells, patch.positions);
    return patch;
  },
  { maxSize: 20 }
);

let gl = function (s) {
  return `precision mediump float;
  ${glLib}\n\n${s[0]}`;
};

export default draw = (
  regl,
  resolution = 256,
  defaultOffset = [0, 0, 0],
  matcap,
  texture
) => {
  const grid = tessellation(resolution);
  const D = 10;
  const r = (x) => x + Math.random() * 20;
  const randomUnit = (D = 1) => {
    let x = [Math.random(), Math.random(), Math.random()];
    const norm = x.reduce((a, b) => a + b, 0);
    return x.map((n) => (D * n) / norm);
  };
  const patch = preparePatch(grid)

  const period = Math.random() / 50;

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

      Patch p =
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

        n = p.normal;
        uv = vec2(position.x, position.y);
        gl_Position = projection * view * vec4(p.vertex+offset, 1.0);
      }`,
    frag: gl`
      precision highp float;
      uniform float time;
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

        // gl_FragColor = texture2D(texture, uv);
      }`,
    blend: {
      enable: true,
      func: {
        srcRGB: "src alpha",
        srcAlpha: 1,
        dstRGB: "one minus src alpha",
        dstAlpha: 1,
      },
      equation: {
        rgb: "add",
        alpha: "add",
      },
      color: [0, 0, 0, 0],
    },

    attributes: {
      position: () => patch.positions,
      normal: () => patch.normals,
    },
    uniforms: {
      matcapTexture: matcap,
      texture,
      eye: (context, props) => props?.eye || [1, 0, 0],

      // time(){ return Date.now() },
      // p0: [0, 0, Math.random()*D],
      // p1: [D, 0, Math.random()*D],
      // p2: [0, D, Math.random()*D],
      // p3: [D, D, Math.random()*D],
      p0: ({ tick }) => [D / 2, 0, 0].map((x) => x * Math.cos(tick * period)),
      p1: randomUnit(D),
      p2: ({ tick }) => [0, D / 2, 0].map((x) => x * Math.sin(tick * period)),
      p3: randomUnit(D),
      w0: randomUnit(2),
      w1: randomUnit(2),
      w2: randomUnit(2),
      w3: randomUnit(2),
      offset: (context, props) =>
        props?.offset
          ? props.offset.map((x) => x * D)
          : defaultOffset.map((x) => x * D),

      // w00: [1, 0, -1],
      // w0: ({tick}) => [Math.sin(tick/150), 1, 0],
      // w10: ({tick}) => [0, 0, 2*Math.sin(tick/150)],
      // w11: ({tick}) => [0, 2*3.1415926*Math.sin(tick/50), -1],
      // time: ({tick}) => {
      //   // console.log(Date.now())
      //   return tick/50
      // }
      // time: 10
    },
    elements: patch.cells,
    count: patch.cells.length * 3,
  });
};

// console.log(matcap)
