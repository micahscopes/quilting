import normals from "angle-normals";
import { fquad, uvGrid, Alg } from "./index.ts";
import Delaunator from "delaunator";
import { chunk } from "lodash-es";

import glLib from "../gpu/lib.glsl";
import moize from "moize";

const tessellation = moize.infinite((resolution) =>
  Delaunator.from(uvGrid(resolution))
);

const prepareMesh = moize(
  (grid) => {
    console.log("preparing patch");
    let mesh = {};
    mesh.cells = chunk(grid.triangles, 3);
    mesh.positions = chunk(grid.coords, 2).map(([u, v]) => [u, v, 0]);
    mesh.normals = normals(mesh.cells, mesh.positions);
    return mesh;
  },
  { maxSize: 40 }
);

let gl = function (s) {
  return `precision mediump float;
  ${glLib}\n\n${s[0]}`;
};

const randomUnit = (D = 1) => {
  let x = [Math.random(), Math.random(), Math.random()];
  const norm = x.reduce((a, b) => a + b, 0);
  return x.map((n) => (D * n) / norm);
};

export default function Patch(regl, resolution, { offset } = {}) {
  const grid = tessellation(resolution);
  let { positions, cells, normals } = prepareMesh(grid);
  offset = offset;
  // positions = regl.buffer(positions);
  // cells = regl.buffer(cells);
  // normals = regl.buffer(normals);
  const count = cells.length * 3;
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
      position: () => positions,
      normal: () => normals,
    },
    uniforms: {
      matcapTexture: regl.prop("matcap"),
      texture: regl.prop("texture"),
      eye: (context, props) => props?.eye || [1, 0, 0],

      p0: (context, props) => props?.p0 || randomUnit(30),
      p1: (context, props) => props?.p1 || randomUnit(30),
      p2: (context, props) => props?.p2 || randomUnit(30),
      p3: (context, props) => props?.p3 || randomUnit(30),
      w0: (context, props) => props?.w0 || randomUnit(2),
      w1: (context, props) => props?.w1 || randomUnit(2),
      w2: (context, props) => props?.w2 || randomUnit(2),
      w3: (context, props) => props?.w3 || randomUnit(2),

      offset: (context, props) =>
        props?.offset || offset || randomUnit(30),
    },
    elements: cells,
    count: cells.length*3,
  });
}
