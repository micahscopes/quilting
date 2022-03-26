import normals from "angle-normals";
import { fquad, uvGrid, Alg } from "./index.ts";
import Delaunator from "delaunator";
import { chunk } from "lodash-es";

import glLib from "../gpu/lib.glsl";
import moize from "moize";
import { quadPatch, triPatch } from "./lod-patches";

const tessellation = moize((type, resolution) => {
  resolution = resolution.length
    ? resolution
    : type === QUAD
    ? [resolution, resolution, resolution, resolution]
    : [resolution, resolution, resolution];
  const { points } =
    type === QUAD ? quadPatch(...resolution) : triPatch(...resolution); // uvGrid(resolution)
  console.log(points);
  return Delaunator.from(points);
});

const prepareMesh = moize(
  (grid) => {
    console.log("preparing patch");
    let mesh = {};
    mesh.cells = chunk(grid.triangles, 3);
    console.log(grid.triangles.length, grid.coords.length);
    mesh.positions = chunk(grid.coords, 2).map(([u, v]) => [u, v, 0]);
    mesh.normals = normals(mesh.cells, mesh.positions);
    return mesh;
  },
  { maxSize: 40 }
);

let gl = function (s, ...values) {
  let str = "";
  s.forEach((string, i) => {
    str += string + (values[i] || "");
  });

  return `precision mediump float;
  ${glLib}\n\n${str}`;
};

const randomUnit = (D = 1, key = null) => {
  let x = [Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5];
  const norm = x.reduce((a, b) => a + b, 0);
  return x.map((n) => (D * n) / norm);
};

const QUAD = "quad";
const TRI = "tri";
const defaultOptions = {
  type: QUAD,
};

export default function Patch(regl, resolution, options = defaultOptions) {
  const { offset, type } = { ...defaultOptions, ...options };
  const grid = tessellation(type, resolution);
  let { positions, cells, normals } = prepareMesh(grid);
  console.log(type, offset);
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
          ${type === QUAD ? "bilinearQuad(" : "bilinearTri("}
            p0,
            p1,
            p2,
          ${type === QUAD ? "p3," : "//"}
            w0,
            w1,
            w2,
          ${type === QUAD ? "w3," : "//"}
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

      p0: (context, props) => props?.p0 || randomUnit(8, "p1"),
      p1: (context, props) => props?.p1 || randomUnit(8, "p2"),
      p2: (context, props) => props?.p2 || randomUnit(8, "p3"),
      p3: (context, props) => props?.p3 || randomUnit(8, "p4"),
      w0: (context, props) => props?.w0 || randomUnit(2, "w1"),
      w1: (context, props) => props?.w1 || randomUnit(2, "w2"),
      w2: (context, props) => props?.w2 || randomUnit(2, "w3"),
      w3: (context, props) => props?.w3 || randomUnit(2, "w4"),

      offset: (context, props) => props?.offset || offset || [0, 0, 0], //|| randomUnit(30, 'offset'),
    },
    elements: cells,
    count: cells.length * 3,
  });
}
