import { uvGrid } from "./tessellation";
import Delaunator from "delaunator";
import { chunk, fromPairs, zip } from "lodash-es";

import glLib from "../gpu/lib.glsl";
import moize from "moize";
import { quadPatch, triPatch } from "./tessellation";

export const randomPoints = (D = 8, W = 2) => ({
  p0: randomUnit(D),
  p1: randomUnit(D),
  p2: randomUnit(D),
  p3: randomUnit(D),
  w0: randomUnit(W),
  w1: randomUnit(W),
  w2: randomUnit(W),
  w3: randomUnit(W),
});

type triResolutions = [number, number, number];
type quadResolutions = [number, number, number, number];
type patchResolutions = triResolutions | quadResolutions;

const tessellation = moize(
  (shape, resolution: number | patchResolutions = 8) => {
    let points;
    if (Array.isArray(resolution)) {
      resolution = resolution as patchResolutions;
      points = (
        shape === QUAD
          ? quadPatch(...(resolution as quadResolutions))
          : triPatch(...(resolution as triResolutions))
      ).points; // uvGrid(resolution)
    } else {
      // resolution = type === QUAD
      //   ? [resolution, resolution, resolution, resolution]
      //   : [resolution, resolution, resolution];
      // const { points } =
      //   type === QUAD ? quadPatch(...resolution) : triPatch(...resolution); // uvGrid(resolution)
      points =
        shape === QUAD
          ? uvGrid(resolution)
          : triPatch(resolution, resolution, resolution).points;
    }
    return Delaunator.from(points);
  }
);

// interface mesh {
//   cells?: number[][];
//   positions?: number[][];
//   normals?: number[][];
// }

const prepareMesh = moize(
  (grid) => {
    let mesh: any = {};
    mesh.cells = chunk(grid.triangles, 3);
    mesh.positions = chunk(grid.coords, 2).map(([u, v]) => [
      u,
      v,
      0,
    ]) as number[][];
    mesh.normals = normals(mesh.cells, mesh.positions);
    return mesh;
  },
  { maxSize: 40 }
);

export const randomUnit = (D = 1, size = 3, fn = Math.abs) => {
  let x = new Array(size).fill(0).map(() => fn(Math.random() - 0.5));
  const norm = x.reduce((a, b) => a + b, 0);
  return x.map((n) => (D * n) / norm);
};


const prepareWeight = (w: any, type = "vec4") => {
  if (type === "vec4") {
    return w?.length === 4 ? w : w ? [0, ...w.slice(0, 3)] : [1, 0, 0, 0];
  } else if (type === "CGA3") {
    return w;
    const result =
      w?.length === 16
        ? w
        : w
        ? [0, ...w.slice(0, 3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    console.log(result);
    return result;
  } else if (type === "vec3") {
    return w || [0, 0, 0];
    return w || [0, 0, 0];
  }
};

export const QUAD = "quad";
export const TRI = "tri";
const defaultOptions = {
  shape: TRI,
};

export default function Patch(regl, resolution, options = defaultOptions) {
  const { offset, shape, pointType, weightType } = {
    ...defaultOptions,
    ...options,
  };
  const weightProps =
    weightType === "CGA3"
      ? fromPairs(
          ["w0", "w1", "w2", "w3"].flatMap((varName) =>
            getStructVarNames(varName, membersCGA3).map((k) => [
              k,
              regl.prop(k),
            ])
          )
        )
      : {
          w0: (context, props) => prepareWeight(props?.w0, weightType), // || randomUnit(2),
          w1: (context, props) => prepareWeight(props?.w1, weightType), // || random), // || randomUnit(2),
          w2: (context, props) => prepareWeight(props?.w2, weightType), // || random), // || randomUnit(2),
          w3: (context, props) => prepareWeight(props?.w3, weightType), // || random), // || randomUnit(2),
        };

  console.log(weightProps);

  const grid = tessellation(shape, resolution);
  let { positions, cells, normals } = prepareMesh(grid);

  const count = cells.length * 3;
  return regl({
    vert:
      options.vert ||
      gl`
      precision highp float;

      attribute vec3 normal;
      attribute vec3 position;

      uniform ${pointType || "vec3"}
        p0, p1, p2, p3;
      uniform ${weightType || "vec4"}
        w0, w1, w2, w3;
      uniform mat4 projection, view;
      uniform vec3 offset;

      varying vec3 n;
      varying vec2 uv;

      void main () {

      Patch p =
          ${shape === QUAD ? "bilinearQuad(" : "bilinearTri("}
            p0,
            p1,
            p2,
          ${shape === QUAD ? "p3," : "//"}
            w0,
            w1,
            w2,
          ${shape === QUAD ? "w3," : "//"}
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

      p0: (context, props) => props?.p0, // || randomUnit(8),
      p1: (context, props) => props?.p1, // || randomUnit(8),
      p2: (context, props) => props?.p2, // || randomUnit(8),
      p3: (context, props) => props?.p3, // || randomUnit(8),
      ...weightProps,

      offset: (context, props) => props?.offset || offset || [0, 0, 0], //|| randomUnit(30, 'offset'),
    },
    elements: cells,
    count: cells.length * 3,
  });
}
