import normals from "angle-normals";
import { fquad, uvGrid, Alg } from "../src";

import Delaunator from "delaunator";
import { chunk } from "lodash-es";

import glLib from "../gpu/lib.glsl";

let gl = function (s) {
  return `precision mediump float;
  ${glLib}\n\n${s[0]}`;
};

export default draw = (regl, resolution = 64, defaultOffset = [0, 0, 0]) => {
  const grid = Delaunator.from(uvGrid(resolution));
  const D = 10;
  const r = (x) => x + Math.random() * 20;
  const randomUnit = (D=1) => {
    let x = [Math.random(), Math.random(), Math.random(),];
    const norm = x.reduce((a,b)=> a+b, 0)
    return x.map(n => D*n/norm);
  };

  const period = Math.random()/10

  let patch = {};
  patch.cells = chunk(grid.triangles, 3);
  patch.positions = chunk(grid.coords, 2).map(([u, v]) => [u, v, 0]);
  patch.normals = normals(patch.cells, patch.positions);

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
      // p0: [0, 0, Math.random()*D],
      // p1: [D, 0, Math.random()*D],
      // p2: [0, D, Math.random()*D],
      // p3: [D, D, Math.random()*D],
      p0: ({tick}) => [D/2,0,0].map(x => x*Math.cos(tick*period)),
      p1: randomUnit(D),
      p2: ({tick}) => [0,D/2,0].map(x => x*Math.sin(tick*period)),
      p3: randomUnit(D),
      w0: randomUnit(2),
      w1: randomUnit(2),
      w2: randomUnit(2),
      w3: randomUnit(2),
      offset: (context, props) =>
        props?.offset ? props.offset.map((x) => x * D) : defaultOffset.map(x => x*D),

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
