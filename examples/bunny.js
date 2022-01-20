import bunny from "bunny";
import normals from "angle-normals";

console.log(bunny)

export default draw = regl => regl({
  vert: `
      precision mediump float;
      attribute vec3 normals;
      attribute vec4 homogeneousPositions;
      uniform mat4 projection, view;
      varying vec3 n;
      void main () {
        n = normals;
        gl_Position = projection * view * homogeneousPositions;
      }`,
  frag: `
      precision mediump float;
      varying vec3 n;
      void main () {
        gl_FragColor = vec4(0.5 + 0.5 * n, 1);
      }`,
  cull: { enable: true, face: "back" },
  attributes: {
    homogeneousPositions: bunny.positions.map(x => [...x, 1]),
    ...bunny,
    normals: normals(bunny.cells, bunny.positions),
  },
  elements: bunny.cells,
  count: bunny.cells.length * 3,
});
