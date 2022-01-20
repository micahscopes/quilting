import normals from "angle-normals";


const controlMesh = {
  positions: [
    [0, 0, 0],
    [5, 0, 0],
    [0, 5, 0],
    [0, 0, 5],
  ],

}

const patch = {
  positions: [
    [-5, 0, 0],
    [0, 5, 0],
    [0, 0, 5],
  ],
  cells: [[0,1,2]]
};

patch.homogeneousPositions = patch.positions.map(x => [...x, 1])
patch.normals = normals(patch.cells, patch.positions)


export default draw = (regl) =>
  regl({
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
    // cull: { enable: true, face: "back" },
    attributes: patch,
    elements: patch.cells,
    count: patch.cells.length * 3,
  });
