import Delaunator from "delaunator";
import { flatten } from "lodash-es";
import { prepareMesh, uvGrid } from "../src/tessellation";
import RBush from "rbush";
// import mda from "mda";
// import boundPoints from "bound-points";

export default function (points = 100, width = 1, height = 1, center = [0, 0]) {
  // const coords = flatten(
  //   new Array(points)
  //     .fill(null)
  //     .map(() => [
  //       center[0] +
  //         (Math.random() - 0.5) * width +
  //         center[1] +
  //         (Math.random() - 0.5) * height,
  //     ])
  // );
  
  const coords = flatten(uvGrid(Math.sqrt(points)).map(([x,y])=>[x-0.5,y-0.5]))

  const delaunay = new Delaunator(coords);

  const mesh = prepareMesh(delaunay);
  const eps = 0.001
  const faceBounds = mesh.mda.vertices.map((vertex) => {
    const [x, y] = mesh.mda.positions[vertex.index]
    return {
      minX: x-eps,
      minY: y-eps,
      maxX: x+eps,
      maxY: y+eps,
      index: vertex.index,
    };

    //.map(([[minX, maxX], [minY, maxY]]) => ({ minX, maxX, minY, maxY }))
  });
  // console.log('faceBounds',faceBounds)
  mesh.rbush = new RBush();
  mesh.rbush.load(faceBounds);
  // mesh.mda.faces
  // return mesh.mda.
  return mesh;
}
