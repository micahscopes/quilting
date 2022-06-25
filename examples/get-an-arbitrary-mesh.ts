import Delaunator from "delaunator";
import { flatten } from "lodash-es";
import { prepareMesh, uvGrid } from "../src/tessellation";
// import RBush from "rbush";

export default function (points = 100, width = 1, height = 1, center = [0, 0]) {
  let coords = flatten(
    new Array(3*points)
      .fill(null)
      .map(() => [
        center[0] +
          (Math.random() - 0.5) * width +
          center[1] +
          (Math.random() - 0.5) * height,
      ])
  );
  
  coords = flatten(uvGrid(Math.sqrt(points)).map(([x,y])=>[x-0.5,y-0.5]))

  const delaunay = new Delaunator(coords);

  const mesh = prepareMesh(delaunay);
  // const eps = 0.001
  // const faceBounds = mesh.mda.vertices.map((vertex) => {
  //   const [x, y] = mesh.mda.positions[vertex.index]
  //   return {
  //     minX: x-eps,
  //     minY: y-eps,
  //     maxX: x+eps,
  //     maxY: y+eps,
  //     index: vertex.index,
  //   };
  // });
  // mesh.rbush = new RBush();
  // mesh.rbush.load(faceBounds);
  return mesh;
}

export function gridMesh(points = 100){
  const coords = flatten(uvGrid(Math.sqrt(points)).map(([x,y])=>[x-0.5,y-0.5]))
  const delaunay = new Delaunator(coords);
  const mesh = prepareMesh(delaunay);

  return mesh;
}