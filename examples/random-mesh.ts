import Delaunator from "delaunator";
import { flatten } from "lodash-es";
import { prepareMesh } from "../src/tessellation";
import RBush from "rbush";
import mda from "mda";
import boundPoints from "bound-points";

export default function (points = 100, width = 1, height = 1, center = [0, 0]) {
  const coords = flatten(
    new Array(points)
      .fill(null)
      .map(() => [
        center[0] +
          (Math.random() - 0.5) * width +
          center[1] +
          (Math.random() - 0.5) * height,
      ])
  );

  const delaunay = new Delaunator(coords);

  const mesh = prepareMesh(delaunay);
  const faceBounds = mesh.mda.faces.map((face) => {
    const bounds = boundPoints(
      mda.FaceVertices(face).map(({ index: i }) => mesh.mda.positions[i])
    );
    const [[minX, minY], [maxX, maxY]] = bounds;
    return {
      minX,
      minY,
      maxX,
      maxY,
      faceIndex: face.index,
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
