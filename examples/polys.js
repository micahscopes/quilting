import bunny from "bunny";
import simplify from "mesh-simplify";
import refine from "refine-mesh";
import {vertexNormals} from "normals";
import moize from "moize";

export const prepareMesh = ({ cells, positions }) =>
  cells.map((cell) => cell.map((i) => [1, ...positions[i]]));

export const simpleBunny = moize((num) =>
  simplify(bunny.cells, bunny.positions)(num)
);
const defaultBunny = simpleBunny(400);

export default (num = 400) => prepareMesh(simpleBunny(num));
export const refinedBunny = (edgeLength = 0.5) => {
  const normals = vertexNormals(bunny.cells, bunny.positions);
  return refine(bunny.cells, bunny.positions, normals, {edgeLength})
}