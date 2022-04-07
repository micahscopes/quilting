import bunny from "bunny";
import simplify from "mesh-simplify";
import moize from "moize";

export const prepareMesh = ({ cells, positions }) =>
  cells.map((cell) => cell.map((i) => [...positions[i]]));

export const simpleBunny = moize((num) =>
  simplify(bunny.cells, bunny.positions)(num)
);
const defaultBunny = simpleBunny(400);

export default (num = 400) => prepareMesh(simpleBunny(num));
