import bunny from "bunny";
import simplify from "mesh-simplify";
import moize from "moize";

export const simpleBunny = moize(num => simplify(bunny.cells, bunny.positions)(num));
// export default simpleBunny
const defaultBunny = simpleBunny(400)
export default (num=400) => simpleBunny(num).cells.map((cell) =>
  cell.map((i) => simpleBunny(num).positions[i])
);
