import bunny from "bunny";
import simplify from "mesh-simplify";

const simpleBunny = simplify(bunny.cells, bunny.positions)(8200);
export default simpleBunny.cells.map((cell) =>
  cell.map((i) => simpleBunny.positions[i])
);
