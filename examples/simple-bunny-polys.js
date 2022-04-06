import bunny from "bunny";
import simplify from "mesh-simplify";

const simpleBunny = simplify(bunny.cells, bunny.positions)(200);
// export default simpleBunny
export default simpleBunny.cells.map((cell) =>
  cell.map((i) => simpleBunny.positions[i])
);
