import { fquad, Alg, barePatchIndices } from "../dist/index.mjs";
import test from "tape";
import { range } from "lodash-es";

const points = [
  [
    Alg.Vector(0, 0, 0),
    Alg.Vector(0, 1, 0),
  ],
  [
    Alg.Vector(1, 0, 0),
    Alg.Vector(1, 1, 0),
  ],
];

const weights = [
  [Alg.Vector(0.5, 0.1), Alg.Vector(-0.5, 0.2, 0.1)],
  [Alg.Vector(0,0,0.1), Alg.Vector(0,0,0.1)],
];

const quad = fquad(2, 2, points, weights)
// console.log('result', range(0, 1, 0.05).map(x => quad(0.5, x).Vector));

console.log(barePatchIndices(4))
// test.skip("simple quad patch", (t) => {
//   t.equal(fquad(2, 2, points, weights)(0, 0.5), 0.5);
// });
