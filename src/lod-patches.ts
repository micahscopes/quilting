/* Generate a function that interpolates between edge weights of a UV square.

  v
  ^
  | ┌──────D──────┐
  | │             │
  | A             C
  | │             │
  | └──────B──────┘
 -|------------------> u

 * @param {number} A - edge A weight.
 * @param {number} B - edge B weight.
 * @param {number} C - edge C weight.
 * @param {number} D - edge D weight.
 * @return {function: } an interpolation function.
 */
export const quadEdgeWeightInterpolator =
  (A: number, B: number, C: number, D: number, n = 4)=>
  ([u, v]: number[]) =>
    (u ** n * C + (1 - v) ** n * D + (1 - u) ** n * A + v ** n * B) ;

// window.quadInt = quadEdgeWeightInterpolator

// export const quadEdgeWeightInterpolator =
//   (A: number, B: number, C: number, D: number) => ([u,v]: number[]) => {
//     // from https://math.stackexchange.com/a/830139/
//     let coefU = 0.5;
//     let coefV = 0.5;
//     const distU = 0.5 - Math.abs(u - 0.5);
//     const distV = 0.5 - Math.abs(v - 0.5);
//     if (distU + distV > 0.000000000000000001) {
//       const distSum = distU + distV;
//       coefU = distU / distSum;
//       coefV = distV / distSum;
//     }
//     return (
//       A * (1 - u) * coefU + B * (1-v) * coefV + C * u * coefU + D * u * coefV
//     );
//   };

/**
 * Generate a function that interpolates between edge weights of a triangle
 * in barycentric coordinates.
 * @param {number} A - edge A weight.
 * @param {number} B - edge B weight.
 * @param {number} C - edge C weight.
 * @return {function: } an interpolation function that takes barycentric coordinates.
 */
export const triEdgeWeightInterpolator =
  (A: number, B: number, C: number) =>
  ([x, y, z]: number[]) =>
    ((1 - x) * A + (1 - y) * B + (1 - z) * C) / 2;

import { KdTreeSet } from "@thi.ng/geom-accel";
import { DensityFunction, samplePoisson } from "@thi.ng/poisson";
import { randMinMax2 } from "@thi.ng/vectors";
import { range } from "lodash-es";

export const quadPatch = (
  resA: number,
  resB: number,
  resC: number,
  resD: number
) => {
  const getRadius = (pt: number[]) => {
    const x = quadEdgeWeightInterpolator(resA, resB, resC, resD)(pt);
    return 1 / x / 2;
  };
  const index = new KdTreeSet(2);
  const addPoint = (radius: number) => (x: number[]) =>
    index.add(
      x.map((x) => x * 1000),
      radius
    );

  // add stitching points
  // range(0, resA + 1, 1 / resA)
  //   .map((a) => [0, a])
  //   .forEach(addPoint(1 / resA));
  // range(0, resB + 1, 1 / resB)
  //   .map((b) => [b, 0])
  //   .forEach(addPoint(1 / resB));
  // range(0, resC + 1, 1 / resC)
  //   .map((c) => [1, c])
  //   .forEach(addPoint(1 / resC));
  // range(0, resD + 1, 1 / resD)
  //   .map((d) => [d, 1])
  //   .forEach(addPoint(1 / resD));

  for (let a = 0; a < 1; a += 1 / resA) {
    index.add([0, a]);
  }
  for (let b = 0; b < 1; b += 1 / resB) {
    index.add([b, 1]);
  }
  for (let c = 0; c < 1; c += 1 / resC) {
    index.add([1, c]);
  }
  for (let d = 0; d < 1; d += 1 / resD) {
    index.add([d, 0]);
  }

  const boundaryPoints = index.keys();

  const points = [
    ...boundaryPoints, //
    ...samplePoisson({
      index,
      points: () => {
        const pt = randMinMax2(null, [0, 0], [1, 1]);
        return pt;
      },
      density: getRadius as DensityFunction,
      iter: 5,
      jitter: 0.001,
      max: 10000,
      quality: 1000,
    }),
  ];

  return { points, index };
};
