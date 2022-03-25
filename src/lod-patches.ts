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
  (A: number, B: number, C: number, D: number, n = 4) =>
  ([u, v]: number[]) =>
    u ** n * C + (1 - v) ** n * D + (1 - u) ** n * A + v ** n * B;

/**
 * Generate a function that interpolates between edge weights of a triangle
 * in barycentric coordinates.
 * @param {number} A - edge A weight.
 * @param {number} B - edge B weight.
 * @param {number} C - edge C weight.
 * @return {function: } an interpolation function that takes barycentric coordinates.
 */
export const triEdgeWeightInterpolator =
  (A: number, B: number, C: number, n = 1) =>
  ([x, y, z]: number[]) => {
    // x = 1-x; y=1-y; z=1-z;
    // float e0 = barys.x * barys.y;
    // float e1 = barys.y * barys.z;
    // float e2 = barys.z * barys.x;
    const e0 = y*z;
    const e1 = x*z
    const e2 = x*y
    // float sum = e0+e1+e2;
    const sum = e0+e1+e2;

    const c0 = A;
    const c1 = B;
    const c2 = C

    return ((e0*c0)**n + (e1*c1)**n + (e2*c2)**n) / sum**n;

    // return (
    //   ((y * z) ** n * A + (x * z) ** n * B + (x * y) ** n * C) *
    //   (y * z + x * z + x * y)
    // );
  };

import { KdTreeSet } from "@thi.ng/geom-accel";
import { DensityFunction, samplePoisson } from "@thi.ng/poisson";
import { randMinMax2 } from "@thi.ng/vectors";
import { range, sum } from "lodash-es";
import bc from "barycentric-coordinates";
import barycentric from "barycentric";

const cartesian = (bc, corners) => [
  bc[0] * corners[0][0] + bc[1] * corners[1][0] + bc[2] * corners[2][0],
  bc[0] * corners[0][1] + bc[1] * corners[1][1] + bc[2] * corners[2][1],
];

const {
  // triangleCartesianCoords,
  triangleBarycentricCoords,
} = bc;
// console.log(bc)

type Point2D = [number, number];

const sampleTriangle = (corners: [Point2D, Point2D, Point2D]) => () => {
  const u = [Math.random(), Math.random()];
  const su0 = Math.sqrt(u[0]);
  const b0 = 1 - su0;
  const b1 = u[1] * su0;

  const pt = cartesian([b0, b1, 1 - b0 - b1], corners);
  return pt;
};

export const quadPatch = (
  resA: number,
  resB: number,
  resC: number,
  resD: number
) => {
  const triangleFanWeights: [number, number, number][] = [
    [resA, (resA + resB) / 2, (resA + resD) / 2],
    [resB, (resB + resC) / 2, (resB + resA) / 2],
    [resC, (resC + resD) / 2, (resC + resB) / 2],
    [resD, (resD + resA) / 2, (resD + resC) / 2],
  ];
  // const triangleFanWeights: [number, number, number][] = [
  //   [resA, 2, 2],
  //   [resB, 2, 2],
  //   [resC, 2, 2],
  //   [resD, 2, 2],
  // ];
  console.log(triangleFanWeights, resA, resB, resC, resD);
  const triangleFanCoords: [Point2D, Point2D, Point2D][] = [
    [
      [0.5, 0.5],
      [0, 1],
      [0, 0],
    ],
    [
      [0.5, 0.5],
      [0, 0],
      [1, 0],
    ],
    [
      [0.5, 0.5],
      [1, 0],
      [1, 1],
    ],
    [
      [0.5, 0.5],
      [1, 1],
      [0, 1],
    ],
  ];

  const getRadiusTri =
    (
      triangle: [Point2D, Point2D, Point2D],
      weights: [number, number, number]
    ) =>
    (pt: Point2D) => {
      // const bary = triangleBarycentricCoords(
      //   [...pt, 1],
      //   triangle.map((x) => [...x, 1])
      // );
      const bary = barycentric(triangle, pt)
      const x = triEdgeWeightInterpolator(
        1 / weights[0],
        1 / weights[1],
        1 / weights[2],
        // 0.45
      )(bary);
      // console.log(x, bary, pt, triangle, weights)
      return x;
    };

  const downward = (x: number) => -(x - 0.5) + 0.5;
  const upward = (x: number) => x - 0.5 + 0.5;

  const getTriangleInfoForPoint = ([x, y]: number[]) => {
    let i = 0;
    const down = downward(x);
    const up = upward(x);
    if (y < down && y > up) {
      i = 0;
    } else if (y < down && y < up) {
      i = 1;
    } else if (y > down && y < up) {
      i = 2;
    } else if (y > down && y > up) {
      i = 3;
    }
    return { weights: triangleFanWeights[i], corners: triangleFanCoords[i], i };
  };

  const getRadiusAuto = (pt: Point2D) => {
    const { corners, weights, i } = getTriangleInfoForPoint(pt);
    return getRadiusTri(corners, weights)(pt);
  };

  const index = new KdTreeSet(2);
  const addPoint = (radius: number) => (x: number[]) =>
    index.add(
      x.map((x) => x * 1000),
      radius
    );

  // add stitching points
  for (let a = 0; a <= 1; a += 1 / resA) {
    index.add([0, a]);
  }
  for (let b = 0; b <= 1; b += 1 / resB) {
    index.add([b, 0]);
  }
  for (let c = 0; c <= 1; c += 1 / resC) {
    index.add([1, c]);
  }
  for (let d = 0; d <= 1; d += 1 / resD) {
    index.add([d, 1]);
  }

  const sampleTri = (
    corners: [Point2D, Point2D, Point2D],
    weights: [number, number, number]
  ) => {
    ////////////////////////
    // const density = getRadiusTri(corners, weights) as DensityFunction
    // const points = sampleTriangle(corners)
    ////////////////////////
    // const points = () => randMinMax2(null, [0, 0], [1, 1])
    const points = () => [Math.random(), Math.random()];
    const density = getRadiusAuto as DensityFunction;
    window.density = density;
    ////////////////////////
    return samplePoisson({
      index,
      ////////////////////////
      points,
      density,
      ///////////////////////////
      iter: 2,
      jitter: 0.00001,
      max: 5000,
      quality: 5000,
    });
  };
  const boundaryPoints = index.keys();

  const points = [
    ...boundaryPoints, //
    ...sampleTri(triangleFanCoords[0], triangleFanWeights[0]),
    // ...sampleTri(triangleFanCoords[1], triangleFanWeights[1]),
    // ...sampleTri(triangleFanCoords[2], triangleFanWeights[2]),
    // ...sampleTri(triangleFanCoords[3], triangleFanWeights[3]),
  ];

  return { points, index };
};
