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

const product = (...a: any[][]) =>
  a.reduce((a, b) => a.flatMap((d) => b.map((e) => [d, e].flat())));

export const uvGrid = (uResolution: number, vResolution: number) =>
  product([...range(0, 1, 1 / uResolution), 1], [...range(0, 1, 1 / vResolution), 1]);

export const quadEdgeWeightInterpolator =
  (A: number, B: number, C: number, D: number, n = 1) =>
  ([x, y]: number[]) => {
    const z = 1 - x;
    const w = 1 - y;
    const e0 = y * z * w;
    const e1 = x * z * w;
    const e2 = x * y * w;
    const e3 = x * y * z;
    const sum = e0 + e1 + e2 + e3;

    const c0 = A;
    const c1 = B;
    const c2 = C;
    const c3 = D;

    return (
      ((e0 * c0) ** n + (e1 * c1) ** n + (e2 * c2) ** n + e3 * c3) / sum ** n
    );
  };

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
    const e0 = y * z;
    const e1 = x * z;
    const e2 = x * y;
    const sum = e0 + e1 + e2;

    const c0 = A;
    const c1 = B;
    const c2 = C;

    return ((e0 * c0) ** n + (e1 * c1) ** n + (e2 * c2) ** n) / sum ** n;
  };

import { KdTreeSet } from "@thi.ng/geom-accel";
import { DensityFunction, samplePoisson } from "@thi.ng/poisson";
import { dist, divN2 } from "@thi.ng/vectors";
import barycentric from "barycentric";
import { range } from "lodash-es";

const cartesian = (bc, corners) => [
  bc[0] * corners[0][0] + bc[1] * corners[1][0] + bc[2] * corners[2][0],
  bc[0] * corners[0][1] + bc[1] * corners[1][1] + bc[2] * corners[2][1],
];


type Point2D = [number, number];

const sampleTriangle = (corners: [Point2D, Point2D, Point2D]) => () => {
  const u = [Math.random(), Math.random()];
  const su0 = Math.sqrt(u[0]);
  const b0 = 1 - su0;
  const b1 = u[1] * su0;

  const pt = cartesian([b0, b1, 1 - b0 - b1], corners);
  return pt;
};

const getRadiusTri =
  (triangle: [Point2D, Point2D, Point2D], weights: [number, number, number]) =>
  (pt: Point2D) => {
    let bary = barycentric(triangle, pt);
    const D = dist([0, 0], [0.5, 0.5]);
    bary = [
      divN2(null, bary[0], 1),
      divN2(null, bary[1], D),
      divN2(null, bary[2], D),
    ];
    const x = triEdgeWeightInterpolator(
      weights[0],
      weights[1],
      weights[2]
      // 0.45
    )(bary);
    // console.log(x, bary, pt, triangle, weights)
    return 1 / x;
  };

const defaultOptions = {
  iter: 1,
  jitter: 0.00001,
  max: 3000000,
  quality: 150,
};

export const quadPatchPrototype = (
  resA: number,
  resB: number,
  resC: number,
  resD: number,
  opts: object
) => {
  opts = { ...defaultOptions, ...opts };
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
    const points = () => [Math.random(), Math.random()];
    const density = getRadiusAuto as DensityFunction;
    return samplePoisson({
      index,
      points,
      density,
      max: 0,
      ...opts,
    });
  };
  const boundaryPoints = index.keys();

  const points = [
    ...boundaryPoints, //
    ...sampleTri(triangleFanCoords[0], triangleFanWeights[0]),
  ];

  return { points, index };
};

export const quadPatch = (
  resA: number,
  resB: number,
  resC: number,
  resD: number,
  opts: object
) => {
  opts = { ...defaultOptions, ...opts };
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

  const interpolation = (x) =>
    1 / quadEdgeWeightInterpolator(resA, resB, resC, resD)(x);
  const sample = () => {
    const points = () => [Math.random(), Math.random()];
    const density = interpolation as DensityFunction;
    return samplePoisson({
      index,
      points,
      density,
      max: 0,
      ...opts,
    });
  };
  const boundaryPoints = index.keys();

  const points = [
    ...boundaryPoints, //
    ...sample(),
  ];

  return { points, index };
};

export const triPatch = (
  resA: number,
  resB: number,
  resC: number,
  opts: object
) => {
  opts = { ...defaultOptions, ...opts };
  const corners: [Point2D, Point2D, Point2D] = [
    [0, 1],
    [1, 0],
    [0, 0],
  ];

  const index = new KdTreeSet(2);
  const addPoint = (radius: number) => (x: number[]) =>
    index.add(
      x.map((x) => x * 1000),
      radius
    );

  // add stitching points
  for (let a = 0; a <= 1; a += 1 / resA) {
    index.add([a, 0]);
  }
  for (let b = 0; b <= 1; b += 1 / resB) {
    index.add([0, b]);
  }
  for (let c = 0; c <= 1; c += 1 / resC) {
    index.add([1-c, c]);
  }

  const interpolation = getRadiusTri(corners, [resA, resB, resC]);
  // const interpolation = (x) =>
  //   1 / triEdgeWeightInterpolator(resA, resB, resC)(x);
  const sampleTri = (
    corners: [Point2D, Point2D, Point2D],
    weights: [number, number, number]
  ) => {
    const points = sampleTriangle(corners);
    const density = interpolation;
    return samplePoisson({
      index,
      points,
      density: interpolation as DensityFunction,
      max: 0,
      ...opts,
    });
  };
  const boundaryPoints = index.keys();

  const points = [
    ...boundaryPoints, //
    ...sampleTri(corners, [resA, resB, resC]),
  ];

  return { points, index };
};
