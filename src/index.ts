import Algebra from "ganja.js";
import { memoize, range, reduce } from "lodash-es";
import { BxB } from "./bernstein";

import { CGA, pointWeight, up, Mul3, ni } from "./ga-helpers.mjs";

const product = (...a: any[][]) =>
  a.reduce((a, b) => a.flatMap((d) => b.map((e) => [d, e].flat())));

// todo: abstract this out
export const Alg = CGA;

export const barePatchIndices = (
  uResolution: number,
  vResolution: number = -1
) => {
  vResolution = vResolution < 0 ? uResolution : vResolution;
  const U = range(0, 1, 1 / uResolution); //.map((u, i) => ({i, u}))
  const V = range(0, 1, 1 / vResolution); //.map((v, j) => ({j, v}))
  let UV = [];
  let quads = [];

  let k = 0;
  // V.forEach((v, i) => {
  //   U.forEach((u, j) => {
  //     UV.push([u,v])
  //     quads.push([[]])
  //     k+=0
  //   })
  // }

  return { UV, U, V };
};

export const fquad =
  // todo: (Alg: any = CGA) =>
  (
    deg_u: number,
    deg_v: number,
    pointsVectorsUV: any[][],
    weightsUV: any[][]
  ) => {
    const patchIndices = product(range(deg_u), range(deg_v));
    const points = patchIndices.map(([i, j]: [number, number]) =>
      up(pointsVectorsUV[i][j])
    );
    const weights = patchIndices.map(([i, j]: [number, number]) =>
      pointWeight(pointsVectorsUV[i][j], weightsUV[i][j])
    );

    const PW = memoize((k) => Alg.Mul(points[k], weights[k]));

    return (u: number, v: number) => {
      const top = reduce(
        patchIndices.map(([i, j]: [number, number], k: number) =>
          Alg.Mul(PW(k), Alg.Scalar(BxB(deg_u, deg_v, i, j, u, v)))
        ),
        (x, y) => Alg.Add(x, y),
        Alg.Scalar(0)
      );
      const bottom = reduce(
        patchIndices.map(([i, j]: [number, number], k: number) =>
          Alg.Mul(weights[k], Alg.Scalar(BxB(deg_u, deg_v, i, j, u, v)))
        ),
        (x, y) => Alg.Add(x, y),
        Alg.Scalar(0)
      );

      return Alg.Div(top, bottom);
    };
  };
