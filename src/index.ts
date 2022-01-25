import Algebra from "ganja.js";
import { memoize, range, reduce } from "lodash-es";
import { BxB } from "./bernstein";

import { CGA, pointWeight, up, Mul3, ni } from "./ga-helpers.mjs";

const product = (...a: any[][]) =>
  a.reduce((a, b) => a.flatMap((d) => b.map((e) => [d, e].flat())));

export const uvGrid = (uResolution: number, vResolution: number) =>
  product([...range(0, 1, 1 / uResolution), 1], [...range(0, 1, 1 / vResolution), 1]);


export const bilinearGrid = (resU0: number, resU1: number, resV0: number, resV1: number) => {
  /*
    Generates a grid of varying resolution

    ┌────resU0────┐
    │             │
  resV0         resV1
    │             │
    └────resU1────┘

  */
  const points = [];

}
// todo: abstract this out
export const Alg = CGA;

export const fquad =
  // todo: (Alg: any = CGA) =>
  (pointsVectorsUV: any[][], weightsUV: any[][]) => {
    const deg_u = pointsVectorsUV.length;
    const deg_v = pointsVectorsUV[0].length;
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
