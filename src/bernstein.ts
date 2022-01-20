import triangle from "pascals-triangle";
const factorials = [
  1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600,
  6227020800, 87178291200, 1307674368000, 20922789888000, 355687428096000,
  6402373705728000, 121645100408832000, 2432902008176640000,
  51090942171709440000, 1124000727777607680000, 25852016738884976640000,
  620448401733239439360000,
];
const binomials = triangle(12);

const fct = (i: number) => factorials[i];
const binomial = (n: number, k: string | number) => binomials[n][k];

export const B = (d: number, i: number, t: number) =>
  binomial(d - 1, i) * Math.pow(1 - t, d - 1 - i) * Math.pow(t, i);

export const Btri = (d: number, i: number, j: number, u: number, v: number) =>
  (fct(d - 1) / fct(d - 1 - i - j) / fct(i) / fct(j)) *
  Math.pow(1 - u - v, d - 1 - i - j) *
  Math.pow(u, i) *
  Math.pow(v, j);

export const BxB = (
  d1: number,
  d2: number,
  i: number,
  j: number,
  t1: number,
  t2: number
) => B(d1, i, t1) * B(d2, j, t2);
