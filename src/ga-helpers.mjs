import Algebra from "ganja.js";
import { reduce, reduceRight } from "lodash-es";

export const CGA = Algebra(4, 1);
const Alg = CGA

export const zero = new Array(CGA.length).fill(0);

export const e4 = Alg.Vector(0,0,0,1)
export const e5 = Alg.Vector(0,0,0,0,1)
export const ni = Alg.Vector(0,0,0,1,1)
export const no = Alg.Vector(0,0,0,-0.5,0.5)
export const Mul3 = (x,y,z) => Alg.Mul(Alg.Mul(x,y), z)

// export const up = x => {
//   console.log('x*x', Alg.Mul(x,x))
//   return Mul3(x, x, ni)
// }

export const up = x => reduce([no, x, Alg.Mul(Mul3(x, x, ni), Alg.Scalar(0.5))], (x,y) => Alg.Add(x,y), Alg.Scalar(0))

export const pointWeight = (point, weight) => Alg.Add(Alg.Mul(Mul3(point, ni, weight), Alg.Scalar(0.5)), weight)

// Create a Clifford Algebra with 4,1 metric for 3D CGA.
// export const { up, pointWeight } = CGA.inline(() => {
//   // We start by defining a null basis in which we will work.
//   var ni = 1e4 + 1e5,
//     no = 0.5e5 - 0.5e4;

//   // To create a conformal point, you upcast :
//   var up = (x) => no + x + (x * x * ni) * 0.5;
//   var pointWeight = (point, weight) =>
//     (point * weight * ni) / 2 + weight;

//   var Mul3 = (u, v, w) => u * v * w;

//   return { up, pointWeight, ni, no, Mul3 };
// })();