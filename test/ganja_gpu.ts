// import util from "util";
// import Algebra from "ganja.js";
// import { GPU } from "gpu.js";

// const A = Algebra({ p: 1 });
// const gpu = new GPU();

// const geoProduct = gpu
//   .createKernel(function (a: number[], b: number[]) {
//     const result = A.Mul(a, b);
//     return result.buffer[this.thread.x];
//   })
//   .setOutput([2, 1]);

// const x = A.Add([1, 2], [1, 2]);
// console.log(geoProduct([1, 2], [2, 3]));
