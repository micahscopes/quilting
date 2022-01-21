import {range as $f3Ts0$range, memoize as $f3Ts0$memoize, reduce as $f3Ts0$reduce} from "lodash-es";
import $f3Ts0$pascalstriangle from "pascals-triangle";
import $f3Ts0$ganjajs from "ganja.js";



const $adf7c9e79b2099a2$var$factorials = [
    1,
    2,
    6,
    24,
    120,
    720,
    5040,
    40320,
    362880,
    3628800,
    39916800,
    479001600,
    6227020800,
    87178291200,
    1307674368000,
    20922789888000,
    355687428096000,
    6402373705728000,
    121645100408832000,
    2432902008176640000,
    51090942171709440000,
    1124000727777607700000,
    25852016738884980000000,
    620448401733239400000000, 
];
const $adf7c9e79b2099a2$var$binomials = $f3Ts0$pascalstriangle(12);
const $adf7c9e79b2099a2$var$fct = (i)=>$adf7c9e79b2099a2$var$factorials[i]
;
const $adf7c9e79b2099a2$var$binomial = (n, k)=>$adf7c9e79b2099a2$var$binomials[n][k]
;
const $adf7c9e79b2099a2$export$ef35774e6d314e91 = (d, i, t)=>$adf7c9e79b2099a2$var$binomial(d - 1, i) * Math.pow(1 - t, d - 1 - i) * Math.pow(t, i)
;
const $adf7c9e79b2099a2$export$38837147ce0b547f = (d, i, j, u, v)=>$adf7c9e79b2099a2$var$fct(d - 1) / $adf7c9e79b2099a2$var$fct(d - 1 - i - j) / $adf7c9e79b2099a2$var$fct(i) / $adf7c9e79b2099a2$var$fct(j) * Math.pow(1 - u - v, d - 1 - i - j) * Math.pow(u, i) * Math.pow(v, j)
;
const $adf7c9e79b2099a2$export$37000325bae273ca = (d1, d2, i, j, t1, t2)=>$adf7c9e79b2099a2$export$ef35774e6d314e91(d1, i, t1) * $adf7c9e79b2099a2$export$ef35774e6d314e91(d2, j, t2)
;




const $893fa29f3010cfec$export$ede4745581eac3cc = $f3Ts0$ganjajs(4, 1);
const $893fa29f3010cfec$var$Alg = $893fa29f3010cfec$export$ede4745581eac3cc;
const $893fa29f3010cfec$export$7f9972325ebfd559 = new Array($893fa29f3010cfec$export$ede4745581eac3cc.length).fill(0);
const $893fa29f3010cfec$export$bee557efb6b6e = $893fa29f3010cfec$var$Alg.Vector(0, 0, 0, 1);
const $893fa29f3010cfec$export$3e0e4217c1341b05 = $893fa29f3010cfec$var$Alg.Vector(0, 0, 0, 0, 1);
const $893fa29f3010cfec$export$8deb4014ab0eb832 = $893fa29f3010cfec$var$Alg.Vector(0, 0, 0, 1, 1);
const $893fa29f3010cfec$export$401451a107dc42ce = $893fa29f3010cfec$var$Alg.Vector(0, 0, 0, -0.5, 0.5);
const $893fa29f3010cfec$export$4fcb62fc1f288b85 = (x, y, z)=>$893fa29f3010cfec$var$Alg.Mul($893fa29f3010cfec$var$Alg.Mul(x, y), z)
;
const $893fa29f3010cfec$export$e1f445cd6eeea85e = (x1)=>$f3Ts0$reduce([
        $893fa29f3010cfec$export$401451a107dc42ce,
        x1,
        $893fa29f3010cfec$var$Alg.Mul($893fa29f3010cfec$export$4fcb62fc1f288b85(x1, x1, $893fa29f3010cfec$export$8deb4014ab0eb832), $893fa29f3010cfec$var$Alg.Scalar(0.5))
    ], (x, y)=>$893fa29f3010cfec$var$Alg.Add(x, y)
    , $893fa29f3010cfec$var$Alg.Scalar(0))
;
const $893fa29f3010cfec$export$135366c179e0830f = (point, weight)=>$893fa29f3010cfec$var$Alg.Add($893fa29f3010cfec$var$Alg.Mul($893fa29f3010cfec$export$4fcb62fc1f288b85(point, $893fa29f3010cfec$export$8deb4014ab0eb832, weight), $893fa29f3010cfec$var$Alg.Scalar(0.5)), weight) // Create a Clifford Algebra with 4,1 metric for 3D CGA.
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
;


const $a8e101027d325e52$var$product = (...a1)=>a1.reduce((a, b)=>a.flatMap((d)=>b.map((e)=>[
                    d,
                    e
                ].flat()
            )
        )
    )
;
const $a8e101027d325e52$export$d3acdbd672dd1635 = (uResolution, vResolution)=>$a8e101027d325e52$var$product($f3Ts0$range(0, 1, 1 / uResolution), $f3Ts0$range(0, 1, 1 / vResolution))
;
const $a8e101027d325e52$export$e5a7308a1ffbe909 = $893fa29f3010cfec$export$ede4745581eac3cc;
const $a8e101027d325e52$export$6ec3a15068727e5f = // todo: (Alg: any = CGA) =>
(pointsVectorsUV, weightsUV)=>{
    const deg_u = pointsVectorsUV.length;
    const deg_v = pointsVectorsUV[0].length;
    const patchIndices = $a8e101027d325e52$var$product($f3Ts0$range(deg_u), $f3Ts0$range(deg_v));
    const points = patchIndices.map(([i, j])=>$893fa29f3010cfec$export$e1f445cd6eeea85e(pointsVectorsUV[i][j])
    );
    const weights = patchIndices.map(([i, j])=>$893fa29f3010cfec$export$135366c179e0830f(pointsVectorsUV[i][j], weightsUV[i][j])
    );
    const PW = $f3Ts0$memoize((k)=>$a8e101027d325e52$export$e5a7308a1ffbe909.Mul(points[k], weights[k])
    );
    return (u, v)=>{
        const top = $f3Ts0$reduce(patchIndices.map(([i, j], k)=>$a8e101027d325e52$export$e5a7308a1ffbe909.Mul(PW(k), $a8e101027d325e52$export$e5a7308a1ffbe909.Scalar($adf7c9e79b2099a2$export$37000325bae273ca(deg_u, deg_v, i, j, u, v)))
        ), (x, y)=>$a8e101027d325e52$export$e5a7308a1ffbe909.Add(x, y)
        , $a8e101027d325e52$export$e5a7308a1ffbe909.Scalar(0));
        const bottom = $f3Ts0$reduce(patchIndices.map(([i, j], k)=>$a8e101027d325e52$export$e5a7308a1ffbe909.Mul(weights[k], $a8e101027d325e52$export$e5a7308a1ffbe909.Scalar($adf7c9e79b2099a2$export$37000325bae273ca(deg_u, deg_v, i, j, u, v)))
        ), (x, y)=>$a8e101027d325e52$export$e5a7308a1ffbe909.Add(x, y)
        , $a8e101027d325e52$export$e5a7308a1ffbe909.Scalar(0));
        return $a8e101027d325e52$export$e5a7308a1ffbe909.Div(top, bottom);
    };
};


export {$a8e101027d325e52$export$d3acdbd672dd1635 as uvGrid, $a8e101027d325e52$export$e5a7308a1ffbe909 as Alg, $a8e101027d325e52$export$6ec3a15068727e5f as fquad};
//# sourceMappingURL=index.mjs.map
