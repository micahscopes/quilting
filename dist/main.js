var $1ZQrD$lodashes = require("lodash-es");
var $1ZQrD$pascalstriangle = require("pascals-triangle");
var $1ZQrD$ganjajs = require("ganja.js");

function $parcel$export(e, n, v, s) {
  Object.defineProperty(e, n, {get: v, set: s, enumerable: true, configurable: true});
}
function $parcel$interopDefault(a) {
  return a && a.__esModule ? a.default : a;
}

$parcel$export(module.exports, "Alg", () => $5e2f01247a5d6f10$export$e5a7308a1ffbe909);
$parcel$export(module.exports, "barePatchIndices", () => $5e2f01247a5d6f10$export$9259594fd24395c0);
$parcel$export(module.exports, "fquad", () => $5e2f01247a5d6f10$export$6ec3a15068727e5f);


const $708e50960befe097$var$factorials = [
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
const $708e50960befe097$var$binomials = ($parcel$interopDefault($1ZQrD$pascalstriangle))(12);
const $708e50960befe097$var$fct = (i)=>$708e50960befe097$var$factorials[i]
;
const $708e50960befe097$var$binomial = (n, k)=>$708e50960befe097$var$binomials[n][k]
;
const $708e50960befe097$export$ef35774e6d314e91 = (d, i, t)=>$708e50960befe097$var$binomial(d - 1, i) * Math.pow(1 - t, d - 1 - i) * Math.pow(t, i)
;
const $708e50960befe097$export$38837147ce0b547f = (d, i, j, u, v)=>$708e50960befe097$var$fct(d - 1) / $708e50960befe097$var$fct(d - 1 - i - j) / $708e50960befe097$var$fct(i) / $708e50960befe097$var$fct(j) * Math.pow(1 - u - v, d - 1 - i - j) * Math.pow(u, i) * Math.pow(v, j)
;
const $708e50960befe097$export$37000325bae273ca = (d1, d2, i, j, t1, t2)=>$708e50960befe097$export$ef35774e6d314e91(d1, i, t1) * $708e50960befe097$export$ef35774e6d314e91(d2, j, t2)
;




const $0e40d3b2684983da$export$ede4745581eac3cc = ($parcel$interopDefault($1ZQrD$ganjajs))(4, 1);
const $0e40d3b2684983da$var$Alg = $0e40d3b2684983da$export$ede4745581eac3cc;
const $0e40d3b2684983da$export$7f9972325ebfd559 = new Array($0e40d3b2684983da$export$ede4745581eac3cc.length).fill(0);
const $0e40d3b2684983da$export$bee557efb6b6e = $0e40d3b2684983da$var$Alg.Vector(0, 0, 0, 1);
const $0e40d3b2684983da$export$3e0e4217c1341b05 = $0e40d3b2684983da$var$Alg.Vector(0, 0, 0, 0, 1);
const $0e40d3b2684983da$export$8deb4014ab0eb832 = $0e40d3b2684983da$var$Alg.Vector(0, 0, 0, 1, 1);
const $0e40d3b2684983da$export$401451a107dc42ce = $0e40d3b2684983da$var$Alg.Vector(0, 0, 0, -0.5, 0.5);
const $0e40d3b2684983da$export$4fcb62fc1f288b85 = (x, y, z)=>$0e40d3b2684983da$var$Alg.Mul($0e40d3b2684983da$var$Alg.Mul(x, y), z)
;
const $0e40d3b2684983da$export$e1f445cd6eeea85e = (x1)=>$1ZQrD$lodashes.reduce([
        $0e40d3b2684983da$export$401451a107dc42ce,
        x1,
        $0e40d3b2684983da$var$Alg.Mul($0e40d3b2684983da$export$4fcb62fc1f288b85(x1, x1, $0e40d3b2684983da$export$8deb4014ab0eb832), $0e40d3b2684983da$var$Alg.Scalar(0.5))
    ], (x, y)=>$0e40d3b2684983da$var$Alg.Add(x, y)
    , $0e40d3b2684983da$var$Alg.Scalar(0))
;
const $0e40d3b2684983da$export$135366c179e0830f = (point, weight)=>$0e40d3b2684983da$var$Alg.Add($0e40d3b2684983da$var$Alg.Mul($0e40d3b2684983da$export$4fcb62fc1f288b85(point, $0e40d3b2684983da$export$8deb4014ab0eb832, weight), $0e40d3b2684983da$var$Alg.Scalar(0.5)), weight) // Create a Clifford Algebra with 4,1 metric for 3D CGA.
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


const $5e2f01247a5d6f10$var$product = (...a1)=>a1.reduce((a, b)=>a.flatMap((d)=>b.map((e)=>[
                    d,
                    e
                ].flat()
            )
        )
    )
;
const $5e2f01247a5d6f10$export$e5a7308a1ffbe909 = $0e40d3b2684983da$export$ede4745581eac3cc;
const $5e2f01247a5d6f10$export$9259594fd24395c0 = (uResolution, vResolution = -1)=>{
    vResolution = vResolution < 0 ? uResolution : vResolution;
    const U = $1ZQrD$lodashes.range(0, 1, 1 / uResolution); //.map((u, i) => ({i, u}))
    const V = $1ZQrD$lodashes.range(0, 1, 1 / vResolution); //.map((v, j) => ({j, v}))
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
    return {
        UV: UV,
        U: U,
        V: V
    };
};
const $5e2f01247a5d6f10$export$6ec3a15068727e5f = // todo: (Alg: any = CGA) =>
(deg_u, deg_v, pointsVectorsUV, weightsUV)=>{
    const patchIndices = $5e2f01247a5d6f10$var$product($1ZQrD$lodashes.range(deg_u), $1ZQrD$lodashes.range(deg_v));
    const points = patchIndices.map(([i, j])=>$0e40d3b2684983da$export$e1f445cd6eeea85e(pointsVectorsUV[i][j])
    );
    const weights = patchIndices.map(([i, j])=>$0e40d3b2684983da$export$135366c179e0830f(pointsVectorsUV[i][j], weightsUV[i][j])
    );
    const PW = $1ZQrD$lodashes.memoize((k)=>$5e2f01247a5d6f10$export$e5a7308a1ffbe909.Mul(points[k], weights[k])
    );
    return (u, v)=>{
        const top = $1ZQrD$lodashes.reduce(patchIndices.map(([i, j], k)=>$5e2f01247a5d6f10$export$e5a7308a1ffbe909.Mul(PW(k), $5e2f01247a5d6f10$export$e5a7308a1ffbe909.Scalar($708e50960befe097$export$37000325bae273ca(deg_u, deg_v, i, j, u, v)))
        ), (x, y)=>$5e2f01247a5d6f10$export$e5a7308a1ffbe909.Add(x, y)
        , $5e2f01247a5d6f10$export$e5a7308a1ffbe909.Scalar(0));
        const bottom = $1ZQrD$lodashes.reduce(patchIndices.map(([i, j], k)=>$5e2f01247a5d6f10$export$e5a7308a1ffbe909.Mul(weights[k], $5e2f01247a5d6f10$export$e5a7308a1ffbe909.Scalar($708e50960befe097$export$37000325bae273ca(deg_u, deg_v, i, j, u, v)))
        ), (x, y)=>$5e2f01247a5d6f10$export$e5a7308a1ffbe909.Add(x, y)
        , $5e2f01247a5d6f10$export$e5a7308a1ffbe909.Scalar(0));
        return $5e2f01247a5d6f10$export$e5a7308a1ffbe909.Div(top, bottom);
    };
};


//# sourceMappingURL=main.js.map
