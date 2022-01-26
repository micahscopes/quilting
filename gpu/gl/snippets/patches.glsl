#pragma glslify: import('./bernstein.glsl')

// CGA3 pointWeight(vec3 controlPoint, vec3 weight) {
//   return mul(
//     add(
//       mul(
//         fromVec(controlPoint),
//         mul( 0.5, INF())
//       ),
//       one()
//     ),
//     fromVec(weight)
//   );
// }

CGA3 inverse(CGA3 x) {
  return mul(1.0/mul(x, reverse(x)).scalar, reverse(x));
}

CGA3 div(CGA3 a, CGA3 b) {
  return mul(a, inverse(b));
}

CGA3 weight(vec3 p1, vec3 p2, vec3 w) {
  // return inverse(fromVec(p1-p2+w));
  return inverse(mul(fromVec(p1-p2), fromVec(w)));
}


CGA3 bilinearQuad(
  vec3 p0, vec3 p1, vec3 p2, vec3 p3,
  vec3 w01, vec3 w12, vec3 w23, vec3 w30,
  float u, float v) {
    CGA3 W01 = mul(bernsteinQuad(0,0,u,v), weight(p0, p1, w01));
    CGA3 W12 = mul(bernsteinQuad(1,0,u,v), weight(p1, p2, w12));
    CGA3 W23 = mul(bernsteinQuad(1,1,u,v), weight(p2, p3, w23));
    CGA3 W30 = mul(bernsteinQuad(0,1,u,v), weight(p3, p0, w30));
    CGA3 top = add(
      mul(fromVec(p0), W01),
      mul(fromVec(p1), W12),
      mul(fromVec(p2), W23),
      mul(fromVec(p3), W30)
    );
    CGA3 bottom = add(
      W01,
      W12,
      W23,
      W30
    );
    return div(top, bottom);
}