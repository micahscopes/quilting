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

CGA3 weight(vec3 p, vec3 w) {
  return inverse(fromVec(w));
}


CGA3 bilinearQuad(
  vec3 p0, vec3 p1, vec3 p2, vec3 p3,
  vec3 w01, vec3 w02, vec3 w13, vec3 w23,
  float u, float v) {
    CGA3 W01 = weight(p0, w01);
    CGA3 W02 = weight(p0, w02);
    CGA3 W13 = weight(p3, w13);
    CGA3 W23 = weight(p3, w23);
    CGA3 top = add(
      mul(bernsteinQuad(0,0,u,v), mul(fromVec(p0), W02)),
      mul(bernsteinQuad(0,1,u,v), mul(fromVec(p1), W01)),
      mul(bernsteinQuad(1,0,u,v), mul(fromVec(p2), W13)),
      mul(bernsteinQuad(1,1,u,v), mul(fromVec(p3), W23))
    );
    CGA3 bottom = add(
      mul(bernsteinQuad(0,0,u,v), W02),
      mul(bernsteinQuad(0,1,u,v), W01),
      mul(bernsteinQuad(1,0,u,v), W13),
      mul(bernsteinQuad(1,1,u,v), W23)
    );
    return div(top, bottom);
}