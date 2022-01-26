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

CGA3 weight(vec3 w) {
  // return one();
  return inverse(fromVec(normalize(w)));
  // return inverse(mul(fromVec(p1-p2), fromVec(w)));
}


CGA3 bilinearQuad(
  vec3 p0, vec3 p1, vec3 p2, vec3 p3,
  vec3 w0, vec3 w1, vec3 w2, vec3 w3,
  float u, float v) {
    CGA3 W0 = mul((1.0-u)*(1.0-v),  weight(w0)); // 0, 0
    CGA3 W1 = mul(u*(1.0-v),        weight(w1)); // 1, 0
    CGA3 W2 = mul((1.0-u)*v,        weight(w2)); // 0, 1
    CGA3 W3 = mul(u*v,              weight(w3)); // 1, 1
    CGA3 top = add(
      mul(fromVec(p0), W0),
      mul(fromVec(p1), W1),
      mul(fromVec(p2), W2),
      mul(fromVec(p3), W3)
    );
    CGA3 bottom = add(
      W0,
      W1,
      W2,
      W3
    );
    return div(top, bottom);
    // vec3 q0 = p0 * (1.0 - u) + p1 * u;
    // vec3 q1 = p2 * (1.0 - u) + p3 * u;
    // vec3 x = q0 * (1.0 - v) + q1*v;

    // return fromVec(x);
}