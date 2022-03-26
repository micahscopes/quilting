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

// CGA3 inverse(CGA3 x) {
//   return mul(1.0/mul(x, reverse(x)).scalar, reverse(x));
// }

// CGA3 div(CGA3 a, CGA3 b) {
//   return mul(a, inverse(b));
// }
CGA3 weight(vec3 w) {
  // return one();
  return inverse(fromVec(normalize(w)));
  // return inverse(mul(fromVec(p1-p2), fromVec(w)));
}

struct Patch {
  vec3 vertex;
  vec3 normal;
};

float eps = 0.01;

Patch bilinearQuad(vec3 p0, vec3 p1, vec3 p2, vec3 p3, vec3 w0, vec3 w1, vec3 w2, vec3 w3, float u, float v) {
  CGA3 weight0 = weight(w0);
  CGA3 weight1 = weight(w1);
  CGA3 weight2 = weight(w2);
  CGA3 weight3 = weight(w3);
  CGA3 point0 = fromVec(p0);
  CGA3 point1 = fromVec(p1);
  CGA3 point2 = fromVec(p2);
  CGA3 point3 = fromVec(p3);

  CGA3 W0 = mul((1.0 - u) * (1.0 - v), weight0); // 0, 0
  CGA3 W1 = mul(u * (1.0 - v), weight1); // 1, 0
  CGA3 W2 = mul((1.0 - u) * v, weight2); // 0, 1
  CGA3 W3 = mul(u * v, weight3); // 1, 1

  CGA3 top = add(mul(point0, W0), mul(point1, W1), mul(point2, W2), mul(point3, W3));

  float uu = u + eps;
  CGA3 W0_uu = mul((1.0 - uu) * (1.0 - v), weight0); // 0, 0
  CGA3 W1_uu = mul((uu) * (1.0 - v), weight1); // 1, 0
  CGA3 W2_uu = mul((1.0 - uu) * v, weight2); // 0, 1
  CGA3 W3_uu = mul((uu) * v, weight3); // 1, 1
  CGA3 top_uu = add(mul(point0, W0_uu), mul(point1, W1_uu), mul(point2, W2_uu), mul(point3, W3_uu));

  float vv = v + eps;
  CGA3 W0_vv = mul((1.0 - u) * (1.0 - vv), weight0); // 0, 0
  CGA3 W1_vv = mul(u * (1.0 - vv), weight1); // 1, 0
  CGA3 W2_vv = mul((1.0 - u) * (vv), weight2); // 0, 1
  CGA3 W3_vv = mul(u * (vv), weight3); // 1, 1

  CGA3 top_vv = add(mul(point0, W0_vv), mul(point1, W1_vv), mul(point2, W2_vv), mul(point3, W3_vv));

  CGA3 bottom = add(W0, W1, W2, W3);

  CGA3 X = div(top, bottom);
    // CGA3 X_vv= div(top_vv, bottom);
  CGA3 X_uu = div(top_uu, bottom);
  CGA3 X_vv = div(top_vv, bottom);
  vec3 x = vec3(X.e1, X.e2, X.e3);
    // CGA3 Normal = outer(sub(X, X_uu), sub(X, X_vv));
    // vec3 normal_alt = normalize(vec3(Normal.e12, Normal.e13, Normal.e23));
  vec3 x_uu = vec3(X_uu.e1, X_uu.e2, X_uu.e3) - x;
  vec3 x_vv = vec3(X_vv.e1, X_vv.e2, X_vv.e3) - x;
  vec3 crossed = cross(x_uu, x_vv);
  vec3 normal = normalize(crossed);
  return Patch(x, normal);
}

Patch bilinearTri(vec3 p0, vec3 p1, vec3 p2, vec3 w0, vec3 w1, vec3 w2, float u, float v) {
  CGA3 weight0 = weight(w0);
  CGA3 weight1 = weight(w1);
  CGA3 weight2 = weight(w2);
  CGA3 point0 = fromVec(p0);
  CGA3 point1 = fromVec(p1);
  CGA3 point2 = fromVec(p2);

  CGA3 W0 = mul((1.0 - u - v), weight0); // 0, 0
  CGA3 W1 = mul(u, weight1); // 1, 0
  CGA3 W2 = mul(v, weight2); // 0, 1

  CGA3 top = add(mul(point0, W0), mul(point1, W1), mul(point2, W2));

  float uu = u + eps;
  CGA3 W0_uu = mul((1.0 - uu - v), weight0); // 0, 0
  CGA3 W1_uu = mul(uu, weight1); // 1, 0
  CGA3 W2_uu = mul(v, weight2); // 0, 1
  CGA3 top_uu = add(mul(point0, W0_uu), mul(point1, W1_uu), mul(point2, W2_uu));

  float vv = v + eps;
  CGA3 W0_vv = mul((1.0 - u - vv), weight0); // 0, 0
  CGA3 W1_vv = mul(u, weight1); // 1, 0
  CGA3 W2_vv = mul(vv, weight2); // 0, 1

  CGA3 top_vv = add(mul(point0, W0_vv), mul(point1, W1_vv), mul(point2, W2_vv));

  CGA3 bottom = add(W0, W1, W2);

  CGA3 X = div(top, bottom);
    // CGA3 X_vv= div(top_vv, bottom);
  CGA3 X_uu = div(top_uu, bottom);
  CGA3 X_vv = div(top_vv, bottom);
  vec3 x = vec3(X.e1, X.e2, X.e3);
  vec3 x_uu = vec3(X_uu.e1, X_uu.e2, X_uu.e3) - x;
  vec3 x_vv = vec3(X_vv.e1, X_vv.e2, X_vv.e3) - x;
  vec3 crossed = cross(x_uu, x_vv);
  vec3 normal = normalize(crossed);
  return Patch(x, normal);
}