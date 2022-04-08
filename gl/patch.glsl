// for some reason `require` only works at top level
#pragma glslify: import('./lib.glsl')

H fromVec(vec4 w) {
  return vecToH(w);
}

H fromVec(vec3 w) {
  return vecToH(w);
}

H weight(vec3 w) {
  return fromVec(w);
  // return invert(fromVec(normalize(w)));
}

H weight(vec4 w) {
  return fromVec(w);
  // return invert(fromVec(normalize(w)));
}

H weight(H w) {
  return w;
  // return invert(mul(sqrt(lcontract(w, w).real), w));
}

struct Patch {
  vec3 vertex;
  vec3 normal;
};

float eps = 0.01;

Patch bilinearQuad(H point0, H point1, H point2, H point3, H weight0, H weight1, H weight2, H weight3, float u, float v) {
  H W0 = mul((1.0 - u) * (1.0 - v), weight0); // 0, 0
  H W1 = mul(u * (1.0 - v), weight1); // 1, 0
  H W2 = mul((1.0 - u) * v, weight2); // 0, 1
  H W3 = mul(u * v, weight3); // 1, 1

  H top = add(mul(point0, W0), mul(point1, W1), mul(point2, W2), mul(point3, W3));

  float uu = u + eps;
  H W0_uu = mul((1.0 - uu) * (1.0 - v), weight0); // 0, 0
  H W1_uu = mul((uu) * (1.0 - v), weight1); // 1, 0
  H W2_uu = mul((1.0 - uu) * v, weight2); // 0, 1
  H W3_uu = mul((uu) * v, weight3); // 1, 1
  H top_uu = add(mul(point0, W0_uu), mul(point1, W1_uu), mul(point2, W2_uu), mul(point3, W3_uu));

  float vv = v + eps;
  H W0_vv = mul((1.0 - u) * (1.0 - vv), weight0); // 0, 0
  H W1_vv = mul(u * (1.0 - vv), weight1); // 1, 0
  H W2_vv = mul((1.0 - u) * (vv), weight2); // 0, 1
  H W3_vv = mul(u * (vv), weight3); // 1, 1

  H top_vv = add(mul(point0, W0_vv), mul(point1, W1_vv), mul(point2, W2_vv), mul(point3, W3_vv));

  H bottom = add(W0, W1, W2, W3);

  H X = div(top, bottom);
  H X_uu = div(top_uu, bottom);
  H X_vv = div(top_vv, bottom);
  vec3 x = vec3(X.i, X.j, X.k);
  vec3 x_uu = vec3(X_uu.i, X_uu.j, X_uu.k) - x;
  vec3 x_vv = vec3(X_vv.i, X_vv.j, X_vv.k) - x;
  vec3 crossed = cross(x_uu, x_vv);
  vec3 normal = normalize(crossed);
  return Patch(x, normal);
}

Patch bilinearQuad(vec3 p0, vec3 p1, vec3 p2, vec3 p3, vec4 w0, vec4 w1, vec4 w2, vec4 w3, float u, float v) {
  return bilinearQuad(fromVec(p0), fromVec(p1), fromVec(p2), fromVec(p3), weight(w0), weight(w1), weight(w2), weight(w3), u, v);
}

Patch bilinearQuad(vec3 p0, vec3 p1, vec3 p2, vec3 p3, vec3 w0, vec3 w1, vec3 w2, vec3 w3, float u, float v) {
  return bilinearQuad(fromVec(p0), fromVec(p1), fromVec(p2), fromVec(p3), weight(w0), weight(w1), weight(w2), weight(w3), u, v);
}

Patch bilinearQuad(vec3 p0, vec3 p1, vec3 p2, vec3 p3, H w0, H w1, H w2, H w3, float u, float v) {
  return bilinearQuad(fromVec(p0), fromVec(p1), fromVec(p2), fromVec(p3), weight(w0), weight(w1), weight(w2), weight(w3), u, v);
}

Patch bilinearTri(H point0, H point1, H point2, H weight0, H weight1, H weight2, float u, float v) {
  H W0 = mul((1.0 - u - v), weight0); // 0, 0
  H W1 = mul(u, weight1); // 1, 0
  H W2 = mul(v, weight2); // 0, 1

  H top = add(mul(point0, W0), mul(point1, W1), mul(point2, W2));

  float uu = u + eps;
  H W0_uu = mul((1.0 - uu - v), weight0); // 0, 0
  H W1_uu = mul(uu, weight1); // 1, 0
  H W2_uu = mul(v, weight2); // 0, 1
  H top_uu = add(mul(point0, W0_uu), mul(point1, W1_uu), mul(point2, W2_uu));

  float vv = v + eps;
  H W0_vv = mul((1.0 - u - vv), weight0); // 0, 0
  H W1_vv = mul(u, weight1); // 1, 0
  H W2_vv = mul(vv, weight2); // 0, 1

  H top_vv = add(mul(point0, W0_vv), mul(point1, W1_vv), mul(point2, W2_vv));

  H bottom = add(W0, W1, W2);

  H X = div(top, bottom);
    // H X_vv= div(top_vv, bottom);
  H X_uu = div(top_uu, bottom);
  H X_vv = div(top_vv, bottom);
  vec3 x = vec3(X.i, X.j, X.k);
  vec3 x_uu = vec3(X_uu.i, X_uu.j, X_uu.k) - x;
  vec3 x_vv = vec3(X_vv.i, X_vv.j, X_vv.k) - x;
  vec3 crossed = cross(x_uu, x_vv);
  vec3 normal = normalize(crossed);
  return Patch(x, normal);
}

Patch bilinearTri(vec4 p0, vec4 p1, vec4 p2, vec4 w0, vec4 w1, vec4 w2, float u, float v) {
  return bilinearTri(fromVec(p0), fromVec(p1), fromVec(p2), weight(w0), weight(w1), weight(w2), u, v);
}

Patch bilinearTri(vec3 p0, vec3 p1, vec3 p2, vec3 w0, vec3 w1, vec3 w2, float u, float v) {
  return bilinearTri(fromVec(p0), fromVec(p1), fromVec(p2), weight(w0), weight(w1), weight(w2), u, v);
}

Patch bilinearTri(vec3 p0, vec3 p1, vec3 p2, H w0, H w1, H w2, float u, float v) {
  return bilinearTri(fromVec(p0), fromVec(p1), fromVec(p2), weight(w0), weight(w1), weight(w2), u, v);
}