#version 300 es
#pragma glslify: import('./lib.glsl')

layout(location = 0) in mat3 points;
layout(location = 3) in mat3x4 weights;

uniform CGA3 transformation;
uniform float angle;

out mat3 transformedPoints;
out mat3x4 transformedWeights;
out float lod;

#define ni CGA3(0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
#define no CGA3(0.0,0.0,0.0,0.0,-0.5,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
#define E outer(no, ni) // could optimize

CGA3 upPoint(vec3 p) {
  CGA3 X = vecToCGA(p);
  return add(X, add(mul(0.5, mul(X, X, ni)), no));
}

vec3 downPoint(CGA3 P) {
  CGA3 X = div(outer(mul(1.0/inner(P,ni).scalar, P), E), E);
  return vec3(X.e1, X.e2, X.e3);
}

vec3 sandwichPoint(CGA3 T, vec3 p, CGA3 hatT) {
  return downPoint(mul(T, upPoint(p), hatT));
}

CGA3 upWeight(vec4 w, vec3 p) {
  CGA3 W = vecToCGA(w);
  CGA3 P = vecToCGA(p);
  return mul(add(mul(0.5, mul(P, ni)), ONE_CGA3), W);
}

vec4 downWeight(CGA3 W, vec3 p) {
  CGA3 P = vecToCGA(p);
  CGA3 X = div(W, add(mul(mul(0.5,P),ni), ONE_CGA3));
  return vec4(X.scalar, X.e1, X.e2, X.e3); }

vec4 sandwichWeight(CGA3 T, vec4 w, CGA3 hatT, vec3 p, vec3 pTransformed) {
  return downWeight(mul(T, upWeight(w, p), hatT), pTransformed);
}

void main() {
  // CGA3 d = ZERO_CGA3;
  // d.e1inf = 0.5;
  // CGA3 T = sub(mul(cos(angle / 2.0), ONE_CGA3),
  //              mul(sin(angle / 2.0), transformation));
  CGA3 T = transformation;
  // CGA3 T = ONE_CGA3;
  // T.e123nilinf = 1.0;
  // CGA3 hatT = ONE_CGA3;
  CGA3 hatT = conjugate(div(T, inner(T, T)));
  transformedPoints[0] = sandwichPoint(T, points[0], hatT);
  transformedPoints[1] = sandwichPoint(T, points[1], hatT);
  transformedPoints[2] = sandwichPoint(T, points[2], hatT);
  transformedWeights[0] = sandwichWeight(T, weights[0], hatT, points[0], transformedPoints[0]);
  transformedWeights[1] = sandwichWeight(T, weights[1], hatT, points[1], transformedPoints[1]);
  transformedWeights[2] = sandwichWeight(T, weights[2], hatT, points[2], transformedPoints[2]);
  lod = transformation.e1;

// override
  // transformedPoints[0] = points[0];
  // transformedPoints[1] = points[1];
  // transformedPoints[2] = points[2];
  // transformedWeights[0] = weights[0];
  // transformedWeights[1] = weights[1];
  // transformedWeights[2] = weights[2];
}