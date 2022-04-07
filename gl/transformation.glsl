#version 300 es
#pragma glslify: import('./lib.glsl')

layout(location = 0) in mat3 points;
layout(location = 3) in mat3x4 weights;

uniform CGA3 transformation;
uniform float angle;

out mat3 transformedPoints;
out mat3x4 transformedWeights;
out float lod;

CGA3 upPoint(vec3 p) {
  CGA3 ni = ZERO_CGA3;
  ni.einf = 1.0;
  CGA3 no = ZERO_CGA3;
  no.enil = 1.0;
  CGA3 X = vecToCGA(p);
  return add(X, add(mul(0.5, mul(X, X, ni)), no));
}

vec3 downPoint(CGA3 P) { return vec3(P.e1, P.e2, P.e3); }

vec3 sandwichPoint(CGA3 T, vec3 p, CGA3 hatT) {
  return downPoint(mul(T, upPoint(p), hatT));
}

CGA3 upWeight(vec4 w, vec3 p) {
  CGA3 ni = ZERO_CGA3;
  ni.einf = 1.0;
  CGA3 no = ZERO_CGA3;
  no.enil = 1.0;
  CGA3 W = vecToCGA(w);
  CGA3 P = vecToCGA(p);
  return mul(add(mul(0.5, mul(P, ni)), ONE_CGA3), W);
}

vec4 downWeight(CGA3 W) { return vec4(W.scalar, W.e1, W.e2, W.e3); }

vec4 sandwichWeight(CGA3 T, vec4 w, CGA3 hatT, vec3 p) {
  return downWeight(mul(T, upWeight(w, p), hatT));
}

void main() {
  CGA3 d = ZERO_CGA3;
  d.e1inf = 0.5;
  CGA3 T =  sub(
    mul(cos(angle / 2.0), ONE_CGA3),
    mul(sin(angle / 2.0), transformation)
  );
  // CGA3 T = ONE_CGA3;
  // T.e123nilinf = 1.0;
  // CGA3 hatT = ONE_CGA3;
  CGA3 hatT = conjugate(mul(T, invert(inner(T, T))));
  transformedPoints[0] = sandwichPoint(T, points[0], hatT);
  transformedPoints[1] = sandwichPoint(T, points[1], hatT);
  transformedPoints[2] = sandwichPoint(T, points[2], hatT);
  transformedWeights[0] = sandwichWeight(T, weights[0], hatT, points[0]);
  transformedWeights[1] = sandwichWeight(T, weights[1], hatT, points[1]);
  transformedWeights[2] = sandwichWeight(T, weights[2], hatT, points[2]);
  lod = transformation.e1;
}