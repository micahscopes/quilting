#version 300 es
#pragma glslify: import('./lib.glsl')

layout(location = 0) in mat3x4 points;
layout(location = 3) in mat3x4 weights;

uniform CGA3 transformation;
uniform float angle;

out mat3x4 transformedPoints;
out mat3x4 transformedWeights;
out float lod;

#define ni CGA3(0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
#define no CGA3(0.0,0.0,0.0,0.0,-0.5,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
#define E outer(no, ni) // could optimize

CGA3 upPoint(vec4 p) {
  CGA3 X = vecToCGA(p);
  return add(X, add(mul(0.5, mul(X, X, ni)), no));
}

CGA3 downPoint(CGA3 P) {
  CGA3 p = mul(div(outer(mul(1.0, invert(lcontract(P,ni))), P), E), E);
  return p;
  // return vec4(X.scalar, X.e1, X.e2, X.e3);
}

// CGA3 sandwichPoint(CGA3 T, CGA3 P, CGA3 hatT) {
//   return downPoint(mul(T, upPoint(p), hatT));
// }

CGA3 upWeight(CGA3 w, vec4 pVec) {
  CGA3 p = vecToCGA(pVec);
  return mul(add(mul(0.5, mul(p, ni)), ONE_CGA3), w);
}

CGA3 upWeight(vec4 w, vec4 pVec) {
  return upWeight(weightVecToCGA(w), pVec);
}

CGA3 downWeight(CGA3 W, vec4 pTransformed) {
  CGA3 transformedP = vecToCGA(pTransformed);
  CGA3 w = mul(invert(add(mul(mul(0.5,transformedP),ni), mul(-1.0, ONE_CGA3))), W);
  return w;
}

CGA3 sandwichWeight(CGA3 T, CGA3 w, CGA3 hatT, vec4 p, vec4 pTransformed) {
  return downWeight(mul(T, upWeight(w, p), hatT), pTransformed);
}

CGA3 locate(CGA3 a, CGA3 M) {
  //  locate( a, M ) = (a⋅M)/M
  return div(inner(a, M), M);
}

CGA3 circumcenter(CGA3 a, CGA3 b, CGA3 c) {
  //  circumcentre( a, b, c) = locate(a, (a∧b + b∧c + c∧a)/(a∧b∧c∧e∞))
  return locate(a, div(add(add(outer(a,b), outer(b,c)), outer(c,a)), outer(a, outer(b, outer(c, ni)))));
}

CGA3 centroid_(CGA3 a, CGA3 b, CGA3 c) {
  return mul(1.0/3.0, add(a, add(b,c)));
}

void main() {
  CGA3 T = transformation;
  CGA3 hatT = conjugate(T);
  // CGA3 hatT = conjugate(div(T, lcontract(T, T)));

  CGA3 p1 = vecToCGA(points[0]);
  CGA3 p2 = vecToCGA(points[1]);
  CGA3 p3 = vecToCGA(points[2]);
  CGA3 P1 = upPoint(points[0]);
  CGA3 P2 = upPoint(points[1]);
  CGA3 P3 = upPoint(points[2]);

  CGA3 w1 = invert(weightVecToCGA(vec4(1.0, 0,0,0)));
  CGA3 w2 = invert(weightVecToCGA(vec4(1.0, 0,0,0)));
  CGA3 w3 = invert(weightVecToCGA(vec4(1.0, 0,0,0)));

  // CGA3 w1 = weightVecToCGA(vec4(0.333333333333, 0,0,0));
  // CGA3 w2 = weightVecToCGA(vec4(0.333333333333, 0,0,0));
  // CGA3 w3 = weightVecToCGA(vec4(0.333333333333, 0,0,0));

////////////////////////////////

  // CGA3 w1 = weightVecToCGA(vec4(1,0,0,0));
  // CGA3 w2 = invert(weightVecToCGA((points[1] - points[2])/2.0 - points[1]));
  // CGA3 w1 = invert(weightVecToCGA((points[1] - points[2])/2.0 - points[1]));
  // CGA3 w2 = invert(weightVecToCGA((points[0] - points[2])/2.0 - points[0]));
  // CGA3 w3 = invert(weightVecToCGA((points[1] - points[0])/2.0 - points[0]));
  // w1 = mul(weightVecToCGA((points[1] - points[2])/2.0 - points[1]), w1);
  // w2 = mul(weightVecToCGA((points[0] - points[2])/2.0 - points[0]), w2);
  // w3 = mul(weightVecToCGA((points[0] - points[1])/2.0 - points[1]), w3);
  // w1 = weightVecToCGA(vec4(1,0,0,0));
  // w2 = div(w2, w1);
  // w3 = div(w3, w1);

  // w2 = weightVecToCGA(vec4(-1,0,0,0));
  // w3 = weightVecToCGA(vec4(-1,0,0,0));

  //////////////////////////////
  // CGA3 wp1 = weightVecToCGA(points[0]);
  // CGA3 wp2 = weightVecToCGA(points[1]);
  // CGA3 wp3 = weightVecToCGA(points[2]);
  // CGA3 center = centroid_(wp1, wp2, wp3);
  // center = wp1;
  // CGA3 inv_center = mul(0.1, invert(center));

  // center = add(center, weightVecToCGA(vec4(0.3)));
  // center = mul(1.0, center);

  // CGA3 w1 = mul(1.0, invert(sub(wp1, inv_center)));
  // CGA3 w2 = mul(1.0, invert(sub(wp2, inv_center)));
  // CGA3 w3 = mul(1.0, invert(sub(wp3, inv_center)));

////////////////////////////

  // H w1H = vecToH(points[0] + normalize(points[1] - points[0]));
  // H w2H = vecToH(points[0] + normalize(points[2] - points[0]));
  // H w3H = vecToH(points[2] + normalize(points[0] - points[2]));

  // w2H = div(w2H, w1H);
  // w3H = div(w3H, w1H);
  // w1H = ONE_H;

  // CGA3 w1 = vecToCGA(vec4(w1H.real, w1H.i, w1H.j, w1H.k));
  // CGA3 w2 = vecToCGA(vec4(w2H.real, w2H.i, w2H.j, w2H.k));
  // CGA3 w3 = vecToCGA(vec4(w3H.real, w3H.i, w3H.j, w3H.k));

////////////////////////

  // CGA3 center = circumcenter(P1, P2, P3);
  // CGA3 inv_center = invert(downPoint(center));

  ////////////////////////////////
  // find weights with quaternions
  // H center = ZERO_H;
  // H inv_center = invert(center);

  // H p1H = vecToH(points[0]);
  // H p2H = vecToH(points[1]);
  // H p3H = vecToH(points[1]);

  // H w1H = mul(1.0, invert(sub(p1H, inv_center)));
  // H w2H = mul(1.0, invert(sub(p2H, inv_center)));
  // H w3H = mul(1.0, invert(sub(p3H, inv_center)));

  // CGA3 w1 = vecToCGA(vec4(w1H.real, w1H.i, w1H.j, w1H.k));
  // CGA3 w2 = vecToCGA(vec4(w2H.real, w2H.i, w2H.j, w2H.k));
  // CGA3 w3 = vecToCGA(vec4(w3H.real, w3H.i, w3H.j, w3H.k));
  ////////////////////////////////////////

  CGA3 p1_ = downPoint(mul(T, P1, hatT));
  CGA3 p2_ = downPoint(mul(T, P2, hatT));
  CGA3 p3_ = downPoint(mul(T, P3, hatT));

  transformedPoints[0] = vec4(p1_.scalar, p1_.e1, p1_.e2, p1_.e3);
  transformedPoints[1] = vec4(p2_.scalar, p2_.e1, p2_.e2, p2_.e3);
  transformedPoints[2] = vec4(p3_.scalar, p3_.e1, p3_.e2, p3_.e3);

  CGA3 w1_ = sandwichWeight(T, w1, hatT, points[0], transformedPoints[0]);
  CGA3 w2_ = sandwichWeight(T, w2, hatT, points[1], transformedPoints[1]);
  CGA3 w3_ = sandwichWeight(T, w3, hatT, points[2], transformedPoints[2]);

  transformedWeights[0] = vec4(w1_.scalar, -w1_.e23, w1_.e13, -w1_.e12);
  transformedWeights[1] = vec4(w2_.scalar, -w2_.e23, w2_.e13, -w2_.e12);
  transformedWeights[2] = vec4(w3_.scalar, -w3_.e23, w3_.e13, -w3_.e12);
  // transformedWeights[0] = vec4(w1_.scalar, w1_.e1, w1.e2, w1_.e3);
  // transformedWeights[1] = vec4(w2_.scalar, w2_.e1, w2.e2, w2_.e3);
  // transformedWeights[2] = vec4(w3_.scalar, w3_.e1, w3.e2, w3_.e3);
  lod = transformation.e1;

// override
  // transformedPoints[0] = points[0];
  // transformedPoints[1] = points[1];
  // transformedPoints[2] = points[2];
  // transformedWeights[0] = vec4(w1.scalar, -w1.e23, w1.e13, -w1.e12);
  // transformedWeights[1] = vec4(w2.scalar, -w2.e23, w2.e13, -w2.e12);
  // transformedWeights[2] = vec4(w3.scalar, -w3.e23, w3.e13, -w3.e12);
}