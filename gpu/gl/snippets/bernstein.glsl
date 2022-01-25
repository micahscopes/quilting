// copied from https://www.shadertoy.com/view/ttsGDM

#define DEGREE 1

// Kronecker delta
#define kd(n, k) ((n == k) ? 1.0 : 0.0)

// Horner-type algorithm for evaluating a Bernstein polynomial, from Farin (http://www.farinhansford.com/books/cagd/materials/allfiles.txt)
float bernstein(int n, float x) {
  int bc = 1;
  float p = 1.0, y = 0.0;
  float x1 = 1.0 - x;

  for(int k = 0; k < DEGREE; k++) {
    y = (y + p * float(bc) * kd(n, k)) * x1;
    p *= x;
    bc = (bc * (DEGREE - k)) / (k + 1);
  }

  return y + p * kd(n, DEGREE);
}

// interpolation weight for a rectangular patch of degrees d_u, d_v
// for indices i, j evaluated at coordinates u, v
float bernsteinQuad(int i, int j, float u, float v) {
  return bernstein(i, u) * bernstein(j, v);
}

// // interpolation weight for a triangular patch of degree d for indices i,j
// // evaluated at barycentric coordinates u,v
// float bernsteinTriangle(int d, int i, int j, int u, int v) {
//   (factorialOf(d) / factorialOf(d - i - j) / factorialOf(i) / factorialOf(j)) *
//   pow(1 - u - v, d - 1 - i - j) *
//   pow(u, i) *
//   pow(v, j);
// }
