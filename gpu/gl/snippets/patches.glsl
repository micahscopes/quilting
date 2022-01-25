#pragma glslify: import('./bernstein.glsl')

CGA3 pointWeight(CGA3 point, CGA3 weight) {
  return add(mul(0.5, mul(point, INF(), weight)), weight);
}

CGA3 bilinearQuad(
  CGA3 p00, CGA3 p01, CGA3 p10, CGA3 p11,
  CGA3 w00, CGA3 w01, CGA3 w10, CGA3 w11,
  float u, float v) {
    // w00 = pointWeight(p00, w00);
    // w01 = pointWeight(p01, w01);
    // w10 = pointWeight(p10, w10);
    // w11 = pointWeight(p11, w11);
    p00 = point(p00); p01 = point(p01); p10 = point(p10); p11 = point(p11);
    CGA3 top = add(
      mul(bernsteinQuad(0,0,u,v), pointWeight(p00, w00)),
      mul(bernsteinQuad(0,1,u,v), pointWeight(p01, w01)),
      mul(bernsteinQuad(1,0,u,v), pointWeight(p10, w10)),
      mul(bernsteinQuad(1,1,u,v), pointWeight(p11, w11))
    );
    CGA3 bottom = add(
      mul(bernsteinQuad(0,0,u,v), w00),
      mul(bernsteinQuad(0,1,u,v), w01),
      mul(bernsteinQuad(1,0,u,v), w10),
      mul(bernsteinQuad(1,1,u,v), w11)
    );
    // return top;
    return mul(1.0/reverse(lcontract(bottom, bottom)).scalar, reverse(top));
}