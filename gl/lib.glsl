// common functions and utilities
#pragma glslify: import('./algebra/index.glsl')

CGA3 vecToCGA(vec3 v){
    CGA3 X = ZERO_CGA3;
    X.e1 = v.x;
    X.e2 = v.y;
    X.e3 = v.z;
    return X;
}

CGA3 vecToCGA(vec4 v){
    CGA3 X = ZERO_CGA3;
    X.scalar = v.x;
    X.e1 = v.y;
    X.e2 = v.z;
    X.e3 = v.w;
    return X;
}

void toVec(out vec3 x, CGA3 X) {
    x = vec3(X.e1, X.e2, X.e3);
}

H vecToH(vec3 v){
    H X = ZERO_H;
    X.real = 1.0;
    X.i = v.x;
    X.j = v.y;
    X.k = v.z;
    return X;
}

H vecToH(vec4 v){
    H X = ZERO_H;
    X.real = v.x;
    X.i = v.y;
    X.j = v.z;
    X.k = v.w;
    return X;
}

void toVec(H X, inout vec4 x) {
    x.x = X.real;
    x.y = X.i;
    x.z = X.j;
    x.w = X.k;
}

CGA3 point(vec3 x){
    return point(vecToCGA(x));
}

vec3 vecFromPoint(CGA3 x) {
    x = point_coords(x);
    return vec3(x.e1, x.e2, x.e3);
}

CGA3 outer(CGA3 p, CGA3 q, CGA3 r, CGA3 s){
    return outer(outer(p,q,r),s);
}

CGA3 mul(CGA3 p, CGA3 q, CGA3 r, CGA3 s){
    return mul(mul(p,q,r),s);
}

// replace this stuff with IPNS sphere stuff
CGA3 sphere(vec3 a, vec3 b, vec3 c, vec3 d) {
    return outer(point(a), point(b), point(c), point(d));
}

vec3 sphere_center(vec3 a, vec3 b, vec3 c, vec3 d){
    CGA3 sphere = sphere(a,b,c,d);
    return vecFromPoint(mul(sphere, INF(), sphere));
}

vec3 reflect_glsl(vec3 x, CGA3 R){
    return vecFromPoint(mul(R,point(x),R));
}

// float sphere_radius(vec3 a, vec3 b, vec3 c, vec3 d){
//     CGA2 circ = sphere(a,b,c,d);
//     return circle_radius(circ);
// }

// CGA3 dual_sphere(vec3 x){
//     float y[3];
//     y[0] = x[0];
//     y[1] = x[1];
//     y[2] = x[2];
//     return dual_sphere(injectOneBlade(zero(), y));
// }
