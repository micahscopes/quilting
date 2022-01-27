#pragma glslify: import('./cga3/index.glsl')


float gbellmf(float x, float a, float b, float c) {
    return 1.0/(1.0 + pow(
    abs((x - c)/a), 2.0 * b
    ));
}

CGA3 outer(CGA3 p, CGA3 q, CGA3 r, CGA3 s){
    return outer(outer(p,q,r),s);
}

CGA3 fromVec(vec3 x){
    float y[3];
    y[0] = x[0];
    y[1] = x[1];
    y[2] = x[2];
    return injectOneBlade(zero(), y);
}

vec3 toVec(CGA3 x) {
    return vec3(x.e1, x.e2, x.e3);
}

CGA3 fromVecInf(vec4 v){
    CGA3 r = zero();
    r.e1 = v.x;
    r.e2 = v.y;
    r.e3 = v.z;
    r.enil = v.w;
    return r;
}

vec4 toVecInf(CGA3 mv){
    return vec4(mv.e1, mv.e2, mv.e3, mv.einf);
}

CGA3 point(vec3 x){
    return point(fromVec(x));
}

vec3 vecFromPoint(CGA3 x) {
    x = point_coords(x);
    return vec3(x.e1, x.e2, x.e3);
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

#pragma glslify: import('./snippets/patches.glsl')