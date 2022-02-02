#pragma glslify: import('./cga3/index.glsl')

CGA3 outer(CGA3 p, CGA3 q, CGA3 r, CGA3 s){
    return outer(outer(p,q,r),s);
}

CGA3 mul(CGA3 p, CGA3 q, CGA3 r, CGA3 s){
    return mul(mul(p,q,r),s);
}

CGA3 fromVec(vec3 v){
    CGA3 X = ZERO_CGA3;
    X.e1 = scalar_Dual(v.x);
    X.e2 = scalar_Dual(v.y);
    X.e3 = scalar_Dual(v.z);
    return X;
}

vec3 toVec(CGA3 x) {
    return vec3(x.e1.scalar, x.e2.scalar, x.e3.scalar);
}

// CGA3 fromVecU(vec3 v){
//     CGA3 X = ZERO_CGA3;
//     X.e1 = scalar_Dual(v.x);
//     X.e1.enil1 = 1.0;
//     X.e2 = scalar_Dual(v.y);
//     X.e2.enil1 = 1.0;
//     X.e3 = scalar_Dual(v.z);
//     X.e3.enil1 = 1.0;
//     return X;
// }

// CGA3 fromVecV(vec3 v){
//     CGA3 X = ZERO_CGA3;
//     X.e1 = scalar_Dual(v.x);
//     X.e1.enil2 = 1.0;
//     X.e2 = scalar_Dual(v.y);
//     X.e2.enil2 = 1.0;
//     X.e3 = scalar_Dual(v.z);
//     X.e3.enil2 = 1.0;
//     return X;
// }

CGA3 point(vec3 x){
    return point(fromVec(x));
}

vec3 vecFromPoint(CGA3 x) {
    // x = point_coords(x);
    return vec3(x.e1.scalar, x.e2.scalar, x.e3.scalar);
}

// replace this stuff with IPNS sphere stuff
// CGA3 sphere(vec3 a, vec3 b, vec3 c, vec3 d) {
//     return outer(point(a), point(b), point(c), point(d));
// }

// vec3 sphere_center(vec3 a, vec3 b, vec3 c, vec3 d){
//     CGA3 sphere = sphere(a,b,c,d);
//     return vecFromPoint(mul(sphere, INF(), sphere));
// }

// vec3 reflect_glsl(vec3 x, CGA3 R){
//     return vecFromPoint(mul(R,point(x),R));
// }


#pragma glslify: import('./snippets/patches.glsl')