#define GLSLIFY 1
// CGA3.glsl
const int I_CGA3_scalar = 0;
const int I_CGA3_e1 = 1;
const int I_CGA3_e2 = 2;
const int I_CGA3_e3 = 3;
const int I_CGA3_enil = 4;
const int I_CGA3_einf = 5;
const int I_CGA3_e12 = 6;
const int I_CGA3_e13 = 7;
const int I_CGA3_e1nil = 8;
const int I_CGA3_e1inf = 9;
const int I_CGA3_e23 = 10;
const int I_CGA3_e2nil = 11;
const int I_CGA3_e2inf = 12;
const int I_CGA3_e3nil = 13;
const int I_CGA3_e3inf = 14;
const int I_CGA3_enilinf = 15;
const int I_CGA3_e123 = 16;
const int I_CGA3_e12nil = 17;
const int I_CGA3_e12inf = 18;
const int I_CGA3_e13nil = 19;
const int I_CGA3_e13inf = 20;
const int I_CGA3_e1nilinf = 21;
const int I_CGA3_e23nil = 22;
const int I_CGA3_e23inf = 23;
const int I_CGA3_e2nilinf = 24;
const int I_CGA3_e3nilinf = 25;
const int I_CGA3_e123nil = 26;
const int I_CGA3_e123inf = 27;
const int I_CGA3_e12nilinf = 28;
const int I_CGA3_e13nilinf = 29;
const int I_CGA3_e23nilinf = 30;
const int I_CGA3_e123nilinf = 31;

struct CGA3 {
    float scalar;
    float e1;
    float e2;
    float e3;
    float enil;
    float einf;
    float e12;
    float e13;
    float e1nil;
    float e1inf;
    float e23;
    float e2nil;
    float e2inf;
    float e3nil;
    float e3inf;
    float enilinf;
    float e123;
    float e12nil;
    float e12inf;
    float e13nil;
    float e13inf;
    float e1nilinf;
    float e23nil;
    float e23inf;
    float e2nilinf;
    float e3nilinf;
    float e123nil;
    float e123inf;
    float e12nilinf;
    float e13nilinf;
    float e23nilinf;
    float e123nilinf;
};

CGA3 fromArray(float x[32]){
    return CGA3(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20], x[21], x[22], x[23], x[24], x[25], x[26], x[27], x[28], x[29], x[30], x[31]);
}

void toArray(CGA3 x, inout float x_ary[32]){
    x_ary[0] = x.scalar;
    x_ary[1] = x.e1;
    x_ary[2] = x.e2;
    x_ary[3] = x.e3;
    x_ary[4] = x.enil;
    x_ary[5] = x.einf;
    x_ary[6] = x.e12;
    x_ary[7] = x.e13;
    x_ary[8] = x.e1nil;
    x_ary[9] = x.e1inf;
    x_ary[10] = x.e23;
    x_ary[11] = x.e2nil;
    x_ary[12] = x.e2inf;
    x_ary[13] = x.e3nil;
    x_ary[14] = x.e3inf;
    x_ary[15] = x.enilinf;
    x_ary[16] = x.e123;
    x_ary[17] = x.e12nil;
    x_ary[18] = x.e12inf;
    x_ary[19] = x.e13nil;
    x_ary[20] = x.e13inf;
    x_ary[21] = x.e1nilinf;
    x_ary[22] = x.e23nil;
    x_ary[23] = x.e23inf;
    x_ary[24] = x.e2nilinf;
    x_ary[25] = x.e3nilinf;
    x_ary[26] = x.e123nil;
    x_ary[27] = x.e123inf;
    x_ary[28] = x.e12nilinf;
    x_ary[29] = x.e13nilinf;
    x_ary[30] = x.e23nilinf;
    x_ary[31] = x.e123nilinf;
}

void zero(inout float x[32]){
    x[0] = 0.0;
    x[1] = 0.0;
    x[2] = 0.0;
    x[3] = 0.0;
    x[4] = 0.0;
    x[5] = 0.0;
    x[6] = 0.0;
    x[7] = 0.0;
    x[8] = 0.0;
    x[9] = 0.0;
    x[10] = 0.0;
    x[11] = 0.0;
    x[12] = 0.0;
    x[13] = 0.0;
    x[14] = 0.0;
    x[15] = 0.0;
    x[16] = 0.0;
    x[17] = 0.0;
    x[18] = 0.0;
    x[19] = 0.0;
    x[20] = 0.0;
    x[21] = 0.0;
    x[22] = 0.0;
    x[23] = 0.0;
    x[24] = 0.0;
    x[25] = 0.0;
    x[26] = 0.0;
    x[27] = 0.0;
    x[28] = 0.0;
    x[29] = 0.0;
    x[30] = 0.0;
    x[31] = 0.0;
}

CGA3 add(CGA3 u, CGA3 v){
    return CGA3(u.scalar + v.scalar, u.e1 + v.e1, u.e2 + v.e2, u.e3 + v.e3, u.enil + v.enil, u.einf + v.einf, u.e12 + v.e12, u.e13 + v.e13, u.e1nil + v.e1nil, u.e1inf + v.e1inf, u.e23 + v.e23, u.e2nil + v.e2nil, u.e2inf + v.e2inf, u.e3nil + v.e3nil, u.e3inf + v.e3inf, u.enilinf + v.enilinf, u.e123 + v.e123, u.e12nil + v.e12nil, u.e12inf + v.e12inf, u.e13nil + v.e13nil, u.e13inf + v.e13inf, u.e1nilinf + v.e1nilinf, u.e23nil + v.e23nil, u.e23inf + v.e23inf, u.e2nilinf + v.e2nilinf, u.e3nilinf + v.e3nilinf, u.e123nil + v.e123nil, u.e123inf + v.e123inf, u.e12nilinf + v.e12nilinf, u.e13nilinf + v.e13nilinf, u.e23nilinf + v.e23nilinf, u.e123nilinf + v.e123nilinf);
}

CGA3 add(CGA3 u, CGA3 v, CGA3 w){
    return add(add(u, v), w);
}

CGA3 add(CGA3 u, CGA3 v, CGA3 w, CGA3 p){
    return add(add(add(u, v), w), p);
}

CGA3 one(){
    return CGA3(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

CGA3 sub(CGA3 u, CGA3 v){
    return CGA3(u.scalar - v.scalar, u.e1 - v.e1, u.e2 - v.e2, u.e3 - v.e3, u.enil - v.enil, u.einf - v.einf, u.e12 - v.e12, u.e13 - v.e13, u.e1nil - v.e1nil, u.e1inf - v.e1inf, u.e23 - v.e23, u.e2nil - v.e2nil, u.e2inf - v.e2inf, u.e3nil - v.e3nil, u.e3inf - v.e3inf, u.enilinf - v.enilinf, u.e123 - v.e123, u.e12nil - v.e12nil, u.e12inf - v.e12inf, u.e13nil - v.e13nil, u.e13inf - v.e13inf, u.e1nilinf - v.e1nilinf, u.e23nil - v.e23nil, u.e23inf - v.e23inf, u.e2nilinf - v.e2nilinf, u.e3nilinf - v.e3nilinf, u.e123nil - v.e123nil, u.e123inf - v.e123inf, u.e12nilinf - v.e12nilinf, u.e13nilinf - v.e13nilinf, u.e23nilinf - v.e23nilinf, u.e123nilinf - v.e123nilinf);
}

CGA3 zero(){
    return CGA3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

CGA3 mul(float a, CGA3 x){
    return CGA3(a*x.scalar, a*x.e1, a*x.e2, a*x.e3, a*x.enil, a*x.einf, a*x.e12, a*x.e13, a*x.e1nil, a*x.e1inf, a*x.e23, a*x.e2nil, a*x.e2inf, a*x.e3nil, a*x.e3inf, a*x.enilinf, a*x.e123, a*x.e12nil, a*x.e12inf, a*x.e13nil, a*x.e13inf, a*x.e1nilinf, a*x.e23nil, a*x.e23inf, a*x.e2nilinf, a*x.e3nilinf, a*x.e123nil, a*x.e123inf, a*x.e12nilinf, a*x.e13nilinf, a*x.e23nilinf, a*x.e123nilinf);
}

CGA3 mul(CGA3 u, CGA3 v){
    return CGA3(u.e1*v.e1 - u.e12*v.e12 - u.e123*v.e123 + u.e123inf*v.e123nil + u.e123nil*v.e123inf - u.e123nilinf*v.e123nilinf - u.e12inf*v.e12nil - u.e12nil*v.e12inf - u.e12nilinf*v.e12nilinf - u.e13*v.e13 - u.e13inf*v.e13nil - u.e13nil*v.e13inf - u.e13nilinf*v.e13nilinf - u.e1inf*v.e1nil - u.e1nil*v.e1inf + u.e1nilinf*v.e1nilinf + u.e2*v.e2 - u.e23*v.e23 - u.e23inf*v.e23nil - u.e23nil*v.e23inf - u.e23nilinf*v.e23nilinf - u.e2inf*v.e2nil - u.e2nil*v.e2inf + u.e2nilinf*v.e2nilinf + u.e3*v.e3 - u.e3inf*v.e3nil - u.e3nil*v.e3inf + u.e3nilinf*v.e3nilinf + u.einf*v.enil + u.enil*v.einf + u.enilinf*v.enilinf + u.scalar*v.scalar, u.e1*v.scalar + u.e12*v.e2 - u.e123*v.e23 - u.e123inf*v.e23nil - u.e123nil*v.e23inf - u.e123nilinf*v.e23nilinf - u.e12inf*v.e2nil - u.e12nil*v.e2inf + u.e12nilinf*v.e2nilinf + u.e13*v.e3 - u.e13inf*v.e3nil - u.e13nil*v.e3inf + u.e13nilinf*v.e3nilinf + u.e1inf*v.enil + u.e1nil*v.einf + u.e1nilinf*v.enilinf - u.e2*v.e12 - u.e23*v.e123 + u.e23inf*v.e123nil + u.e23nil*v.e123inf - u.e23nilinf*v.e123nilinf - u.e2inf*v.e12nil - u.e2nil*v.e12inf - u.e2nilinf*v.e12nilinf - u.e3*v.e13 - u.e3inf*v.e13nil - u.e3nil*v.e13inf - u.e3nilinf*v.e13nilinf - u.einf*v.e1nil - u.enil*v.e1inf + u.enilinf*v.e1nilinf + u.scalar*v.e1, u.e1*v.e12 - u.e12*v.e1 + u.e123*v.e13 + u.e123inf*v.e13nil + u.e123nil*v.e13inf + u.e123nilinf*v.e13nilinf + u.e12inf*v.e1nil + u.e12nil*v.e1inf - u.e12nilinf*v.e1nilinf + u.e13*v.e123 - u.e13inf*v.e123nil - u.e13nil*v.e123inf + u.e13nilinf*v.e123nilinf + u.e1inf*v.e12nil + u.e1nil*v.e12inf + u.e1nilinf*v.e12nilinf + u.e2*v.scalar + u.e23*v.e3 - u.e23inf*v.e3nil - u.e23nil*v.e3inf + u.e23nilinf*v.e3nilinf + u.e2inf*v.enil + u.e2nil*v.einf + u.e2nilinf*v.enilinf - u.e3*v.e23 - u.e3inf*v.e23nil - u.e3nil*v.e23inf - u.e3nilinf*v.e23nilinf - u.einf*v.e2nil - u.enil*v.e2inf + u.enilinf*v.e2nilinf + u.scalar*v.e2, u.e1*v.e13 - u.e12*v.e123 - u.e123*v.e12 - u.e123inf*v.e12nil - u.e123nil*v.e12inf - u.e123nilinf*v.e12nilinf + u.e12inf*v.e123nil + u.e12nil*v.e123inf - u.e12nilinf*v.e123nilinf - u.e13*v.e1 + u.e13inf*v.e1nil + u.e13nil*v.e1inf - u.e13nilinf*v.e1nilinf + u.e1inf*v.e13nil + u.e1nil*v.e13inf + u.e1nilinf*v.e13nilinf + u.e2*v.e23 - u.e23*v.e2 + u.e23inf*v.e2nil + u.e23nil*v.e2inf - u.e23nilinf*v.e2nilinf + u.e2inf*v.e23nil + u.e2nil*v.e23inf + u.e2nilinf*v.e23nilinf + u.e3*v.scalar + u.e3inf*v.enil + u.e3nil*v.einf + u.e3nilinf*v.enilinf - u.einf*v.e3nil - u.enil*v.e3inf + u.enilinf*v.e3nilinf + u.scalar*v.e3, u.e1*v.e1nil - u.e12*v.e12nil - u.e123*v.e123nil + u.e123nil*v.e123 - u.e123nil*v.e123nilinf - u.e123nilinf*v.e123nil - u.e12nil*v.e12 + u.e12nil*v.e12nilinf - u.e12nilinf*v.e12nil - u.e13*v.e13nil - u.e13nil*v.e13 + u.e13nil*v.e13nilinf - u.e13nilinf*v.e13nil - u.e1nil*v.e1 + u.e1nil*v.e1nilinf + u.e1nilinf*v.e1nil + u.e2*v.e2nil - u.e23*v.e23nil - u.e23nil*v.e23 + u.e23nil*v.e23nilinf - u.e23nilinf*v.e23nil - u.e2nil*v.e2 + u.e2nil*v.e2nilinf + u.e2nilinf*v.e2nil + u.e3*v.e3nil - u.e3nil*v.e3 + u.e3nil*v.e3nilinf + u.e3nilinf*v.e3nil - u.enil*v.enilinf + u.enil*v.scalar + u.enilinf*v.enil + u.scalar*v.enil, u.e1*v.e1inf - u.e12*v.e12inf - u.e123*v.e123inf + u.e123inf*v.e123 + u.e123inf*v.e123nilinf + u.e123nilinf*v.e123inf - u.e12inf*v.e12 - u.e12inf*v.e12nilinf + u.e12nilinf*v.e12inf - u.e13*v.e13inf - u.e13inf*v.e13 - u.e13inf*v.e13nilinf + u.e13nilinf*v.e13inf - u.e1inf*v.e1 - u.e1inf*v.e1nilinf - u.e1nilinf*v.e1inf + u.e2*v.e2inf - u.e23*v.e23inf - u.e23inf*v.e23 - u.e23inf*v.e23nilinf + u.e23nilinf*v.e23inf - u.e2inf*v.e2 - u.e2inf*v.e2nilinf - u.e2nilinf*v.e2inf + u.e3*v.e3inf - u.e3inf*v.e3 - u.e3inf*v.e3nilinf - u.e3nilinf*v.e3inf + u.einf*v.enilinf + u.einf*v.scalar - u.enilinf*v.einf + u.scalar*v.einf, u.e1*v.e2 + u.e12*v.scalar + u.e123*v.e3 - u.e123inf*v.e3nil - u.e123nil*v.e3inf + u.e123nilinf*v.e3nilinf + u.e12inf*v.enil + u.e12nil*v.einf + u.e12nilinf*v.enilinf - u.e13*v.e23 - u.e13inf*v.e23nil - u.e13nil*v.e23inf - u.e13nilinf*v.e23nilinf - u.e1inf*v.e2nil - u.e1nil*v.e2inf + u.e1nilinf*v.e2nilinf - u.e2*v.e1 + u.e23*v.e13 + u.e23inf*v.e13nil + u.e23nil*v.e13inf + u.e23nilinf*v.e13nilinf + u.e2inf*v.e1nil + u.e2nil*v.e1inf - u.e2nilinf*v.e1nilinf + u.e3*v.e123 - u.e3inf*v.e123nil - u.e3nil*v.e123inf + u.e3nilinf*v.e123nilinf + u.einf*v.e12nil + u.enil*v.e12inf + u.enilinf*v.e12nilinf + u.scalar*v.e12, u.e1*v.e3 + u.e12*v.e23 - u.e123*v.e2 + u.e123inf*v.e2nil + u.e123nil*v.e2inf - u.e123nilinf*v.e2nilinf + u.e12inf*v.e23nil + u.e12nil*v.e23inf + u.e12nilinf*v.e23nilinf + u.e13*v.scalar + u.e13inf*v.enil + u.e13nil*v.einf + u.e13nilinf*v.enilinf - u.e1inf*v.e3nil - u.e1nil*v.e3inf + u.e1nilinf*v.e3nilinf - u.e2*v.e123 - u.e23*v.e12 - u.e23inf*v.e12nil - u.e23nil*v.e12inf - u.e23nilinf*v.e12nilinf + u.e2inf*v.e123nil + u.e2nil*v.e123inf - u.e2nilinf*v.e123nilinf - u.e3*v.e1 + u.e3inf*v.e1nil + u.e3nil*v.e1inf - u.e3nilinf*v.e1nilinf + u.einf*v.e13nil + u.enil*v.e13inf + u.enilinf*v.e13nilinf + u.scalar*v.e13, u.e1*v.enil + u.e12*v.e2nil - u.e123*v.e23nil - u.e123nil*v.e23 + u.e123nil*v.e23nilinf - u.e123nilinf*v.e23nil - u.e12nil*v.e2 + u.e12nil*v.e2nilinf + u.e12nilinf*v.e2nil + u.e13*v.e3nil - u.e13nil*v.e3 + u.e13nil*v.e3nilinf + u.e13nilinf*v.e3nil - u.e1nil*v.enilinf + u.e1nil*v.scalar + u.e1nilinf*v.enil - u.e2*v.e12nil - u.e23*v.e123nil + u.e23nil*v.e123 - u.e23nil*v.e123nilinf - u.e23nilinf*v.e123nil - u.e2nil*v.e12 + u.e2nil*v.e12nilinf - u.e2nilinf*v.e12nil - u.e3*v.e13nil - u.e3nil*v.e13 + u.e3nil*v.e13nilinf - u.e3nilinf*v.e13nil - u.enil*v.e1 + u.enil*v.e1nilinf + u.enilinf*v.e1nil + u.scalar*v.e1nil, u.e1*v.einf + u.e12*v.e2inf - u.e123*v.e23inf - u.e123inf*v.e23 - u.e123inf*v.e23nilinf + u.e123nilinf*v.e23inf - u.e12inf*v.e2 - u.e12inf*v.e2nilinf - u.e12nilinf*v.e2inf + u.e13*v.e3inf - u.e13inf*v.e3 - u.e13inf*v.e3nilinf - u.e13nilinf*v.e3inf + u.e1inf*v.enilinf + u.e1inf*v.scalar - u.e1nilinf*v.einf - u.e2*v.e12inf - u.e23*v.e123inf + u.e23inf*v.e123 + u.e23inf*v.e123nilinf + u.e23nilinf*v.e123inf - u.e2inf*v.e12 - u.e2inf*v.e12nilinf + u.e2nilinf*v.e12inf - u.e3*v.e13inf - u.e3inf*v.e13 - u.e3inf*v.e13nilinf + u.e3nilinf*v.e13inf - u.einf*v.e1 - u.einf*v.e1nilinf - u.enilinf*v.e1inf + u.scalar*v.e1inf, u.e1*v.e123 - u.e12*v.e13 + u.e123*v.e1 - u.e123inf*v.e1nil - u.e123nil*v.e1inf + u.e123nilinf*v.e1nilinf - u.e12inf*v.e13nil - u.e12nil*v.e13inf - u.e12nilinf*v.e13nilinf + u.e13*v.e12 + u.e13inf*v.e12nil + u.e13nil*v.e12inf + u.e13nilinf*v.e12nilinf - u.e1inf*v.e123nil - u.e1nil*v.e123inf + u.e1nilinf*v.e123nilinf + u.e2*v.e3 + u.e23*v.scalar + u.e23inf*v.enil + u.e23nil*v.einf + u.e23nilinf*v.enilinf - u.e2inf*v.e3nil - u.e2nil*v.e3inf + u.e2nilinf*v.e3nilinf - u.e3*v.e2 + u.e3inf*v.e2nil + u.e3nil*v.e2inf - u.e3nilinf*v.e2nilinf + u.einf*v.e23nil + u.enil*v.e23inf + u.enilinf*v.e23nilinf + u.scalar*v.e23, u.e1*v.e12nil - u.e12*v.e1nil + u.e123*v.e13nil + u.e123nil*v.e13 - u.e123nil*v.e13nilinf + u.e123nilinf*v.e13nil + u.e12nil*v.e1 - u.e12nil*v.e1nilinf - u.e12nilinf*v.e1nil + u.e13*v.e123nil - u.e13nil*v.e123 + u.e13nil*v.e123nilinf + u.e13nilinf*v.e123nil + u.e1nil*v.e12 - u.e1nil*v.e12nilinf + u.e1nilinf*v.e12nil + u.e2*v.enil + u.e23*v.e3nil - u.e23nil*v.e3 + u.e23nil*v.e3nilinf + u.e23nilinf*v.e3nil - u.e2nil*v.enilinf + u.e2nil*v.scalar + u.e2nilinf*v.enil - u.e3*v.e23nil - u.e3nil*v.e23 + u.e3nil*v.e23nilinf - u.e3nilinf*v.e23nil - u.enil*v.e2 + u.enil*v.e2nilinf + u.enilinf*v.e2nil + u.scalar*v.e2nil, u.e1*v.e12inf - u.e12*v.e1inf + u.e123*v.e13inf + u.e123inf*v.e13 + u.e123inf*v.e13nilinf - u.e123nilinf*v.e13inf + u.e12inf*v.e1 + u.e12inf*v.e1nilinf + u.e12nilinf*v.e1inf + u.e13*v.e123inf - u.e13inf*v.e123 - u.e13inf*v.e123nilinf - u.e13nilinf*v.e123inf + u.e1inf*v.e12 + u.e1inf*v.e12nilinf - u.e1nilinf*v.e12inf + u.e2*v.einf + u.e23*v.e3inf - u.e23inf*v.e3 - u.e23inf*v.e3nilinf - u.e23nilinf*v.e3inf + u.e2inf*v.enilinf + u.e2inf*v.scalar - u.e2nilinf*v.einf - u.e3*v.e23inf - u.e3inf*v.e23 - u.e3inf*v.e23nilinf + u.e3nilinf*v.e23inf - u.einf*v.e2 - u.einf*v.e2nilinf - u.enilinf*v.e2inf + u.scalar*v.e2inf, u.e1*v.e13nil - u.e12*v.e123nil - u.e123*v.e12nil - u.e123nil*v.e12 + u.e123nil*v.e12nilinf - u.e123nilinf*v.e12nil + u.e12nil*v.e123 - u.e12nil*v.e123nilinf - u.e12nilinf*v.e123nil - u.e13*v.e1nil + u.e13nil*v.e1 - u.e13nil*v.e1nilinf - u.e13nilinf*v.e1nil + u.e1nil*v.e13 - u.e1nil*v.e13nilinf + u.e1nilinf*v.e13nil + u.e2*v.e23nil - u.e23*v.e2nil + u.e23nil*v.e2 - u.e23nil*v.e2nilinf - u.e23nilinf*v.e2nil + u.e2nil*v.e23 - u.e2nil*v.e23nilinf + u.e2nilinf*v.e23nil + u.e3*v.enil - u.e3nil*v.enilinf + u.e3nil*v.scalar + u.e3nilinf*v.enil - u.enil*v.e3 + u.enil*v.e3nilinf + u.enilinf*v.e3nil + u.scalar*v.e3nil, u.e1*v.e13inf - u.e12*v.e123inf - u.e123*v.e12inf - u.e123inf*v.e12 - u.e123inf*v.e12nilinf + u.e123nilinf*v.e12inf + u.e12inf*v.e123 + u.e12inf*v.e123nilinf + u.e12nilinf*v.e123inf - u.e13*v.e1inf + u.e13inf*v.e1 + u.e13inf*v.e1nilinf + u.e13nilinf*v.e1inf + u.e1inf*v.e13 + u.e1inf*v.e13nilinf - u.e1nilinf*v.e13inf + u.e2*v.e23inf - u.e23*v.e2inf + u.e23inf*v.e2 + u.e23inf*v.e2nilinf + u.e23nilinf*v.e2inf + u.e2inf*v.e23 + u.e2inf*v.e23nilinf - u.e2nilinf*v.e23inf + u.e3*v.einf + u.e3inf*v.enilinf + u.e3inf*v.scalar - u.e3nilinf*v.einf - u.einf*v.e3 - u.einf*v.e3nilinf - u.enilinf*v.e3inf + u.scalar*v.e3inf, u.e1*v.e1nilinf - u.e12*v.e12nilinf - u.e123*v.e123nilinf - u.e123inf*v.e123nil + u.e123nil*v.e123inf - u.e123nilinf*v.e123 + u.e12inf*v.e12nil - u.e12nil*v.e12inf - u.e12nilinf*v.e12 - u.e13*v.e13nilinf + u.e13inf*v.e13nil - u.e13nil*v.e13inf - u.e13nilinf*v.e13 + u.e1inf*v.e1nil - u.e1nil*v.e1inf + u.e1nilinf*v.e1 + u.e2*v.e2nilinf - u.e23*v.e23nilinf + u.e23inf*v.e23nil - u.e23nil*v.e23inf - u.e23nilinf*v.e23 + u.e2inf*v.e2nil - u.e2nil*v.e2inf + u.e2nilinf*v.e2 + u.e3*v.e3nilinf + u.e3inf*v.e3nil - u.e3nil*v.e3inf + u.e3nilinf*v.e3 - u.einf*v.enil + u.enil*v.einf + u.enilinf*v.scalar + u.scalar*v.enilinf, u.e1*v.e23 + u.e12*v.e3 + u.e123*v.scalar + u.e123inf*v.enil + u.e123nil*v.einf + u.e123nilinf*v.enilinf - u.e12inf*v.e3nil - u.e12nil*v.e3inf + u.e12nilinf*v.e3nilinf - u.e13*v.e2 + u.e13inf*v.e2nil + u.e13nil*v.e2inf - u.e13nilinf*v.e2nilinf + u.e1inf*v.e23nil + u.e1nil*v.e23inf + u.e1nilinf*v.e23nilinf - u.e2*v.e13 + u.e23*v.e1 - u.e23inf*v.e1nil - u.e23nil*v.e1inf + u.e23nilinf*v.e1nilinf - u.e2inf*v.e13nil - u.e2nil*v.e13inf - u.e2nilinf*v.e13nilinf + u.e3*v.e12 + u.e3inf*v.e12nil + u.e3nil*v.e12inf + u.e3nilinf*v.e12nilinf - u.einf*v.e123nil - u.enil*v.e123inf + u.enilinf*v.e123nilinf + u.scalar*v.e123, u.e1*v.e2nil + u.e12*v.enil + u.e123*v.e3nil - u.e123nil*v.e3 + u.e123nil*v.e3nilinf + u.e123nilinf*v.e3nil - u.e12nil*v.enilinf + u.e12nil*v.scalar + u.e12nilinf*v.enil - u.e13*v.e23nil - u.e13nil*v.e23 + u.e13nil*v.e23nilinf - u.e13nilinf*v.e23nil - u.e1nil*v.e2 + u.e1nil*v.e2nilinf + u.e1nilinf*v.e2nil - u.e2*v.e1nil + u.e23*v.e13nil + u.e23nil*v.e13 - u.e23nil*v.e13nilinf + u.e23nilinf*v.e13nil + u.e2nil*v.e1 - u.e2nil*v.e1nilinf - u.e2nilinf*v.e1nil + u.e3*v.e123nil - u.e3nil*v.e123 + u.e3nil*v.e123nilinf + u.e3nilinf*v.e123nil + u.enil*v.e12 - u.enil*v.e12nilinf + u.enilinf*v.e12nil + u.scalar*v.e12nil, u.e1*v.e2inf + u.e12*v.einf + u.e123*v.e3inf - u.e123inf*v.e3 - u.e123inf*v.e3nilinf - u.e123nilinf*v.e3inf + u.e12inf*v.enilinf + u.e12inf*v.scalar - u.e12nilinf*v.einf - u.e13*v.e23inf - u.e13inf*v.e23 - u.e13inf*v.e23nilinf + u.e13nilinf*v.e23inf - u.e1inf*v.e2 - u.e1inf*v.e2nilinf - u.e1nilinf*v.e2inf - u.e2*v.e1inf + u.e23*v.e13inf + u.e23inf*v.e13 + u.e23inf*v.e13nilinf - u.e23nilinf*v.e13inf + u.e2inf*v.e1 + u.e2inf*v.e1nilinf + u.e2nilinf*v.e1inf + u.e3*v.e123inf - u.e3inf*v.e123 - u.e3inf*v.e123nilinf - u.e3nilinf*v.e123inf + u.einf*v.e12 + u.einf*v.e12nilinf - u.enilinf*v.e12inf + u.scalar*v.e12inf, u.e1*v.e3nil + u.e12*v.e23nil - u.e123*v.e2nil + u.e123nil*v.e2 - u.e123nil*v.e2nilinf - u.e123nilinf*v.e2nil + u.e12nil*v.e23 - u.e12nil*v.e23nilinf + u.e12nilinf*v.e23nil + u.e13*v.enil - u.e13nil*v.enilinf + u.e13nil*v.scalar + u.e13nilinf*v.enil - u.e1nil*v.e3 + u.e1nil*v.e3nilinf + u.e1nilinf*v.e3nil - u.e2*v.e123nil - u.e23*v.e12nil - u.e23nil*v.e12 + u.e23nil*v.e12nilinf - u.e23nilinf*v.e12nil + u.e2nil*v.e123 - u.e2nil*v.e123nilinf - u.e2nilinf*v.e123nil - u.e3*v.e1nil + u.e3nil*v.e1 - u.e3nil*v.e1nilinf - u.e3nilinf*v.e1nil + u.enil*v.e13 - u.enil*v.e13nilinf + u.enilinf*v.e13nil + u.scalar*v.e13nil, u.e1*v.e3inf + u.e12*v.e23inf - u.e123*v.e2inf + u.e123inf*v.e2 + u.e123inf*v.e2nilinf + u.e123nilinf*v.e2inf + u.e12inf*v.e23 + u.e12inf*v.e23nilinf - u.e12nilinf*v.e23inf + u.e13*v.einf + u.e13inf*v.enilinf + u.e13inf*v.scalar - u.e13nilinf*v.einf - u.e1inf*v.e3 - u.e1inf*v.e3nilinf - u.e1nilinf*v.e3inf - u.e2*v.e123inf - u.e23*v.e12inf - u.e23inf*v.e12 - u.e23inf*v.e12nilinf + u.e23nilinf*v.e12inf + u.e2inf*v.e123 + u.e2inf*v.e123nilinf + u.e2nilinf*v.e123inf - u.e3*v.e1inf + u.e3inf*v.e1 + u.e3inf*v.e1nilinf + u.e3nilinf*v.e1inf + u.einf*v.e13 + u.einf*v.e13nilinf - u.enilinf*v.e13inf + u.scalar*v.e13inf, u.e1*v.enilinf + u.e12*v.e2nilinf - u.e123*v.e23nilinf + u.e123inf*v.e23nil - u.e123nil*v.e23inf - u.e123nilinf*v.e23 + u.e12inf*v.e2nil - u.e12nil*v.e2inf + u.e12nilinf*v.e2 + u.e13*v.e3nilinf + u.e13inf*v.e3nil - u.e13nil*v.e3inf + u.e13nilinf*v.e3 - u.e1inf*v.enil + u.e1nil*v.einf + u.e1nilinf*v.scalar - u.e2*v.e12nilinf - u.e23*v.e123nilinf - u.e23inf*v.e123nil + u.e23nil*v.e123inf - u.e23nilinf*v.e123 + u.e2inf*v.e12nil - u.e2nil*v.e12inf - u.e2nilinf*v.e12 - u.e3*v.e13nilinf + u.e3inf*v.e13nil - u.e3nil*v.e13inf - u.e3nilinf*v.e13 + u.einf*v.e1nil - u.enil*v.e1inf + u.enilinf*v.e1 + u.scalar*v.e1nilinf, u.e1*v.e123nil - u.e12*v.e13nil + u.e123*v.e1nil - u.e123nil*v.e1 + u.e123nil*v.e1nilinf + u.e123nilinf*v.e1nil - u.e12nil*v.e13 + u.e12nil*v.e13nilinf - u.e12nilinf*v.e13nil + u.e13*v.e12nil + u.e13nil*v.e12 - u.e13nil*v.e12nilinf + u.e13nilinf*v.e12nil - u.e1nil*v.e123 + u.e1nil*v.e123nilinf + u.e1nilinf*v.e123nil + u.e2*v.e3nil + u.e23*v.enil - u.e23nil*v.enilinf + u.e23nil*v.scalar + u.e23nilinf*v.enil - u.e2nil*v.e3 + u.e2nil*v.e3nilinf + u.e2nilinf*v.e3nil - u.e3*v.e2nil + u.e3nil*v.e2 - u.e3nil*v.e2nilinf - u.e3nilinf*v.e2nil + u.enil*v.e23 - u.enil*v.e23nilinf + u.enilinf*v.e23nil + u.scalar*v.e23nil, u.e1*v.e123inf - u.e12*v.e13inf + u.e123*v.e1inf - u.e123inf*v.e1 - u.e123inf*v.e1nilinf - u.e123nilinf*v.e1inf - u.e12inf*v.e13 - u.e12inf*v.e13nilinf + u.e12nilinf*v.e13inf + u.e13*v.e12inf + u.e13inf*v.e12 + u.e13inf*v.e12nilinf - u.e13nilinf*v.e12inf - u.e1inf*v.e123 - u.e1inf*v.e123nilinf - u.e1nilinf*v.e123inf + u.e2*v.e3inf + u.e23*v.einf + u.e23inf*v.enilinf + u.e23inf*v.scalar - u.e23nilinf*v.einf - u.e2inf*v.e3 - u.e2inf*v.e3nilinf - u.e2nilinf*v.e3inf - u.e3*v.e2inf + u.e3inf*v.e2 + u.e3inf*v.e2nilinf + u.e3nilinf*v.e2inf + u.einf*v.e23 + u.einf*v.e23nilinf - u.enilinf*v.e23inf + u.scalar*v.e23inf, u.e1*v.e12nilinf - u.e12*v.e1nilinf + u.e123*v.e13nilinf - u.e123inf*v.e13nil + u.e123nil*v.e13inf + u.e123nilinf*v.e13 - u.e12inf*v.e1nil + u.e12nil*v.e1inf - u.e12nilinf*v.e1 + u.e13*v.e123nilinf + u.e13inf*v.e123nil - u.e13nil*v.e123inf + u.e13nilinf*v.e123 - u.e1inf*v.e12nil + u.e1nil*v.e12inf + u.e1nilinf*v.e12 + u.e2*v.enilinf + u.e23*v.e3nilinf + u.e23inf*v.e3nil - u.e23nil*v.e3inf + u.e23nilinf*v.e3 - u.e2inf*v.enil + u.e2nil*v.einf + u.e2nilinf*v.scalar - u.e3*v.e23nilinf + u.e3inf*v.e23nil - u.e3nil*v.e23inf - u.e3nilinf*v.e23 + u.einf*v.e2nil - u.enil*v.e2inf + u.enilinf*v.e2 + u.scalar*v.e2nilinf, u.e1*v.e13nilinf - u.e12*v.e123nilinf - u.e123*v.e12nilinf + u.e123inf*v.e12nil - u.e123nil*v.e12inf - u.e123nilinf*v.e12 - u.e12inf*v.e123nil + u.e12nil*v.e123inf - u.e12nilinf*v.e123 - u.e13*v.e1nilinf - u.e13inf*v.e1nil + u.e13nil*v.e1inf - u.e13nilinf*v.e1 - u.e1inf*v.e13nil + u.e1nil*v.e13inf + u.e1nilinf*v.e13 + u.e2*v.e23nilinf - u.e23*v.e2nilinf - u.e23inf*v.e2nil + u.e23nil*v.e2inf - u.e23nilinf*v.e2 - u.e2inf*v.e23nil + u.e2nil*v.e23inf + u.e2nilinf*v.e23 + u.e3*v.enilinf - u.e3inf*v.enil + u.e3nil*v.einf + u.e3nilinf*v.scalar + u.einf*v.e3nil - u.enil*v.e3inf + u.enilinf*v.e3 + u.scalar*v.e3nilinf, u.e1*v.e23nil + u.e12*v.e3nil + u.e123*v.enil - u.e123nil*v.enilinf + u.e123nil*v.scalar + u.e123nilinf*v.enil - u.e12nil*v.e3 + u.e12nil*v.e3nilinf + u.e12nilinf*v.e3nil - u.e13*v.e2nil + u.e13nil*v.e2 - u.e13nil*v.e2nilinf - u.e13nilinf*v.e2nil + u.e1nil*v.e23 - u.e1nil*v.e23nilinf + u.e1nilinf*v.e23nil - u.e2*v.e13nil + u.e23*v.e1nil - u.e23nil*v.e1 + u.e23nil*v.e1nilinf + u.e23nilinf*v.e1nil - u.e2nil*v.e13 + u.e2nil*v.e13nilinf - u.e2nilinf*v.e13nil + u.e3*v.e12nil + u.e3nil*v.e12 - u.e3nil*v.e12nilinf + u.e3nilinf*v.e12nil - u.enil*v.e123 + u.enil*v.e123nilinf + u.enilinf*v.e123nil + u.scalar*v.e123nil, u.e1*v.e23inf + u.e12*v.e3inf + u.e123*v.einf + u.e123inf*v.enilinf + u.e123inf*v.scalar - u.e123nilinf*v.einf - u.e12inf*v.e3 - u.e12inf*v.e3nilinf - u.e12nilinf*v.e3inf - u.e13*v.e2inf + u.e13inf*v.e2 + u.e13inf*v.e2nilinf + u.e13nilinf*v.e2inf + u.e1inf*v.e23 + u.e1inf*v.e23nilinf - u.e1nilinf*v.e23inf - u.e2*v.e13inf + u.e23*v.e1inf - u.e23inf*v.e1 - u.e23inf*v.e1nilinf - u.e23nilinf*v.e1inf - u.e2inf*v.e13 - u.e2inf*v.e13nilinf + u.e2nilinf*v.e13inf + u.e3*v.e12inf + u.e3inf*v.e12 + u.e3inf*v.e12nilinf - u.e3nilinf*v.e12inf - u.einf*v.e123 - u.einf*v.e123nilinf - u.enilinf*v.e123inf + u.scalar*v.e123inf, u.e1*v.e2nilinf + u.e12*v.enilinf + u.e123*v.e3nilinf + u.e123inf*v.e3nil - u.e123nil*v.e3inf + u.e123nilinf*v.e3 - u.e12inf*v.enil + u.e12nil*v.einf + u.e12nilinf*v.scalar - u.e13*v.e23nilinf + u.e13inf*v.e23nil - u.e13nil*v.e23inf - u.e13nilinf*v.e23 + u.e1inf*v.e2nil - u.e1nil*v.e2inf + u.e1nilinf*v.e2 - u.e2*v.e1nilinf + u.e23*v.e13nilinf - u.e23inf*v.e13nil + u.e23nil*v.e13inf + u.e23nilinf*v.e13 - u.e2inf*v.e1nil + u.e2nil*v.e1inf - u.e2nilinf*v.e1 + u.e3*v.e123nilinf + u.e3inf*v.e123nil - u.e3nil*v.e123inf + u.e3nilinf*v.e123 - u.einf*v.e12nil + u.enil*v.e12inf + u.enilinf*v.e12 + u.scalar*v.e12nilinf, u.e1*v.e3nilinf + u.e12*v.e23nilinf - u.e123*v.e2nilinf - u.e123inf*v.e2nil + u.e123nil*v.e2inf - u.e123nilinf*v.e2 - u.e12inf*v.e23nil + u.e12nil*v.e23inf + u.e12nilinf*v.e23 + u.e13*v.enilinf - u.e13inf*v.enil + u.e13nil*v.einf + u.e13nilinf*v.scalar + u.e1inf*v.e3nil - u.e1nil*v.e3inf + u.e1nilinf*v.e3 - u.e2*v.e123nilinf - u.e23*v.e12nilinf + u.e23inf*v.e12nil - u.e23nil*v.e12inf - u.e23nilinf*v.e12 - u.e2inf*v.e123nil + u.e2nil*v.e123inf - u.e2nilinf*v.e123 - u.e3*v.e1nilinf - u.e3inf*v.e1nil + u.e3nil*v.e1inf - u.e3nilinf*v.e1 - u.einf*v.e13nil + u.enil*v.e13inf + u.enilinf*v.e13 + u.scalar*v.e13nilinf, u.e1*v.e123nilinf - u.e12*v.e13nilinf + u.e123*v.e1nilinf + u.e123inf*v.e1nil - u.e123nil*v.e1inf + u.e123nilinf*v.e1 + u.e12inf*v.e13nil - u.e12nil*v.e13inf - u.e12nilinf*v.e13 + u.e13*v.e12nilinf - u.e13inf*v.e12nil + u.e13nil*v.e12inf + u.e13nilinf*v.e12 + u.e1inf*v.e123nil - u.e1nil*v.e123inf + u.e1nilinf*v.e123 + u.e2*v.e3nilinf + u.e23*v.enilinf - u.e23inf*v.enil + u.e23nil*v.einf + u.e23nilinf*v.scalar + u.e2inf*v.e3nil - u.e2nil*v.e3inf + u.e2nilinf*v.e3 - u.e3*v.e2nilinf - u.e3inf*v.e2nil + u.e3nil*v.e2inf - u.e3nilinf*v.e2 - u.einf*v.e23nil + u.enil*v.e23inf + u.enilinf*v.e23 + u.scalar*v.e23nilinf, u.e1*v.e23nilinf + u.e12*v.e3nilinf + u.e123*v.enilinf - u.e123inf*v.enil + u.e123nil*v.einf + u.e123nilinf*v.scalar + u.e12inf*v.e3nil - u.e12nil*v.e3inf + u.e12nilinf*v.e3 - u.e13*v.e2nilinf - u.e13inf*v.e2nil + u.e13nil*v.e2inf - u.e13nilinf*v.e2 - u.e1inf*v.e23nil + u.e1nil*v.e23inf + u.e1nilinf*v.e23 - u.e2*v.e13nilinf + u.e23*v.e1nilinf + u.e23inf*v.e1nil - u.e23nil*v.e1inf + u.e23nilinf*v.e1 + u.e2inf*v.e13nil - u.e2nil*v.e13inf - u.e2nilinf*v.e13 + u.e3*v.e12nilinf - u.e3inf*v.e12nil + u.e3nil*v.e12inf + u.e3nilinf*v.e12 + u.einf*v.e123nil - u.enil*v.e123inf + u.enilinf*v.e123 + u.scalar*v.e123nilinf);
}

CGA3 mul(int a, CGA3 x){
    return mul(float(a), x);
}

CGA3 mul(CGA3 u, CGA3 v, CGA3 w){
    return mul(mul(u, v), w);
}

CGA3 dual(CGA3 u){
    return CGA3(u.e123nilinf, u.e23nilinf, -u.e13nilinf, u.e12nilinf, u.e123nil, -u.e123inf, -u.e3nilinf, u.e2nilinf, u.e23nil, -u.e23inf, -u.e1nilinf, -u.e13nil, u.e13inf, u.e12nil, -u.e12inf, u.e123, -u.enilinf, -u.e3nil, u.e3inf, u.e2nil, -u.e2inf, u.e23, -u.e1nil, u.e1inf, -u.e13, u.e12, -u.enil, u.einf, -u.e3, u.e2, -u.e1, -u.scalar);
}

CGA3 involve(CGA3 u){
    return CGA3(u.scalar, -u.e1, -u.e2, -u.e3, -u.enil, -u.einf, u.e12, u.e13, u.e1nil, u.e1inf, u.e23, u.e2nil, u.e2inf, u.e3nil, u.e3inf, u.enilinf, -u.e123, -u.e12nil, -u.e12inf, -u.e13nil, -u.e13inf, -u.e1nilinf, -u.e23nil, -u.e23inf, -u.e2nilinf, -u.e3nilinf, u.e123nil, u.e123inf, u.e12nilinf, u.e13nilinf, u.e23nilinf, -u.e123nilinf);
}

CGA3 inner(CGA3 u, CGA3 v){
    return CGA3(u.e1*v.e1 - u.e12*v.e12 - u.e123*v.e123 + u.e123inf*v.e123nil + u.e123nil*v.e123inf - u.e123nilinf*v.e123nilinf - u.e12inf*v.e12nil - u.e12nil*v.e12inf - u.e12nilinf*v.e12nilinf - u.e13*v.e13 - u.e13inf*v.e13nil - u.e13nil*v.e13inf - u.e13nilinf*v.e13nilinf - u.e1inf*v.e1nil - u.e1nil*v.e1inf + u.e1nilinf*v.e1nilinf + u.e2*v.e2 - u.e23*v.e23 - u.e23inf*v.e23nil - u.e23nil*v.e23inf - u.e23nilinf*v.e23nilinf - u.e2inf*v.e2nil - u.e2nil*v.e2inf + u.e2nilinf*v.e2nilinf + u.e3*v.e3 - u.e3inf*v.e3nil - u.e3nil*v.e3inf + u.e3nilinf*v.e3nilinf + u.einf*v.enil + u.enil*v.einf + u.enilinf*v.enilinf, u.e12*v.e2 - u.e123*v.e23 - u.e123inf*v.e23nil - u.e123nil*v.e23inf - u.e123nilinf*v.e23nilinf - u.e12inf*v.e2nil - u.e12nil*v.e2inf + u.e12nilinf*v.e2nilinf + u.e13*v.e3 - u.e13inf*v.e3nil - u.e13nil*v.e3inf + u.e13nilinf*v.e3nilinf + u.e1inf*v.enil + u.e1nil*v.einf + u.e1nilinf*v.enilinf - u.e2*v.e12 - u.e23*v.e123 + u.e23inf*v.e123nil + u.e23nil*v.e123inf - u.e23nilinf*v.e123nilinf - u.e2inf*v.e12nil - u.e2nil*v.e12inf - u.e2nilinf*v.e12nilinf - u.e3*v.e13 - u.e3inf*v.e13nil - u.e3nil*v.e13inf - u.e3nilinf*v.e13nilinf - u.einf*v.e1nil - u.enil*v.e1inf + u.enilinf*v.e1nilinf, u.e1*v.e12 - u.e12*v.e1 + u.e123*v.e13 + u.e123inf*v.e13nil + u.e123nil*v.e13inf + u.e123nilinf*v.e13nilinf + u.e12inf*v.e1nil + u.e12nil*v.e1inf - u.e12nilinf*v.e1nilinf + u.e13*v.e123 - u.e13inf*v.e123nil - u.e13nil*v.e123inf + u.e13nilinf*v.e123nilinf + u.e1inf*v.e12nil + u.e1nil*v.e12inf + u.e1nilinf*v.e12nilinf + u.e23*v.e3 - u.e23inf*v.e3nil - u.e23nil*v.e3inf + u.e23nilinf*v.e3nilinf + u.e2inf*v.enil + u.e2nil*v.einf + u.e2nilinf*v.enilinf - u.e3*v.e23 - u.e3inf*v.e23nil - u.e3nil*v.e23inf - u.e3nilinf*v.e23nilinf - u.einf*v.e2nil - u.enil*v.e2inf + u.enilinf*v.e2nilinf, u.e1*v.e13 - u.e12*v.e123 - u.e123*v.e12 - u.e123inf*v.e12nil - u.e123nil*v.e12inf - u.e123nilinf*v.e12nilinf + u.e12inf*v.e123nil + u.e12nil*v.e123inf - u.e12nilinf*v.e123nilinf - u.e13*v.e1 + u.e13inf*v.e1nil + u.e13nil*v.e1inf - u.e13nilinf*v.e1nilinf + u.e1inf*v.e13nil + u.e1nil*v.e13inf + u.e1nilinf*v.e13nilinf + u.e2*v.e23 - u.e23*v.e2 + u.e23inf*v.e2nil + u.e23nil*v.e2inf - u.e23nilinf*v.e2nilinf + u.e2inf*v.e23nil + u.e2nil*v.e23inf + u.e2nilinf*v.e23nilinf + u.e3inf*v.enil + u.e3nil*v.einf + u.e3nilinf*v.enilinf - u.einf*v.e3nil - u.enil*v.e3inf + u.enilinf*v.e3nilinf, u.e1*v.e1nil - u.e12*v.e12nil - u.e123*v.e123nil + u.e123nil*v.e123 - u.e123nil*v.e123nilinf - u.e123nilinf*v.e123nil - u.e12nil*v.e12 + u.e12nil*v.e12nilinf - u.e12nilinf*v.e12nil - u.e13*v.e13nil - u.e13nil*v.e13 + u.e13nil*v.e13nilinf - u.e13nilinf*v.e13nil - u.e1nil*v.e1 + u.e1nil*v.e1nilinf + u.e1nilinf*v.e1nil + u.e2*v.e2nil - u.e23*v.e23nil - u.e23nil*v.e23 + u.e23nil*v.e23nilinf - u.e23nilinf*v.e23nil - u.e2nil*v.e2 + u.e2nil*v.e2nilinf + u.e2nilinf*v.e2nil + u.e3*v.e3nil - u.e3nil*v.e3 + u.e3nil*v.e3nilinf + u.e3nilinf*v.e3nil - u.enil*v.enilinf + u.enilinf*v.enil, u.e1*v.e1inf - u.e12*v.e12inf - u.e123*v.e123inf + u.e123inf*v.e123 + u.e123inf*v.e123nilinf + u.e123nilinf*v.e123inf - u.e12inf*v.e12 - u.e12inf*v.e12nilinf + u.e12nilinf*v.e12inf - u.e13*v.e13inf - u.e13inf*v.e13 - u.e13inf*v.e13nilinf + u.e13nilinf*v.e13inf - u.e1inf*v.e1 - u.e1inf*v.e1nilinf - u.e1nilinf*v.e1inf + u.e2*v.e2inf - u.e23*v.e23inf - u.e23inf*v.e23 - u.e23inf*v.e23nilinf + u.e23nilinf*v.e23inf - u.e2inf*v.e2 - u.e2inf*v.e2nilinf - u.e2nilinf*v.e2inf + u.e3*v.e3inf - u.e3inf*v.e3 - u.e3inf*v.e3nilinf - u.e3nilinf*v.e3inf + u.einf*v.enilinf - u.enilinf*v.einf, u.e123*v.e3 - u.e123inf*v.e3nil - u.e123nil*v.e3inf + u.e123nilinf*v.e3nilinf + u.e12inf*v.enil + u.e12nil*v.einf + u.e12nilinf*v.enilinf + u.e3*v.e123 - u.e3inf*v.e123nil - u.e3nil*v.e123inf + u.e3nilinf*v.e123nilinf + u.einf*v.e12nil + u.enil*v.e12inf + u.enilinf*v.e12nilinf, -u.e123*v.e2 + u.e123inf*v.e2nil + u.e123nil*v.e2inf - u.e123nilinf*v.e2nilinf + u.e13inf*v.enil + u.e13nil*v.einf + u.e13nilinf*v.enilinf - u.e2*v.e123 + u.e2inf*v.e123nil + u.e2nil*v.e123inf - u.e2nilinf*v.e123nilinf + u.einf*v.e13nil + u.enil*v.e13inf + u.enilinf*v.e13nilinf, -u.e123nil*v.e23 - u.e123nilinf*v.e23nil - u.e12nil*v.e2 + u.e12nilinf*v.e2nil - u.e13nil*v.e3 + u.e13nilinf*v.e3nil + u.e1nilinf*v.enil - u.e2*v.e12nil - u.e23*v.e123nil - u.e23nil*v.e123nilinf + u.e2nil*v.e12nilinf - u.e3*v.e13nil + u.e3nil*v.e13nilinf + u.enil*v.e1nilinf, -u.e123inf*v.e23 + u.e123nilinf*v.e23inf - u.e12inf*v.e2 - u.e12nilinf*v.e2inf - u.e13inf*v.e3 - u.e13nilinf*v.e3inf - u.e1nilinf*v.einf - u.e2*v.e12inf - u.e23*v.e123inf + u.e23inf*v.e123nilinf - u.e2inf*v.e12nilinf - u.e3*v.e13inf - u.e3inf*v.e13nilinf - u.einf*v.e1nilinf, u.e1*v.e123 + u.e123*v.e1 - u.e123inf*v.e1nil - u.e123nil*v.e1inf + u.e123nilinf*v.e1nilinf - u.e1inf*v.e123nil - u.e1nil*v.e123inf + u.e1nilinf*v.e123nilinf + u.e23inf*v.enil + u.e23nil*v.einf + u.e23nilinf*v.enilinf + u.einf*v.e23nil + u.enil*v.e23inf + u.enilinf*v.e23nilinf, u.e1*v.e12nil + u.e123nil*v.e13 + u.e123nilinf*v.e13nil + u.e12nil*v.e1 - u.e12nilinf*v.e1nil + u.e13*v.e123nil + u.e13nil*v.e123nilinf - u.e1nil*v.e12nilinf - u.e23nil*v.e3 + u.e23nilinf*v.e3nil + u.e2nilinf*v.enil - u.e3*v.e23nil + u.e3nil*v.e23nilinf + u.enil*v.e2nilinf, u.e1*v.e12inf + u.e123inf*v.e13 - u.e123nilinf*v.e13inf + u.e12inf*v.e1 + u.e12nilinf*v.e1inf + u.e13*v.e123inf - u.e13inf*v.e123nilinf + u.e1inf*v.e12nilinf - u.e23inf*v.e3 - u.e23nilinf*v.e3inf - u.e2nilinf*v.einf - u.e3*v.e23inf - u.e3inf*v.e23nilinf - u.einf*v.e2nilinf, u.e1*v.e13nil - u.e12*v.e123nil - u.e123nil*v.e12 - u.e123nilinf*v.e12nil - u.e12nil*v.e123nilinf + u.e13nil*v.e1 - u.e13nilinf*v.e1nil - u.e1nil*v.e13nilinf + u.e2*v.e23nil + u.e23nil*v.e2 - u.e23nilinf*v.e2nil - u.e2nil*v.e23nilinf + u.e3nilinf*v.enil + u.enil*v.e3nilinf, u.e1*v.e13inf - u.e12*v.e123inf - u.e123inf*v.e12 + u.e123nilinf*v.e12inf + u.e12inf*v.e123nilinf + u.e13inf*v.e1 + u.e13nilinf*v.e1inf + u.e1inf*v.e13nilinf + u.e2*v.e23inf + u.e23inf*v.e2 + u.e23nilinf*v.e2inf + u.e2inf*v.e23nilinf - u.e3nilinf*v.einf - u.einf*v.e3nilinf, u.e1*v.e1nilinf - u.e12*v.e12nilinf - u.e123*v.e123nilinf - u.e123nilinf*v.e123 - u.e12nilinf*v.e12 - u.e13*v.e13nilinf - u.e13nilinf*v.e13 + u.e1nilinf*v.e1 + u.e2*v.e2nilinf - u.e23*v.e23nilinf - u.e23nilinf*v.e23 + u.e2nilinf*v.e2 + u.e3*v.e3nilinf + u.e3nilinf*v.e3, u.e123inf*v.enil + u.e123nil*v.einf + u.e123nilinf*v.enilinf - u.einf*v.e123nil - u.enil*v.e123inf + u.enilinf*v.e123nilinf, -u.e123nil*v.e3 + u.e123nilinf*v.e3nil + u.e12nilinf*v.enil + u.e3*v.e123nil + u.e3nil*v.e123nilinf - u.enil*v.e12nilinf, -u.e123inf*v.e3 - u.e123nilinf*v.e3inf - u.e12nilinf*v.einf + u.e3*v.e123inf - u.e3inf*v.e123nilinf + u.einf*v.e12nilinf, u.e123nil*v.e2 - u.e123nilinf*v.e2nil + u.e13nilinf*v.enil - u.e2*v.e123nil - u.e2nil*v.e123nilinf - u.enil*v.e13nilinf, u.e123inf*v.e2 + u.e123nilinf*v.e2inf - u.e13nilinf*v.einf - u.e2*v.e123inf + u.e2inf*v.e123nilinf + u.einf*v.e13nilinf, -u.e123nilinf*v.e23 + u.e12nilinf*v.e2 + u.e13nilinf*v.e3 - u.e2*v.e12nilinf - u.e23*v.e123nilinf - u.e3*v.e13nilinf, u.e1*v.e123nil - u.e123nil*v.e1 + u.e123nilinf*v.e1nil + u.e1nil*v.e123nilinf + u.e23nilinf*v.enil - u.enil*v.e23nilinf, u.e1*v.e123inf - u.e123inf*v.e1 - u.e123nilinf*v.e1inf - u.e1inf*v.e123nilinf - u.e23nilinf*v.einf + u.einf*v.e23nilinf, u.e1*v.e12nilinf + u.e123nilinf*v.e13 - u.e12nilinf*v.e1 + u.e13*v.e123nilinf + u.e23nilinf*v.e3 - u.e3*v.e23nilinf, u.e1*v.e13nilinf - u.e12*v.e123nilinf - u.e123nilinf*v.e12 - u.e13nilinf*v.e1 + u.e2*v.e23nilinf - u.e23nilinf*v.e2, u.e123nilinf*v.enil + u.enil*v.e123nilinf, -u.e123nilinf*v.einf - u.einf*v.e123nilinf, u.e123nilinf*v.e3 + u.e3*v.e123nilinf, -u.e123nilinf*v.e2 - u.e2*v.e123nilinf, u.e1*v.e123nilinf + u.e123nilinf*v.e1, 0.0);
}

CGA3 lcontract(CGA3 u, CGA3 v){
    return CGA3(u.e1*v.e1 - u.e12*v.e12 - u.e123*v.e123 + u.e123inf*v.e123nil + u.e123nil*v.e123inf - u.e123nilinf*v.e123nilinf - u.e12inf*v.e12nil - u.e12nil*v.e12inf - u.e12nilinf*v.e12nilinf - u.e13*v.e13 - u.e13inf*v.e13nil - u.e13nil*v.e13inf - u.e13nilinf*v.e13nilinf - u.e1inf*v.e1nil - u.e1nil*v.e1inf + u.e1nilinf*v.e1nilinf + u.e2*v.e2 - u.e23*v.e23 - u.e23inf*v.e23nil - u.e23nil*v.e23inf - u.e23nilinf*v.e23nilinf - u.e2inf*v.e2nil - u.e2nil*v.e2inf + u.e2nilinf*v.e2nilinf + u.e3*v.e3 - u.e3inf*v.e3nil - u.e3nil*v.e3inf + u.e3nilinf*v.e3nilinf + u.einf*v.enil + u.enil*v.einf + u.enilinf*v.enilinf + u.scalar*v.scalar, -u.e2*v.e12 - u.e23*v.e123 + u.e23inf*v.e123nil + u.e23nil*v.e123inf - u.e23nilinf*v.e123nilinf - u.e2inf*v.e12nil - u.e2nil*v.e12inf - u.e2nilinf*v.e12nilinf - u.e3*v.e13 - u.e3inf*v.e13nil - u.e3nil*v.e13inf - u.e3nilinf*v.e13nilinf - u.einf*v.e1nil - u.enil*v.e1inf + u.enilinf*v.e1nilinf + u.scalar*v.e1, u.e1*v.e12 + u.e13*v.e123 - u.e13inf*v.e123nil - u.e13nil*v.e123inf + u.e13nilinf*v.e123nilinf + u.e1inf*v.e12nil + u.e1nil*v.e12inf + u.e1nilinf*v.e12nilinf - u.e3*v.e23 - u.e3inf*v.e23nil - u.e3nil*v.e23inf - u.e3nilinf*v.e23nilinf - u.einf*v.e2nil - u.enil*v.e2inf + u.enilinf*v.e2nilinf + u.scalar*v.e2, u.e1*v.e13 - u.e12*v.e123 + u.e12inf*v.e123nil + u.e12nil*v.e123inf - u.e12nilinf*v.e123nilinf + u.e1inf*v.e13nil + u.e1nil*v.e13inf + u.e1nilinf*v.e13nilinf + u.e2*v.e23 + u.e2inf*v.e23nil + u.e2nil*v.e23inf + u.e2nilinf*v.e23nilinf - u.einf*v.e3nil - u.enil*v.e3inf + u.enilinf*v.e3nilinf + u.scalar*v.e3, u.e1*v.e1nil - u.e12*v.e12nil - u.e123*v.e123nil - u.e123nil*v.e123nilinf + u.e12nil*v.e12nilinf - u.e13*v.e13nil + u.e13nil*v.e13nilinf + u.e1nil*v.e1nilinf + u.e2*v.e2nil - u.e23*v.e23nil + u.e23nil*v.e23nilinf + u.e2nil*v.e2nilinf + u.e3*v.e3nil + u.e3nil*v.e3nilinf - u.enil*v.enilinf + u.scalar*v.enil, u.e1*v.e1inf - u.e12*v.e12inf - u.e123*v.e123inf + u.e123inf*v.e123nilinf - u.e12inf*v.e12nilinf - u.e13*v.e13inf - u.e13inf*v.e13nilinf - u.e1inf*v.e1nilinf + u.e2*v.e2inf - u.e23*v.e23inf - u.e23inf*v.e23nilinf - u.e2inf*v.e2nilinf + u.e3*v.e3inf - u.e3inf*v.e3nilinf + u.einf*v.enilinf + u.scalar*v.einf, u.e3*v.e123 - u.e3inf*v.e123nil - u.e3nil*v.e123inf + u.e3nilinf*v.e123nilinf + u.einf*v.e12nil + u.enil*v.e12inf + u.enilinf*v.e12nilinf + u.scalar*v.e12, -u.e2*v.e123 + u.e2inf*v.e123nil + u.e2nil*v.e123inf - u.e2nilinf*v.e123nilinf + u.einf*v.e13nil + u.enil*v.e13inf + u.enilinf*v.e13nilinf + u.scalar*v.e13, -u.e2*v.e12nil - u.e23*v.e123nil - u.e23nil*v.e123nilinf + u.e2nil*v.e12nilinf - u.e3*v.e13nil + u.e3nil*v.e13nilinf + u.enil*v.e1nilinf + u.scalar*v.e1nil, -u.e2*v.e12inf - u.e23*v.e123inf + u.e23inf*v.e123nilinf - u.e2inf*v.e12nilinf - u.e3*v.e13inf - u.e3inf*v.e13nilinf - u.einf*v.e1nilinf + u.scalar*v.e1inf, u.e1*v.e123 - u.e1inf*v.e123nil - u.e1nil*v.e123inf + u.e1nilinf*v.e123nilinf + u.einf*v.e23nil + u.enil*v.e23inf + u.enilinf*v.e23nilinf + u.scalar*v.e23, u.e1*v.e12nil + u.e13*v.e123nil + u.e13nil*v.e123nilinf - u.e1nil*v.e12nilinf - u.e3*v.e23nil + u.e3nil*v.e23nilinf + u.enil*v.e2nilinf + u.scalar*v.e2nil, u.e1*v.e12inf + u.e13*v.e123inf - u.e13inf*v.e123nilinf + u.e1inf*v.e12nilinf - u.e3*v.e23inf - u.e3inf*v.e23nilinf - u.einf*v.e2nilinf + u.scalar*v.e2inf, u.e1*v.e13nil - u.e12*v.e123nil - u.e12nil*v.e123nilinf - u.e1nil*v.e13nilinf + u.e2*v.e23nil - u.e2nil*v.e23nilinf + u.enil*v.e3nilinf + u.scalar*v.e3nil, u.e1*v.e13inf - u.e12*v.e123inf + u.e12inf*v.e123nilinf + u.e1inf*v.e13nilinf + u.e2*v.e23inf + u.e2inf*v.e23nilinf - u.einf*v.e3nilinf + u.scalar*v.e3inf, u.e1*v.e1nilinf - u.e12*v.e12nilinf - u.e123*v.e123nilinf - u.e13*v.e13nilinf + u.e2*v.e2nilinf - u.e23*v.e23nilinf + u.e3*v.e3nilinf + u.scalar*v.enilinf, -u.einf*v.e123nil - u.enil*v.e123inf + u.enilinf*v.e123nilinf + u.scalar*v.e123, u.e3*v.e123nil + u.e3nil*v.e123nilinf - u.enil*v.e12nilinf + u.scalar*v.e12nil, u.e3*v.e123inf - u.e3inf*v.e123nilinf + u.einf*v.e12nilinf + u.scalar*v.e12inf, -u.e2*v.e123nil - u.e2nil*v.e123nilinf - u.enil*v.e13nilinf + u.scalar*v.e13nil, -u.e2*v.e123inf + u.e2inf*v.e123nilinf + u.einf*v.e13nilinf + u.scalar*v.e13inf, -u.e2*v.e12nilinf - u.e23*v.e123nilinf - u.e3*v.e13nilinf + u.scalar*v.e1nilinf, u.e1*v.e123nil + u.e1nil*v.e123nilinf - u.enil*v.e23nilinf + u.scalar*v.e23nil, u.e1*v.e123inf - u.e1inf*v.e123nilinf + u.einf*v.e23nilinf + u.scalar*v.e23inf, u.e1*v.e12nilinf + u.e13*v.e123nilinf - u.e3*v.e23nilinf + u.scalar*v.e2nilinf, u.e1*v.e13nilinf - u.e12*v.e123nilinf + u.e2*v.e23nilinf + u.scalar*v.e3nilinf, u.enil*v.e123nilinf + u.scalar*v.e123nil, -u.einf*v.e123nilinf + u.scalar*v.e123inf, u.e3*v.e123nilinf + u.scalar*v.e12nilinf, -u.e2*v.e123nilinf + u.scalar*v.e13nilinf, u.e1*v.e123nilinf + u.scalar*v.e23nilinf, u.scalar*v.e123nilinf);
}

CGA3 outer(CGA3 u, CGA3 v){
    return CGA3(u.scalar*v.scalar, u.e1*v.scalar + u.scalar*v.e1, u.e2*v.scalar + u.scalar*v.e2, u.e3*v.scalar + u.scalar*v.e3, u.enil*v.scalar + u.scalar*v.enil, u.einf*v.scalar + u.scalar*v.einf, u.e1*v.e2 + u.e12*v.scalar - u.e2*v.e1 + u.scalar*v.e12, u.e1*v.e3 + u.e13*v.scalar - u.e3*v.e1 + u.scalar*v.e13, u.e1*v.enil + u.e1nil*v.scalar - u.enil*v.e1 + u.scalar*v.e1nil, u.e1*v.einf + u.e1inf*v.scalar - u.einf*v.e1 + u.scalar*v.e1inf, u.e2*v.e3 + u.e23*v.scalar - u.e3*v.e2 + u.scalar*v.e23, u.e2*v.enil + u.e2nil*v.scalar - u.enil*v.e2 + u.scalar*v.e2nil, u.e2*v.einf + u.e2inf*v.scalar - u.einf*v.e2 + u.scalar*v.e2inf, u.e3*v.enil + u.e3nil*v.scalar - u.enil*v.e3 + u.scalar*v.e3nil, u.e3*v.einf + u.e3inf*v.scalar - u.einf*v.e3 + u.scalar*v.e3inf, -u.einf*v.enil + u.enil*v.einf + u.enilinf*v.scalar + u.scalar*v.enilinf, u.e1*v.e23 + u.e12*v.e3 + u.e123*v.scalar - u.e13*v.e2 - u.e2*v.e13 + u.e23*v.e1 + u.e3*v.e12 + u.scalar*v.e123, u.e1*v.e2nil + u.e12*v.enil + u.e12nil*v.scalar - u.e1nil*v.e2 - u.e2*v.e1nil + u.e2nil*v.e1 + u.enil*v.e12 + u.scalar*v.e12nil, u.e1*v.e2inf + u.e12*v.einf + u.e12inf*v.scalar - u.e1inf*v.e2 - u.e2*v.e1inf + u.e2inf*v.e1 + u.einf*v.e12 + u.scalar*v.e12inf, u.e1*v.e3nil + u.e13*v.enil + u.e13nil*v.scalar - u.e1nil*v.e3 - u.e3*v.e1nil + u.e3nil*v.e1 + u.enil*v.e13 + u.scalar*v.e13nil, u.e1*v.e3inf + u.e13*v.einf + u.e13inf*v.scalar - u.e1inf*v.e3 - u.e3*v.e1inf + u.e3inf*v.e1 + u.einf*v.e13 + u.scalar*v.e13inf, u.e1*v.enilinf - u.e1inf*v.enil + u.e1nil*v.einf + u.e1nilinf*v.scalar + u.einf*v.e1nil - u.enil*v.e1inf + u.enilinf*v.e1 + u.scalar*v.e1nilinf, u.e2*v.e3nil + u.e23*v.enil + u.e23nil*v.scalar - u.e2nil*v.e3 - u.e3*v.e2nil + u.e3nil*v.e2 + u.enil*v.e23 + u.scalar*v.e23nil, u.e2*v.e3inf + u.e23*v.einf + u.e23inf*v.scalar - u.e2inf*v.e3 - u.e3*v.e2inf + u.e3inf*v.e2 + u.einf*v.e23 + u.scalar*v.e23inf, u.e2*v.enilinf - u.e2inf*v.enil + u.e2nil*v.einf + u.e2nilinf*v.scalar + u.einf*v.e2nil - u.enil*v.e2inf + u.enilinf*v.e2 + u.scalar*v.e2nilinf, u.e3*v.enilinf - u.e3inf*v.enil + u.e3nil*v.einf + u.e3nilinf*v.scalar + u.einf*v.e3nil - u.enil*v.e3inf + u.enilinf*v.e3 + u.scalar*v.e3nilinf, u.e1*v.e23nil + u.e12*v.e3nil + u.e123*v.enil + u.e123nil*v.scalar - u.e12nil*v.e3 - u.e13*v.e2nil + u.e13nil*v.e2 + u.e1nil*v.e23 - u.e2*v.e13nil + u.e23*v.e1nil - u.e23nil*v.e1 - u.e2nil*v.e13 + u.e3*v.e12nil + u.e3nil*v.e12 - u.enil*v.e123 + u.scalar*v.e123nil, u.e1*v.e23inf + u.e12*v.e3inf + u.e123*v.einf + u.e123inf*v.scalar - u.e12inf*v.e3 - u.e13*v.e2inf + u.e13inf*v.e2 + u.e1inf*v.e23 - u.e2*v.e13inf + u.e23*v.e1inf - u.e23inf*v.e1 - u.e2inf*v.e13 + u.e3*v.e12inf + u.e3inf*v.e12 - u.einf*v.e123 + u.scalar*v.e123inf, u.e1*v.e2nilinf + u.e12*v.enilinf - u.e12inf*v.enil + u.e12nil*v.einf + u.e12nilinf*v.scalar + u.e1inf*v.e2nil - u.e1nil*v.e2inf + u.e1nilinf*v.e2 - u.e2*v.e1nilinf - u.e2inf*v.e1nil + u.e2nil*v.e1inf - u.e2nilinf*v.e1 - u.einf*v.e12nil + u.enil*v.e12inf + u.enilinf*v.e12 + u.scalar*v.e12nilinf, u.e1*v.e3nilinf + u.e13*v.enilinf - u.e13inf*v.enil + u.e13nil*v.einf + u.e13nilinf*v.scalar + u.e1inf*v.e3nil - u.e1nil*v.e3inf + u.e1nilinf*v.e3 - u.e3*v.e1nilinf - u.e3inf*v.e1nil + u.e3nil*v.e1inf - u.e3nilinf*v.e1 - u.einf*v.e13nil + u.enil*v.e13inf + u.enilinf*v.e13 + u.scalar*v.e13nilinf, u.e2*v.e3nilinf + u.e23*v.enilinf - u.e23inf*v.enil + u.e23nil*v.einf + u.e23nilinf*v.scalar + u.e2inf*v.e3nil - u.e2nil*v.e3inf + u.e2nilinf*v.e3 - u.e3*v.e2nilinf - u.e3inf*v.e2nil + u.e3nil*v.e2inf - u.e3nilinf*v.e2 - u.einf*v.e23nil + u.enil*v.e23inf + u.enilinf*v.e23 + u.scalar*v.e23nilinf, u.e1*v.e23nilinf + u.e12*v.e3nilinf + u.e123*v.enilinf - u.e123inf*v.enil + u.e123nil*v.einf + u.e123nilinf*v.scalar + u.e12inf*v.e3nil - u.e12nil*v.e3inf + u.e12nilinf*v.e3 - u.e13*v.e2nilinf - u.e13inf*v.e2nil + u.e13nil*v.e2inf - u.e13nilinf*v.e2 - u.e1inf*v.e23nil + u.e1nil*v.e23inf + u.e1nilinf*v.e23 - u.e2*v.e13nilinf + u.e23*v.e1nilinf + u.e23inf*v.e1nil - u.e23nil*v.e1inf + u.e23nilinf*v.e1 + u.e2inf*v.e13nil - u.e2nil*v.e13inf - u.e2nilinf*v.e13 + u.e3*v.e12nilinf - u.e3inf*v.e12nil + u.e3nil*v.e12inf + u.e3nilinf*v.e12 + u.einf*v.e123nil - u.enil*v.e123inf + u.enilinf*v.e123 + u.scalar*v.e123nilinf);
}

CGA3 I(){
    return CGA3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
}

CGA3 rcontract(CGA3 u, CGA3 v){
    return CGA3(u.e1*v.e1 - u.e12*v.e12 - u.e123*v.e123 + u.e123inf*v.e123nil + u.e123nil*v.e123inf - u.e123nilinf*v.e123nilinf - u.e12inf*v.e12nil - u.e12nil*v.e12inf - u.e12nilinf*v.e12nilinf - u.e13*v.e13 - u.e13inf*v.e13nil - u.e13nil*v.e13inf - u.e13nilinf*v.e13nilinf - u.e1inf*v.e1nil - u.e1nil*v.e1inf + u.e1nilinf*v.e1nilinf + u.e2*v.e2 - u.e23*v.e23 - u.e23inf*v.e23nil - u.e23nil*v.e23inf - u.e23nilinf*v.e23nilinf - u.e2inf*v.e2nil - u.e2nil*v.e2inf + u.e2nilinf*v.e2nilinf + u.e3*v.e3 - u.e3inf*v.e3nil - u.e3nil*v.e3inf + u.e3nilinf*v.e3nilinf + u.einf*v.enil + u.enil*v.einf + u.enilinf*v.enilinf + u.scalar*v.scalar, u.e1*v.scalar + u.e12*v.e2 - u.e123*v.e23 - u.e123inf*v.e23nil - u.e123nil*v.e23inf - u.e123nilinf*v.e23nilinf - u.e12inf*v.e2nil - u.e12nil*v.e2inf + u.e12nilinf*v.e2nilinf + u.e13*v.e3 - u.e13inf*v.e3nil - u.e13nil*v.e3inf + u.e13nilinf*v.e3nilinf + u.e1inf*v.enil + u.e1nil*v.einf + u.e1nilinf*v.enilinf, -u.e12*v.e1 + u.e123*v.e13 + u.e123inf*v.e13nil + u.e123nil*v.e13inf + u.e123nilinf*v.e13nilinf + u.e12inf*v.e1nil + u.e12nil*v.e1inf - u.e12nilinf*v.e1nilinf + u.e2*v.scalar + u.e23*v.e3 - u.e23inf*v.e3nil - u.e23nil*v.e3inf + u.e23nilinf*v.e3nilinf + u.e2inf*v.enil + u.e2nil*v.einf + u.e2nilinf*v.enilinf, -u.e123*v.e12 - u.e123inf*v.e12nil - u.e123nil*v.e12inf - u.e123nilinf*v.e12nilinf - u.e13*v.e1 + u.e13inf*v.e1nil + u.e13nil*v.e1inf - u.e13nilinf*v.e1nilinf - u.e23*v.e2 + u.e23inf*v.e2nil + u.e23nil*v.e2inf - u.e23nilinf*v.e2nilinf + u.e3*v.scalar + u.e3inf*v.enil + u.e3nil*v.einf + u.e3nilinf*v.enilinf, u.e123nil*v.e123 - u.e123nilinf*v.e123nil - u.e12nil*v.e12 - u.e12nilinf*v.e12nil - u.e13nil*v.e13 - u.e13nilinf*v.e13nil - u.e1nil*v.e1 + u.e1nilinf*v.e1nil - u.e23nil*v.e23 - u.e23nilinf*v.e23nil - u.e2nil*v.e2 + u.e2nilinf*v.e2nil - u.e3nil*v.e3 + u.e3nilinf*v.e3nil + u.enil*v.scalar + u.enilinf*v.enil, u.e123inf*v.e123 + u.e123nilinf*v.e123inf - u.e12inf*v.e12 + u.e12nilinf*v.e12inf - u.e13inf*v.e13 + u.e13nilinf*v.e13inf - u.e1inf*v.e1 - u.e1nilinf*v.e1inf - u.e23inf*v.e23 + u.e23nilinf*v.e23inf - u.e2inf*v.e2 - u.e2nilinf*v.e2inf - u.e3inf*v.e3 - u.e3nilinf*v.e3inf + u.einf*v.scalar - u.enilinf*v.einf, u.e12*v.scalar + u.e123*v.e3 - u.e123inf*v.e3nil - u.e123nil*v.e3inf + u.e123nilinf*v.e3nilinf + u.e12inf*v.enil + u.e12nil*v.einf + u.e12nilinf*v.enilinf, -u.e123*v.e2 + u.e123inf*v.e2nil + u.e123nil*v.e2inf - u.e123nilinf*v.e2nilinf + u.e13*v.scalar + u.e13inf*v.enil + u.e13nil*v.einf + u.e13nilinf*v.enilinf, -u.e123nil*v.e23 - u.e123nilinf*v.e23nil - u.e12nil*v.e2 + u.e12nilinf*v.e2nil - u.e13nil*v.e3 + u.e13nilinf*v.e3nil + u.e1nil*v.scalar + u.e1nilinf*v.enil, -u.e123inf*v.e23 + u.e123nilinf*v.e23inf - u.e12inf*v.e2 - u.e12nilinf*v.e2inf - u.e13inf*v.e3 - u.e13nilinf*v.e3inf + u.e1inf*v.scalar - u.e1nilinf*v.einf, u.e123*v.e1 - u.e123inf*v.e1nil - u.e123nil*v.e1inf + u.e123nilinf*v.e1nilinf + u.e23*v.scalar + u.e23inf*v.enil + u.e23nil*v.einf + u.e23nilinf*v.enilinf, u.e123nil*v.e13 + u.e123nilinf*v.e13nil + u.e12nil*v.e1 - u.e12nilinf*v.e1nil - u.e23nil*v.e3 + u.e23nilinf*v.e3nil + u.e2nil*v.scalar + u.e2nilinf*v.enil, u.e123inf*v.e13 - u.e123nilinf*v.e13inf + u.e12inf*v.e1 + u.e12nilinf*v.e1inf - u.e23inf*v.e3 - u.e23nilinf*v.e3inf + u.e2inf*v.scalar - u.e2nilinf*v.einf, -u.e123nil*v.e12 - u.e123nilinf*v.e12nil + u.e13nil*v.e1 - u.e13nilinf*v.e1nil + u.e23nil*v.e2 - u.e23nilinf*v.e2nil + u.e3nil*v.scalar + u.e3nilinf*v.enil, -u.e123inf*v.e12 + u.e123nilinf*v.e12inf + u.e13inf*v.e1 + u.e13nilinf*v.e1inf + u.e23inf*v.e2 + u.e23nilinf*v.e2inf + u.e3inf*v.scalar - u.e3nilinf*v.einf, -u.e123nilinf*v.e123 - u.e12nilinf*v.e12 - u.e13nilinf*v.e13 + u.e1nilinf*v.e1 - u.e23nilinf*v.e23 + u.e2nilinf*v.e2 + u.e3nilinf*v.e3 + u.enilinf*v.scalar, u.e123*v.scalar + u.e123inf*v.enil + u.e123nil*v.einf + u.e123nilinf*v.enilinf, -u.e123nil*v.e3 + u.e123nilinf*v.e3nil + u.e12nil*v.scalar + u.e12nilinf*v.enil, -u.e123inf*v.e3 - u.e123nilinf*v.e3inf + u.e12inf*v.scalar - u.e12nilinf*v.einf, u.e123nil*v.e2 - u.e123nilinf*v.e2nil + u.e13nil*v.scalar + u.e13nilinf*v.enil, u.e123inf*v.e2 + u.e123nilinf*v.e2inf + u.e13inf*v.scalar - u.e13nilinf*v.einf, -u.e123nilinf*v.e23 + u.e12nilinf*v.e2 + u.e13nilinf*v.e3 + u.e1nilinf*v.scalar, -u.e123nil*v.e1 + u.e123nilinf*v.e1nil + u.e23nil*v.scalar + u.e23nilinf*v.enil, -u.e123inf*v.e1 - u.e123nilinf*v.e1inf + u.e23inf*v.scalar - u.e23nilinf*v.einf, u.e123nilinf*v.e13 - u.e12nilinf*v.e1 + u.e23nilinf*v.e3 + u.e2nilinf*v.scalar, -u.e123nilinf*v.e12 - u.e13nilinf*v.e1 - u.e23nilinf*v.e2 + u.e3nilinf*v.scalar, u.e123nil*v.scalar + u.e123nilinf*v.enil, u.e123inf*v.scalar - u.e123nilinf*v.einf, u.e123nilinf*v.e3 + u.e12nilinf*v.scalar, -u.e123nilinf*v.e2 + u.e13nilinf*v.scalar, u.e123nilinf*v.e1 + u.e23nilinf*v.scalar, u.e123nilinf*v.scalar);
}

CGA3 reverse(CGA3 u){
    return CGA3(u.scalar, u.e1, u.e2, u.e3, u.enil, u.einf, -u.e12, -u.e13, -u.e1nil, -u.e1inf, -u.e23, -u.e2nil, -u.e2inf, -u.e3nil, -u.e3inf, -u.enilinf, -u.e123, -u.e12nil, -u.e12inf, -u.e13nil, -u.e13inf, -u.e1nilinf, -u.e23nil, -u.e23inf, -u.e2nilinf, -u.e3nilinf, u.e123nil, u.e123inf, u.e12nilinf, u.e13nilinf, u.e23nilinf, u.e123nilinf);
}

CGA3 conjugate(CGA3 u){
    return reverse(involve(u));
}

CGA3 outer(CGA3 u, CGA3 v, CGA3 w){
    return outer(outer(u, v), w);
}

CGA3 INF(){
    return CGA3(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

CGA3 MNK(){
    return CGA3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

CGA3 NIL(){
    return CGA3(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

CGA3 point(CGA3 u){
    return CGA3(-u.e12*u.e12nil - u.e123nil*u.e123nilinf - u.e13*u.e13nil + u.e1nil*u.e1nilinf - u.e23*u.e23nil + u.e2nil*u.e2nilinf + u.e3nil*u.e3nilinf + u.scalar, u.e1 - u.e123nil*u.e23 - u.e123nilinf*u.e23nil - u.e12nil*u.e2 + u.e12nilinf*u.e2nil - u.e13nil*u.e3 + u.e13nilinf*u.e3nil + u.e1nilinf*u.enil, u.e1*u.e12nil + u.e123nil*u.e13 + u.e123nilinf*u.e13nil - u.e12nilinf*u.e1nil + u.e2 - u.e23nil*u.e3 + u.e23nilinf*u.e3nil + u.e2nilinf*u.enil, u.e1*u.e13nil - u.e12*u.e123nil - u.e123nilinf*u.e12nil - u.e13nilinf*u.e1nil + u.e2*u.e23nil - u.e23nilinf*u.e2nil + u.e3 + u.e3nilinf*u.enil, u.enil + 1.0, 0.5*pow(u.e1, 2.0) - u.e1*u.e1nilinf - 0.5*pow(u.e12, 2.0) + u.e12*u.e12nilinf - 0.5*pow(u.e123, 2.0) + u.e123*u.e123nilinf + u.e123inf*u.e123nil - 0.5*pow(u.e123nilinf, 2.0) - u.e12inf*u.e12nil - 0.5*pow(u.e12nilinf, 2.0) - 0.5*pow(u.e13, 2.0) + u.e13*u.e13nilinf - u.e13inf*u.e13nil - 0.5*pow(u.e13nilinf, 2.0) - u.e1inf*u.e1nil + 0.5*pow(u.e1nilinf, 2.0) + 0.5*pow(u.e2, 2.0) - u.e2*u.e2nilinf - 0.5*pow(u.e23, 2.0) + u.e23*u.e23nilinf - u.e23inf*u.e23nil - 0.5*pow(u.e23nilinf, 2.0) - u.e2inf*u.e2nil + 0.5*pow(u.e2nilinf, 2.0) + 0.5*pow(u.e3, 2.0) - u.e3*u.e3nilinf - u.e3inf*u.e3nil + 0.5*pow(u.e3nilinf, 2.0) + u.einf*u.enil + u.einf + 0.5*pow(u.enilinf, 2.0), u.e12 + u.e123nilinf*u.e3nil, -u.e123nilinf*u.e2nil + u.e13, u.e1nil, -u.e123*u.e23 + u.e123nilinf*u.e23 - u.e123nilinf*u.e23nilinf - u.e12inf*u.e2nil - u.e12nil*u.e2inf - u.e13inf*u.e3nil - u.e13nil*u.e3inf + u.e1inf + u.e1nilinf*u.enilinf, u.e123nilinf*u.e1nil + u.e23, u.e2nil, u.e123*u.e13 - u.e123nilinf*u.e13 + u.e123nilinf*u.e13nilinf + u.e12inf*u.e1nil + u.e12nil*u.e1inf - u.e23inf*u.e3nil - u.e23nil*u.e3inf + u.e2inf + u.e2nilinf*u.enilinf, u.e3nil, -u.e12*u.e123 + u.e12*u.e123nilinf - u.e123nilinf*u.e12nilinf + u.e13inf*u.e1nil + u.e13nil*u.e1inf + u.e23inf*u.e2nil + u.e23nil*u.e2inf + u.e3inf + u.e3nilinf*u.enilinf, -u.e12*u.e12nil - u.e123nil*u.e123nilinf - u.e13*u.e13nil + u.e1nil*u.e1nilinf - u.e23*u.e23nil + u.e2nil*u.e2nilinf + u.e3nil*u.e3nilinf + u.enilinf, u.e123 + u.e123nilinf*u.enil, u.e12nil, u.e123*u.e3 - u.e123inf*u.e3nil - u.e123nil*u.e3inf - u.e123nilinf*u.e3 + u.e123nilinf*u.e3nilinf + u.e12inf*u.enil + u.e12inf + u.e12nil*u.einf + u.e12nilinf*u.enilinf, u.e13nil, -u.e123*u.e2 + u.e123inf*u.e2nil + u.e123nil*u.e2inf + u.e123nilinf*u.e2 - u.e123nilinf*u.e2nilinf + u.e13inf*u.enil + u.e13inf + u.e13nil*u.einf + u.e13nilinf*u.enilinf, -u.e123nil*u.e23 - u.e123nilinf*u.e23nil - u.e12nil*u.e2 + u.e12nilinf*u.e2nil - u.e13nil*u.e3 + u.e13nilinf*u.e3nil + u.e1nilinf*u.enil + u.e1nilinf, u.e23nil, u.e1*u.e123 - u.e1*u.e123nilinf - u.e123inf*u.e1nil - u.e123nil*u.e1inf + u.e123nilinf*u.e1nilinf + u.e23inf*u.enil + u.e23inf + u.e23nil*u.einf + u.e23nilinf*u.enilinf, u.e1*u.e12nil + u.e123nil*u.e13 + u.e123nilinf*u.e13nil - u.e12nilinf*u.e1nil - u.e23nil*u.e3 + u.e23nilinf*u.e3nil + u.e2nilinf*u.enil + u.e2nilinf, u.e1*u.e13nil - u.e12*u.e123nil - u.e123nilinf*u.e12nil - u.e13nilinf*u.e1nil + u.e2*u.e23nil - u.e23nilinf*u.e2nil + u.e3nilinf*u.enil + u.e3nilinf, u.e123nil, u.e123inf + u.e123nilinf*u.enilinf, u.e123nilinf*u.e3nil + u.e12nilinf, -u.e123nilinf*u.e2nil + u.e13nilinf, u.e123nilinf*u.e1nil + u.e23nilinf, u.e123nilinf*u.enil + u.e123nilinf);
}

CGA3 point_coords(CGA3 u){
    return CGA3(u.scalar/u.enil, u.e1/u.enil, u.e2/u.enil, u.e3/u.enil, 1.0, u.einf/u.enil, u.e12/u.enil, u.e13/u.enil, u.e1nil/u.enil, u.e1inf/u.enil, u.e23/u.enil, u.e2nil/u.enil, u.e2inf/u.enil, u.e3nil/u.enil, u.e3inf/u.enil, u.enilinf/u.enil, u.e123/u.enil, u.e12nil/u.enil, u.e12inf/u.enil, u.e13nil/u.enil, u.e13inf/u.enil, u.e1nilinf/u.enil, u.e23nil/u.enil, u.e23inf/u.enil, u.e2nilinf/u.enil, u.e3nilinf/u.enil, u.e123nil/u.enil, u.e123inf/u.enil, u.e12nilinf/u.enil, u.e13nilinf/u.enil, u.e23nilinf/u.enil, u.e123nilinf/u.enil);
}

// injectOneBlade.glsl
/* Inject array v at the indices for e1, e2, e3 in array u */
void injectOneBladeArray(inout float u[32], float v[3]){
    u[I_CGA3_e1] = v[0];
    u[I_CGA3_e2] = v[1];
    u[I_CGA3_e3] = v[2];
}

/* Inject array v into e1, e2, e3 of struct u */
CGA3 injectOneBlade(CGA3 u, float v[3]){
    float u_ary[32];
    toArray(u, u_ary);
    injectOneBladeArray(u_ary, v);
    return fromArray(u_ary);
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

CGA3 inverse(CGA3 x) {
  return mul(1.0/mul(x, reverse(x)).scalar, reverse(x));
}

CGA3 div(CGA3 a, CGA3 b) {
  return mul(a, inverse(b));
}

CGA3 weight(vec3 w) {
  // return one();
  return inverse(fromVec(normalize(w)));
  // return inverse(mul(fromVec(p1-p2), fromVec(w)));
}

CGA3 bilinearQuad(
  vec3 p0, vec3 p1, vec3 p2, vec3 p3,
  vec3 w0, vec3 w1, vec3 w2, vec3 w3,
  float u, float v) {
    CGA3 W0 = mul((1.0-u)*(1.0-v),  weight(w0)); // 0, 0
    CGA3 W1 = mul(u*(1.0-v),        weight(w1)); // 1, 0
    CGA3 W2 = mul((1.0-u)*v,        weight(w2)); // 0, 1
    CGA3 W3 = mul(u*v,              weight(w3)); // 1, 1
    CGA3 top = add(
      mul(fromVec(p0), W0),
      mul(fromVec(p1), W1),
      mul(fromVec(p2), W2),
      mul(fromVec(p3), W3)
    );
    CGA3 bottom = add(
      W0,
      W1,
      W2,
      W3
    );
    return div(top, bottom);
    // vec3 q0 = p0 * (1.0 - u) + p1 * u;
    // vec3 q1 = p2 * (1.0 - u) + p3 * u;
    // vec3 x = q0 * (1.0 - v) + q1*v;

    // return fromVec(x);
}