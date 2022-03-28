#define GLSLIFY 1
// CGA3.glsl
const int Idx_CGA3_scalar = 0;
const int Idx_CGA3_e1 = 1;
const int Idx_CGA3_e2 = 2;
const int Idx_CGA3_e3 = 3;
const int Idx_CGA3_enil = 4;
const int Idx_CGA3_einf = 5;
const int Idx_CGA3_e12 = 6;
const int Idx_CGA3_e13 = 7;
const int Idx_CGA3_e1nil = 8;
const int Idx_CGA3_e1inf = 9;
const int Idx_CGA3_e23 = 10;
const int Idx_CGA3_e2nil = 11;
const int Idx_CGA3_e2inf = 12;
const int Idx_CGA3_e3nil = 13;
const int Idx_CGA3_e3inf = 14;
const int Idx_CGA3_enilinf = 15;
const int Idx_CGA3_e123 = 16;
const int Idx_CGA3_e12nil = 17;
const int Idx_CGA3_e12inf = 18;
const int Idx_CGA3_e13nil = 19;
const int Idx_CGA3_e13inf = 20;
const int Idx_CGA3_e1nilinf = 21;
const int Idx_CGA3_e23nil = 22;
const int Idx_CGA3_e23inf = 23;
const int Idx_CGA3_e2nilinf = 24;
const int Idx_CGA3_e3nilinf = 25;
const int Idx_CGA3_e123nil = 26;
const int Idx_CGA3_e123inf = 27;
const int Idx_CGA3_e12nilinf = 28;
const int Idx_CGA3_e13nilinf = 29;
const int Idx_CGA3_e23nilinf = 30;
const int Idx_CGA3_e123nilinf = 31;

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

CGA3 fromArray(float X[32]){
    return CGA3(X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8], X[9], X[10], X[11], X[12], X[13], X[14], X[15], X[16], X[17], X[18], X[19], X[20], X[21], X[22], X[23], X[24], X[25], X[26], X[27], X[28], X[29], X[30], X[31]);
}

void toArray(CGA3 X, inout float X_ary[32]){
    X_ary[0] = X.scalar;
    X_ary[1] = X.e1;
    X_ary[2] = X.e2;
    X_ary[3] = X.e3;
    X_ary[4] = X.enil;
    X_ary[5] = X.einf;
    X_ary[6] = X.e12;
    X_ary[7] = X.e13;
    X_ary[8] = X.e1nil;
    X_ary[9] = X.e1inf;
    X_ary[10] = X.e23;
    X_ary[11] = X.e2nil;
    X_ary[12] = X.e2inf;
    X_ary[13] = X.e3nil;
    X_ary[14] = X.e3inf;
    X_ary[15] = X.enilinf;
    X_ary[16] = X.e123;
    X_ary[17] = X.e12nil;
    X_ary[18] = X.e12inf;
    X_ary[19] = X.e13nil;
    X_ary[20] = X.e13inf;
    X_ary[21] = X.e1nilinf;
    X_ary[22] = X.e23nil;
    X_ary[23] = X.e23inf;
    X_ary[24] = X.e2nilinf;
    X_ary[25] = X.e3nilinf;
    X_ary[26] = X.e123nil;
    X_ary[27] = X.e123inf;
    X_ary[28] = X.e12nilinf;
    X_ary[29] = X.e13nilinf;
    X_ary[30] = X.e23nilinf;
    X_ary[31] = X.e123nilinf;
}

void zero(inout float X[32]){
    X[0] = 0.0;
    X[1] = 0.0;
    X[2] = 0.0;
    X[3] = 0.0;
    X[4] = 0.0;
    X[5] = 0.0;
    X[6] = 0.0;
    X[7] = 0.0;
    X[8] = 0.0;
    X[9] = 0.0;
    X[10] = 0.0;
    X[11] = 0.0;
    X[12] = 0.0;
    X[13] = 0.0;
    X[14] = 0.0;
    X[15] = 0.0;
    X[16] = 0.0;
    X[17] = 0.0;
    X[18] = 0.0;
    X[19] = 0.0;
    X[20] = 0.0;
    X[21] = 0.0;
    X[22] = 0.0;
    X[23] = 0.0;
    X[24] = 0.0;
    X[25] = 0.0;
    X[26] = 0.0;
    X[27] = 0.0;
    X[28] = 0.0;
    X[29] = 0.0;
    X[30] = 0.0;
    X[31] = 0.0;
}

CGA3 add(CGA3 X, CGA3 Y){
    return CGA3(X.scalar + Y.scalar, X.e1 + Y.e1, X.e2 + Y.e2, X.e3 + Y.e3, X.enil + Y.enil, X.einf + Y.einf, X.e12 + Y.e12, X.e13 + Y.e13, X.e1nil + Y.e1nil, X.e1inf + Y.e1inf, X.e23 + Y.e23, X.e2nil + Y.e2nil, X.e2inf + Y.e2inf, X.e3nil + Y.e3nil, X.e3inf + Y.e3inf, X.enilinf + Y.enilinf, X.e123 + Y.e123, X.e12nil + Y.e12nil, X.e12inf + Y.e12inf, X.e13nil + Y.e13nil, X.e13inf + Y.e13inf, X.e1nilinf + Y.e1nilinf, X.e23nil + Y.e23nil, X.e23inf + Y.e23inf, X.e2nilinf + Y.e2nilinf, X.e3nilinf + Y.e3nilinf, X.e123nil + Y.e123nil, X.e123inf + Y.e123inf, X.e12nilinf + Y.e12nilinf, X.e13nilinf + Y.e13nilinf, X.e23nilinf + Y.e23nilinf, X.e123nilinf + Y.e123nilinf);
}

CGA3 add(CGA3 X, CGA3 Y, CGA3 Z){
    return add(add(X, Y), Z);
}

CGA3 add(CGA3 X, CGA3 Y, CGA3 Z, CGA3 P){
    return add(add(add(X, Y), Z), P);
}

#define ONE_CGA3 CGA3(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

CGA3 mul(float a, CGA3 X){
    return CGA3(X.scalar*a, X.e1*a, X.e2*a, X.e3*a, X.enil*a, X.einf*a, X.e12*a, X.e13*a, X.e1nil*a, X.e1inf*a, X.e23*a, X.e2nil*a, X.e2inf*a, X.e3nil*a, X.e3inf*a, X.enilinf*a, X.e123*a, X.e12nil*a, X.e12inf*a, X.e13nil*a, X.e13inf*a, X.e1nilinf*a, X.e23nil*a, X.e23inf*a, X.e2nilinf*a, X.e3nilinf*a, X.e123nil*a, X.e123inf*a, X.e12nilinf*a, X.e13nilinf*a, X.e23nilinf*a, X.e123nilinf*a);
}

CGA3 sub(CGA3 X, CGA3 Y){
    return CGA3(X.scalar - Y.scalar, X.e1 - Y.e1, X.e2 - Y.e2, X.e3 - Y.e3, X.enil - Y.enil, X.einf - Y.einf, X.e12 - Y.e12, X.e13 - Y.e13, X.e1nil - Y.e1nil, X.e1inf - Y.e1inf, X.e23 - Y.e23, X.e2nil - Y.e2nil, X.e2inf - Y.e2inf, X.e3nil - Y.e3nil, X.e3inf - Y.e3inf, X.enilinf - Y.enilinf, X.e123 - Y.e123, X.e12nil - Y.e12nil, X.e12inf - Y.e12inf, X.e13nil - Y.e13nil, X.e13inf - Y.e13inf, X.e1nilinf - Y.e1nilinf, X.e23nil - Y.e23nil, X.e23inf - Y.e23inf, X.e2nilinf - Y.e2nilinf, X.e3nilinf - Y.e3nilinf, X.e123nil - Y.e123nil, X.e123inf - Y.e123inf, X.e12nilinf - Y.e12nilinf, X.e13nilinf - Y.e13nilinf, X.e23nilinf - Y.e23nilinf, X.e123nilinf - Y.e123nilinf);
}

#define ZERO_CGA3 CGA3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

CGA3 mul(int a, CGA3 X){
    return mul(float(a), X);
}

CGA3 mul(CGA3 X, CGA3 Y){
    return CGA3(X.e1*Y.e1 - X.e12*Y.e12 - X.e123*Y.e123 + X.e123inf*Y.e123nil + X.e123nil*Y.e123inf - X.e123nilinf*Y.e123nilinf - X.e12inf*Y.e12nil - X.e12nil*Y.e12inf - X.e12nilinf*Y.e12nilinf - X.e13*Y.e13 - X.e13inf*Y.e13nil - X.e13nil*Y.e13inf - X.e13nilinf*Y.e13nilinf - X.e1inf*Y.e1nil - X.e1nil*Y.e1inf + X.e1nilinf*Y.e1nilinf + X.e2*Y.e2 - X.e23*Y.e23 - X.e23inf*Y.e23nil - X.e23nil*Y.e23inf - X.e23nilinf*Y.e23nilinf - X.e2inf*Y.e2nil - X.e2nil*Y.e2inf + X.e2nilinf*Y.e2nilinf + X.e3*Y.e3 - X.e3inf*Y.e3nil - X.e3nil*Y.e3inf + X.e3nilinf*Y.e3nilinf + X.einf*Y.enil + X.enil*Y.einf + X.enilinf*Y.enilinf + X.scalar*Y.scalar, X.e1*Y.scalar + X.e12*Y.e2 - X.e123*Y.e23 - X.e123inf*Y.e23nil - X.e123nil*Y.e23inf - X.e123nilinf*Y.e23nilinf - X.e12inf*Y.e2nil - X.e12nil*Y.e2inf + X.e12nilinf*Y.e2nilinf + X.e13*Y.e3 - X.e13inf*Y.e3nil - X.e13nil*Y.e3inf + X.e13nilinf*Y.e3nilinf + X.e1inf*Y.enil + X.e1nil*Y.einf + X.e1nilinf*Y.enilinf - X.e2*Y.e12 - X.e23*Y.e123 + X.e23inf*Y.e123nil + X.e23nil*Y.e123inf - X.e23nilinf*Y.e123nilinf - X.e2inf*Y.e12nil - X.e2nil*Y.e12inf - X.e2nilinf*Y.e12nilinf - X.e3*Y.e13 - X.e3inf*Y.e13nil - X.e3nil*Y.e13inf - X.e3nilinf*Y.e13nilinf - X.einf*Y.e1nil - X.enil*Y.e1inf + X.enilinf*Y.e1nilinf + X.scalar*Y.e1, X.e1*Y.e12 - X.e12*Y.e1 + X.e123*Y.e13 + X.e123inf*Y.e13nil + X.e123nil*Y.e13inf + X.e123nilinf*Y.e13nilinf + X.e12inf*Y.e1nil + X.e12nil*Y.e1inf - X.e12nilinf*Y.e1nilinf + X.e13*Y.e123 - X.e13inf*Y.e123nil - X.e13nil*Y.e123inf + X.e13nilinf*Y.e123nilinf + X.e1inf*Y.e12nil + X.e1nil*Y.e12inf + X.e1nilinf*Y.e12nilinf + X.e2*Y.scalar + X.e23*Y.e3 - X.e23inf*Y.e3nil - X.e23nil*Y.e3inf + X.e23nilinf*Y.e3nilinf + X.e2inf*Y.enil + X.e2nil*Y.einf + X.e2nilinf*Y.enilinf - X.e3*Y.e23 - X.e3inf*Y.e23nil - X.e3nil*Y.e23inf - X.e3nilinf*Y.e23nilinf - X.einf*Y.e2nil - X.enil*Y.e2inf + X.enilinf*Y.e2nilinf + X.scalar*Y.e2, X.e1*Y.e13 - X.e12*Y.e123 - X.e123*Y.e12 - X.e123inf*Y.e12nil - X.e123nil*Y.e12inf - X.e123nilinf*Y.e12nilinf + X.e12inf*Y.e123nil + X.e12nil*Y.e123inf - X.e12nilinf*Y.e123nilinf - X.e13*Y.e1 + X.e13inf*Y.e1nil + X.e13nil*Y.e1inf - X.e13nilinf*Y.e1nilinf + X.e1inf*Y.e13nil + X.e1nil*Y.e13inf + X.e1nilinf*Y.e13nilinf + X.e2*Y.e23 - X.e23*Y.e2 + X.e23inf*Y.e2nil + X.e23nil*Y.e2inf - X.e23nilinf*Y.e2nilinf + X.e2inf*Y.e23nil + X.e2nil*Y.e23inf + X.e2nilinf*Y.e23nilinf + X.e3*Y.scalar + X.e3inf*Y.enil + X.e3nil*Y.einf + X.e3nilinf*Y.enilinf - X.einf*Y.e3nil - X.enil*Y.e3inf + X.enilinf*Y.e3nilinf + X.scalar*Y.e3, X.e1*Y.e1nil - X.e12*Y.e12nil - X.e123*Y.e123nil + X.e123nil*Y.e123 - X.e123nil*Y.e123nilinf - X.e123nilinf*Y.e123nil - X.e12nil*Y.e12 + X.e12nil*Y.e12nilinf - X.e12nilinf*Y.e12nil - X.e13*Y.e13nil - X.e13nil*Y.e13 + X.e13nil*Y.e13nilinf - X.e13nilinf*Y.e13nil - X.e1nil*Y.e1 + X.e1nil*Y.e1nilinf + X.e1nilinf*Y.e1nil + X.e2*Y.e2nil - X.e23*Y.e23nil - X.e23nil*Y.e23 + X.e23nil*Y.e23nilinf - X.e23nilinf*Y.e23nil - X.e2nil*Y.e2 + X.e2nil*Y.e2nilinf + X.e2nilinf*Y.e2nil + X.e3*Y.e3nil - X.e3nil*Y.e3 + X.e3nil*Y.e3nilinf + X.e3nilinf*Y.e3nil - X.enil*Y.enilinf + X.enil*Y.scalar + X.enilinf*Y.enil + X.scalar*Y.enil, X.e1*Y.e1inf - X.e12*Y.e12inf - X.e123*Y.e123inf + X.e123inf*Y.e123 + X.e123inf*Y.e123nilinf + X.e123nilinf*Y.e123inf - X.e12inf*Y.e12 - X.e12inf*Y.e12nilinf + X.e12nilinf*Y.e12inf - X.e13*Y.e13inf - X.e13inf*Y.e13 - X.e13inf*Y.e13nilinf + X.e13nilinf*Y.e13inf - X.e1inf*Y.e1 - X.e1inf*Y.e1nilinf - X.e1nilinf*Y.e1inf + X.e2*Y.e2inf - X.e23*Y.e23inf - X.e23inf*Y.e23 - X.e23inf*Y.e23nilinf + X.e23nilinf*Y.e23inf - X.e2inf*Y.e2 - X.e2inf*Y.e2nilinf - X.e2nilinf*Y.e2inf + X.e3*Y.e3inf - X.e3inf*Y.e3 - X.e3inf*Y.e3nilinf - X.e3nilinf*Y.e3inf + X.einf*Y.enilinf + X.einf*Y.scalar - X.enilinf*Y.einf + X.scalar*Y.einf, X.e1*Y.e2 + X.e12*Y.scalar + X.e123*Y.e3 - X.e123inf*Y.e3nil - X.e123nil*Y.e3inf + X.e123nilinf*Y.e3nilinf + X.e12inf*Y.enil + X.e12nil*Y.einf + X.e12nilinf*Y.enilinf - X.e13*Y.e23 - X.e13inf*Y.e23nil - X.e13nil*Y.e23inf - X.e13nilinf*Y.e23nilinf - X.e1inf*Y.e2nil - X.e1nil*Y.e2inf + X.e1nilinf*Y.e2nilinf - X.e2*Y.e1 + X.e23*Y.e13 + X.e23inf*Y.e13nil + X.e23nil*Y.e13inf + X.e23nilinf*Y.e13nilinf + X.e2inf*Y.e1nil + X.e2nil*Y.e1inf - X.e2nilinf*Y.e1nilinf + X.e3*Y.e123 - X.e3inf*Y.e123nil - X.e3nil*Y.e123inf + X.e3nilinf*Y.e123nilinf + X.einf*Y.e12nil + X.enil*Y.e12inf + X.enilinf*Y.e12nilinf + X.scalar*Y.e12, X.e1*Y.e3 + X.e12*Y.e23 - X.e123*Y.e2 + X.e123inf*Y.e2nil + X.e123nil*Y.e2inf - X.e123nilinf*Y.e2nilinf + X.e12inf*Y.e23nil + X.e12nil*Y.e23inf + X.e12nilinf*Y.e23nilinf + X.e13*Y.scalar + X.e13inf*Y.enil + X.e13nil*Y.einf + X.e13nilinf*Y.enilinf - X.e1inf*Y.e3nil - X.e1nil*Y.e3inf + X.e1nilinf*Y.e3nilinf - X.e2*Y.e123 - X.e23*Y.e12 - X.e23inf*Y.e12nil - X.e23nil*Y.e12inf - X.e23nilinf*Y.e12nilinf + X.e2inf*Y.e123nil + X.e2nil*Y.e123inf - X.e2nilinf*Y.e123nilinf - X.e3*Y.e1 + X.e3inf*Y.e1nil + X.e3nil*Y.e1inf - X.e3nilinf*Y.e1nilinf + X.einf*Y.e13nil + X.enil*Y.e13inf + X.enilinf*Y.e13nilinf + X.scalar*Y.e13, X.e1*Y.enil + X.e12*Y.e2nil - X.e123*Y.e23nil - X.e123nil*Y.e23 + X.e123nil*Y.e23nilinf - X.e123nilinf*Y.e23nil - X.e12nil*Y.e2 + X.e12nil*Y.e2nilinf + X.e12nilinf*Y.e2nil + X.e13*Y.e3nil - X.e13nil*Y.e3 + X.e13nil*Y.e3nilinf + X.e13nilinf*Y.e3nil - X.e1nil*Y.enilinf + X.e1nil*Y.scalar + X.e1nilinf*Y.enil - X.e2*Y.e12nil - X.e23*Y.e123nil + X.e23nil*Y.e123 - X.e23nil*Y.e123nilinf - X.e23nilinf*Y.e123nil - X.e2nil*Y.e12 + X.e2nil*Y.e12nilinf - X.e2nilinf*Y.e12nil - X.e3*Y.e13nil - X.e3nil*Y.e13 + X.e3nil*Y.e13nilinf - X.e3nilinf*Y.e13nil - X.enil*Y.e1 + X.enil*Y.e1nilinf + X.enilinf*Y.e1nil + X.scalar*Y.e1nil, X.e1*Y.einf + X.e12*Y.e2inf - X.e123*Y.e23inf - X.e123inf*Y.e23 - X.e123inf*Y.e23nilinf + X.e123nilinf*Y.e23inf - X.e12inf*Y.e2 - X.e12inf*Y.e2nilinf - X.e12nilinf*Y.e2inf + X.e13*Y.e3inf - X.e13inf*Y.e3 - X.e13inf*Y.e3nilinf - X.e13nilinf*Y.e3inf + X.e1inf*Y.enilinf + X.e1inf*Y.scalar - X.e1nilinf*Y.einf - X.e2*Y.e12inf - X.e23*Y.e123inf + X.e23inf*Y.e123 + X.e23inf*Y.e123nilinf + X.e23nilinf*Y.e123inf - X.e2inf*Y.e12 - X.e2inf*Y.e12nilinf + X.e2nilinf*Y.e12inf - X.e3*Y.e13inf - X.e3inf*Y.e13 - X.e3inf*Y.e13nilinf + X.e3nilinf*Y.e13inf - X.einf*Y.e1 - X.einf*Y.e1nilinf - X.enilinf*Y.e1inf + X.scalar*Y.e1inf, X.e1*Y.e123 - X.e12*Y.e13 + X.e123*Y.e1 - X.e123inf*Y.e1nil - X.e123nil*Y.e1inf + X.e123nilinf*Y.e1nilinf - X.e12inf*Y.e13nil - X.e12nil*Y.e13inf - X.e12nilinf*Y.e13nilinf + X.e13*Y.e12 + X.e13inf*Y.e12nil + X.e13nil*Y.e12inf + X.e13nilinf*Y.e12nilinf - X.e1inf*Y.e123nil - X.e1nil*Y.e123inf + X.e1nilinf*Y.e123nilinf + X.e2*Y.e3 + X.e23*Y.scalar + X.e23inf*Y.enil + X.e23nil*Y.einf + X.e23nilinf*Y.enilinf - X.e2inf*Y.e3nil - X.e2nil*Y.e3inf + X.e2nilinf*Y.e3nilinf - X.e3*Y.e2 + X.e3inf*Y.e2nil + X.e3nil*Y.e2inf - X.e3nilinf*Y.e2nilinf + X.einf*Y.e23nil + X.enil*Y.e23inf + X.enilinf*Y.e23nilinf + X.scalar*Y.e23, X.e1*Y.e12nil - X.e12*Y.e1nil + X.e123*Y.e13nil + X.e123nil*Y.e13 - X.e123nil*Y.e13nilinf + X.e123nilinf*Y.e13nil + X.e12nil*Y.e1 - X.e12nil*Y.e1nilinf - X.e12nilinf*Y.e1nil + X.e13*Y.e123nil - X.e13nil*Y.e123 + X.e13nil*Y.e123nilinf + X.e13nilinf*Y.e123nil + X.e1nil*Y.e12 - X.e1nil*Y.e12nilinf + X.e1nilinf*Y.e12nil + X.e2*Y.enil + X.e23*Y.e3nil - X.e23nil*Y.e3 + X.e23nil*Y.e3nilinf + X.e23nilinf*Y.e3nil - X.e2nil*Y.enilinf + X.e2nil*Y.scalar + X.e2nilinf*Y.enil - X.e3*Y.e23nil - X.e3nil*Y.e23 + X.e3nil*Y.e23nilinf - X.e3nilinf*Y.e23nil - X.enil*Y.e2 + X.enil*Y.e2nilinf + X.enilinf*Y.e2nil + X.scalar*Y.e2nil, X.e1*Y.e12inf - X.e12*Y.e1inf + X.e123*Y.e13inf + X.e123inf*Y.e13 + X.e123inf*Y.e13nilinf - X.e123nilinf*Y.e13inf + X.e12inf*Y.e1 + X.e12inf*Y.e1nilinf + X.e12nilinf*Y.e1inf + X.e13*Y.e123inf - X.e13inf*Y.e123 - X.e13inf*Y.e123nilinf - X.e13nilinf*Y.e123inf + X.e1inf*Y.e12 + X.e1inf*Y.e12nilinf - X.e1nilinf*Y.e12inf + X.e2*Y.einf + X.e23*Y.e3inf - X.e23inf*Y.e3 - X.e23inf*Y.e3nilinf - X.e23nilinf*Y.e3inf + X.e2inf*Y.enilinf + X.e2inf*Y.scalar - X.e2nilinf*Y.einf - X.e3*Y.e23inf - X.e3inf*Y.e23 - X.e3inf*Y.e23nilinf + X.e3nilinf*Y.e23inf - X.einf*Y.e2 - X.einf*Y.e2nilinf - X.enilinf*Y.e2inf + X.scalar*Y.e2inf, X.e1*Y.e13nil - X.e12*Y.e123nil - X.e123*Y.e12nil - X.e123nil*Y.e12 + X.e123nil*Y.e12nilinf - X.e123nilinf*Y.e12nil + X.e12nil*Y.e123 - X.e12nil*Y.e123nilinf - X.e12nilinf*Y.e123nil - X.e13*Y.e1nil + X.e13nil*Y.e1 - X.e13nil*Y.e1nilinf - X.e13nilinf*Y.e1nil + X.e1nil*Y.e13 - X.e1nil*Y.e13nilinf + X.e1nilinf*Y.e13nil + X.e2*Y.e23nil - X.e23*Y.e2nil + X.e23nil*Y.e2 - X.e23nil*Y.e2nilinf - X.e23nilinf*Y.e2nil + X.e2nil*Y.e23 - X.e2nil*Y.e23nilinf + X.e2nilinf*Y.e23nil + X.e3*Y.enil - X.e3nil*Y.enilinf + X.e3nil*Y.scalar + X.e3nilinf*Y.enil - X.enil*Y.e3 + X.enil*Y.e3nilinf + X.enilinf*Y.e3nil + X.scalar*Y.e3nil, X.e1*Y.e13inf - X.e12*Y.e123inf - X.e123*Y.e12inf - X.e123inf*Y.e12 - X.e123inf*Y.e12nilinf + X.e123nilinf*Y.e12inf + X.e12inf*Y.e123 + X.e12inf*Y.e123nilinf + X.e12nilinf*Y.e123inf - X.e13*Y.e1inf + X.e13inf*Y.e1 + X.e13inf*Y.e1nilinf + X.e13nilinf*Y.e1inf + X.e1inf*Y.e13 + X.e1inf*Y.e13nilinf - X.e1nilinf*Y.e13inf + X.e2*Y.e23inf - X.e23*Y.e2inf + X.e23inf*Y.e2 + X.e23inf*Y.e2nilinf + X.e23nilinf*Y.e2inf + X.e2inf*Y.e23 + X.e2inf*Y.e23nilinf - X.e2nilinf*Y.e23inf + X.e3*Y.einf + X.e3inf*Y.enilinf + X.e3inf*Y.scalar - X.e3nilinf*Y.einf - X.einf*Y.e3 - X.einf*Y.e3nilinf - X.enilinf*Y.e3inf + X.scalar*Y.e3inf, X.e1*Y.e1nilinf - X.e12*Y.e12nilinf - X.e123*Y.e123nilinf - X.e123inf*Y.e123nil + X.e123nil*Y.e123inf - X.e123nilinf*Y.e123 + X.e12inf*Y.e12nil - X.e12nil*Y.e12inf - X.e12nilinf*Y.e12 - X.e13*Y.e13nilinf + X.e13inf*Y.e13nil - X.e13nil*Y.e13inf - X.e13nilinf*Y.e13 + X.e1inf*Y.e1nil - X.e1nil*Y.e1inf + X.e1nilinf*Y.e1 + X.e2*Y.e2nilinf - X.e23*Y.e23nilinf + X.e23inf*Y.e23nil - X.e23nil*Y.e23inf - X.e23nilinf*Y.e23 + X.e2inf*Y.e2nil - X.e2nil*Y.e2inf + X.e2nilinf*Y.e2 + X.e3*Y.e3nilinf + X.e3inf*Y.e3nil - X.e3nil*Y.e3inf + X.e3nilinf*Y.e3 - X.einf*Y.enil + X.enil*Y.einf + X.enilinf*Y.scalar + X.scalar*Y.enilinf, X.e1*Y.e23 + X.e12*Y.e3 + X.e123*Y.scalar + X.e123inf*Y.enil + X.e123nil*Y.einf + X.e123nilinf*Y.enilinf - X.e12inf*Y.e3nil - X.e12nil*Y.e3inf + X.e12nilinf*Y.e3nilinf - X.e13*Y.e2 + X.e13inf*Y.e2nil + X.e13nil*Y.e2inf - X.e13nilinf*Y.e2nilinf + X.e1inf*Y.e23nil + X.e1nil*Y.e23inf + X.e1nilinf*Y.e23nilinf - X.e2*Y.e13 + X.e23*Y.e1 - X.e23inf*Y.e1nil - X.e23nil*Y.e1inf + X.e23nilinf*Y.e1nilinf - X.e2inf*Y.e13nil - X.e2nil*Y.e13inf - X.e2nilinf*Y.e13nilinf + X.e3*Y.e12 + X.e3inf*Y.e12nil + X.e3nil*Y.e12inf + X.e3nilinf*Y.e12nilinf - X.einf*Y.e123nil - X.enil*Y.e123inf + X.enilinf*Y.e123nilinf + X.scalar*Y.e123, X.e1*Y.e2nil + X.e12*Y.enil + X.e123*Y.e3nil - X.e123nil*Y.e3 + X.e123nil*Y.e3nilinf + X.e123nilinf*Y.e3nil - X.e12nil*Y.enilinf + X.e12nil*Y.scalar + X.e12nilinf*Y.enil - X.e13*Y.e23nil - X.e13nil*Y.e23 + X.e13nil*Y.e23nilinf - X.e13nilinf*Y.e23nil - X.e1nil*Y.e2 + X.e1nil*Y.e2nilinf + X.e1nilinf*Y.e2nil - X.e2*Y.e1nil + X.e23*Y.e13nil + X.e23nil*Y.e13 - X.e23nil*Y.e13nilinf + X.e23nilinf*Y.e13nil + X.e2nil*Y.e1 - X.e2nil*Y.e1nilinf - X.e2nilinf*Y.e1nil + X.e3*Y.e123nil - X.e3nil*Y.e123 + X.e3nil*Y.e123nilinf + X.e3nilinf*Y.e123nil + X.enil*Y.e12 - X.enil*Y.e12nilinf + X.enilinf*Y.e12nil + X.scalar*Y.e12nil, X.e1*Y.e2inf + X.e12*Y.einf + X.e123*Y.e3inf - X.e123inf*Y.e3 - X.e123inf*Y.e3nilinf - X.e123nilinf*Y.e3inf + X.e12inf*Y.enilinf + X.e12inf*Y.scalar - X.e12nilinf*Y.einf - X.e13*Y.e23inf - X.e13inf*Y.e23 - X.e13inf*Y.e23nilinf + X.e13nilinf*Y.e23inf - X.e1inf*Y.e2 - X.e1inf*Y.e2nilinf - X.e1nilinf*Y.e2inf - X.e2*Y.e1inf + X.e23*Y.e13inf + X.e23inf*Y.e13 + X.e23inf*Y.e13nilinf - X.e23nilinf*Y.e13inf + X.e2inf*Y.e1 + X.e2inf*Y.e1nilinf + X.e2nilinf*Y.e1inf + X.e3*Y.e123inf - X.e3inf*Y.e123 - X.e3inf*Y.e123nilinf - X.e3nilinf*Y.e123inf + X.einf*Y.e12 + X.einf*Y.e12nilinf - X.enilinf*Y.e12inf + X.scalar*Y.e12inf, X.e1*Y.e3nil + X.e12*Y.e23nil - X.e123*Y.e2nil + X.e123nil*Y.e2 - X.e123nil*Y.e2nilinf - X.e123nilinf*Y.e2nil + X.e12nil*Y.e23 - X.e12nil*Y.e23nilinf + X.e12nilinf*Y.e23nil + X.e13*Y.enil - X.e13nil*Y.enilinf + X.e13nil*Y.scalar + X.e13nilinf*Y.enil - X.e1nil*Y.e3 + X.e1nil*Y.e3nilinf + X.e1nilinf*Y.e3nil - X.e2*Y.e123nil - X.e23*Y.e12nil - X.e23nil*Y.e12 + X.e23nil*Y.e12nilinf - X.e23nilinf*Y.e12nil + X.e2nil*Y.e123 - X.e2nil*Y.e123nilinf - X.e2nilinf*Y.e123nil - X.e3*Y.e1nil + X.e3nil*Y.e1 - X.e3nil*Y.e1nilinf - X.e3nilinf*Y.e1nil + X.enil*Y.e13 - X.enil*Y.e13nilinf + X.enilinf*Y.e13nil + X.scalar*Y.e13nil, X.e1*Y.e3inf + X.e12*Y.e23inf - X.e123*Y.e2inf + X.e123inf*Y.e2 + X.e123inf*Y.e2nilinf + X.e123nilinf*Y.e2inf + X.e12inf*Y.e23 + X.e12inf*Y.e23nilinf - X.e12nilinf*Y.e23inf + X.e13*Y.einf + X.e13inf*Y.enilinf + X.e13inf*Y.scalar - X.e13nilinf*Y.einf - X.e1inf*Y.e3 - X.e1inf*Y.e3nilinf - X.e1nilinf*Y.e3inf - X.e2*Y.e123inf - X.e23*Y.e12inf - X.e23inf*Y.e12 - X.e23inf*Y.e12nilinf + X.e23nilinf*Y.e12inf + X.e2inf*Y.e123 + X.e2inf*Y.e123nilinf + X.e2nilinf*Y.e123inf - X.e3*Y.e1inf + X.e3inf*Y.e1 + X.e3inf*Y.e1nilinf + X.e3nilinf*Y.e1inf + X.einf*Y.e13 + X.einf*Y.e13nilinf - X.enilinf*Y.e13inf + X.scalar*Y.e13inf, X.e1*Y.enilinf + X.e12*Y.e2nilinf - X.e123*Y.e23nilinf + X.e123inf*Y.e23nil - X.e123nil*Y.e23inf - X.e123nilinf*Y.e23 + X.e12inf*Y.e2nil - X.e12nil*Y.e2inf + X.e12nilinf*Y.e2 + X.e13*Y.e3nilinf + X.e13inf*Y.e3nil - X.e13nil*Y.e3inf + X.e13nilinf*Y.e3 - X.e1inf*Y.enil + X.e1nil*Y.einf + X.e1nilinf*Y.scalar - X.e2*Y.e12nilinf - X.e23*Y.e123nilinf - X.e23inf*Y.e123nil + X.e23nil*Y.e123inf - X.e23nilinf*Y.e123 + X.e2inf*Y.e12nil - X.e2nil*Y.e12inf - X.e2nilinf*Y.e12 - X.e3*Y.e13nilinf + X.e3inf*Y.e13nil - X.e3nil*Y.e13inf - X.e3nilinf*Y.e13 + X.einf*Y.e1nil - X.enil*Y.e1inf + X.enilinf*Y.e1 + X.scalar*Y.e1nilinf, X.e1*Y.e123nil - X.e12*Y.e13nil + X.e123*Y.e1nil - X.e123nil*Y.e1 + X.e123nil*Y.e1nilinf + X.e123nilinf*Y.e1nil - X.e12nil*Y.e13 + X.e12nil*Y.e13nilinf - X.e12nilinf*Y.e13nil + X.e13*Y.e12nil + X.e13nil*Y.e12 - X.e13nil*Y.e12nilinf + X.e13nilinf*Y.e12nil - X.e1nil*Y.e123 + X.e1nil*Y.e123nilinf + X.e1nilinf*Y.e123nil + X.e2*Y.e3nil + X.e23*Y.enil - X.e23nil*Y.enilinf + X.e23nil*Y.scalar + X.e23nilinf*Y.enil - X.e2nil*Y.e3 + X.e2nil*Y.e3nilinf + X.e2nilinf*Y.e3nil - X.e3*Y.e2nil + X.e3nil*Y.e2 - X.e3nil*Y.e2nilinf - X.e3nilinf*Y.e2nil + X.enil*Y.e23 - X.enil*Y.e23nilinf + X.enilinf*Y.e23nil + X.scalar*Y.e23nil, X.e1*Y.e123inf - X.e12*Y.e13inf + X.e123*Y.e1inf - X.e123inf*Y.e1 - X.e123inf*Y.e1nilinf - X.e123nilinf*Y.e1inf - X.e12inf*Y.e13 - X.e12inf*Y.e13nilinf + X.e12nilinf*Y.e13inf + X.e13*Y.e12inf + X.e13inf*Y.e12 + X.e13inf*Y.e12nilinf - X.e13nilinf*Y.e12inf - X.e1inf*Y.e123 - X.e1inf*Y.e123nilinf - X.e1nilinf*Y.e123inf + X.e2*Y.e3inf + X.e23*Y.einf + X.e23inf*Y.enilinf + X.e23inf*Y.scalar - X.e23nilinf*Y.einf - X.e2inf*Y.e3 - X.e2inf*Y.e3nilinf - X.e2nilinf*Y.e3inf - X.e3*Y.e2inf + X.e3inf*Y.e2 + X.e3inf*Y.e2nilinf + X.e3nilinf*Y.e2inf + X.einf*Y.e23 + X.einf*Y.e23nilinf - X.enilinf*Y.e23inf + X.scalar*Y.e23inf, X.e1*Y.e12nilinf - X.e12*Y.e1nilinf + X.e123*Y.e13nilinf - X.e123inf*Y.e13nil + X.e123nil*Y.e13inf + X.e123nilinf*Y.e13 - X.e12inf*Y.e1nil + X.e12nil*Y.e1inf - X.e12nilinf*Y.e1 + X.e13*Y.e123nilinf + X.e13inf*Y.e123nil - X.e13nil*Y.e123inf + X.e13nilinf*Y.e123 - X.e1inf*Y.e12nil + X.e1nil*Y.e12inf + X.e1nilinf*Y.e12 + X.e2*Y.enilinf + X.e23*Y.e3nilinf + X.e23inf*Y.e3nil - X.e23nil*Y.e3inf + X.e23nilinf*Y.e3 - X.e2inf*Y.enil + X.e2nil*Y.einf + X.e2nilinf*Y.scalar - X.e3*Y.e23nilinf + X.e3inf*Y.e23nil - X.e3nil*Y.e23inf - X.e3nilinf*Y.e23 + X.einf*Y.e2nil - X.enil*Y.e2inf + X.enilinf*Y.e2 + X.scalar*Y.e2nilinf, X.e1*Y.e13nilinf - X.e12*Y.e123nilinf - X.e123*Y.e12nilinf + X.e123inf*Y.e12nil - X.e123nil*Y.e12inf - X.e123nilinf*Y.e12 - X.e12inf*Y.e123nil + X.e12nil*Y.e123inf - X.e12nilinf*Y.e123 - X.e13*Y.e1nilinf - X.e13inf*Y.e1nil + X.e13nil*Y.e1inf - X.e13nilinf*Y.e1 - X.e1inf*Y.e13nil + X.e1nil*Y.e13inf + X.e1nilinf*Y.e13 + X.e2*Y.e23nilinf - X.e23*Y.e2nilinf - X.e23inf*Y.e2nil + X.e23nil*Y.e2inf - X.e23nilinf*Y.e2 - X.e2inf*Y.e23nil + X.e2nil*Y.e23inf + X.e2nilinf*Y.e23 + X.e3*Y.enilinf - X.e3inf*Y.enil + X.e3nil*Y.einf + X.e3nilinf*Y.scalar + X.einf*Y.e3nil - X.enil*Y.e3inf + X.enilinf*Y.e3 + X.scalar*Y.e3nilinf, X.e1*Y.e23nil + X.e12*Y.e3nil + X.e123*Y.enil - X.e123nil*Y.enilinf + X.e123nil*Y.scalar + X.e123nilinf*Y.enil - X.e12nil*Y.e3 + X.e12nil*Y.e3nilinf + X.e12nilinf*Y.e3nil - X.e13*Y.e2nil + X.e13nil*Y.e2 - X.e13nil*Y.e2nilinf - X.e13nilinf*Y.e2nil + X.e1nil*Y.e23 - X.e1nil*Y.e23nilinf + X.e1nilinf*Y.e23nil - X.e2*Y.e13nil + X.e23*Y.e1nil - X.e23nil*Y.e1 + X.e23nil*Y.e1nilinf + X.e23nilinf*Y.e1nil - X.e2nil*Y.e13 + X.e2nil*Y.e13nilinf - X.e2nilinf*Y.e13nil + X.e3*Y.e12nil + X.e3nil*Y.e12 - X.e3nil*Y.e12nilinf + X.e3nilinf*Y.e12nil - X.enil*Y.e123 + X.enil*Y.e123nilinf + X.enilinf*Y.e123nil + X.scalar*Y.e123nil, X.e1*Y.e23inf + X.e12*Y.e3inf + X.e123*Y.einf + X.e123inf*Y.enilinf + X.e123inf*Y.scalar - X.e123nilinf*Y.einf - X.e12inf*Y.e3 - X.e12inf*Y.e3nilinf - X.e12nilinf*Y.e3inf - X.e13*Y.e2inf + X.e13inf*Y.e2 + X.e13inf*Y.e2nilinf + X.e13nilinf*Y.e2inf + X.e1inf*Y.e23 + X.e1inf*Y.e23nilinf - X.e1nilinf*Y.e23inf - X.e2*Y.e13inf + X.e23*Y.e1inf - X.e23inf*Y.e1 - X.e23inf*Y.e1nilinf - X.e23nilinf*Y.e1inf - X.e2inf*Y.e13 - X.e2inf*Y.e13nilinf + X.e2nilinf*Y.e13inf + X.e3*Y.e12inf + X.e3inf*Y.e12 + X.e3inf*Y.e12nilinf - X.e3nilinf*Y.e12inf - X.einf*Y.e123 - X.einf*Y.e123nilinf - X.enilinf*Y.e123inf + X.scalar*Y.e123inf, X.e1*Y.e2nilinf + X.e12*Y.enilinf + X.e123*Y.e3nilinf + X.e123inf*Y.e3nil - X.e123nil*Y.e3inf + X.e123nilinf*Y.e3 - X.e12inf*Y.enil + X.e12nil*Y.einf + X.e12nilinf*Y.scalar - X.e13*Y.e23nilinf + X.e13inf*Y.e23nil - X.e13nil*Y.e23inf - X.e13nilinf*Y.e23 + X.e1inf*Y.e2nil - X.e1nil*Y.e2inf + X.e1nilinf*Y.e2 - X.e2*Y.e1nilinf + X.e23*Y.e13nilinf - X.e23inf*Y.e13nil + X.e23nil*Y.e13inf + X.e23nilinf*Y.e13 - X.e2inf*Y.e1nil + X.e2nil*Y.e1inf - X.e2nilinf*Y.e1 + X.e3*Y.e123nilinf + X.e3inf*Y.e123nil - X.e3nil*Y.e123inf + X.e3nilinf*Y.e123 - X.einf*Y.e12nil + X.enil*Y.e12inf + X.enilinf*Y.e12 + X.scalar*Y.e12nilinf, X.e1*Y.e3nilinf + X.e12*Y.e23nilinf - X.e123*Y.e2nilinf - X.e123inf*Y.e2nil + X.e123nil*Y.e2inf - X.e123nilinf*Y.e2 - X.e12inf*Y.e23nil + X.e12nil*Y.e23inf + X.e12nilinf*Y.e23 + X.e13*Y.enilinf - X.e13inf*Y.enil + X.e13nil*Y.einf + X.e13nilinf*Y.scalar + X.e1inf*Y.e3nil - X.e1nil*Y.e3inf + X.e1nilinf*Y.e3 - X.e2*Y.e123nilinf - X.e23*Y.e12nilinf + X.e23inf*Y.e12nil - X.e23nil*Y.e12inf - X.e23nilinf*Y.e12 - X.e2inf*Y.e123nil + X.e2nil*Y.e123inf - X.e2nilinf*Y.e123 - X.e3*Y.e1nilinf - X.e3inf*Y.e1nil + X.e3nil*Y.e1inf - X.e3nilinf*Y.e1 - X.einf*Y.e13nil + X.enil*Y.e13inf + X.enilinf*Y.e13 + X.scalar*Y.e13nilinf, X.e1*Y.e123nilinf - X.e12*Y.e13nilinf + X.e123*Y.e1nilinf + X.e123inf*Y.e1nil - X.e123nil*Y.e1inf + X.e123nilinf*Y.e1 + X.e12inf*Y.e13nil - X.e12nil*Y.e13inf - X.e12nilinf*Y.e13 + X.e13*Y.e12nilinf - X.e13inf*Y.e12nil + X.e13nil*Y.e12inf + X.e13nilinf*Y.e12 + X.e1inf*Y.e123nil - X.e1nil*Y.e123inf + X.e1nilinf*Y.e123 + X.e2*Y.e3nilinf + X.e23*Y.enilinf - X.e23inf*Y.enil + X.e23nil*Y.einf + X.e23nilinf*Y.scalar + X.e2inf*Y.e3nil - X.e2nil*Y.e3inf + X.e2nilinf*Y.e3 - X.e3*Y.e2nilinf - X.e3inf*Y.e2nil + X.e3nil*Y.e2inf - X.e3nilinf*Y.e2 - X.einf*Y.e23nil + X.enil*Y.e23inf + X.enilinf*Y.e23 + X.scalar*Y.e23nilinf, X.e1*Y.e23nilinf + X.e12*Y.e3nilinf + X.e123*Y.enilinf - X.e123inf*Y.enil + X.e123nil*Y.einf + X.e123nilinf*Y.scalar + X.e12inf*Y.e3nil - X.e12nil*Y.e3inf + X.e12nilinf*Y.e3 - X.e13*Y.e2nilinf - X.e13inf*Y.e2nil + X.e13nil*Y.e2inf - X.e13nilinf*Y.e2 - X.e1inf*Y.e23nil + X.e1nil*Y.e23inf + X.e1nilinf*Y.e23 - X.e2*Y.e13nilinf + X.e23*Y.e1nilinf + X.e23inf*Y.e1nil - X.e23nil*Y.e1inf + X.e23nilinf*Y.e1 + X.e2inf*Y.e13nil - X.e2nil*Y.e13inf - X.e2nilinf*Y.e13 + X.e3*Y.e12nilinf - X.e3inf*Y.e12nil + X.e3nil*Y.e12inf + X.e3nilinf*Y.e12 + X.einf*Y.e123nil - X.enil*Y.e123inf + X.enilinf*Y.e123 + X.scalar*Y.e123nilinf);
}

CGA3 scalar_CGA3(float a){
    return mul(a, ONE_CGA3);
}

CGA3 mul(CGA3 X, CGA3 Y, CGA3 Z){
    return mul(mul(X, Y), Z);
}

CGA3 involve(CGA3 X){
    return CGA3(X.scalar, -X.e1, -X.e2, -X.e3, -X.enil, -X.einf, X.e12, X.e13, X.e1nil, X.e1inf, X.e23, X.e2nil, X.e2inf, X.e3nil, X.e3inf, X.enilinf, -X.e123, -X.e12nil, -X.e12inf, -X.e13nil, -X.e13inf, -X.e1nilinf, -X.e23nil, -X.e23inf, -X.e2nilinf, -X.e3nilinf, X.e123nil, X.e123inf, X.e12nilinf, X.e13nilinf, X.e23nilinf, -X.e123nilinf);
}

CGA3 inner(CGA3 X, CGA3 Y){
    return CGA3(X.e1*Y.e1 - X.e12*Y.e12 - X.e123*Y.e123 + X.e123inf*Y.e123nil + X.e123nil*Y.e123inf - X.e123nilinf*Y.e123nilinf - X.e12inf*Y.e12nil - X.e12nil*Y.e12inf - X.e12nilinf*Y.e12nilinf - X.e13*Y.e13 - X.e13inf*Y.e13nil - X.e13nil*Y.e13inf - X.e13nilinf*Y.e13nilinf - X.e1inf*Y.e1nil - X.e1nil*Y.e1inf + X.e1nilinf*Y.e1nilinf + X.e2*Y.e2 - X.e23*Y.e23 - X.e23inf*Y.e23nil - X.e23nil*Y.e23inf - X.e23nilinf*Y.e23nilinf - X.e2inf*Y.e2nil - X.e2nil*Y.e2inf + X.e2nilinf*Y.e2nilinf + X.e3*Y.e3 - X.e3inf*Y.e3nil - X.e3nil*Y.e3inf + X.e3nilinf*Y.e3nilinf + X.einf*Y.enil + X.enil*Y.einf + X.enilinf*Y.enilinf, X.e12*Y.e2 - X.e123*Y.e23 - X.e123inf*Y.e23nil - X.e123nil*Y.e23inf - X.e123nilinf*Y.e23nilinf - X.e12inf*Y.e2nil - X.e12nil*Y.e2inf + X.e12nilinf*Y.e2nilinf + X.e13*Y.e3 - X.e13inf*Y.e3nil - X.e13nil*Y.e3inf + X.e13nilinf*Y.e3nilinf + X.e1inf*Y.enil + X.e1nil*Y.einf + X.e1nilinf*Y.enilinf - X.e2*Y.e12 - X.e23*Y.e123 + X.e23inf*Y.e123nil + X.e23nil*Y.e123inf - X.e23nilinf*Y.e123nilinf - X.e2inf*Y.e12nil - X.e2nil*Y.e12inf - X.e2nilinf*Y.e12nilinf - X.e3*Y.e13 - X.e3inf*Y.e13nil - X.e3nil*Y.e13inf - X.e3nilinf*Y.e13nilinf - X.einf*Y.e1nil - X.enil*Y.e1inf + X.enilinf*Y.e1nilinf, X.e1*Y.e12 - X.e12*Y.e1 + X.e123*Y.e13 + X.e123inf*Y.e13nil + X.e123nil*Y.e13inf + X.e123nilinf*Y.e13nilinf + X.e12inf*Y.e1nil + X.e12nil*Y.e1inf - X.e12nilinf*Y.e1nilinf + X.e13*Y.e123 - X.e13inf*Y.e123nil - X.e13nil*Y.e123inf + X.e13nilinf*Y.e123nilinf + X.e1inf*Y.e12nil + X.e1nil*Y.e12inf + X.e1nilinf*Y.e12nilinf + X.e23*Y.e3 - X.e23inf*Y.e3nil - X.e23nil*Y.e3inf + X.e23nilinf*Y.e3nilinf + X.e2inf*Y.enil + X.e2nil*Y.einf + X.e2nilinf*Y.enilinf - X.e3*Y.e23 - X.e3inf*Y.e23nil - X.e3nil*Y.e23inf - X.e3nilinf*Y.e23nilinf - X.einf*Y.e2nil - X.enil*Y.e2inf + X.enilinf*Y.e2nilinf, X.e1*Y.e13 - X.e12*Y.e123 - X.e123*Y.e12 - X.e123inf*Y.e12nil - X.e123nil*Y.e12inf - X.e123nilinf*Y.e12nilinf + X.e12inf*Y.e123nil + X.e12nil*Y.e123inf - X.e12nilinf*Y.e123nilinf - X.e13*Y.e1 + X.e13inf*Y.e1nil + X.e13nil*Y.e1inf - X.e13nilinf*Y.e1nilinf + X.e1inf*Y.e13nil + X.e1nil*Y.e13inf + X.e1nilinf*Y.e13nilinf + X.e2*Y.e23 - X.e23*Y.e2 + X.e23inf*Y.e2nil + X.e23nil*Y.e2inf - X.e23nilinf*Y.e2nilinf + X.e2inf*Y.e23nil + X.e2nil*Y.e23inf + X.e2nilinf*Y.e23nilinf + X.e3inf*Y.enil + X.e3nil*Y.einf + X.e3nilinf*Y.enilinf - X.einf*Y.e3nil - X.enil*Y.e3inf + X.enilinf*Y.e3nilinf, X.e1*Y.e1nil - X.e12*Y.e12nil - X.e123*Y.e123nil + X.e123nil*Y.e123 - X.e123nil*Y.e123nilinf - X.e123nilinf*Y.e123nil - X.e12nil*Y.e12 + X.e12nil*Y.e12nilinf - X.e12nilinf*Y.e12nil - X.e13*Y.e13nil - X.e13nil*Y.e13 + X.e13nil*Y.e13nilinf - X.e13nilinf*Y.e13nil - X.e1nil*Y.e1 + X.e1nil*Y.e1nilinf + X.e1nilinf*Y.e1nil + X.e2*Y.e2nil - X.e23*Y.e23nil - X.e23nil*Y.e23 + X.e23nil*Y.e23nilinf - X.e23nilinf*Y.e23nil - X.e2nil*Y.e2 + X.e2nil*Y.e2nilinf + X.e2nilinf*Y.e2nil + X.e3*Y.e3nil - X.e3nil*Y.e3 + X.e3nil*Y.e3nilinf + X.e3nilinf*Y.e3nil - X.enil*Y.enilinf + X.enilinf*Y.enil, X.e1*Y.e1inf - X.e12*Y.e12inf - X.e123*Y.e123inf + X.e123inf*Y.e123 + X.e123inf*Y.e123nilinf + X.e123nilinf*Y.e123inf - X.e12inf*Y.e12 - X.e12inf*Y.e12nilinf + X.e12nilinf*Y.e12inf - X.e13*Y.e13inf - X.e13inf*Y.e13 - X.e13inf*Y.e13nilinf + X.e13nilinf*Y.e13inf - X.e1inf*Y.e1 - X.e1inf*Y.e1nilinf - X.e1nilinf*Y.e1inf + X.e2*Y.e2inf - X.e23*Y.e23inf - X.e23inf*Y.e23 - X.e23inf*Y.e23nilinf + X.e23nilinf*Y.e23inf - X.e2inf*Y.e2 - X.e2inf*Y.e2nilinf - X.e2nilinf*Y.e2inf + X.e3*Y.e3inf - X.e3inf*Y.e3 - X.e3inf*Y.e3nilinf - X.e3nilinf*Y.e3inf + X.einf*Y.enilinf - X.enilinf*Y.einf, X.e123*Y.e3 - X.e123inf*Y.e3nil - X.e123nil*Y.e3inf + X.e123nilinf*Y.e3nilinf + X.e12inf*Y.enil + X.e12nil*Y.einf + X.e12nilinf*Y.enilinf + X.e3*Y.e123 - X.e3inf*Y.e123nil - X.e3nil*Y.e123inf + X.e3nilinf*Y.e123nilinf + X.einf*Y.e12nil + X.enil*Y.e12inf + X.enilinf*Y.e12nilinf, -X.e123*Y.e2 + X.e123inf*Y.e2nil + X.e123nil*Y.e2inf - X.e123nilinf*Y.e2nilinf + X.e13inf*Y.enil + X.e13nil*Y.einf + X.e13nilinf*Y.enilinf - X.e2*Y.e123 + X.e2inf*Y.e123nil + X.e2nil*Y.e123inf - X.e2nilinf*Y.e123nilinf + X.einf*Y.e13nil + X.enil*Y.e13inf + X.enilinf*Y.e13nilinf, -X.e123nil*Y.e23 - X.e123nilinf*Y.e23nil - X.e12nil*Y.e2 + X.e12nilinf*Y.e2nil - X.e13nil*Y.e3 + X.e13nilinf*Y.e3nil + X.e1nilinf*Y.enil - X.e2*Y.e12nil - X.e23*Y.e123nil - X.e23nil*Y.e123nilinf + X.e2nil*Y.e12nilinf - X.e3*Y.e13nil + X.e3nil*Y.e13nilinf + X.enil*Y.e1nilinf, -X.e123inf*Y.e23 + X.e123nilinf*Y.e23inf - X.e12inf*Y.e2 - X.e12nilinf*Y.e2inf - X.e13inf*Y.e3 - X.e13nilinf*Y.e3inf - X.e1nilinf*Y.einf - X.e2*Y.e12inf - X.e23*Y.e123inf + X.e23inf*Y.e123nilinf - X.e2inf*Y.e12nilinf - X.e3*Y.e13inf - X.e3inf*Y.e13nilinf - X.einf*Y.e1nilinf, X.e1*Y.e123 + X.e123*Y.e1 - X.e123inf*Y.e1nil - X.e123nil*Y.e1inf + X.e123nilinf*Y.e1nilinf - X.e1inf*Y.e123nil - X.e1nil*Y.e123inf + X.e1nilinf*Y.e123nilinf + X.e23inf*Y.enil + X.e23nil*Y.einf + X.e23nilinf*Y.enilinf + X.einf*Y.e23nil + X.enil*Y.e23inf + X.enilinf*Y.e23nilinf, X.e1*Y.e12nil + X.e123nil*Y.e13 + X.e123nilinf*Y.e13nil + X.e12nil*Y.e1 - X.e12nilinf*Y.e1nil + X.e13*Y.e123nil + X.e13nil*Y.e123nilinf - X.e1nil*Y.e12nilinf - X.e23nil*Y.e3 + X.e23nilinf*Y.e3nil + X.e2nilinf*Y.enil - X.e3*Y.e23nil + X.e3nil*Y.e23nilinf + X.enil*Y.e2nilinf, X.e1*Y.e12inf + X.e123inf*Y.e13 - X.e123nilinf*Y.e13inf + X.e12inf*Y.e1 + X.e12nilinf*Y.e1inf + X.e13*Y.e123inf - X.e13inf*Y.e123nilinf + X.e1inf*Y.e12nilinf - X.e23inf*Y.e3 - X.e23nilinf*Y.e3inf - X.e2nilinf*Y.einf - X.e3*Y.e23inf - X.e3inf*Y.e23nilinf - X.einf*Y.e2nilinf, X.e1*Y.e13nil - X.e12*Y.e123nil - X.e123nil*Y.e12 - X.e123nilinf*Y.e12nil - X.e12nil*Y.e123nilinf + X.e13nil*Y.e1 - X.e13nilinf*Y.e1nil - X.e1nil*Y.e13nilinf + X.e2*Y.e23nil + X.e23nil*Y.e2 - X.e23nilinf*Y.e2nil - X.e2nil*Y.e23nilinf + X.e3nilinf*Y.enil + X.enil*Y.e3nilinf, X.e1*Y.e13inf - X.e12*Y.e123inf - X.e123inf*Y.e12 + X.e123nilinf*Y.e12inf + X.e12inf*Y.e123nilinf + X.e13inf*Y.e1 + X.e13nilinf*Y.e1inf + X.e1inf*Y.e13nilinf + X.e2*Y.e23inf + X.e23inf*Y.e2 + X.e23nilinf*Y.e2inf + X.e2inf*Y.e23nilinf - X.e3nilinf*Y.einf - X.einf*Y.e3nilinf, X.e1*Y.e1nilinf - X.e12*Y.e12nilinf - X.e123*Y.e123nilinf - X.e123nilinf*Y.e123 - X.e12nilinf*Y.e12 - X.e13*Y.e13nilinf - X.e13nilinf*Y.e13 + X.e1nilinf*Y.e1 + X.e2*Y.e2nilinf - X.e23*Y.e23nilinf - X.e23nilinf*Y.e23 + X.e2nilinf*Y.e2 + X.e3*Y.e3nilinf + X.e3nilinf*Y.e3, X.e123inf*Y.enil + X.e123nil*Y.einf + X.e123nilinf*Y.enilinf - X.einf*Y.e123nil - X.enil*Y.e123inf + X.enilinf*Y.e123nilinf, -X.e123nil*Y.e3 + X.e123nilinf*Y.e3nil + X.e12nilinf*Y.enil + X.e3*Y.e123nil + X.e3nil*Y.e123nilinf - X.enil*Y.e12nilinf, -X.e123inf*Y.e3 - X.e123nilinf*Y.e3inf - X.e12nilinf*Y.einf + X.e3*Y.e123inf - X.e3inf*Y.e123nilinf + X.einf*Y.e12nilinf, X.e123nil*Y.e2 - X.e123nilinf*Y.e2nil + X.e13nilinf*Y.enil - X.e2*Y.e123nil - X.e2nil*Y.e123nilinf - X.enil*Y.e13nilinf, X.e123inf*Y.e2 + X.e123nilinf*Y.e2inf - X.e13nilinf*Y.einf - X.e2*Y.e123inf + X.e2inf*Y.e123nilinf + X.einf*Y.e13nilinf, -X.e123nilinf*Y.e23 + X.e12nilinf*Y.e2 + X.e13nilinf*Y.e3 - X.e2*Y.e12nilinf - X.e23*Y.e123nilinf - X.e3*Y.e13nilinf, X.e1*Y.e123nil - X.e123nil*Y.e1 + X.e123nilinf*Y.e1nil + X.e1nil*Y.e123nilinf + X.e23nilinf*Y.enil - X.enil*Y.e23nilinf, X.e1*Y.e123inf - X.e123inf*Y.e1 - X.e123nilinf*Y.e1inf - X.e1inf*Y.e123nilinf - X.e23nilinf*Y.einf + X.einf*Y.e23nilinf, X.e1*Y.e12nilinf + X.e123nilinf*Y.e13 - X.e12nilinf*Y.e1 + X.e13*Y.e123nilinf + X.e23nilinf*Y.e3 - X.e3*Y.e23nilinf, X.e1*Y.e13nilinf - X.e12*Y.e123nilinf - X.e123nilinf*Y.e12 - X.e13nilinf*Y.e1 + X.e2*Y.e23nilinf - X.e23nilinf*Y.e2, X.e123nilinf*Y.enil + X.enil*Y.e123nilinf, -X.e123nilinf*Y.einf - X.einf*Y.e123nilinf, X.e123nilinf*Y.e3 + X.e3*Y.e123nilinf, -X.e123nilinf*Y.e2 - X.e2*Y.e123nilinf, X.e1*Y.e123nilinf + X.e123nilinf*Y.e1, 0.0);
}

CGA3 lcontract(CGA3 X, CGA3 Y){
    return CGA3(X.e1*Y.e1 - X.e12*Y.e12 - X.e123*Y.e123 + X.e123inf*Y.e123nil + X.e123nil*Y.e123inf - X.e123nilinf*Y.e123nilinf - X.e12inf*Y.e12nil - X.e12nil*Y.e12inf - X.e12nilinf*Y.e12nilinf - X.e13*Y.e13 - X.e13inf*Y.e13nil - X.e13nil*Y.e13inf - X.e13nilinf*Y.e13nilinf - X.e1inf*Y.e1nil - X.e1nil*Y.e1inf + X.e1nilinf*Y.e1nilinf + X.e2*Y.e2 - X.e23*Y.e23 - X.e23inf*Y.e23nil - X.e23nil*Y.e23inf - X.e23nilinf*Y.e23nilinf - X.e2inf*Y.e2nil - X.e2nil*Y.e2inf + X.e2nilinf*Y.e2nilinf + X.e3*Y.e3 - X.e3inf*Y.e3nil - X.e3nil*Y.e3inf + X.e3nilinf*Y.e3nilinf + X.einf*Y.enil + X.enil*Y.einf + X.enilinf*Y.enilinf + X.scalar*Y.scalar, -X.e2*Y.e12 - X.e23*Y.e123 + X.e23inf*Y.e123nil + X.e23nil*Y.e123inf - X.e23nilinf*Y.e123nilinf - X.e2inf*Y.e12nil - X.e2nil*Y.e12inf - X.e2nilinf*Y.e12nilinf - X.e3*Y.e13 - X.e3inf*Y.e13nil - X.e3nil*Y.e13inf - X.e3nilinf*Y.e13nilinf - X.einf*Y.e1nil - X.enil*Y.e1inf + X.enilinf*Y.e1nilinf + X.scalar*Y.e1, X.e1*Y.e12 + X.e13*Y.e123 - X.e13inf*Y.e123nil - X.e13nil*Y.e123inf + X.e13nilinf*Y.e123nilinf + X.e1inf*Y.e12nil + X.e1nil*Y.e12inf + X.e1nilinf*Y.e12nilinf - X.e3*Y.e23 - X.e3inf*Y.e23nil - X.e3nil*Y.e23inf - X.e3nilinf*Y.e23nilinf - X.einf*Y.e2nil - X.enil*Y.e2inf + X.enilinf*Y.e2nilinf + X.scalar*Y.e2, X.e1*Y.e13 - X.e12*Y.e123 + X.e12inf*Y.e123nil + X.e12nil*Y.e123inf - X.e12nilinf*Y.e123nilinf + X.e1inf*Y.e13nil + X.e1nil*Y.e13inf + X.e1nilinf*Y.e13nilinf + X.e2*Y.e23 + X.e2inf*Y.e23nil + X.e2nil*Y.e23inf + X.e2nilinf*Y.e23nilinf - X.einf*Y.e3nil - X.enil*Y.e3inf + X.enilinf*Y.e3nilinf + X.scalar*Y.e3, X.e1*Y.e1nil - X.e12*Y.e12nil - X.e123*Y.e123nil - X.e123nil*Y.e123nilinf + X.e12nil*Y.e12nilinf - X.e13*Y.e13nil + X.e13nil*Y.e13nilinf + X.e1nil*Y.e1nilinf + X.e2*Y.e2nil - X.e23*Y.e23nil + X.e23nil*Y.e23nilinf + X.e2nil*Y.e2nilinf + X.e3*Y.e3nil + X.e3nil*Y.e3nilinf - X.enil*Y.enilinf + X.scalar*Y.enil, X.e1*Y.e1inf - X.e12*Y.e12inf - X.e123*Y.e123inf + X.e123inf*Y.e123nilinf - X.e12inf*Y.e12nilinf - X.e13*Y.e13inf - X.e13inf*Y.e13nilinf - X.e1inf*Y.e1nilinf + X.e2*Y.e2inf - X.e23*Y.e23inf - X.e23inf*Y.e23nilinf - X.e2inf*Y.e2nilinf + X.e3*Y.e3inf - X.e3inf*Y.e3nilinf + X.einf*Y.enilinf + X.scalar*Y.einf, X.e3*Y.e123 - X.e3inf*Y.e123nil - X.e3nil*Y.e123inf + X.e3nilinf*Y.e123nilinf + X.einf*Y.e12nil + X.enil*Y.e12inf + X.enilinf*Y.e12nilinf + X.scalar*Y.e12, -X.e2*Y.e123 + X.e2inf*Y.e123nil + X.e2nil*Y.e123inf - X.e2nilinf*Y.e123nilinf + X.einf*Y.e13nil + X.enil*Y.e13inf + X.enilinf*Y.e13nilinf + X.scalar*Y.e13, -X.e2*Y.e12nil - X.e23*Y.e123nil - X.e23nil*Y.e123nilinf + X.e2nil*Y.e12nilinf - X.e3*Y.e13nil + X.e3nil*Y.e13nilinf + X.enil*Y.e1nilinf + X.scalar*Y.e1nil, -X.e2*Y.e12inf - X.e23*Y.e123inf + X.e23inf*Y.e123nilinf - X.e2inf*Y.e12nilinf - X.e3*Y.e13inf - X.e3inf*Y.e13nilinf - X.einf*Y.e1nilinf + X.scalar*Y.e1inf, X.e1*Y.e123 - X.e1inf*Y.e123nil - X.e1nil*Y.e123inf + X.e1nilinf*Y.e123nilinf + X.einf*Y.e23nil + X.enil*Y.e23inf + X.enilinf*Y.e23nilinf + X.scalar*Y.e23, X.e1*Y.e12nil + X.e13*Y.e123nil + X.e13nil*Y.e123nilinf - X.e1nil*Y.e12nilinf - X.e3*Y.e23nil + X.e3nil*Y.e23nilinf + X.enil*Y.e2nilinf + X.scalar*Y.e2nil, X.e1*Y.e12inf + X.e13*Y.e123inf - X.e13inf*Y.e123nilinf + X.e1inf*Y.e12nilinf - X.e3*Y.e23inf - X.e3inf*Y.e23nilinf - X.einf*Y.e2nilinf + X.scalar*Y.e2inf, X.e1*Y.e13nil - X.e12*Y.e123nil - X.e12nil*Y.e123nilinf - X.e1nil*Y.e13nilinf + X.e2*Y.e23nil - X.e2nil*Y.e23nilinf + X.enil*Y.e3nilinf + X.scalar*Y.e3nil, X.e1*Y.e13inf - X.e12*Y.e123inf + X.e12inf*Y.e123nilinf + X.e1inf*Y.e13nilinf + X.e2*Y.e23inf + X.e2inf*Y.e23nilinf - X.einf*Y.e3nilinf + X.scalar*Y.e3inf, X.e1*Y.e1nilinf - X.e12*Y.e12nilinf - X.e123*Y.e123nilinf - X.e13*Y.e13nilinf + X.e2*Y.e2nilinf - X.e23*Y.e23nilinf + X.e3*Y.e3nilinf + X.scalar*Y.enilinf, -X.einf*Y.e123nil - X.enil*Y.e123inf + X.enilinf*Y.e123nilinf + X.scalar*Y.e123, X.e3*Y.e123nil + X.e3nil*Y.e123nilinf - X.enil*Y.e12nilinf + X.scalar*Y.e12nil, X.e3*Y.e123inf - X.e3inf*Y.e123nilinf + X.einf*Y.e12nilinf + X.scalar*Y.e12inf, -X.e2*Y.e123nil - X.e2nil*Y.e123nilinf - X.enil*Y.e13nilinf + X.scalar*Y.e13nil, -X.e2*Y.e123inf + X.e2inf*Y.e123nilinf + X.einf*Y.e13nilinf + X.scalar*Y.e13inf, -X.e2*Y.e12nilinf - X.e23*Y.e123nilinf - X.e3*Y.e13nilinf + X.scalar*Y.e1nilinf, X.e1*Y.e123nil + X.e1nil*Y.e123nilinf - X.enil*Y.e23nilinf + X.scalar*Y.e23nil, X.e1*Y.e123inf - X.e1inf*Y.e123nilinf + X.einf*Y.e23nilinf + X.scalar*Y.e23inf, X.e1*Y.e12nilinf + X.e13*Y.e123nilinf - X.e3*Y.e23nilinf + X.scalar*Y.e2nilinf, X.e1*Y.e13nilinf - X.e12*Y.e123nilinf + X.e2*Y.e23nilinf + X.scalar*Y.e3nilinf, X.enil*Y.e123nilinf + X.scalar*Y.e123nil, -X.einf*Y.e123nilinf + X.scalar*Y.e123inf, X.e3*Y.e123nilinf + X.scalar*Y.e12nilinf, -X.e2*Y.e123nilinf + X.scalar*Y.e13nilinf, X.e1*Y.e123nilinf + X.scalar*Y.e23nilinf, X.scalar*Y.e123nilinf);
}

CGA3 outer(CGA3 X, CGA3 Y){
    return CGA3(X.scalar*Y.scalar, X.e1*Y.scalar + X.scalar*Y.e1, X.e2*Y.scalar + X.scalar*Y.e2, X.e3*Y.scalar + X.scalar*Y.e3, X.enil*Y.scalar + X.scalar*Y.enil, X.einf*Y.scalar + X.scalar*Y.einf, X.e1*Y.e2 + X.e12*Y.scalar - X.e2*Y.e1 + X.scalar*Y.e12, X.e1*Y.e3 + X.e13*Y.scalar - X.e3*Y.e1 + X.scalar*Y.e13, X.e1*Y.enil + X.e1nil*Y.scalar - X.enil*Y.e1 + X.scalar*Y.e1nil, X.e1*Y.einf + X.e1inf*Y.scalar - X.einf*Y.e1 + X.scalar*Y.e1inf, X.e2*Y.e3 + X.e23*Y.scalar - X.e3*Y.e2 + X.scalar*Y.e23, X.e2*Y.enil + X.e2nil*Y.scalar - X.enil*Y.e2 + X.scalar*Y.e2nil, X.e2*Y.einf + X.e2inf*Y.scalar - X.einf*Y.e2 + X.scalar*Y.e2inf, X.e3*Y.enil + X.e3nil*Y.scalar - X.enil*Y.e3 + X.scalar*Y.e3nil, X.e3*Y.einf + X.e3inf*Y.scalar - X.einf*Y.e3 + X.scalar*Y.e3inf, -X.einf*Y.enil + X.enil*Y.einf + X.enilinf*Y.scalar + X.scalar*Y.enilinf, X.e1*Y.e23 + X.e12*Y.e3 + X.e123*Y.scalar - X.e13*Y.e2 - X.e2*Y.e13 + X.e23*Y.e1 + X.e3*Y.e12 + X.scalar*Y.e123, X.e1*Y.e2nil + X.e12*Y.enil + X.e12nil*Y.scalar - X.e1nil*Y.e2 - X.e2*Y.e1nil + X.e2nil*Y.e1 + X.enil*Y.e12 + X.scalar*Y.e12nil, X.e1*Y.e2inf + X.e12*Y.einf + X.e12inf*Y.scalar - X.e1inf*Y.e2 - X.e2*Y.e1inf + X.e2inf*Y.e1 + X.einf*Y.e12 + X.scalar*Y.e12inf, X.e1*Y.e3nil + X.e13*Y.enil + X.e13nil*Y.scalar - X.e1nil*Y.e3 - X.e3*Y.e1nil + X.e3nil*Y.e1 + X.enil*Y.e13 + X.scalar*Y.e13nil, X.e1*Y.e3inf + X.e13*Y.einf + X.e13inf*Y.scalar - X.e1inf*Y.e3 - X.e3*Y.e1inf + X.e3inf*Y.e1 + X.einf*Y.e13 + X.scalar*Y.e13inf, X.e1*Y.enilinf - X.e1inf*Y.enil + X.e1nil*Y.einf + X.e1nilinf*Y.scalar + X.einf*Y.e1nil - X.enil*Y.e1inf + X.enilinf*Y.e1 + X.scalar*Y.e1nilinf, X.e2*Y.e3nil + X.e23*Y.enil + X.e23nil*Y.scalar - X.e2nil*Y.e3 - X.e3*Y.e2nil + X.e3nil*Y.e2 + X.enil*Y.e23 + X.scalar*Y.e23nil, X.e2*Y.e3inf + X.e23*Y.einf + X.e23inf*Y.scalar - X.e2inf*Y.e3 - X.e3*Y.e2inf + X.e3inf*Y.e2 + X.einf*Y.e23 + X.scalar*Y.e23inf, X.e2*Y.enilinf - X.e2inf*Y.enil + X.e2nil*Y.einf + X.e2nilinf*Y.scalar + X.einf*Y.e2nil - X.enil*Y.e2inf + X.enilinf*Y.e2 + X.scalar*Y.e2nilinf, X.e3*Y.enilinf - X.e3inf*Y.enil + X.e3nil*Y.einf + X.e3nilinf*Y.scalar + X.einf*Y.e3nil - X.enil*Y.e3inf + X.enilinf*Y.e3 + X.scalar*Y.e3nilinf, X.e1*Y.e23nil + X.e12*Y.e3nil + X.e123*Y.enil + X.e123nil*Y.scalar - X.e12nil*Y.e3 - X.e13*Y.e2nil + X.e13nil*Y.e2 + X.e1nil*Y.e23 - X.e2*Y.e13nil + X.e23*Y.e1nil - X.e23nil*Y.e1 - X.e2nil*Y.e13 + X.e3*Y.e12nil + X.e3nil*Y.e12 - X.enil*Y.e123 + X.scalar*Y.e123nil, X.e1*Y.e23inf + X.e12*Y.e3inf + X.e123*Y.einf + X.e123inf*Y.scalar - X.e12inf*Y.e3 - X.e13*Y.e2inf + X.e13inf*Y.e2 + X.e1inf*Y.e23 - X.e2*Y.e13inf + X.e23*Y.e1inf - X.e23inf*Y.e1 - X.e2inf*Y.e13 + X.e3*Y.e12inf + X.e3inf*Y.e12 - X.einf*Y.e123 + X.scalar*Y.e123inf, X.e1*Y.e2nilinf + X.e12*Y.enilinf - X.e12inf*Y.enil + X.e12nil*Y.einf + X.e12nilinf*Y.scalar + X.e1inf*Y.e2nil - X.e1nil*Y.e2inf + X.e1nilinf*Y.e2 - X.e2*Y.e1nilinf - X.e2inf*Y.e1nil + X.e2nil*Y.e1inf - X.e2nilinf*Y.e1 - X.einf*Y.e12nil + X.enil*Y.e12inf + X.enilinf*Y.e12 + X.scalar*Y.e12nilinf, X.e1*Y.e3nilinf + X.e13*Y.enilinf - X.e13inf*Y.enil + X.e13nil*Y.einf + X.e13nilinf*Y.scalar + X.e1inf*Y.e3nil - X.e1nil*Y.e3inf + X.e1nilinf*Y.e3 - X.e3*Y.e1nilinf - X.e3inf*Y.e1nil + X.e3nil*Y.e1inf - X.e3nilinf*Y.e1 - X.einf*Y.e13nil + X.enil*Y.e13inf + X.enilinf*Y.e13 + X.scalar*Y.e13nilinf, X.e2*Y.e3nilinf + X.e23*Y.enilinf - X.e23inf*Y.enil + X.e23nil*Y.einf + X.e23nilinf*Y.scalar + X.e2inf*Y.e3nil - X.e2nil*Y.e3inf + X.e2nilinf*Y.e3 - X.e3*Y.e2nilinf - X.e3inf*Y.e2nil + X.e3nil*Y.e2inf - X.e3nilinf*Y.e2 - X.einf*Y.e23nil + X.enil*Y.e23inf + X.enilinf*Y.e23 + X.scalar*Y.e23nilinf, X.e1*Y.e23nilinf + X.e12*Y.e3nilinf + X.e123*Y.enilinf - X.e123inf*Y.enil + X.e123nil*Y.einf + X.e123nilinf*Y.scalar + X.e12inf*Y.e3nil - X.e12nil*Y.e3inf + X.e12nilinf*Y.e3 - X.e13*Y.e2nilinf - X.e13inf*Y.e2nil + X.e13nil*Y.e2inf - X.e13nilinf*Y.e2 - X.e1inf*Y.e23nil + X.e1nil*Y.e23inf + X.e1nilinf*Y.e23 - X.e2*Y.e13nilinf + X.e23*Y.e1nilinf + X.e23inf*Y.e1nil - X.e23nil*Y.e1inf + X.e23nilinf*Y.e1 + X.e2inf*Y.e13nil - X.e2nil*Y.e13inf - X.e2nilinf*Y.e13 + X.e3*Y.e12nilinf - X.e3inf*Y.e12nil + X.e3nil*Y.e12inf + X.e3nilinf*Y.e12 + X.einf*Y.e123nil - X.enil*Y.e123inf + X.enilinf*Y.e123 + X.scalar*Y.e123nilinf);
}

#define I_CGA3 CGA3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

CGA3 rcontract(CGA3 X, CGA3 Y){
    return CGA3(X.e1*Y.e1 - X.e12*Y.e12 - X.e123*Y.e123 + X.e123inf*Y.e123nil + X.e123nil*Y.e123inf - X.e123nilinf*Y.e123nilinf - X.e12inf*Y.e12nil - X.e12nil*Y.e12inf - X.e12nilinf*Y.e12nilinf - X.e13*Y.e13 - X.e13inf*Y.e13nil - X.e13nil*Y.e13inf - X.e13nilinf*Y.e13nilinf - X.e1inf*Y.e1nil - X.e1nil*Y.e1inf + X.e1nilinf*Y.e1nilinf + X.e2*Y.e2 - X.e23*Y.e23 - X.e23inf*Y.e23nil - X.e23nil*Y.e23inf - X.e23nilinf*Y.e23nilinf - X.e2inf*Y.e2nil - X.e2nil*Y.e2inf + X.e2nilinf*Y.e2nilinf + X.e3*Y.e3 - X.e3inf*Y.e3nil - X.e3nil*Y.e3inf + X.e3nilinf*Y.e3nilinf + X.einf*Y.enil + X.enil*Y.einf + X.enilinf*Y.enilinf + X.scalar*Y.scalar, X.e1*Y.scalar + X.e12*Y.e2 - X.e123*Y.e23 - X.e123inf*Y.e23nil - X.e123nil*Y.e23inf - X.e123nilinf*Y.e23nilinf - X.e12inf*Y.e2nil - X.e12nil*Y.e2inf + X.e12nilinf*Y.e2nilinf + X.e13*Y.e3 - X.e13inf*Y.e3nil - X.e13nil*Y.e3inf + X.e13nilinf*Y.e3nilinf + X.e1inf*Y.enil + X.e1nil*Y.einf + X.e1nilinf*Y.enilinf, -X.e12*Y.e1 + X.e123*Y.e13 + X.e123inf*Y.e13nil + X.e123nil*Y.e13inf + X.e123nilinf*Y.e13nilinf + X.e12inf*Y.e1nil + X.e12nil*Y.e1inf - X.e12nilinf*Y.e1nilinf + X.e2*Y.scalar + X.e23*Y.e3 - X.e23inf*Y.e3nil - X.e23nil*Y.e3inf + X.e23nilinf*Y.e3nilinf + X.e2inf*Y.enil + X.e2nil*Y.einf + X.e2nilinf*Y.enilinf, -X.e123*Y.e12 - X.e123inf*Y.e12nil - X.e123nil*Y.e12inf - X.e123nilinf*Y.e12nilinf - X.e13*Y.e1 + X.e13inf*Y.e1nil + X.e13nil*Y.e1inf - X.e13nilinf*Y.e1nilinf - X.e23*Y.e2 + X.e23inf*Y.e2nil + X.e23nil*Y.e2inf - X.e23nilinf*Y.e2nilinf + X.e3*Y.scalar + X.e3inf*Y.enil + X.e3nil*Y.einf + X.e3nilinf*Y.enilinf, X.e123nil*Y.e123 - X.e123nilinf*Y.e123nil - X.e12nil*Y.e12 - X.e12nilinf*Y.e12nil - X.e13nil*Y.e13 - X.e13nilinf*Y.e13nil - X.e1nil*Y.e1 + X.e1nilinf*Y.e1nil - X.e23nil*Y.e23 - X.e23nilinf*Y.e23nil - X.e2nil*Y.e2 + X.e2nilinf*Y.e2nil - X.e3nil*Y.e3 + X.e3nilinf*Y.e3nil + X.enil*Y.scalar + X.enilinf*Y.enil, X.e123inf*Y.e123 + X.e123nilinf*Y.e123inf - X.e12inf*Y.e12 + X.e12nilinf*Y.e12inf - X.e13inf*Y.e13 + X.e13nilinf*Y.e13inf - X.e1inf*Y.e1 - X.e1nilinf*Y.e1inf - X.e23inf*Y.e23 + X.e23nilinf*Y.e23inf - X.e2inf*Y.e2 - X.e2nilinf*Y.e2inf - X.e3inf*Y.e3 - X.e3nilinf*Y.e3inf + X.einf*Y.scalar - X.enilinf*Y.einf, X.e12*Y.scalar + X.e123*Y.e3 - X.e123inf*Y.e3nil - X.e123nil*Y.e3inf + X.e123nilinf*Y.e3nilinf + X.e12inf*Y.enil + X.e12nil*Y.einf + X.e12nilinf*Y.enilinf, -X.e123*Y.e2 + X.e123inf*Y.e2nil + X.e123nil*Y.e2inf - X.e123nilinf*Y.e2nilinf + X.e13*Y.scalar + X.e13inf*Y.enil + X.e13nil*Y.einf + X.e13nilinf*Y.enilinf, -X.e123nil*Y.e23 - X.e123nilinf*Y.e23nil - X.e12nil*Y.e2 + X.e12nilinf*Y.e2nil - X.e13nil*Y.e3 + X.e13nilinf*Y.e3nil + X.e1nil*Y.scalar + X.e1nilinf*Y.enil, -X.e123inf*Y.e23 + X.e123nilinf*Y.e23inf - X.e12inf*Y.e2 - X.e12nilinf*Y.e2inf - X.e13inf*Y.e3 - X.e13nilinf*Y.e3inf + X.e1inf*Y.scalar - X.e1nilinf*Y.einf, X.e123*Y.e1 - X.e123inf*Y.e1nil - X.e123nil*Y.e1inf + X.e123nilinf*Y.e1nilinf + X.e23*Y.scalar + X.e23inf*Y.enil + X.e23nil*Y.einf + X.e23nilinf*Y.enilinf, X.e123nil*Y.e13 + X.e123nilinf*Y.e13nil + X.e12nil*Y.e1 - X.e12nilinf*Y.e1nil - X.e23nil*Y.e3 + X.e23nilinf*Y.e3nil + X.e2nil*Y.scalar + X.e2nilinf*Y.enil, X.e123inf*Y.e13 - X.e123nilinf*Y.e13inf + X.e12inf*Y.e1 + X.e12nilinf*Y.e1inf - X.e23inf*Y.e3 - X.e23nilinf*Y.e3inf + X.e2inf*Y.scalar - X.e2nilinf*Y.einf, -X.e123nil*Y.e12 - X.e123nilinf*Y.e12nil + X.e13nil*Y.e1 - X.e13nilinf*Y.e1nil + X.e23nil*Y.e2 - X.e23nilinf*Y.e2nil + X.e3nil*Y.scalar + X.e3nilinf*Y.enil, -X.e123inf*Y.e12 + X.e123nilinf*Y.e12inf + X.e13inf*Y.e1 + X.e13nilinf*Y.e1inf + X.e23inf*Y.e2 + X.e23nilinf*Y.e2inf + X.e3inf*Y.scalar - X.e3nilinf*Y.einf, -X.e123nilinf*Y.e123 - X.e12nilinf*Y.e12 - X.e13nilinf*Y.e13 + X.e1nilinf*Y.e1 - X.e23nilinf*Y.e23 + X.e2nilinf*Y.e2 + X.e3nilinf*Y.e3 + X.enilinf*Y.scalar, X.e123*Y.scalar + X.e123inf*Y.enil + X.e123nil*Y.einf + X.e123nilinf*Y.enilinf, -X.e123nil*Y.e3 + X.e123nilinf*Y.e3nil + X.e12nil*Y.scalar + X.e12nilinf*Y.enil, -X.e123inf*Y.e3 - X.e123nilinf*Y.e3inf + X.e12inf*Y.scalar - X.e12nilinf*Y.einf, X.e123nil*Y.e2 - X.e123nilinf*Y.e2nil + X.e13nil*Y.scalar + X.e13nilinf*Y.enil, X.e123inf*Y.e2 + X.e123nilinf*Y.e2inf + X.e13inf*Y.scalar - X.e13nilinf*Y.einf, -X.e123nilinf*Y.e23 + X.e12nilinf*Y.e2 + X.e13nilinf*Y.e3 + X.e1nilinf*Y.scalar, -X.e123nil*Y.e1 + X.e123nilinf*Y.e1nil + X.e23nil*Y.scalar + X.e23nilinf*Y.enil, -X.e123inf*Y.e1 - X.e123nilinf*Y.e1inf + X.e23inf*Y.scalar - X.e23nilinf*Y.einf, X.e123nilinf*Y.e13 - X.e12nilinf*Y.e1 + X.e23nilinf*Y.e3 + X.e2nilinf*Y.scalar, -X.e123nilinf*Y.e12 - X.e13nilinf*Y.e1 - X.e23nilinf*Y.e2 + X.e3nilinf*Y.scalar, X.e123nil*Y.scalar + X.e123nilinf*Y.enil, X.e123inf*Y.scalar - X.e123nilinf*Y.einf, X.e123nilinf*Y.e3 + X.e12nilinf*Y.scalar, -X.e123nilinf*Y.e2 + X.e13nilinf*Y.scalar, X.e123nilinf*Y.e1 + X.e23nilinf*Y.scalar, X.e123nilinf*Y.scalar);
}

CGA3 reverse(CGA3 X){
    return CGA3(X.scalar, X.e1, X.e2, X.e3, X.enil, X.einf, -X.e12, -X.e13, -X.e1nil, -X.e1inf, -X.e23, -X.e2nil, -X.e2inf, -X.e3nil, -X.e3inf, -X.enilinf, -X.e123, -X.e12nil, -X.e12inf, -X.e13nil, -X.e13inf, -X.e1nilinf, -X.e23nil, -X.e23inf, -X.e2nilinf, -X.e3nilinf, X.e123nil, X.e123inf, X.e12nilinf, X.e13nilinf, X.e23nilinf, X.e123nilinf);
}

CGA3 conjugate(CGA3 X){
    return reverse(involve(X));
}

CGA3 outer(CGA3 X, CGA3 Y, CGA3 Z){
    return outer(outer(X, Y), Z);
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

CGA3 point(CGA3 X){
    return CGA3(-X.e12*X.e12nil - X.e123nil*X.e123nilinf - X.e13*X.e13nil + X.e1nil*X.e1nilinf - X.e23*X.e23nil + X.e2nil*X.e2nilinf + X.e3nil*X.e3nilinf + X.scalar, X.e1 - X.e123nil*X.e23 - X.e123nilinf*X.e23nil - X.e12nil*X.e2 + X.e12nilinf*X.e2nil - X.e13nil*X.e3 + X.e13nilinf*X.e3nil + X.e1nilinf*X.enil, X.e1*X.e12nil + X.e123nil*X.e13 + X.e123nilinf*X.e13nil - X.e12nilinf*X.e1nil + X.e2 - X.e23nil*X.e3 + X.e23nilinf*X.e3nil + X.e2nilinf*X.enil, X.e1*X.e13nil - X.e12*X.e123nil - X.e123nilinf*X.e12nil - X.e13nilinf*X.e1nil + X.e2*X.e23nil - X.e23nilinf*X.e2nil + X.e3 + X.e3nilinf*X.enil, X.enil + 1.0, 0.5*pow(X.e1, 2.0) - X.e1*X.e1nilinf - 0.5*pow(X.e12, 2.0) + X.e12*X.e12nilinf - 0.5*pow(X.e123, 2.0) + X.e123*X.e123nilinf + X.e123inf*X.e123nil - 0.5*pow(X.e123nilinf, 2.0) - X.e12inf*X.e12nil - 0.5*pow(X.e12nilinf, 2.0) - 0.5*pow(X.e13, 2.0) + X.e13*X.e13nilinf - X.e13inf*X.e13nil - 0.5*pow(X.e13nilinf, 2.0) - X.e1inf*X.e1nil + 0.5*pow(X.e1nilinf, 2.0) + 0.5*pow(X.e2, 2.0) - X.e2*X.e2nilinf - 0.5*pow(X.e23, 2.0) + X.e23*X.e23nilinf - X.e23inf*X.e23nil - 0.5*pow(X.e23nilinf, 2.0) - X.e2inf*X.e2nil + 0.5*pow(X.e2nilinf, 2.0) + 0.5*pow(X.e3, 2.0) - X.e3*X.e3nilinf - X.e3inf*X.e3nil + 0.5*pow(X.e3nilinf, 2.0) + X.einf*X.enil + X.einf + 0.5*pow(X.enilinf, 2.0), X.e12 + X.e123nilinf*X.e3nil, -X.e123nilinf*X.e2nil + X.e13, X.e1nil, -X.e123*X.e23 + X.e123nilinf*X.e23 - X.e123nilinf*X.e23nilinf - X.e12inf*X.e2nil - X.e12nil*X.e2inf - X.e13inf*X.e3nil - X.e13nil*X.e3inf + X.e1inf + X.e1nilinf*X.enilinf, X.e123nilinf*X.e1nil + X.e23, X.e2nil, X.e123*X.e13 - X.e123nilinf*X.e13 + X.e123nilinf*X.e13nilinf + X.e12inf*X.e1nil + X.e12nil*X.e1inf - X.e23inf*X.e3nil - X.e23nil*X.e3inf + X.e2inf + X.e2nilinf*X.enilinf, X.e3nil, -X.e12*X.e123 + X.e12*X.e123nilinf - X.e123nilinf*X.e12nilinf + X.e13inf*X.e1nil + X.e13nil*X.e1inf + X.e23inf*X.e2nil + X.e23nil*X.e2inf + X.e3inf + X.e3nilinf*X.enilinf, -X.e12*X.e12nil - X.e123nil*X.e123nilinf - X.e13*X.e13nil + X.e1nil*X.e1nilinf - X.e23*X.e23nil + X.e2nil*X.e2nilinf + X.e3nil*X.e3nilinf + X.enilinf, X.e123 + X.e123nilinf*X.enil, X.e12nil, X.e123*X.e3 - X.e123inf*X.e3nil - X.e123nil*X.e3inf - X.e123nilinf*X.e3 + X.e123nilinf*X.e3nilinf + X.e12inf*X.enil + X.e12inf + X.e12nil*X.einf + X.e12nilinf*X.enilinf, X.e13nil, -X.e123*X.e2 + X.e123inf*X.e2nil + X.e123nil*X.e2inf + X.e123nilinf*X.e2 - X.e123nilinf*X.e2nilinf + X.e13inf*X.enil + X.e13inf + X.e13nil*X.einf + X.e13nilinf*X.enilinf, -X.e123nil*X.e23 - X.e123nilinf*X.e23nil - X.e12nil*X.e2 + X.e12nilinf*X.e2nil - X.e13nil*X.e3 + X.e13nilinf*X.e3nil + X.e1nilinf*X.enil + X.e1nilinf, X.e23nil, X.e1*X.e123 - X.e1*X.e123nilinf - X.e123inf*X.e1nil - X.e123nil*X.e1inf + X.e123nilinf*X.e1nilinf + X.e23inf*X.enil + X.e23inf + X.e23nil*X.einf + X.e23nilinf*X.enilinf, X.e1*X.e12nil + X.e123nil*X.e13 + X.e123nilinf*X.e13nil - X.e12nilinf*X.e1nil - X.e23nil*X.e3 + X.e23nilinf*X.e3nil + X.e2nilinf*X.enil + X.e2nilinf, X.e1*X.e13nil - X.e12*X.e123nil - X.e123nilinf*X.e12nil - X.e13nilinf*X.e1nil + X.e2*X.e23nil - X.e23nilinf*X.e2nil + X.e3nilinf*X.enil + X.e3nilinf, X.e123nil, X.e123inf + X.e123nilinf*X.enilinf, X.e123nilinf*X.e3nil + X.e12nilinf, -X.e123nilinf*X.e2nil + X.e13nilinf, X.e123nilinf*X.e1nil + X.e23nilinf, X.e123nilinf*X.enil + X.e123nilinf);
}

CGA3 point_coords(CGA3 X){
    return CGA3(X.scalar/X.enil, X.e1/X.enil, X.e2/X.enil, X.e3/X.enil, 1.0, X.einf/X.enil, X.e12/X.enil, X.e13/X.enil, X.e1nil/X.enil, X.e1inf/X.enil, X.e23/X.enil, X.e2nil/X.enil, X.e2inf/X.enil, X.e3nil/X.enil, X.e3inf/X.enil, X.enilinf/X.enil, X.e123/X.enil, X.e12nil/X.enil, X.e12inf/X.enil, X.e13nil/X.enil, X.e13inf/X.enil, X.e1nilinf/X.enil, X.e23nil/X.enil, X.e23inf/X.enil, X.e2nilinf/X.enil, X.e3nilinf/X.enil, X.e123nil/X.enil, X.e123inf/X.enil, X.e12nilinf/X.enil, X.e13nilinf/X.enil, X.e23nilinf/X.enil, X.e123nilinf/X.enil);
}

CGA3 inverse(CGA3 X){
    return mul(1.0/lcontract(X,conjugate(X)).scalar, conjugate(X));
}

CGA3 div(CGA3 X, CGA3 Y){
    return mul(X, inverse(Y));
}

CGA3 dual(CGA3 X){
    return div(X, I_CGA3);
}

vec2 matcap(vec3 eye, vec3 normal) {
  vec3 reflected = reflect(eye, normal);
  float m = 2.8284271247461903 * sqrt( reflected.z+1.0 );
  return reflected.xy / m + 0.5;
}

CGA3 outer(CGA3 p, CGA3 q, CGA3 r, CGA3 s){
    return outer(outer(p,q,r),s);
}

CGA3 mul(CGA3 p, CGA3 q, CGA3 r, CGA3 s){
    return mul(mul(p,q,r),s);
}

CGA3 fromVec(vec3 v){
    CGA3 X = ZERO_CGA3;
    X.e1 = v.x;
    X.e2 = v.y;
    X.e3 = v.z;
    return X;
}

CGA3 fromVec(vec4 v){
    CGA3 X = ZERO_CGA3;
    X.scalar = v.w;
    X.e1 = v.x;
    X.e2 = v.y;
    X.e3 = v.z;
    return X;
}

vec3 toVec(CGA3 x) {
    return vec3(x.e1, x.e2, x.e3);
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

// CGA3 inverse(CGA3 x) {
//   return mul(1.0/mul(x, reverse(x)).scalar, reverse(x));
// }

// CGA3 div(CGA3 a, CGA3 b) {
//   return mul(a, inverse(b));
// }
CGA3 weight(vec3 w) {
  // return one();
  return inverse(fromVec(normalize(w)));
  // return inverse(mul(fromVec(p1-p2), fromVec(w)));
}
CGA3 weight(vec4 w) {
  // return one();
  return inverse(fromVec(normalize(w)));
  // return inverse(mul(fromVec(p1-p2), fromVec(w)));
}

CGA3 weight(CGA3 w) {
  // return one();
  return inverse(w);
  // return inverse(mul(fromVec(p1-p2), fromVec(w)));
}

struct Patch {
  vec3 vertex;
  vec3 normal;
};

float eps = 0.01;

Patch bilinearQuad(CGA3 point0, CGA3 point1, CGA3 point2, CGA3 point3, CGA3 weight0, CGA3 weight1, CGA3 weight2, CGA3 weight3, float u, float v) {
  CGA3 W0 = mul((1.0 - u) * (1.0 - v), weight0); // 0, 0
  CGA3 W1 = mul(u * (1.0 - v), weight1); // 1, 0
  CGA3 W2 = mul((1.0 - u) * v, weight2); // 0, 1
  CGA3 W3 = mul(u * v, weight3); // 1, 1

  CGA3 top = add(mul(point0, W0), mul(point1, W1), mul(point2, W2), mul(point3, W3));

  float uu = u + eps;
  CGA3 W0_uu = mul((1.0 - uu) * (1.0 - v), weight0); // 0, 0
  CGA3 W1_uu = mul((uu) * (1.0 - v), weight1); // 1, 0
  CGA3 W2_uu = mul((1.0 - uu) * v, weight2); // 0, 1
  CGA3 W3_uu = mul((uu) * v, weight3); // 1, 1
  CGA3 top_uu = add(mul(point0, W0_uu), mul(point1, W1_uu), mul(point2, W2_uu), mul(point3, W3_uu));

  float vv = v + eps;
  CGA3 W0_vv = mul((1.0 - u) * (1.0 - vv), weight0); // 0, 0
  CGA3 W1_vv = mul(u * (1.0 - vv), weight1); // 1, 0
  CGA3 W2_vv = mul((1.0 - u) * (vv), weight2); // 0, 1
  CGA3 W3_vv = mul(u * (vv), weight3); // 1, 1

  CGA3 top_vv = add(mul(point0, W0_vv), mul(point1, W1_vv), mul(point2, W2_vv), mul(point3, W3_vv));

  CGA3 bottom = add(W0, W1, W2, W3);

  CGA3 X = div(top, bottom);
  CGA3 X_uu = div(top_uu, bottom);
  CGA3 X_vv = div(top_vv, bottom);
  vec3 x = vec3(X.e1, X.e2, X.e3);
  vec3 x_uu = vec3(X_uu.e1, X_uu.e2, X_uu.e3) - x;
  vec3 x_vv = vec3(X_vv.e1, X_vv.e2, X_vv.e3) - x;
  vec3 crossed = cross(x_uu, x_vv);
  vec3 normal = normalize(crossed);
  return Patch(x, normal);
}

Patch bilinearQuad(vec3 p0, vec3 p1, vec3 p2, vec3 p3, vec4 w0, vec4 w1, vec4 w2, vec4 w3, float u, float v) {
  return bilinearQuad(fromVec(p0), fromVec(p1), fromVec(p2), fromVec(p3), weight(w0), weight(w1), weight(w2), weight(w3), u, v);
}

Patch bilinearQuad(vec3 p0, vec3 p1, vec3 p2, vec3 p3, CGA3 w0, CGA3 w1, CGA3 w2, CGA3 w3, float u, float v) {
  return bilinearQuad(fromVec(p0), fromVec(p1), fromVec(p2), fromVec(p3), weight(w0), weight(w1), weight(w2), weight(w3), u, v);
}

Patch bilinearTri(CGA3 point0, CGA3 point1, CGA3 point2, CGA3 weight0, CGA3 weight1, CGA3 weight2, float u, float v) {
  CGA3 W0 = mul((1.0 - u - v), weight0); // 0, 0
  CGA3 W1 = mul(u, weight1); // 1, 0
  CGA3 W2 = mul(v, weight2); // 0, 1

  CGA3 top = add(mul(point0, W0), mul(point1, W1), mul(point2, W2));

  float uu = u + eps;
  CGA3 W0_uu = mul((1.0 - uu - v), weight0); // 0, 0
  CGA3 W1_uu = mul(uu, weight1); // 1, 0
  CGA3 W2_uu = mul(v, weight2); // 0, 1
  CGA3 top_uu = add(mul(point0, W0_uu), mul(point1, W1_uu), mul(point2, W2_uu));

  float vv = v + eps;
  CGA3 W0_vv = mul((1.0 - u - vv), weight0); // 0, 0
  CGA3 W1_vv = mul(u, weight1); // 1, 0
  CGA3 W2_vv = mul(vv, weight2); // 0, 1

  CGA3 top_vv = add(mul(point0, W0_vv), mul(point1, W1_vv), mul(point2, W2_vv));

  CGA3 bottom = add(W0, W1, W2);

  CGA3 X = div(top, bottom);
    // CGA3 X_vv= div(top_vv, bottom);
  CGA3 X_uu = div(top_uu, bottom);
  CGA3 X_vv = div(top_vv, bottom);
  vec3 x = vec3(X.e1, X.e2, X.e3);
  vec3 x_uu = vec3(X_uu.e1, X_uu.e2, X_uu.e3) - x;
  vec3 x_vv = vec3(X_vv.e1, X_vv.e2, X_vv.e3) - x;
  vec3 crossed = cross(x_uu, x_vv);
  vec3 normal = normalize(crossed);
  return Patch(x, normal);
}

Patch bilinearTri(vec3 p0, vec3 p1, vec3 p2, vec4 w0, vec4 w1, vec4 w2, float u, float v) {
  return bilinearTri(fromVec(p0), fromVec(p1), fromVec(p2), weight(w0), weight(w1), weight(w2), u, v);
}

Patch bilinearTri(vec3 p0, vec3 p1, vec3 p2, CGA3 w0, CGA3 w1, CGA3 w2, float u, float v) {
  return bilinearTri(fromVec(p0), fromVec(p1), fromVec(p2), weight(w0), weight(w1), weight(w2), u, v);
}