#define GLSLIFY 1
// Dual.glsl
const int Idx_Dual_scalar = 0;
const int Idx_Dual_enil1 = 1;
const int Idx_Dual_enil2 = 2;
const int Idx_Dual_enil1nil2 = 3;

struct Dual {
    float scalar;
    float enil1;
    float enil2;
    float enil1nil2;
};

Dual fromArray(float X[4]){
    return Dual(X[0], X[1], X[2], X[3]);
}

void toArray(Dual X, inout float X_ary[4]){
    X_ary[0] = X.scalar;
    X_ary[1] = X.enil1;
    X_ary[2] = X.enil2;
    X_ary[3] = X.enil1nil2;
}

void zero(inout float X[4]){
    X[0] = 0.0;
    X[1] = 0.0;
    X[2] = 0.0;
    X[3] = 0.0;
}

Dual add(Dual X, Dual Y){
    return Dual(X.scalar + Y.scalar, X.enil1 + Y.enil1, X.enil2 + Y.enil2, X.enil1nil2 + Y.enil1nil2);
}

Dual add(Dual X, Dual Y, Dual Z){
    return add(add(X, Y), Z);
}

Dual add(Dual X, Dual Y, Dual Z, Dual P){
    return add(add(add(X, Y), Z), P);
}

#define ONE_Dual Dual(1.0, 0.0, 0.0, 0.0)

Dual mul(float a, Dual X){
    return Dual(X.scalar*a, X.enil1*a, X.enil2*a, X.enil1nil2*a);
}

Dual sub(Dual X, Dual Y){
    return Dual(X.scalar - Y.scalar, X.enil1 - Y.enil1, X.enil2 - Y.enil2, X.enil1nil2 - Y.enil1nil2);
}

#define ZERO_Dual Dual(0.0, 0.0, 0.0, 0.0)

Dual mul(int a, Dual X){
    return mul(float(a), X);
}

Dual mul(Dual X, Dual Y){
    return Dual(X.scalar*Y.scalar, X.enil1*Y.scalar + X.scalar*Y.enil1, X.enil2*Y.scalar + X.scalar*Y.enil2, X.enil1*Y.enil2 + X.enil1nil2*Y.scalar - X.enil2*Y.enil1 + X.scalar*Y.enil1nil2);
}

Dual scalar_Dual(float a){
    return mul(a, ONE_Dual);
}

Dual mul(Dual X, Dual Y, Dual Z){
    return mul(mul(X, Y), Z);
}

Dual involve(Dual X){
    return Dual(X.scalar, -X.enil1, -X.enil2, X.enil1nil2);
}

Dual inner(Dual X, Dual Y){
    return Dual(0.0, 0.0, 0.0, 0.0);
}

Dual lcontract(Dual X, Dual Y){
    return Dual(X.scalar*Y.scalar, X.scalar*Y.enil1, X.scalar*Y.enil2, X.scalar*Y.enil1nil2);
}

Dual outer(Dual X, Dual Y){
    return Dual(X.scalar*Y.scalar, X.enil1*Y.scalar + X.scalar*Y.enil1, X.enil2*Y.scalar + X.scalar*Y.enil2, X.enil1*Y.enil2 + X.enil1nil2*Y.scalar - X.enil2*Y.enil1 + X.scalar*Y.enil1nil2);
}

#define I_Dual Dual(0.0, 0.0, 0.0, 1.0)

Dual rcontract(Dual X, Dual Y){
    return Dual(X.scalar*Y.scalar, X.enil1*Y.scalar, X.enil2*Y.scalar, X.enil1nil2*Y.scalar);
}

Dual reverse(Dual X){
    return Dual(X.scalar, X.enil1, X.enil2, -X.enil1nil2);
}

Dual conjugate(Dual X){
    return reverse(involve(X));
}

Dual outer(Dual X, Dual Y, Dual Z){
    return outer(outer(X, Y), Z);
}

Dual inv(Dual X){
    return mul(1.0/lcontract(X,conjugate(X)).scalar, conjugate(X));
}

Dual div(Dual X, Dual Y){
    return mul(X, inv(Y));
}

Dual dual(Dual X){
    return div(X, I_Dual);
}

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
    Dual scalar;
    Dual e1;
    Dual e2;
    Dual e3;
    Dual enil;
    Dual einf;
    Dual e12;
    Dual e13;
    Dual e1nil;
    Dual e1inf;
    Dual e23;
    Dual e2nil;
    Dual e2inf;
    Dual e3nil;
    Dual e3inf;
    Dual enilinf;
    Dual e123;
    Dual e12nil;
    Dual e12inf;
    Dual e13nil;
    Dual e13inf;
    Dual e1nilinf;
    Dual e23nil;
    Dual e23inf;
    Dual e2nilinf;
    Dual e3nilinf;
    Dual e123nil;
    Dual e123inf;
    Dual e12nilinf;
    Dual e13nilinf;
    Dual e23nilinf;
    Dual e123nilinf;
};

CGA3 fromArray(Dual X[32]){
    return CGA3(X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7], X[8], X[9], X[10], X[11], X[12], X[13], X[14], X[15], X[16], X[17], X[18], X[19], X[20], X[21], X[22], X[23], X[24], X[25], X[26], X[27], X[28], X[29], X[30], X[31]);
}

void toArray(CGA3 X, inout Dual X_ary[32]){
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

void zero(inout Dual X[32]){
    X[0] = ZERO_Dual;
    X[1] = ZERO_Dual;
    X[2] = ZERO_Dual;
    X[3] = ZERO_Dual;
    X[4] = ZERO_Dual;
    X[5] = ZERO_Dual;
    X[6] = ZERO_Dual;
    X[7] = ZERO_Dual;
    X[8] = ZERO_Dual;
    X[9] = ZERO_Dual;
    X[10] = ZERO_Dual;
    X[11] = ZERO_Dual;
    X[12] = ZERO_Dual;
    X[13] = ZERO_Dual;
    X[14] = ZERO_Dual;
    X[15] = ZERO_Dual;
    X[16] = ZERO_Dual;
    X[17] = ZERO_Dual;
    X[18] = ZERO_Dual;
    X[19] = ZERO_Dual;
    X[20] = ZERO_Dual;
    X[21] = ZERO_Dual;
    X[22] = ZERO_Dual;
    X[23] = ZERO_Dual;
    X[24] = ZERO_Dual;
    X[25] = ZERO_Dual;
    X[26] = ZERO_Dual;
    X[27] = ZERO_Dual;
    X[28] = ZERO_Dual;
    X[29] = ZERO_Dual;
    X[30] = ZERO_Dual;
    X[31] = ZERO_Dual;
}

CGA3 add(CGA3 X, CGA3 Y){
    return CGA3(add(X.scalar, Y.scalar), add(X.e1, Y.e1), add(X.e2, Y.e2), add(X.e3, Y.e3), add(X.enil, Y.enil), add(X.einf, Y.einf), add(X.e12, Y.e12), add(X.e13, Y.e13), add(X.e1nil, Y.e1nil), add(X.e1inf, Y.e1inf), add(X.e23, Y.e23), add(X.e2nil, Y.e2nil), add(X.e2inf, Y.e2inf), add(X.e3nil, Y.e3nil), add(X.e3inf, Y.e3inf), add(X.enilinf, Y.enilinf), add(X.e123, Y.e123), add(X.e12nil, Y.e12nil), add(X.e12inf, Y.e12inf), add(X.e13nil, Y.e13nil), add(X.e13inf, Y.e13inf), add(X.e1nilinf, Y.e1nilinf), add(X.e23nil, Y.e23nil), add(X.e23inf, Y.e23inf), add(X.e2nilinf, Y.e2nilinf), add(X.e3nilinf, Y.e3nilinf), add(X.e123nil, Y.e123nil), add(X.e123inf, Y.e123inf), add(X.e12nilinf, Y.e12nilinf), add(X.e13nilinf, Y.e13nilinf), add(X.e23nilinf, Y.e23nilinf), add(X.e123nilinf, Y.e123nilinf));
}

CGA3 add(CGA3 X, CGA3 Y, CGA3 Z){
    return add(add(X, Y), Z);
}

CGA3 add(CGA3 X, CGA3 Y, CGA3 Z, CGA3 P){
    return add(add(add(X, Y), Z), P);
}

#define ONE_CGA3 CGA3(ONE_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual)

CGA3 mul(Dual a, CGA3 X){
    return CGA3(mul(X.scalar, a), mul(X.e1, a), mul(X.e2, a), mul(X.e3, a), mul(X.enil, a), mul(X.einf, a), mul(X.e12, a), mul(X.e13, a), mul(X.e1nil, a), mul(X.e1inf, a), mul(X.e23, a), mul(X.e2nil, a), mul(X.e2inf, a), mul(X.e3nil, a), mul(X.e3inf, a), mul(X.enilinf, a), mul(X.e123, a), mul(X.e12nil, a), mul(X.e12inf, a), mul(X.e13nil, a), mul(X.e13inf, a), mul(X.e1nilinf, a), mul(X.e23nil, a), mul(X.e23inf, a), mul(X.e2nilinf, a), mul(X.e3nilinf, a), mul(X.e123nil, a), mul(X.e123inf, a), mul(X.e12nilinf, a), mul(X.e13nilinf, a), mul(X.e23nilinf, a), mul(X.e123nilinf, a));
}

CGA3 sub(CGA3 X, CGA3 Y){
    return CGA3(sub(X.scalar, Y.scalar), sub(X.e1, Y.e1), sub(X.e2, Y.e2), sub(X.e3, Y.e3), sub(X.enil, Y.enil), sub(X.einf, Y.einf), sub(X.e12, Y.e12), sub(X.e13, Y.e13), sub(X.e1nil, Y.e1nil), sub(X.e1inf, Y.e1inf), sub(X.e23, Y.e23), sub(X.e2nil, Y.e2nil), sub(X.e2inf, Y.e2inf), sub(X.e3nil, Y.e3nil), sub(X.e3inf, Y.e3inf), sub(X.enilinf, Y.enilinf), sub(X.e123, Y.e123), sub(X.e12nil, Y.e12nil), sub(X.e12inf, Y.e12inf), sub(X.e13nil, Y.e13nil), sub(X.e13inf, Y.e13inf), sub(X.e1nilinf, Y.e1nilinf), sub(X.e23nil, Y.e23nil), sub(X.e23inf, Y.e23inf), sub(X.e2nilinf, Y.e2nilinf), sub(X.e3nilinf, Y.e3nilinf), sub(X.e123nil, Y.e123nil), sub(X.e123inf, Y.e123inf), sub(X.e12nilinf, Y.e12nilinf), sub(X.e13nilinf, Y.e13nilinf), sub(X.e23nilinf, Y.e23nilinf), sub(X.e123nilinf, Y.e123nilinf));
}

#define ZERO_CGA3 CGA3(ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual)

CGA3 mul(float a, CGA3 X){
    return mul(mul(a, ONE_Dual), X);
}

CGA3 mul(int a, CGA3 X){
    return mul(float(a), X);
}

CGA3 mul(CGA3 X, CGA3 Y){
    return CGA3(sub(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e1), mul(X.e123inf, Y.e123nil)), mul(X.e123nil, Y.e123inf)), mul(X.e1nilinf, Y.e1nilinf)), mul(X.e2, Y.e2)), mul(X.e2nilinf, Y.e2nilinf)), mul(X.e3, Y.e3)), mul(X.e3nilinf, Y.e3nilinf)), mul(X.einf, Y.enil)), mul(X.enil, Y.einf)), mul(X.enilinf, Y.enilinf)), mul(X.scalar, Y.scalar)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e12), mul(X.e123, Y.e123)), mul(X.e123nilinf, Y.e123nilinf)), mul(X.e12inf, Y.e12nil)), mul(X.e12nil, Y.e12inf)), mul(X.e12nilinf, Y.e12nilinf)), mul(X.e13, Y.e13)), mul(X.e13inf, Y.e13nil)), mul(X.e13nil, Y.e13inf)), mul(X.e13nilinf, Y.e13nilinf)), mul(X.e1inf, Y.e1nil)), mul(X.e1nil, Y.e1inf)), mul(X.e23, Y.e23)), mul(X.e23inf, Y.e23nil)), mul(X.e23nil, Y.e23inf)), mul(X.e23nilinf, Y.e23nilinf)), mul(X.e2inf, Y.e2nil)), mul(X.e2nil, Y.e2inf)), mul(X.e3inf, Y.e3nil)), mul(X.e3nil, Y.e3inf))), sub(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.scalar), mul(X.e12, Y.e2)), mul(X.e12nilinf, Y.e2nilinf)), mul(X.e13, Y.e3)), mul(X.e13nilinf, Y.e3nilinf)), mul(X.e1inf, Y.enil)), mul(X.e1nil, Y.einf)), mul(X.e1nilinf, Y.enilinf)), mul(X.e23inf, Y.e123nil)), mul(X.e23nil, Y.e123inf)), mul(X.enilinf, Y.e1nilinf)), mul(X.scalar, Y.e1)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e123, Y.e23), mul(X.e123inf, Y.e23nil)), mul(X.e123nil, Y.e23inf)), mul(X.e123nilinf, Y.e23nilinf)), mul(X.e12inf, Y.e2nil)), mul(X.e12nil, Y.e2inf)), mul(X.e13inf, Y.e3nil)), mul(X.e13nil, Y.e3inf)), mul(X.e2, Y.e12)), mul(X.e23, Y.e123)), mul(X.e23nilinf, Y.e123nilinf)), mul(X.e2inf, Y.e12nil)), mul(X.e2nil, Y.e12inf)), mul(X.e2nilinf, Y.e12nilinf)), mul(X.e3, Y.e13)), mul(X.e3inf, Y.e13nil)), mul(X.e3nil, Y.e13inf)), mul(X.e3nilinf, Y.e13nilinf)), mul(X.einf, Y.e1nil)), mul(X.enil, Y.e1inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e12), mul(X.e123, Y.e13)), mul(X.e123inf, Y.e13nil)), mul(X.e123nil, Y.e13inf)), mul(X.e123nilinf, Y.e13nilinf)), mul(X.e12inf, Y.e1nil)), mul(X.e12nil, Y.e1inf)), mul(X.e13, Y.e123)), mul(X.e13nilinf, Y.e123nilinf)), mul(X.e1inf, Y.e12nil)), mul(X.e1nil, Y.e12inf)), mul(X.e1nilinf, Y.e12nilinf)), mul(X.e2, Y.scalar)), mul(X.e23, Y.e3)), mul(X.e23nilinf, Y.e3nilinf)), mul(X.e2inf, Y.enil)), mul(X.e2nil, Y.einf)), mul(X.e2nilinf, Y.enilinf)), mul(X.enilinf, Y.e2nilinf)), mul(X.scalar, Y.e2)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e1), mul(X.e12nilinf, Y.e1nilinf)), mul(X.e13inf, Y.e123nil)), mul(X.e13nil, Y.e123inf)), mul(X.e23inf, Y.e3nil)), mul(X.e23nil, Y.e3inf)), mul(X.e3, Y.e23)), mul(X.e3inf, Y.e23nil)), mul(X.e3nil, Y.e23inf)), mul(X.e3nilinf, Y.e23nilinf)), mul(X.einf, Y.e2nil)), mul(X.enil, Y.e2inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e13), mul(X.e12inf, Y.e123nil)), mul(X.e12nil, Y.e123inf)), mul(X.e13inf, Y.e1nil)), mul(X.e13nil, Y.e1inf)), mul(X.e1inf, Y.e13nil)), mul(X.e1nil, Y.e13inf)), mul(X.e1nilinf, Y.e13nilinf)), mul(X.e2, Y.e23)), mul(X.e23inf, Y.e2nil)), mul(X.e23nil, Y.e2inf)), mul(X.e2inf, Y.e23nil)), mul(X.e2nil, Y.e23inf)), mul(X.e2nilinf, Y.e23nilinf)), mul(X.e3, Y.scalar)), mul(X.e3inf, Y.enil)), mul(X.e3nil, Y.einf)), mul(X.e3nilinf, Y.enilinf)), mul(X.enilinf, Y.e3nilinf)), mul(X.scalar, Y.e3)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e123), mul(X.e123, Y.e12)), mul(X.e123inf, Y.e12nil)), mul(X.e123nil, Y.e12inf)), mul(X.e123nilinf, Y.e12nilinf)), mul(X.e12nilinf, Y.e123nilinf)), mul(X.e13, Y.e1)), mul(X.e13nilinf, Y.e1nilinf)), mul(X.e23, Y.e2)), mul(X.e23nilinf, Y.e2nilinf)), mul(X.einf, Y.e3nil)), mul(X.enil, Y.e3inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e1nil), mul(X.e123nil, Y.e123)), mul(X.e12nil, Y.e12nilinf)), mul(X.e13nil, Y.e13nilinf)), mul(X.e1nil, Y.e1nilinf)), mul(X.e1nilinf, Y.e1nil)), mul(X.e2, Y.e2nil)), mul(X.e23nil, Y.e23nilinf)), mul(X.e2nil, Y.e2nilinf)), mul(X.e2nilinf, Y.e2nil)), mul(X.e3, Y.e3nil)), mul(X.e3nil, Y.e3nilinf)), mul(X.e3nilinf, Y.e3nil)), mul(X.enil, Y.scalar)), mul(X.enilinf, Y.enil)), mul(X.scalar, Y.enil)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e12nil), mul(X.e123, Y.e123nil)), mul(X.e123nil, Y.e123nilinf)), mul(X.e123nilinf, Y.e123nil)), mul(X.e12nil, Y.e12)), mul(X.e12nilinf, Y.e12nil)), mul(X.e13, Y.e13nil)), mul(X.e13nil, Y.e13)), mul(X.e13nilinf, Y.e13nil)), mul(X.e1nil, Y.e1)), mul(X.e23, Y.e23nil)), mul(X.e23nil, Y.e23)), mul(X.e23nilinf, Y.e23nil)), mul(X.e2nil, Y.e2)), mul(X.e3nil, Y.e3)), mul(X.enil, Y.enilinf))), sub(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e1inf), mul(X.e123inf, Y.e123)), mul(X.e123inf, Y.e123nilinf)), mul(X.e123nilinf, Y.e123inf)), mul(X.e12nilinf, Y.e12inf)), mul(X.e13nilinf, Y.e13inf)), mul(X.e2, Y.e2inf)), mul(X.e23nilinf, Y.e23inf)), mul(X.e3, Y.e3inf)), mul(X.einf, Y.enilinf)), mul(X.einf, Y.scalar)), mul(X.scalar, Y.einf)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e12inf), mul(X.e123, Y.e123inf)), mul(X.e12inf, Y.e12)), mul(X.e12inf, Y.e12nilinf)), mul(X.e13, Y.e13inf)), mul(X.e13inf, Y.e13)), mul(X.e13inf, Y.e13nilinf)), mul(X.e1inf, Y.e1)), mul(X.e1inf, Y.e1nilinf)), mul(X.e1nilinf, Y.e1inf)), mul(X.e23, Y.e23inf)), mul(X.e23inf, Y.e23)), mul(X.e23inf, Y.e23nilinf)), mul(X.e2inf, Y.e2)), mul(X.e2inf, Y.e2nilinf)), mul(X.e2nilinf, Y.e2inf)), mul(X.e3inf, Y.e3)), mul(X.e3inf, Y.e3nilinf)), mul(X.e3nilinf, Y.e3inf)), mul(X.enilinf, Y.einf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e2), mul(X.e12, Y.scalar)), mul(X.e123, Y.e3)), mul(X.e123nilinf, Y.e3nilinf)), mul(X.e12inf, Y.enil)), mul(X.e12nil, Y.einf)), mul(X.e12nilinf, Y.enilinf)), mul(X.e1nilinf, Y.e2nilinf)), mul(X.e23, Y.e13)), mul(X.e23inf, Y.e13nil)), mul(X.e23nil, Y.e13inf)), mul(X.e23nilinf, Y.e13nilinf)), mul(X.e2inf, Y.e1nil)), mul(X.e2nil, Y.e1inf)), mul(X.e3, Y.e123)), mul(X.e3nilinf, Y.e123nilinf)), mul(X.einf, Y.e12nil)), mul(X.enil, Y.e12inf)), mul(X.enilinf, Y.e12nilinf)), mul(X.scalar, Y.e12)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e123inf, Y.e3nil), mul(X.e123nil, Y.e3inf)), mul(X.e13, Y.e23)), mul(X.e13inf, Y.e23nil)), mul(X.e13nil, Y.e23inf)), mul(X.e13nilinf, Y.e23nilinf)), mul(X.e1inf, Y.e2nil)), mul(X.e1nil, Y.e2inf)), mul(X.e2, Y.e1)), mul(X.e2nilinf, Y.e1nilinf)), mul(X.e3inf, Y.e123nil)), mul(X.e3nil, Y.e123inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e3), mul(X.e12, Y.e23)), mul(X.e123inf, Y.e2nil)), mul(X.e123nil, Y.e2inf)), mul(X.e12inf, Y.e23nil)), mul(X.e12nil, Y.e23inf)), mul(X.e12nilinf, Y.e23nilinf)), mul(X.e13, Y.scalar)), mul(X.e13inf, Y.enil)), mul(X.e13nil, Y.einf)), mul(X.e13nilinf, Y.enilinf)), mul(X.e1nilinf, Y.e3nilinf)), mul(X.e2inf, Y.e123nil)), mul(X.e2nil, Y.e123inf)), mul(X.e3inf, Y.e1nil)), mul(X.e3nil, Y.e1inf)), mul(X.einf, Y.e13nil)), mul(X.enil, Y.e13inf)), mul(X.enilinf, Y.e13nilinf)), mul(X.scalar, Y.e13)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e123, Y.e2), mul(X.e123nilinf, Y.e2nilinf)), mul(X.e1inf, Y.e3nil)), mul(X.e1nil, Y.e3inf)), mul(X.e2, Y.e123)), mul(X.e23, Y.e12)), mul(X.e23inf, Y.e12nil)), mul(X.e23nil, Y.e12inf)), mul(X.e23nilinf, Y.e12nilinf)), mul(X.e2nilinf, Y.e123nilinf)), mul(X.e3, Y.e1)), mul(X.e3nilinf, Y.e1nilinf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.enil), mul(X.e12, Y.e2nil)), mul(X.e123nil, Y.e23nilinf)), mul(X.e12nil, Y.e2nilinf)), mul(X.e12nilinf, Y.e2nil)), mul(X.e13, Y.e3nil)), mul(X.e13nil, Y.e3nilinf)), mul(X.e13nilinf, Y.e3nil)), mul(X.e1nil, Y.scalar)), mul(X.e1nilinf, Y.enil)), mul(X.e23nil, Y.e123)), mul(X.e2nil, Y.e12nilinf)), mul(X.e3nil, Y.e13nilinf)), mul(X.enil, Y.e1nilinf)), mul(X.enilinf, Y.e1nil)), mul(X.scalar, Y.e1nil)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e123, Y.e23nil), mul(X.e123nil, Y.e23)), mul(X.e123nilinf, Y.e23nil)), mul(X.e12nil, Y.e2)), mul(X.e13nil, Y.e3)), mul(X.e1nil, Y.enilinf)), mul(X.e2, Y.e12nil)), mul(X.e23, Y.e123nil)), mul(X.e23nil, Y.e123nilinf)), mul(X.e23nilinf, Y.e123nil)), mul(X.e2nil, Y.e12)), mul(X.e2nilinf, Y.e12nil)), mul(X.e3, Y.e13nil)), mul(X.e3nil, Y.e13)), mul(X.e3nilinf, Y.e13nil)), mul(X.enil, Y.e1))), sub(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.einf), mul(X.e12, Y.e2inf)), mul(X.e123nilinf, Y.e23inf)), mul(X.e13, Y.e3inf)), mul(X.e1inf, Y.enilinf)), mul(X.e1inf, Y.scalar)), mul(X.e23inf, Y.e123)), mul(X.e23inf, Y.e123nilinf)), mul(X.e23nilinf, Y.e123inf)), mul(X.e2nilinf, Y.e12inf)), mul(X.e3nilinf, Y.e13inf)), mul(X.scalar, Y.e1inf)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e123, Y.e23inf), mul(X.e123inf, Y.e23)), mul(X.e123inf, Y.e23nilinf)), mul(X.e12inf, Y.e2)), mul(X.e12inf, Y.e2nilinf)), mul(X.e12nilinf, Y.e2inf)), mul(X.e13inf, Y.e3)), mul(X.e13inf, Y.e3nilinf)), mul(X.e13nilinf, Y.e3inf)), mul(X.e1nilinf, Y.einf)), mul(X.e2, Y.e12inf)), mul(X.e23, Y.e123inf)), mul(X.e2inf, Y.e12)), mul(X.e2inf, Y.e12nilinf)), mul(X.e3, Y.e13inf)), mul(X.e3inf, Y.e13)), mul(X.e3inf, Y.e13nilinf)), mul(X.einf, Y.e1)), mul(X.einf, Y.e1nilinf)), mul(X.enilinf, Y.e1inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e123), mul(X.e123, Y.e1)), mul(X.e123nilinf, Y.e1nilinf)), mul(X.e13, Y.e12)), mul(X.e13inf, Y.e12nil)), mul(X.e13nil, Y.e12inf)), mul(X.e13nilinf, Y.e12nilinf)), mul(X.e1nilinf, Y.e123nilinf)), mul(X.e2, Y.e3)), mul(X.e23, Y.scalar)), mul(X.e23inf, Y.enil)), mul(X.e23nil, Y.einf)), mul(X.e23nilinf, Y.enilinf)), mul(X.e2nilinf, Y.e3nilinf)), mul(X.e3inf, Y.e2nil)), mul(X.e3nil, Y.e2inf)), mul(X.einf, Y.e23nil)), mul(X.enil, Y.e23inf)), mul(X.enilinf, Y.e23nilinf)), mul(X.scalar, Y.e23)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e13), mul(X.e123inf, Y.e1nil)), mul(X.e123nil, Y.e1inf)), mul(X.e12inf, Y.e13nil)), mul(X.e12nil, Y.e13inf)), mul(X.e12nilinf, Y.e13nilinf)), mul(X.e1inf, Y.e123nil)), mul(X.e1nil, Y.e123inf)), mul(X.e2inf, Y.e3nil)), mul(X.e2nil, Y.e3inf)), mul(X.e3, Y.e2)), mul(X.e3nilinf, Y.e2nilinf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e12nil), mul(X.e123, Y.e13nil)), mul(X.e123nil, Y.e13)), mul(X.e123nilinf, Y.e13nil)), mul(X.e12nil, Y.e1)), mul(X.e13, Y.e123nil)), mul(X.e13nil, Y.e123nilinf)), mul(X.e13nilinf, Y.e123nil)), mul(X.e1nil, Y.e12)), mul(X.e1nilinf, Y.e12nil)), mul(X.e2, Y.enil)), mul(X.e23, Y.e3nil)), mul(X.e23nil, Y.e3nilinf)), mul(X.e23nilinf, Y.e3nil)), mul(X.e2nil, Y.scalar)), mul(X.e2nilinf, Y.enil)), mul(X.e3nil, Y.e23nilinf)), mul(X.enil, Y.e2nilinf)), mul(X.enilinf, Y.e2nil)), mul(X.scalar, Y.e2nil)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e1nil), mul(X.e123nil, Y.e13nilinf)), mul(X.e12nil, Y.e1nilinf)), mul(X.e12nilinf, Y.e1nil)), mul(X.e13nil, Y.e123)), mul(X.e1nil, Y.e12nilinf)), mul(X.e23nil, Y.e3)), mul(X.e2nil, Y.enilinf)), mul(X.e3, Y.e23nil)), mul(X.e3nil, Y.e23)), mul(X.e3nilinf, Y.e23nil)), mul(X.enil, Y.e2))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e12inf), mul(X.e123, Y.e13inf)), mul(X.e123inf, Y.e13)), mul(X.e123inf, Y.e13nilinf)), mul(X.e12inf, Y.e1)), mul(X.e12inf, Y.e1nilinf)), mul(X.e12nilinf, Y.e1inf)), mul(X.e13, Y.e123inf)), mul(X.e1inf, Y.e12)), mul(X.e1inf, Y.e12nilinf)), mul(X.e2, Y.einf)), mul(X.e23, Y.e3inf)), mul(X.e2inf, Y.enilinf)), mul(X.e2inf, Y.scalar)), mul(X.e3nilinf, Y.e23inf)), mul(X.scalar, Y.e2inf)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e1inf), mul(X.e123nilinf, Y.e13inf)), mul(X.e13inf, Y.e123)), mul(X.e13inf, Y.e123nilinf)), mul(X.e13nilinf, Y.e123inf)), mul(X.e1nilinf, Y.e12inf)), mul(X.e23inf, Y.e3)), mul(X.e23inf, Y.e3nilinf)), mul(X.e23nilinf, Y.e3inf)), mul(X.e2nilinf, Y.einf)), mul(X.e3, Y.e23inf)), mul(X.e3inf, Y.e23)), mul(X.e3inf, Y.e23nilinf)), mul(X.einf, Y.e2)), mul(X.einf, Y.e2nilinf)), mul(X.enilinf, Y.e2inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e13nil), mul(X.e123nil, Y.e12nilinf)), mul(X.e12nil, Y.e123)), mul(X.e13nil, Y.e1)), mul(X.e1nil, Y.e13)), mul(X.e1nilinf, Y.e13nil)), mul(X.e2, Y.e23nil)), mul(X.e23nil, Y.e2)), mul(X.e2nil, Y.e23)), mul(X.e2nilinf, Y.e23nil)), mul(X.e3, Y.enil)), mul(X.e3nil, Y.scalar)), mul(X.e3nilinf, Y.enil)), mul(X.enil, Y.e3nilinf)), mul(X.enilinf, Y.e3nil)), mul(X.scalar, Y.e3nil)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e123nil), mul(X.e123, Y.e12nil)), mul(X.e123nil, Y.e12)), mul(X.e123nilinf, Y.e12nil)), mul(X.e12nil, Y.e123nilinf)), mul(X.e12nilinf, Y.e123nil)), mul(X.e13, Y.e1nil)), mul(X.e13nil, Y.e1nilinf)), mul(X.e13nilinf, Y.e1nil)), mul(X.e1nil, Y.e13nilinf)), mul(X.e23, Y.e2nil)), mul(X.e23nil, Y.e2nilinf)), mul(X.e23nilinf, Y.e2nil)), mul(X.e2nil, Y.e23nilinf)), mul(X.e3nil, Y.enilinf)), mul(X.enil, Y.e3))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e13inf), mul(X.e123nilinf, Y.e12inf)), mul(X.e12inf, Y.e123)), mul(X.e12inf, Y.e123nilinf)), mul(X.e12nilinf, Y.e123inf)), mul(X.e13inf, Y.e1)), mul(X.e13inf, Y.e1nilinf)), mul(X.e13nilinf, Y.e1inf)), mul(X.e1inf, Y.e13)), mul(X.e1inf, Y.e13nilinf)), mul(X.e2, Y.e23inf)), mul(X.e23inf, Y.e2)), mul(X.e23inf, Y.e2nilinf)), mul(X.e23nilinf, Y.e2inf)), mul(X.e2inf, Y.e23)), mul(X.e2inf, Y.e23nilinf)), mul(X.e3, Y.einf)), mul(X.e3inf, Y.enilinf)), mul(X.e3inf, Y.scalar)), mul(X.scalar, Y.e3inf)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e123inf), mul(X.e123, Y.e12inf)), mul(X.e123inf, Y.e12)), mul(X.e123inf, Y.e12nilinf)), mul(X.e13, Y.e1inf)), mul(X.e1nilinf, Y.e13inf)), mul(X.e23, Y.e2inf)), mul(X.e2nilinf, Y.e23inf)), mul(X.e3nilinf, Y.einf)), mul(X.einf, Y.e3)), mul(X.einf, Y.e3nilinf)), mul(X.enilinf, Y.e3inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e1nilinf), mul(X.e123nil, Y.e123inf)), mul(X.e12inf, Y.e12nil)), mul(X.e13inf, Y.e13nil)), mul(X.e1inf, Y.e1nil)), mul(X.e1nilinf, Y.e1)), mul(X.e2, Y.e2nilinf)), mul(X.e23inf, Y.e23nil)), mul(X.e2inf, Y.e2nil)), mul(X.e2nilinf, Y.e2)), mul(X.e3, Y.e3nilinf)), mul(X.e3inf, Y.e3nil)), mul(X.e3nilinf, Y.e3)), mul(X.enil, Y.einf)), mul(X.enilinf, Y.scalar)), mul(X.scalar, Y.enilinf)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e12nilinf), mul(X.e123, Y.e123nilinf)), mul(X.e123inf, Y.e123nil)), mul(X.e123nilinf, Y.e123)), mul(X.e12nil, Y.e12inf)), mul(X.e12nilinf, Y.e12)), mul(X.e13, Y.e13nilinf)), mul(X.e13nil, Y.e13inf)), mul(X.e13nilinf, Y.e13)), mul(X.e1nil, Y.e1inf)), mul(X.e23, Y.e23nilinf)), mul(X.e23nil, Y.e23inf)), mul(X.e23nilinf, Y.e23)), mul(X.e2nil, Y.e2inf)), mul(X.e3nil, Y.e3inf)), mul(X.einf, Y.enil))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e23), mul(X.e12, Y.e3)), mul(X.e123, Y.scalar)), mul(X.e123inf, Y.enil)), mul(X.e123nil, Y.einf)), mul(X.e123nilinf, Y.enilinf)), mul(X.e12nilinf, Y.e3nilinf)), mul(X.e13inf, Y.e2nil)), mul(X.e13nil, Y.e2inf)), mul(X.e1inf, Y.e23nil)), mul(X.e1nil, Y.e23inf)), mul(X.e1nilinf, Y.e23nilinf)), mul(X.e23, Y.e1)), mul(X.e23nilinf, Y.e1nilinf)), mul(X.e3, Y.e12)), mul(X.e3inf, Y.e12nil)), mul(X.e3nil, Y.e12inf)), mul(X.e3nilinf, Y.e12nilinf)), mul(X.enilinf, Y.e123nilinf)), mul(X.scalar, Y.e123)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e12inf, Y.e3nil), mul(X.e12nil, Y.e3inf)), mul(X.e13, Y.e2)), mul(X.e13nilinf, Y.e2nilinf)), mul(X.e2, Y.e13)), mul(X.e23inf, Y.e1nil)), mul(X.e23nil, Y.e1inf)), mul(X.e2inf, Y.e13nil)), mul(X.e2nil, Y.e13inf)), mul(X.e2nilinf, Y.e13nilinf)), mul(X.einf, Y.e123nil)), mul(X.enil, Y.e123inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e2nil), mul(X.e12, Y.enil)), mul(X.e123, Y.e3nil)), mul(X.e123nil, Y.e3nilinf)), mul(X.e123nilinf, Y.e3nil)), mul(X.e12nil, Y.scalar)), mul(X.e12nilinf, Y.enil)), mul(X.e13nil, Y.e23nilinf)), mul(X.e1nil, Y.e2nilinf)), mul(X.e1nilinf, Y.e2nil)), mul(X.e23, Y.e13nil)), mul(X.e23nil, Y.e13)), mul(X.e23nilinf, Y.e13nil)), mul(X.e2nil, Y.e1)), mul(X.e3, Y.e123nil)), mul(X.e3nil, Y.e123nilinf)), mul(X.e3nilinf, Y.e123nil)), mul(X.enil, Y.e12)), mul(X.enilinf, Y.e12nil)), mul(X.scalar, Y.e12nil)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e123nil, Y.e3), mul(X.e12nil, Y.enilinf)), mul(X.e13, Y.e23nil)), mul(X.e13nil, Y.e23)), mul(X.e13nilinf, Y.e23nil)), mul(X.e1nil, Y.e2)), mul(X.e2, Y.e1nil)), mul(X.e23nil, Y.e13nilinf)), mul(X.e2nil, Y.e1nilinf)), mul(X.e2nilinf, Y.e1nil)), mul(X.e3nil, Y.e123)), mul(X.enil, Y.e12nilinf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e2inf), mul(X.e12, Y.einf)), mul(X.e123, Y.e3inf)), mul(X.e12inf, Y.enilinf)), mul(X.e12inf, Y.scalar)), mul(X.e13nilinf, Y.e23inf)), mul(X.e23, Y.e13inf)), mul(X.e23inf, Y.e13)), mul(X.e23inf, Y.e13nilinf)), mul(X.e2inf, Y.e1)), mul(X.e2inf, Y.e1nilinf)), mul(X.e2nilinf, Y.e1inf)), mul(X.e3, Y.e123inf)), mul(X.einf, Y.e12)), mul(X.einf, Y.e12nilinf)), mul(X.scalar, Y.e12inf)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e123inf, Y.e3), mul(X.e123inf, Y.e3nilinf)), mul(X.e123nilinf, Y.e3inf)), mul(X.e12nilinf, Y.einf)), mul(X.e13, Y.e23inf)), mul(X.e13inf, Y.e23)), mul(X.e13inf, Y.e23nilinf)), mul(X.e1inf, Y.e2)), mul(X.e1inf, Y.e2nilinf)), mul(X.e1nilinf, Y.e2inf)), mul(X.e2, Y.e1inf)), mul(X.e23nilinf, Y.e13inf)), mul(X.e3inf, Y.e123)), mul(X.e3inf, Y.e123nilinf)), mul(X.e3nilinf, Y.e123inf)), mul(X.enilinf, Y.e12inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e3nil), mul(X.e12, Y.e23nil)), mul(X.e123nil, Y.e2)), mul(X.e12nil, Y.e23)), mul(X.e12nilinf, Y.e23nil)), mul(X.e13, Y.enil)), mul(X.e13nil, Y.scalar)), mul(X.e13nilinf, Y.enil)), mul(X.e1nil, Y.e3nilinf)), mul(X.e1nilinf, Y.e3nil)), mul(X.e23nil, Y.e12nilinf)), mul(X.e2nil, Y.e123)), mul(X.e3nil, Y.e1)), mul(X.enil, Y.e13)), mul(X.enilinf, Y.e13nil)), mul(X.scalar, Y.e13nil)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e123, Y.e2nil), mul(X.e123nil, Y.e2nilinf)), mul(X.e123nilinf, Y.e2nil)), mul(X.e12nil, Y.e23nilinf)), mul(X.e13nil, Y.enilinf)), mul(X.e1nil, Y.e3)), mul(X.e2, Y.e123nil)), mul(X.e23, Y.e12nil)), mul(X.e23nil, Y.e12)), mul(X.e23nilinf, Y.e12nil)), mul(X.e2nil, Y.e123nilinf)), mul(X.e2nilinf, Y.e123nil)), mul(X.e3, Y.e1nil)), mul(X.e3nil, Y.e1nilinf)), mul(X.e3nilinf, Y.e1nil)), mul(X.enil, Y.e13nilinf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e3inf), mul(X.e12, Y.e23inf)), mul(X.e123inf, Y.e2)), mul(X.e123inf, Y.e2nilinf)), mul(X.e123nilinf, Y.e2inf)), mul(X.e12inf, Y.e23)), mul(X.e12inf, Y.e23nilinf)), mul(X.e13, Y.einf)), mul(X.e13inf, Y.enilinf)), mul(X.e13inf, Y.scalar)), mul(X.e23nilinf, Y.e12inf)), mul(X.e2inf, Y.e123)), mul(X.e2inf, Y.e123nilinf)), mul(X.e2nilinf, Y.e123inf)), mul(X.e3inf, Y.e1)), mul(X.e3inf, Y.e1nilinf)), mul(X.e3nilinf, Y.e1inf)), mul(X.einf, Y.e13)), mul(X.einf, Y.e13nilinf)), mul(X.scalar, Y.e13inf)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e123, Y.e2inf), mul(X.e12nilinf, Y.e23inf)), mul(X.e13nilinf, Y.einf)), mul(X.e1inf, Y.e3)), mul(X.e1inf, Y.e3nilinf)), mul(X.e1nilinf, Y.e3inf)), mul(X.e2, Y.e123inf)), mul(X.e23, Y.e12inf)), mul(X.e23inf, Y.e12)), mul(X.e23inf, Y.e12nilinf)), mul(X.e3, Y.e1inf)), mul(X.enilinf, Y.e13inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.enilinf), mul(X.e12, Y.e2nilinf)), mul(X.e123inf, Y.e23nil)), mul(X.e12inf, Y.e2nil)), mul(X.e12nilinf, Y.e2)), mul(X.e13, Y.e3nilinf)), mul(X.e13inf, Y.e3nil)), mul(X.e13nilinf, Y.e3)), mul(X.e1nil, Y.einf)), mul(X.e1nilinf, Y.scalar)), mul(X.e23nil, Y.e123inf)), mul(X.e2inf, Y.e12nil)), mul(X.e3inf, Y.e13nil)), mul(X.einf, Y.e1nil)), mul(X.enilinf, Y.e1)), mul(X.scalar, Y.e1nilinf)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e123, Y.e23nilinf), mul(X.e123nil, Y.e23inf)), mul(X.e123nilinf, Y.e23)), mul(X.e12nil, Y.e2inf)), mul(X.e13nil, Y.e3inf)), mul(X.e1inf, Y.enil)), mul(X.e2, Y.e12nilinf)), mul(X.e23, Y.e123nilinf)), mul(X.e23inf, Y.e123nil)), mul(X.e23nilinf, Y.e123)), mul(X.e2nil, Y.e12inf)), mul(X.e2nilinf, Y.e12)), mul(X.e3, Y.e13nilinf)), mul(X.e3nil, Y.e13inf)), mul(X.e3nilinf, Y.e13)), mul(X.enil, Y.e1inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e123nil), mul(X.e123, Y.e1nil)), mul(X.e123nil, Y.e1nilinf)), mul(X.e123nilinf, Y.e1nil)), mul(X.e12nil, Y.e13nilinf)), mul(X.e13, Y.e12nil)), mul(X.e13nil, Y.e12)), mul(X.e13nilinf, Y.e12nil)), mul(X.e1nil, Y.e123nilinf)), mul(X.e1nilinf, Y.e123nil)), mul(X.e2, Y.e3nil)), mul(X.e23, Y.enil)), mul(X.e23nil, Y.scalar)), mul(X.e23nilinf, Y.enil)), mul(X.e2nil, Y.e3nilinf)), mul(X.e2nilinf, Y.e3nil)), mul(X.e3nil, Y.e2)), mul(X.enil, Y.e23)), mul(X.enilinf, Y.e23nil)), mul(X.scalar, Y.e23nil)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e13nil), mul(X.e123nil, Y.e1)), mul(X.e12nil, Y.e13)), mul(X.e12nilinf, Y.e13nil)), mul(X.e13nil, Y.e12nilinf)), mul(X.e1nil, Y.e123)), mul(X.e23nil, Y.enilinf)), mul(X.e2nil, Y.e3)), mul(X.e3, Y.e2nil)), mul(X.e3nil, Y.e2nilinf)), mul(X.e3nilinf, Y.e2nil)), mul(X.enil, Y.e23nilinf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e123inf), mul(X.e123, Y.e1inf)), mul(X.e12nilinf, Y.e13inf)), mul(X.e13, Y.e12inf)), mul(X.e13inf, Y.e12)), mul(X.e13inf, Y.e12nilinf)), mul(X.e2, Y.e3inf)), mul(X.e23, Y.einf)), mul(X.e23inf, Y.enilinf)), mul(X.e23inf, Y.scalar)), mul(X.e3inf, Y.e2)), mul(X.e3inf, Y.e2nilinf)), mul(X.e3nilinf, Y.e2inf)), mul(X.einf, Y.e23)), mul(X.einf, Y.e23nilinf)), mul(X.scalar, Y.e23inf)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e13inf), mul(X.e123inf, Y.e1)), mul(X.e123inf, Y.e1nilinf)), mul(X.e123nilinf, Y.e1inf)), mul(X.e12inf, Y.e13)), mul(X.e12inf, Y.e13nilinf)), mul(X.e13nilinf, Y.e12inf)), mul(X.e1inf, Y.e123)), mul(X.e1inf, Y.e123nilinf)), mul(X.e1nilinf, Y.e123inf)), mul(X.e23nilinf, Y.einf)), mul(X.e2inf, Y.e3)), mul(X.e2inf, Y.e3nilinf)), mul(X.e2nilinf, Y.e3inf)), mul(X.e3, Y.e2inf)), mul(X.enilinf, Y.e23inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e12nilinf), mul(X.e123, Y.e13nilinf)), mul(X.e123nil, Y.e13inf)), mul(X.e123nilinf, Y.e13)), mul(X.e12nil, Y.e1inf)), mul(X.e13, Y.e123nilinf)), mul(X.e13inf, Y.e123nil)), mul(X.e13nilinf, Y.e123)), mul(X.e1nil, Y.e12inf)), mul(X.e1nilinf, Y.e12)), mul(X.e2, Y.enilinf)), mul(X.e23, Y.e3nilinf)), mul(X.e23inf, Y.e3nil)), mul(X.e23nilinf, Y.e3)), mul(X.e2nil, Y.einf)), mul(X.e2nilinf, Y.scalar)), mul(X.e3inf, Y.e23nil)), mul(X.einf, Y.e2nil)), mul(X.enilinf, Y.e2)), mul(X.scalar, Y.e2nilinf)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e1nilinf), mul(X.e123inf, Y.e13nil)), mul(X.e12inf, Y.e1nil)), mul(X.e12nilinf, Y.e1)), mul(X.e13nil, Y.e123inf)), mul(X.e1inf, Y.e12nil)), mul(X.e23nil, Y.e3inf)), mul(X.e2inf, Y.enil)), mul(X.e3, Y.e23nilinf)), mul(X.e3nil, Y.e23inf)), mul(X.e3nilinf, Y.e23)), mul(X.enil, Y.e2inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e13nilinf), mul(X.e123inf, Y.e12nil)), mul(X.e12nil, Y.e123inf)), mul(X.e13nil, Y.e1inf)), mul(X.e1nil, Y.e13inf)), mul(X.e1nilinf, Y.e13)), mul(X.e2, Y.e23nilinf)), mul(X.e23nil, Y.e2inf)), mul(X.e2nil, Y.e23inf)), mul(X.e2nilinf, Y.e23)), mul(X.e3, Y.enilinf)), mul(X.e3nil, Y.einf)), mul(X.e3nilinf, Y.scalar)), mul(X.einf, Y.e3nil)), mul(X.enilinf, Y.e3)), mul(X.scalar, Y.e3nilinf)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e123nilinf), mul(X.e123, Y.e12nilinf)), mul(X.e123nil, Y.e12inf)), mul(X.e123nilinf, Y.e12)), mul(X.e12inf, Y.e123nil)), mul(X.e12nilinf, Y.e123)), mul(X.e13, Y.e1nilinf)), mul(X.e13inf, Y.e1nil)), mul(X.e13nilinf, Y.e1)), mul(X.e1inf, Y.e13nil)), mul(X.e23, Y.e2nilinf)), mul(X.e23inf, Y.e2nil)), mul(X.e23nilinf, Y.e2)), mul(X.e2inf, Y.e23nil)), mul(X.e3inf, Y.enil)), mul(X.enil, Y.e3inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e23nil), mul(X.e12, Y.e3nil)), mul(X.e123, Y.enil)), mul(X.e123nil, Y.scalar)), mul(X.e123nilinf, Y.enil)), mul(X.e12nil, Y.e3nilinf)), mul(X.e12nilinf, Y.e3nil)), mul(X.e13nil, Y.e2)), mul(X.e1nil, Y.e23)), mul(X.e1nilinf, Y.e23nil)), mul(X.e23, Y.e1nil)), mul(X.e23nil, Y.e1nilinf)), mul(X.e23nilinf, Y.e1nil)), mul(X.e2nil, Y.e13nilinf)), mul(X.e3, Y.e12nil)), mul(X.e3nil, Y.e12)), mul(X.e3nilinf, Y.e12nil)), mul(X.enil, Y.e123nilinf)), mul(X.enilinf, Y.e123nil)), mul(X.scalar, Y.e123nil)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e123nil, Y.enilinf), mul(X.e12nil, Y.e3)), mul(X.e13, Y.e2nil)), mul(X.e13nil, Y.e2nilinf)), mul(X.e13nilinf, Y.e2nil)), mul(X.e1nil, Y.e23nilinf)), mul(X.e2, Y.e13nil)), mul(X.e23nil, Y.e1)), mul(X.e2nil, Y.e13)), mul(X.e2nilinf, Y.e13nil)), mul(X.e3nil, Y.e12nilinf)), mul(X.enil, Y.e123))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e23inf), mul(X.e12, Y.e3inf)), mul(X.e123, Y.einf)), mul(X.e123inf, Y.enilinf)), mul(X.e123inf, Y.scalar)), mul(X.e13inf, Y.e2)), mul(X.e13inf, Y.e2nilinf)), mul(X.e13nilinf, Y.e2inf)), mul(X.e1inf, Y.e23)), mul(X.e1inf, Y.e23nilinf)), mul(X.e23, Y.e1inf)), mul(X.e2nilinf, Y.e13inf)), mul(X.e3, Y.e12inf)), mul(X.e3inf, Y.e12)), mul(X.e3inf, Y.e12nilinf)), mul(X.scalar, Y.e123inf)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e123nilinf, Y.einf), mul(X.e12inf, Y.e3)), mul(X.e12inf, Y.e3nilinf)), mul(X.e12nilinf, Y.e3inf)), mul(X.e13, Y.e2inf)), mul(X.e1nilinf, Y.e23inf)), mul(X.e2, Y.e13inf)), mul(X.e23inf, Y.e1)), mul(X.e23inf, Y.e1nilinf)), mul(X.e23nilinf, Y.e1inf)), mul(X.e2inf, Y.e13)), mul(X.e2inf, Y.e13nilinf)), mul(X.e3nilinf, Y.e12inf)), mul(X.einf, Y.e123)), mul(X.einf, Y.e123nilinf)), mul(X.enilinf, Y.e123inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e2nilinf), mul(X.e12, Y.enilinf)), mul(X.e123, Y.e3nilinf)), mul(X.e123inf, Y.e3nil)), mul(X.e123nilinf, Y.e3)), mul(X.e12nil, Y.einf)), mul(X.e12nilinf, Y.scalar)), mul(X.e13inf, Y.e23nil)), mul(X.e1inf, Y.e2nil)), mul(X.e1nilinf, Y.e2)), mul(X.e23, Y.e13nilinf)), mul(X.e23nil, Y.e13inf)), mul(X.e23nilinf, Y.e13)), mul(X.e2nil, Y.e1inf)), mul(X.e3, Y.e123nilinf)), mul(X.e3inf, Y.e123nil)), mul(X.e3nilinf, Y.e123)), mul(X.enil, Y.e12inf)), mul(X.enilinf, Y.e12)), mul(X.scalar, Y.e12nilinf)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e123nil, Y.e3inf), mul(X.e12inf, Y.enil)), mul(X.e13, Y.e23nilinf)), mul(X.e13nil, Y.e23inf)), mul(X.e13nilinf, Y.e23)), mul(X.e1nil, Y.e2inf)), mul(X.e2, Y.e1nilinf)), mul(X.e23inf, Y.e13nil)), mul(X.e2inf, Y.e1nil)), mul(X.e2nilinf, Y.e1)), mul(X.e3nil, Y.e123inf)), mul(X.einf, Y.e12nil))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e3nilinf), mul(X.e12, Y.e23nilinf)), mul(X.e123nil, Y.e2inf)), mul(X.e12nil, Y.e23inf)), mul(X.e12nilinf, Y.e23)), mul(X.e13, Y.enilinf)), mul(X.e13nil, Y.einf)), mul(X.e13nilinf, Y.scalar)), mul(X.e1inf, Y.e3nil)), mul(X.e1nilinf, Y.e3)), mul(X.e23inf, Y.e12nil)), mul(X.e2nil, Y.e123inf)), mul(X.e3nil, Y.e1inf)), mul(X.enil, Y.e13inf)), mul(X.enilinf, Y.e13)), mul(X.scalar, Y.e13nilinf)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e123, Y.e2nilinf), mul(X.e123inf, Y.e2nil)), mul(X.e123nilinf, Y.e2)), mul(X.e12inf, Y.e23nil)), mul(X.e13inf, Y.enil)), mul(X.e1nil, Y.e3inf)), mul(X.e2, Y.e123nilinf)), mul(X.e23, Y.e12nilinf)), mul(X.e23nil, Y.e12inf)), mul(X.e23nilinf, Y.e12)), mul(X.e2inf, Y.e123nil)), mul(X.e2nilinf, Y.e123)), mul(X.e3, Y.e1nilinf)), mul(X.e3inf, Y.e1nil)), mul(X.e3nilinf, Y.e1)), mul(X.einf, Y.e13nil))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e123nilinf), mul(X.e123, Y.e1nilinf)), mul(X.e123inf, Y.e1nil)), mul(X.e123nilinf, Y.e1)), mul(X.e12inf, Y.e13nil)), mul(X.e13, Y.e12nilinf)), mul(X.e13nil, Y.e12inf)), mul(X.e13nilinf, Y.e12)), mul(X.e1inf, Y.e123nil)), mul(X.e1nilinf, Y.e123)), mul(X.e2, Y.e3nilinf)), mul(X.e23, Y.enilinf)), mul(X.e23nil, Y.einf)), mul(X.e23nilinf, Y.scalar)), mul(X.e2inf, Y.e3nil)), mul(X.e2nilinf, Y.e3)), mul(X.e3nil, Y.e2inf)), mul(X.enil, Y.e23inf)), mul(X.enilinf, Y.e23)), mul(X.scalar, Y.e23nilinf)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e13nilinf), mul(X.e123nil, Y.e1inf)), mul(X.e12nil, Y.e13inf)), mul(X.e12nilinf, Y.e13)), mul(X.e13inf, Y.e12nil)), mul(X.e1nil, Y.e123inf)), mul(X.e23inf, Y.enil)), mul(X.e2nil, Y.e3inf)), mul(X.e3, Y.e2nilinf)), mul(X.e3inf, Y.e2nil)), mul(X.e3nilinf, Y.e2)), mul(X.einf, Y.e23nil))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e23nilinf), mul(X.e12, Y.e3nilinf)), mul(X.e123, Y.enilinf)), mul(X.e123nil, Y.einf)), mul(X.e123nilinf, Y.scalar)), mul(X.e12inf, Y.e3nil)), mul(X.e12nilinf, Y.e3)), mul(X.e13nil, Y.e2inf)), mul(X.e1nil, Y.e23inf)), mul(X.e1nilinf, Y.e23)), mul(X.e23, Y.e1nilinf)), mul(X.e23inf, Y.e1nil)), mul(X.e23nilinf, Y.e1)), mul(X.e2inf, Y.e13nil)), mul(X.e3, Y.e12nilinf)), mul(X.e3nil, Y.e12inf)), mul(X.e3nilinf, Y.e12)), mul(X.einf, Y.e123nil)), mul(X.enilinf, Y.e123)), mul(X.scalar, Y.e123nilinf)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e123inf, Y.enil), mul(X.e12nil, Y.e3inf)), mul(X.e13, Y.e2nilinf)), mul(X.e13inf, Y.e2nil)), mul(X.e13nilinf, Y.e2)), mul(X.e1inf, Y.e23nil)), mul(X.e2, Y.e13nilinf)), mul(X.e23nil, Y.e1inf)), mul(X.e2nil, Y.e13inf)), mul(X.e2nilinf, Y.e13)), mul(X.e3inf, Y.e12nil)), mul(X.enil, Y.e123inf))));
}

CGA3 scalar_CGA3(float a){
    return mul(a, ONE_CGA3);
}

CGA3 mul(CGA3 X, CGA3 Y, CGA3 Z){
    return mul(mul(X, Y), Z);
}

CGA3 involve(CGA3 X){
    return CGA3(X.scalar, mul(-1, X.e1), mul(-1, X.e2), mul(-1, X.e3), mul(-1, X.enil), mul(-1, X.einf), X.e12, X.e13, X.e1nil, X.e1inf, X.e23, X.e2nil, X.e2inf, X.e3nil, X.e3inf, X.enilinf, mul(-1, X.e123), mul(-1, X.e12nil), mul(-1, X.e12inf), mul(-1, X.e13nil), mul(-1, X.e13inf), mul(-1, X.e1nilinf), mul(-1, X.e23nil), mul(-1, X.e23inf), mul(-1, X.e2nilinf), mul(-1, X.e3nilinf), X.e123nil, X.e123inf, X.e12nilinf, X.e13nilinf, X.e23nilinf, mul(-1, X.e123nilinf));
}

CGA3 inner(CGA3 X, CGA3 Y){
    return CGA3(sub(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e1), mul(X.e123inf, Y.e123nil)), mul(X.e123nil, Y.e123inf)), mul(X.e1nilinf, Y.e1nilinf)), mul(X.e2, Y.e2)), mul(X.e2nilinf, Y.e2nilinf)), mul(X.e3, Y.e3)), mul(X.e3nilinf, Y.e3nilinf)), mul(X.einf, Y.enil)), mul(X.enil, Y.einf)), mul(X.enilinf, Y.enilinf)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e12), mul(X.e123, Y.e123)), mul(X.e123nilinf, Y.e123nilinf)), mul(X.e12inf, Y.e12nil)), mul(X.e12nil, Y.e12inf)), mul(X.e12nilinf, Y.e12nilinf)), mul(X.e13, Y.e13)), mul(X.e13inf, Y.e13nil)), mul(X.e13nil, Y.e13inf)), mul(X.e13nilinf, Y.e13nilinf)), mul(X.e1inf, Y.e1nil)), mul(X.e1nil, Y.e1inf)), mul(X.e23, Y.e23)), mul(X.e23inf, Y.e23nil)), mul(X.e23nil, Y.e23inf)), mul(X.e23nilinf, Y.e23nilinf)), mul(X.e2inf, Y.e2nil)), mul(X.e2nil, Y.e2inf)), mul(X.e3inf, Y.e3nil)), mul(X.e3nil, Y.e3inf))), sub(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e2), mul(X.e12nilinf, Y.e2nilinf)), mul(X.e13, Y.e3)), mul(X.e13nilinf, Y.e3nilinf)), mul(X.e1inf, Y.enil)), mul(X.e1nil, Y.einf)), mul(X.e1nilinf, Y.enilinf)), mul(X.e23inf, Y.e123nil)), mul(X.e23nil, Y.e123inf)), mul(X.enilinf, Y.e1nilinf)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e123, Y.e23), mul(X.e123inf, Y.e23nil)), mul(X.e123nil, Y.e23inf)), mul(X.e123nilinf, Y.e23nilinf)), mul(X.e12inf, Y.e2nil)), mul(X.e12nil, Y.e2inf)), mul(X.e13inf, Y.e3nil)), mul(X.e13nil, Y.e3inf)), mul(X.e2, Y.e12)), mul(X.e23, Y.e123)), mul(X.e23nilinf, Y.e123nilinf)), mul(X.e2inf, Y.e12nil)), mul(X.e2nil, Y.e12inf)), mul(X.e2nilinf, Y.e12nilinf)), mul(X.e3, Y.e13)), mul(X.e3inf, Y.e13nil)), mul(X.e3nil, Y.e13inf)), mul(X.e3nilinf, Y.e13nilinf)), mul(X.einf, Y.e1nil)), mul(X.enil, Y.e1inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e12), mul(X.e123, Y.e13)), mul(X.e123inf, Y.e13nil)), mul(X.e123nil, Y.e13inf)), mul(X.e123nilinf, Y.e13nilinf)), mul(X.e12inf, Y.e1nil)), mul(X.e12nil, Y.e1inf)), mul(X.e13, Y.e123)), mul(X.e13nilinf, Y.e123nilinf)), mul(X.e1inf, Y.e12nil)), mul(X.e1nil, Y.e12inf)), mul(X.e1nilinf, Y.e12nilinf)), mul(X.e23, Y.e3)), mul(X.e23nilinf, Y.e3nilinf)), mul(X.e2inf, Y.enil)), mul(X.e2nil, Y.einf)), mul(X.e2nilinf, Y.enilinf)), mul(X.enilinf, Y.e2nilinf)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e1), mul(X.e12nilinf, Y.e1nilinf)), mul(X.e13inf, Y.e123nil)), mul(X.e13nil, Y.e123inf)), mul(X.e23inf, Y.e3nil)), mul(X.e23nil, Y.e3inf)), mul(X.e3, Y.e23)), mul(X.e3inf, Y.e23nil)), mul(X.e3nil, Y.e23inf)), mul(X.e3nilinf, Y.e23nilinf)), mul(X.einf, Y.e2nil)), mul(X.enil, Y.e2inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e13), mul(X.e12inf, Y.e123nil)), mul(X.e12nil, Y.e123inf)), mul(X.e13inf, Y.e1nil)), mul(X.e13nil, Y.e1inf)), mul(X.e1inf, Y.e13nil)), mul(X.e1nil, Y.e13inf)), mul(X.e1nilinf, Y.e13nilinf)), mul(X.e2, Y.e23)), mul(X.e23inf, Y.e2nil)), mul(X.e23nil, Y.e2inf)), mul(X.e2inf, Y.e23nil)), mul(X.e2nil, Y.e23inf)), mul(X.e2nilinf, Y.e23nilinf)), mul(X.e3inf, Y.enil)), mul(X.e3nil, Y.einf)), mul(X.e3nilinf, Y.enilinf)), mul(X.enilinf, Y.e3nilinf)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e123), mul(X.e123, Y.e12)), mul(X.e123inf, Y.e12nil)), mul(X.e123nil, Y.e12inf)), mul(X.e123nilinf, Y.e12nilinf)), mul(X.e12nilinf, Y.e123nilinf)), mul(X.e13, Y.e1)), mul(X.e13nilinf, Y.e1nilinf)), mul(X.e23, Y.e2)), mul(X.e23nilinf, Y.e2nilinf)), mul(X.einf, Y.e3nil)), mul(X.enil, Y.e3inf))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e1nil), mul(X.e123nil, Y.e123)), mul(X.e12nil, Y.e12nilinf)), mul(X.e13nil, Y.e13nilinf)), mul(X.e1nil, Y.e1nilinf)), mul(X.e1nilinf, Y.e1nil)), mul(X.e2, Y.e2nil)), mul(X.e23nil, Y.e23nilinf)), mul(X.e2nil, Y.e2nilinf)), mul(X.e2nilinf, Y.e2nil)), mul(X.e3, Y.e3nil)), mul(X.e3nil, Y.e3nilinf)), mul(X.e3nilinf, Y.e3nil)), mul(X.enilinf, Y.enil)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e12nil), mul(X.e123, Y.e123nil)), mul(X.e123nil, Y.e123nilinf)), mul(X.e123nilinf, Y.e123nil)), mul(X.e12nil, Y.e12)), mul(X.e12nilinf, Y.e12nil)), mul(X.e13, Y.e13nil)), mul(X.e13nil, Y.e13)), mul(X.e13nilinf, Y.e13nil)), mul(X.e1nil, Y.e1)), mul(X.e23, Y.e23nil)), mul(X.e23nil, Y.e23)), mul(X.e23nilinf, Y.e23nil)), mul(X.e2nil, Y.e2)), mul(X.e3nil, Y.e3)), mul(X.enil, Y.enilinf))), sub(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e1inf), mul(X.e123inf, Y.e123)), mul(X.e123inf, Y.e123nilinf)), mul(X.e123nilinf, Y.e123inf)), mul(X.e12nilinf, Y.e12inf)), mul(X.e13nilinf, Y.e13inf)), mul(X.e2, Y.e2inf)), mul(X.e23nilinf, Y.e23inf)), mul(X.e3, Y.e3inf)), mul(X.einf, Y.enilinf)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e12inf), mul(X.e123, Y.e123inf)), mul(X.e12inf, Y.e12)), mul(X.e12inf, Y.e12nilinf)), mul(X.e13, Y.e13inf)), mul(X.e13inf, Y.e13)), mul(X.e13inf, Y.e13nilinf)), mul(X.e1inf, Y.e1)), mul(X.e1inf, Y.e1nilinf)), mul(X.e1nilinf, Y.e1inf)), mul(X.e23, Y.e23inf)), mul(X.e23inf, Y.e23)), mul(X.e23inf, Y.e23nilinf)), mul(X.e2inf, Y.e2)), mul(X.e2inf, Y.e2nilinf)), mul(X.e2nilinf, Y.e2inf)), mul(X.e3inf, Y.e3)), mul(X.e3inf, Y.e3nilinf)), mul(X.e3nilinf, Y.e3inf)), mul(X.enilinf, Y.einf))), sub(add(add(add(add(add(add(add(add(add(mul(X.e123, Y.e3), mul(X.e123nilinf, Y.e3nilinf)), mul(X.e12inf, Y.enil)), mul(X.e12nil, Y.einf)), mul(X.e12nilinf, Y.enilinf)), mul(X.e3, Y.e123)), mul(X.e3nilinf, Y.e123nilinf)), mul(X.einf, Y.e12nil)), mul(X.enil, Y.e12inf)), mul(X.enilinf, Y.e12nilinf)), add(add(add(mul(X.e123inf, Y.e3nil), mul(X.e123nil, Y.e3inf)), mul(X.e3inf, Y.e123nil)), mul(X.e3nil, Y.e123inf))), sub(add(add(add(add(add(add(add(add(add(mul(X.e123inf, Y.e2nil), mul(X.e123nil, Y.e2inf)), mul(X.e13inf, Y.enil)), mul(X.e13nil, Y.einf)), mul(X.e13nilinf, Y.enilinf)), mul(X.e2inf, Y.e123nil)), mul(X.e2nil, Y.e123inf)), mul(X.einf, Y.e13nil)), mul(X.enil, Y.e13inf)), mul(X.enilinf, Y.e13nilinf)), add(add(add(mul(X.e123, Y.e2), mul(X.e123nilinf, Y.e2nilinf)), mul(X.e2, Y.e123)), mul(X.e2nilinf, Y.e123nilinf))), sub(add(add(add(add(add(mul(X.e12nilinf, Y.e2nil), mul(X.e13nilinf, Y.e3nil)), mul(X.e1nilinf, Y.enil)), mul(X.e2nil, Y.e12nilinf)), mul(X.e3nil, Y.e13nilinf)), mul(X.enil, Y.e1nilinf)), add(add(add(add(add(add(add(mul(X.e123nil, Y.e23), mul(X.e123nilinf, Y.e23nil)), mul(X.e12nil, Y.e2)), mul(X.e13nil, Y.e3)), mul(X.e2, Y.e12nil)), mul(X.e23, Y.e123nil)), mul(X.e23nil, Y.e123nilinf)), mul(X.e3, Y.e13nil))), sub(add(mul(X.e123nilinf, Y.e23inf), mul(X.e23inf, Y.e123nilinf)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e123inf, Y.e23), mul(X.e12inf, Y.e2)), mul(X.e12nilinf, Y.e2inf)), mul(X.e13inf, Y.e3)), mul(X.e13nilinf, Y.e3inf)), mul(X.e1nilinf, Y.einf)), mul(X.e2, Y.e12inf)), mul(X.e23, Y.e123inf)), mul(X.e2inf, Y.e12nilinf)), mul(X.e3, Y.e13inf)), mul(X.e3inf, Y.e13nilinf)), mul(X.einf, Y.e1nilinf))), sub(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e123), mul(X.e123, Y.e1)), mul(X.e123nilinf, Y.e1nilinf)), mul(X.e1nilinf, Y.e123nilinf)), mul(X.e23inf, Y.enil)), mul(X.e23nil, Y.einf)), mul(X.e23nilinf, Y.enilinf)), mul(X.einf, Y.e23nil)), mul(X.enil, Y.e23inf)), mul(X.enilinf, Y.e23nilinf)), add(add(add(mul(X.e123inf, Y.e1nil), mul(X.e123nil, Y.e1inf)), mul(X.e1inf, Y.e123nil)), mul(X.e1nil, Y.e123inf))), sub(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e12nil), mul(X.e123nil, Y.e13)), mul(X.e123nilinf, Y.e13nil)), mul(X.e12nil, Y.e1)), mul(X.e13, Y.e123nil)), mul(X.e13nil, Y.e123nilinf)), mul(X.e23nilinf, Y.e3nil)), mul(X.e2nilinf, Y.enil)), mul(X.e3nil, Y.e23nilinf)), mul(X.enil, Y.e2nilinf)), add(add(add(mul(X.e12nilinf, Y.e1nil), mul(X.e1nil, Y.e12nilinf)), mul(X.e23nil, Y.e3)), mul(X.e3, Y.e23nil))), sub(add(add(add(add(add(mul(X.e1, Y.e12inf), mul(X.e123inf, Y.e13)), mul(X.e12inf, Y.e1)), mul(X.e12nilinf, Y.e1inf)), mul(X.e13, Y.e123inf)), mul(X.e1inf, Y.e12nilinf)), add(add(add(add(add(add(add(mul(X.e123nilinf, Y.e13inf), mul(X.e13inf, Y.e123nilinf)), mul(X.e23inf, Y.e3)), mul(X.e23nilinf, Y.e3inf)), mul(X.e2nilinf, Y.einf)), mul(X.e3, Y.e23inf)), mul(X.e3inf, Y.e23nilinf)), mul(X.einf, Y.e2nilinf))), sub(add(add(add(add(add(mul(X.e1, Y.e13nil), mul(X.e13nil, Y.e1)), mul(X.e2, Y.e23nil)), mul(X.e23nil, Y.e2)), mul(X.e3nilinf, Y.enil)), mul(X.enil, Y.e3nilinf)), add(add(add(add(add(add(add(mul(X.e12, Y.e123nil), mul(X.e123nil, Y.e12)), mul(X.e123nilinf, Y.e12nil)), mul(X.e12nil, Y.e123nilinf)), mul(X.e13nilinf, Y.e1nil)), mul(X.e1nil, Y.e13nilinf)), mul(X.e23nilinf, Y.e2nil)), mul(X.e2nil, Y.e23nilinf))), sub(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e13inf), mul(X.e123nilinf, Y.e12inf)), mul(X.e12inf, Y.e123nilinf)), mul(X.e13inf, Y.e1)), mul(X.e13nilinf, Y.e1inf)), mul(X.e1inf, Y.e13nilinf)), mul(X.e2, Y.e23inf)), mul(X.e23inf, Y.e2)), mul(X.e23nilinf, Y.e2inf)), mul(X.e2inf, Y.e23nilinf)), add(add(add(mul(X.e12, Y.e123inf), mul(X.e123inf, Y.e12)), mul(X.e3nilinf, Y.einf)), mul(X.einf, Y.e3nilinf))), sub(add(add(add(add(add(mul(X.e1, Y.e1nilinf), mul(X.e1nilinf, Y.e1)), mul(X.e2, Y.e2nilinf)), mul(X.e2nilinf, Y.e2)), mul(X.e3, Y.e3nilinf)), mul(X.e3nilinf, Y.e3)), add(add(add(add(add(add(add(mul(X.e12, Y.e12nilinf), mul(X.e123, Y.e123nilinf)), mul(X.e123nilinf, Y.e123)), mul(X.e12nilinf, Y.e12)), mul(X.e13, Y.e13nilinf)), mul(X.e13nilinf, Y.e13)), mul(X.e23, Y.e23nilinf)), mul(X.e23nilinf, Y.e23))), sub(add(add(add(mul(X.e123inf, Y.enil), mul(X.e123nil, Y.einf)), mul(X.e123nilinf, Y.enilinf)), mul(X.enilinf, Y.e123nilinf)), add(mul(X.einf, Y.e123nil), mul(X.enil, Y.e123inf))), sub(add(add(add(mul(X.e123nilinf, Y.e3nil), mul(X.e12nilinf, Y.enil)), mul(X.e3, Y.e123nil)), mul(X.e3nil, Y.e123nilinf)), add(mul(X.e123nil, Y.e3), mul(X.enil, Y.e12nilinf))), sub(add(mul(X.e3, Y.e123inf), mul(X.einf, Y.e12nilinf)), add(add(add(mul(X.e123inf, Y.e3), mul(X.e123nilinf, Y.e3inf)), mul(X.e12nilinf, Y.einf)), mul(X.e3inf, Y.e123nilinf))), sub(add(mul(X.e123nil, Y.e2), mul(X.e13nilinf, Y.enil)), add(add(add(mul(X.e123nilinf, Y.e2nil), mul(X.e2, Y.e123nil)), mul(X.e2nil, Y.e123nilinf)), mul(X.enil, Y.e13nilinf))), sub(add(add(add(mul(X.e123inf, Y.e2), mul(X.e123nilinf, Y.e2inf)), mul(X.e2inf, Y.e123nilinf)), mul(X.einf, Y.e13nilinf)), add(mul(X.e13nilinf, Y.einf), mul(X.e2, Y.e123inf))), sub(add(mul(X.e12nilinf, Y.e2), mul(X.e13nilinf, Y.e3)), add(add(add(mul(X.e123nilinf, Y.e23), mul(X.e2, Y.e12nilinf)), mul(X.e23, Y.e123nilinf)), mul(X.e3, Y.e13nilinf))), sub(add(add(add(mul(X.e1, Y.e123nil), mul(X.e123nilinf, Y.e1nil)), mul(X.e1nil, Y.e123nilinf)), mul(X.e23nilinf, Y.enil)), add(mul(X.e123nil, Y.e1), mul(X.enil, Y.e23nilinf))), sub(add(mul(X.e1, Y.e123inf), mul(X.einf, Y.e23nilinf)), add(add(add(mul(X.e123inf, Y.e1), mul(X.e123nilinf, Y.e1inf)), mul(X.e1inf, Y.e123nilinf)), mul(X.e23nilinf, Y.einf))), sub(add(add(add(mul(X.e1, Y.e12nilinf), mul(X.e123nilinf, Y.e13)), mul(X.e13, Y.e123nilinf)), mul(X.e23nilinf, Y.e3)), add(mul(X.e12nilinf, Y.e1), mul(X.e3, Y.e23nilinf))), sub(add(mul(X.e1, Y.e13nilinf), mul(X.e2, Y.e23nilinf)), add(add(add(mul(X.e12, Y.e123nilinf), mul(X.e123nilinf, Y.e12)), mul(X.e13nilinf, Y.e1)), mul(X.e23nilinf, Y.e2))), add(mul(X.e123nilinf, Y.enil), mul(X.enil, Y.e123nilinf)), sub(ZERO_Dual, add(mul(X.e123nilinf, Y.einf), mul(X.einf, Y.e123nilinf))), add(mul(X.e123nilinf, Y.e3), mul(X.e3, Y.e123nilinf)), sub(ZERO_Dual, add(mul(X.e123nilinf, Y.e2), mul(X.e2, Y.e123nilinf))), add(mul(X.e1, Y.e123nilinf), mul(X.e123nilinf, Y.e1)), ZERO_Dual);
}

CGA3 lcontract(CGA3 X, CGA3 Y){
    return CGA3(sub(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e1), mul(X.e123inf, Y.e123nil)), mul(X.e123nil, Y.e123inf)), mul(X.e1nilinf, Y.e1nilinf)), mul(X.e2, Y.e2)), mul(X.e2nilinf, Y.e2nilinf)), mul(X.e3, Y.e3)), mul(X.e3nilinf, Y.e3nilinf)), mul(X.einf, Y.enil)), mul(X.enil, Y.einf)), mul(X.enilinf, Y.enilinf)), mul(X.scalar, Y.scalar)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e12), mul(X.e123, Y.e123)), mul(X.e123nilinf, Y.e123nilinf)), mul(X.e12inf, Y.e12nil)), mul(X.e12nil, Y.e12inf)), mul(X.e12nilinf, Y.e12nilinf)), mul(X.e13, Y.e13)), mul(X.e13inf, Y.e13nil)), mul(X.e13nil, Y.e13inf)), mul(X.e13nilinf, Y.e13nilinf)), mul(X.e1inf, Y.e1nil)), mul(X.e1nil, Y.e1inf)), mul(X.e23, Y.e23)), mul(X.e23inf, Y.e23nil)), mul(X.e23nil, Y.e23inf)), mul(X.e23nilinf, Y.e23nilinf)), mul(X.e2inf, Y.e2nil)), mul(X.e2nil, Y.e2inf)), mul(X.e3inf, Y.e3nil)), mul(X.e3nil, Y.e3inf))), sub(add(add(add(mul(X.e23inf, Y.e123nil), mul(X.e23nil, Y.e123inf)), mul(X.enilinf, Y.e1nilinf)), mul(X.scalar, Y.e1)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e2, Y.e12), mul(X.e23, Y.e123)), mul(X.e23nilinf, Y.e123nilinf)), mul(X.e2inf, Y.e12nil)), mul(X.e2nil, Y.e12inf)), mul(X.e2nilinf, Y.e12nilinf)), mul(X.e3, Y.e13)), mul(X.e3inf, Y.e13nil)), mul(X.e3nil, Y.e13inf)), mul(X.e3nilinf, Y.e13nilinf)), mul(X.einf, Y.e1nil)), mul(X.enil, Y.e1inf))), sub(add(add(add(add(add(add(add(mul(X.e1, Y.e12), mul(X.e13, Y.e123)), mul(X.e13nilinf, Y.e123nilinf)), mul(X.e1inf, Y.e12nil)), mul(X.e1nil, Y.e12inf)), mul(X.e1nilinf, Y.e12nilinf)), mul(X.enilinf, Y.e2nilinf)), mul(X.scalar, Y.e2)), add(add(add(add(add(add(add(mul(X.e13inf, Y.e123nil), mul(X.e13nil, Y.e123inf)), mul(X.e3, Y.e23)), mul(X.e3inf, Y.e23nil)), mul(X.e3nil, Y.e23inf)), mul(X.e3nilinf, Y.e23nilinf)), mul(X.einf, Y.e2nil)), mul(X.enil, Y.e2inf))), sub(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e13), mul(X.e12inf, Y.e123nil)), mul(X.e12nil, Y.e123inf)), mul(X.e1inf, Y.e13nil)), mul(X.e1nil, Y.e13inf)), mul(X.e1nilinf, Y.e13nilinf)), mul(X.e2, Y.e23)), mul(X.e2inf, Y.e23nil)), mul(X.e2nil, Y.e23inf)), mul(X.e2nilinf, Y.e23nilinf)), mul(X.enilinf, Y.e3nilinf)), mul(X.scalar, Y.e3)), add(add(add(mul(X.e12, Y.e123), mul(X.e12nilinf, Y.e123nilinf)), mul(X.einf, Y.e3nil)), mul(X.enil, Y.e3inf))), sub(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e1nil), mul(X.e12nil, Y.e12nilinf)), mul(X.e13nil, Y.e13nilinf)), mul(X.e1nil, Y.e1nilinf)), mul(X.e2, Y.e2nil)), mul(X.e23nil, Y.e23nilinf)), mul(X.e2nil, Y.e2nilinf)), mul(X.e3, Y.e3nil)), mul(X.e3nil, Y.e3nilinf)), mul(X.scalar, Y.enil)), add(add(add(add(add(mul(X.e12, Y.e12nil), mul(X.e123, Y.e123nil)), mul(X.e123nil, Y.e123nilinf)), mul(X.e13, Y.e13nil)), mul(X.e23, Y.e23nil)), mul(X.enil, Y.enilinf))), sub(add(add(add(add(add(mul(X.e1, Y.e1inf), mul(X.e123inf, Y.e123nilinf)), mul(X.e2, Y.e2inf)), mul(X.e3, Y.e3inf)), mul(X.einf, Y.enilinf)), mul(X.scalar, Y.einf)), add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e12inf), mul(X.e123, Y.e123inf)), mul(X.e12inf, Y.e12nilinf)), mul(X.e13, Y.e13inf)), mul(X.e13inf, Y.e13nilinf)), mul(X.e1inf, Y.e1nilinf)), mul(X.e23, Y.e23inf)), mul(X.e23inf, Y.e23nilinf)), mul(X.e2inf, Y.e2nilinf)), mul(X.e3inf, Y.e3nilinf))), sub(add(add(add(add(add(mul(X.e3, Y.e123), mul(X.e3nilinf, Y.e123nilinf)), mul(X.einf, Y.e12nil)), mul(X.enil, Y.e12inf)), mul(X.enilinf, Y.e12nilinf)), mul(X.scalar, Y.e12)), add(mul(X.e3inf, Y.e123nil), mul(X.e3nil, Y.e123inf))), sub(add(add(add(add(add(mul(X.e2inf, Y.e123nil), mul(X.e2nil, Y.e123inf)), mul(X.einf, Y.e13nil)), mul(X.enil, Y.e13inf)), mul(X.enilinf, Y.e13nilinf)), mul(X.scalar, Y.e13)), add(mul(X.e2, Y.e123), mul(X.e2nilinf, Y.e123nilinf))), sub(add(add(add(mul(X.e2nil, Y.e12nilinf), mul(X.e3nil, Y.e13nilinf)), mul(X.enil, Y.e1nilinf)), mul(X.scalar, Y.e1nil)), add(add(add(mul(X.e2, Y.e12nil), mul(X.e23, Y.e123nil)), mul(X.e23nil, Y.e123nilinf)), mul(X.e3, Y.e13nil))), sub(add(mul(X.e23inf, Y.e123nilinf), mul(X.scalar, Y.e1inf)), add(add(add(add(add(mul(X.e2, Y.e12inf), mul(X.e23, Y.e123inf)), mul(X.e2inf, Y.e12nilinf)), mul(X.e3, Y.e13inf)), mul(X.e3inf, Y.e13nilinf)), mul(X.einf, Y.e1nilinf))), sub(add(add(add(add(add(mul(X.e1, Y.e123), mul(X.e1nilinf, Y.e123nilinf)), mul(X.einf, Y.e23nil)), mul(X.enil, Y.e23inf)), mul(X.enilinf, Y.e23nilinf)), mul(X.scalar, Y.e23)), add(mul(X.e1inf, Y.e123nil), mul(X.e1nil, Y.e123inf))), sub(add(add(add(add(add(mul(X.e1, Y.e12nil), mul(X.e13, Y.e123nil)), mul(X.e13nil, Y.e123nilinf)), mul(X.e3nil, Y.e23nilinf)), mul(X.enil, Y.e2nilinf)), mul(X.scalar, Y.e2nil)), add(mul(X.e1nil, Y.e12nilinf), mul(X.e3, Y.e23nil))), sub(add(add(add(mul(X.e1, Y.e12inf), mul(X.e13, Y.e123inf)), mul(X.e1inf, Y.e12nilinf)), mul(X.scalar, Y.e2inf)), add(add(add(mul(X.e13inf, Y.e123nilinf), mul(X.e3, Y.e23inf)), mul(X.e3inf, Y.e23nilinf)), mul(X.einf, Y.e2nilinf))), sub(add(add(add(mul(X.e1, Y.e13nil), mul(X.e2, Y.e23nil)), mul(X.enil, Y.e3nilinf)), mul(X.scalar, Y.e3nil)), add(add(add(mul(X.e12, Y.e123nil), mul(X.e12nil, Y.e123nilinf)), mul(X.e1nil, Y.e13nilinf)), mul(X.e2nil, Y.e23nilinf))), sub(add(add(add(add(add(mul(X.e1, Y.e13inf), mul(X.e12inf, Y.e123nilinf)), mul(X.e1inf, Y.e13nilinf)), mul(X.e2, Y.e23inf)), mul(X.e2inf, Y.e23nilinf)), mul(X.scalar, Y.e3inf)), add(mul(X.e12, Y.e123inf), mul(X.einf, Y.e3nilinf))), sub(add(add(add(mul(X.e1, Y.e1nilinf), mul(X.e2, Y.e2nilinf)), mul(X.e3, Y.e3nilinf)), mul(X.scalar, Y.enilinf)), add(add(add(mul(X.e12, Y.e12nilinf), mul(X.e123, Y.e123nilinf)), mul(X.e13, Y.e13nilinf)), mul(X.e23, Y.e23nilinf))), sub(add(mul(X.enilinf, Y.e123nilinf), mul(X.scalar, Y.e123)), add(mul(X.einf, Y.e123nil), mul(X.enil, Y.e123inf))), sub(add(add(mul(X.e3, Y.e123nil), mul(X.e3nil, Y.e123nilinf)), mul(X.scalar, Y.e12nil)), mul(X.enil, Y.e12nilinf)), sub(add(add(mul(X.e3, Y.e123inf), mul(X.einf, Y.e12nilinf)), mul(X.scalar, Y.e12inf)), mul(X.e3inf, Y.e123nilinf)), sub(mul(X.scalar, Y.e13nil), add(add(mul(X.e2, Y.e123nil), mul(X.e2nil, Y.e123nilinf)), mul(X.enil, Y.e13nilinf))), sub(add(add(mul(X.e2inf, Y.e123nilinf), mul(X.einf, Y.e13nilinf)), mul(X.scalar, Y.e13inf)), mul(X.e2, Y.e123inf)), sub(mul(X.scalar, Y.e1nilinf), add(add(mul(X.e2, Y.e12nilinf), mul(X.e23, Y.e123nilinf)), mul(X.e3, Y.e13nilinf))), sub(add(add(mul(X.e1, Y.e123nil), mul(X.e1nil, Y.e123nilinf)), mul(X.scalar, Y.e23nil)), mul(X.enil, Y.e23nilinf)), sub(add(add(mul(X.e1, Y.e123inf), mul(X.einf, Y.e23nilinf)), mul(X.scalar, Y.e23inf)), mul(X.e1inf, Y.e123nilinf)), sub(add(add(mul(X.e1, Y.e12nilinf), mul(X.e13, Y.e123nilinf)), mul(X.scalar, Y.e2nilinf)), mul(X.e3, Y.e23nilinf)), sub(add(add(mul(X.e1, Y.e13nilinf), mul(X.e2, Y.e23nilinf)), mul(X.scalar, Y.e3nilinf)), mul(X.e12, Y.e123nilinf)), add(mul(X.enil, Y.e123nilinf), mul(X.scalar, Y.e123nil)), sub(mul(X.scalar, Y.e123inf), mul(X.einf, Y.e123nilinf)), add(mul(X.e3, Y.e123nilinf), mul(X.scalar, Y.e12nilinf)), sub(mul(X.scalar, Y.e13nilinf), mul(X.e2, Y.e123nilinf)), add(mul(X.e1, Y.e123nilinf), mul(X.scalar, Y.e23nilinf)), mul(X.scalar, Y.e123nilinf));
}

CGA3 outer(CGA3 X, CGA3 Y){
    return CGA3(mul(X.scalar, Y.scalar), add(mul(X.e1, Y.scalar), mul(X.scalar, Y.e1)), add(mul(X.e2, Y.scalar), mul(X.scalar, Y.e2)), add(mul(X.e3, Y.scalar), mul(X.scalar, Y.e3)), add(mul(X.enil, Y.scalar), mul(X.scalar, Y.enil)), add(mul(X.einf, Y.scalar), mul(X.scalar, Y.einf)), sub(add(add(mul(X.e1, Y.e2), mul(X.e12, Y.scalar)), mul(X.scalar, Y.e12)), mul(X.e2, Y.e1)), sub(add(add(mul(X.e1, Y.e3), mul(X.e13, Y.scalar)), mul(X.scalar, Y.e13)), mul(X.e3, Y.e1)), sub(add(add(mul(X.e1, Y.enil), mul(X.e1nil, Y.scalar)), mul(X.scalar, Y.e1nil)), mul(X.enil, Y.e1)), sub(add(add(mul(X.e1, Y.einf), mul(X.e1inf, Y.scalar)), mul(X.scalar, Y.e1inf)), mul(X.einf, Y.e1)), sub(add(add(mul(X.e2, Y.e3), mul(X.e23, Y.scalar)), mul(X.scalar, Y.e23)), mul(X.e3, Y.e2)), sub(add(add(mul(X.e2, Y.enil), mul(X.e2nil, Y.scalar)), mul(X.scalar, Y.e2nil)), mul(X.enil, Y.e2)), sub(add(add(mul(X.e2, Y.einf), mul(X.e2inf, Y.scalar)), mul(X.scalar, Y.e2inf)), mul(X.einf, Y.e2)), sub(add(add(mul(X.e3, Y.enil), mul(X.e3nil, Y.scalar)), mul(X.scalar, Y.e3nil)), mul(X.enil, Y.e3)), sub(add(add(mul(X.e3, Y.einf), mul(X.e3inf, Y.scalar)), mul(X.scalar, Y.e3inf)), mul(X.einf, Y.e3)), sub(add(add(mul(X.enil, Y.einf), mul(X.enilinf, Y.scalar)), mul(X.scalar, Y.enilinf)), mul(X.einf, Y.enil)), sub(add(add(add(add(add(mul(X.e1, Y.e23), mul(X.e12, Y.e3)), mul(X.e123, Y.scalar)), mul(X.e23, Y.e1)), mul(X.e3, Y.e12)), mul(X.scalar, Y.e123)), add(mul(X.e13, Y.e2), mul(X.e2, Y.e13))), sub(add(add(add(add(add(mul(X.e1, Y.e2nil), mul(X.e12, Y.enil)), mul(X.e12nil, Y.scalar)), mul(X.e2nil, Y.e1)), mul(X.enil, Y.e12)), mul(X.scalar, Y.e12nil)), add(mul(X.e1nil, Y.e2), mul(X.e2, Y.e1nil))), sub(add(add(add(add(add(mul(X.e1, Y.e2inf), mul(X.e12, Y.einf)), mul(X.e12inf, Y.scalar)), mul(X.e2inf, Y.e1)), mul(X.einf, Y.e12)), mul(X.scalar, Y.e12inf)), add(mul(X.e1inf, Y.e2), mul(X.e2, Y.e1inf))), sub(add(add(add(add(add(mul(X.e1, Y.e3nil), mul(X.e13, Y.enil)), mul(X.e13nil, Y.scalar)), mul(X.e3nil, Y.e1)), mul(X.enil, Y.e13)), mul(X.scalar, Y.e13nil)), add(mul(X.e1nil, Y.e3), mul(X.e3, Y.e1nil))), sub(add(add(add(add(add(mul(X.e1, Y.e3inf), mul(X.e13, Y.einf)), mul(X.e13inf, Y.scalar)), mul(X.e3inf, Y.e1)), mul(X.einf, Y.e13)), mul(X.scalar, Y.e13inf)), add(mul(X.e1inf, Y.e3), mul(X.e3, Y.e1inf))), sub(add(add(add(add(add(mul(X.e1, Y.enilinf), mul(X.e1nil, Y.einf)), mul(X.e1nilinf, Y.scalar)), mul(X.einf, Y.e1nil)), mul(X.enilinf, Y.e1)), mul(X.scalar, Y.e1nilinf)), add(mul(X.e1inf, Y.enil), mul(X.enil, Y.e1inf))), sub(add(add(add(add(add(mul(X.e2, Y.e3nil), mul(X.e23, Y.enil)), mul(X.e23nil, Y.scalar)), mul(X.e3nil, Y.e2)), mul(X.enil, Y.e23)), mul(X.scalar, Y.e23nil)), add(mul(X.e2nil, Y.e3), mul(X.e3, Y.e2nil))), sub(add(add(add(add(add(mul(X.e2, Y.e3inf), mul(X.e23, Y.einf)), mul(X.e23inf, Y.scalar)), mul(X.e3inf, Y.e2)), mul(X.einf, Y.e23)), mul(X.scalar, Y.e23inf)), add(mul(X.e2inf, Y.e3), mul(X.e3, Y.e2inf))), sub(add(add(add(add(add(mul(X.e2, Y.enilinf), mul(X.e2nil, Y.einf)), mul(X.e2nilinf, Y.scalar)), mul(X.einf, Y.e2nil)), mul(X.enilinf, Y.e2)), mul(X.scalar, Y.e2nilinf)), add(mul(X.e2inf, Y.enil), mul(X.enil, Y.e2inf))), sub(add(add(add(add(add(mul(X.e3, Y.enilinf), mul(X.e3nil, Y.einf)), mul(X.e3nilinf, Y.scalar)), mul(X.einf, Y.e3nil)), mul(X.enilinf, Y.e3)), mul(X.scalar, Y.e3nilinf)), add(mul(X.e3inf, Y.enil), mul(X.enil, Y.e3inf))), sub(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e23nil), mul(X.e12, Y.e3nil)), mul(X.e123, Y.enil)), mul(X.e123nil, Y.scalar)), mul(X.e13nil, Y.e2)), mul(X.e1nil, Y.e23)), mul(X.e23, Y.e1nil)), mul(X.e3, Y.e12nil)), mul(X.e3nil, Y.e12)), mul(X.scalar, Y.e123nil)), add(add(add(add(add(mul(X.e12nil, Y.e3), mul(X.e13, Y.e2nil)), mul(X.e2, Y.e13nil)), mul(X.e23nil, Y.e1)), mul(X.e2nil, Y.e13)), mul(X.enil, Y.e123))), sub(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e23inf), mul(X.e12, Y.e3inf)), mul(X.e123, Y.einf)), mul(X.e123inf, Y.scalar)), mul(X.e13inf, Y.e2)), mul(X.e1inf, Y.e23)), mul(X.e23, Y.e1inf)), mul(X.e3, Y.e12inf)), mul(X.e3inf, Y.e12)), mul(X.scalar, Y.e123inf)), add(add(add(add(add(mul(X.e12inf, Y.e3), mul(X.e13, Y.e2inf)), mul(X.e2, Y.e13inf)), mul(X.e23inf, Y.e1)), mul(X.e2inf, Y.e13)), mul(X.einf, Y.e123))), sub(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e2nilinf), mul(X.e12, Y.enilinf)), mul(X.e12nil, Y.einf)), mul(X.e12nilinf, Y.scalar)), mul(X.e1inf, Y.e2nil)), mul(X.e1nilinf, Y.e2)), mul(X.e2nil, Y.e1inf)), mul(X.enil, Y.e12inf)), mul(X.enilinf, Y.e12)), mul(X.scalar, Y.e12nilinf)), add(add(add(add(add(mul(X.e12inf, Y.enil), mul(X.e1nil, Y.e2inf)), mul(X.e2, Y.e1nilinf)), mul(X.e2inf, Y.e1nil)), mul(X.e2nilinf, Y.e1)), mul(X.einf, Y.e12nil))), sub(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e3nilinf), mul(X.e13, Y.enilinf)), mul(X.e13nil, Y.einf)), mul(X.e13nilinf, Y.scalar)), mul(X.e1inf, Y.e3nil)), mul(X.e1nilinf, Y.e3)), mul(X.e3nil, Y.e1inf)), mul(X.enil, Y.e13inf)), mul(X.enilinf, Y.e13)), mul(X.scalar, Y.e13nilinf)), add(add(add(add(add(mul(X.e13inf, Y.enil), mul(X.e1nil, Y.e3inf)), mul(X.e3, Y.e1nilinf)), mul(X.e3inf, Y.e1nil)), mul(X.e3nilinf, Y.e1)), mul(X.einf, Y.e13nil))), sub(add(add(add(add(add(add(add(add(add(mul(X.e2, Y.e3nilinf), mul(X.e23, Y.enilinf)), mul(X.e23nil, Y.einf)), mul(X.e23nilinf, Y.scalar)), mul(X.e2inf, Y.e3nil)), mul(X.e2nilinf, Y.e3)), mul(X.e3nil, Y.e2inf)), mul(X.enil, Y.e23inf)), mul(X.enilinf, Y.e23)), mul(X.scalar, Y.e23nilinf)), add(add(add(add(add(mul(X.e23inf, Y.enil), mul(X.e2nil, Y.e3inf)), mul(X.e3, Y.e2nilinf)), mul(X.e3inf, Y.e2nil)), mul(X.e3nilinf, Y.e2)), mul(X.einf, Y.e23nil))), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e23nilinf), mul(X.e12, Y.e3nilinf)), mul(X.e123, Y.enilinf)), mul(X.e123nil, Y.einf)), mul(X.e123nilinf, Y.scalar)), mul(X.e12inf, Y.e3nil)), mul(X.e12nilinf, Y.e3)), mul(X.e13nil, Y.e2inf)), mul(X.e1nil, Y.e23inf)), mul(X.e1nilinf, Y.e23)), mul(X.e23, Y.e1nilinf)), mul(X.e23inf, Y.e1nil)), mul(X.e23nilinf, Y.e1)), mul(X.e2inf, Y.e13nil)), mul(X.e3, Y.e12nilinf)), mul(X.e3nil, Y.e12inf)), mul(X.e3nilinf, Y.e12)), mul(X.einf, Y.e123nil)), mul(X.enilinf, Y.e123)), mul(X.scalar, Y.e123nilinf)), add(add(add(add(add(add(add(add(add(add(add(mul(X.e123inf, Y.enil), mul(X.e12nil, Y.e3inf)), mul(X.e13, Y.e2nilinf)), mul(X.e13inf, Y.e2nil)), mul(X.e13nilinf, Y.e2)), mul(X.e1inf, Y.e23nil)), mul(X.e2, Y.e13nilinf)), mul(X.e23nil, Y.e1inf)), mul(X.e2nil, Y.e13inf)), mul(X.e2nilinf, Y.e13)), mul(X.e3inf, Y.e12nil)), mul(X.enil, Y.e123inf))));
}

#define I_CGA3 CGA3(ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ONE_Dual)

CGA3 rcontract(CGA3 X, CGA3 Y){
    return CGA3(sub(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, Y.e1), mul(X.e123inf, Y.e123nil)), mul(X.e123nil, Y.e123inf)), mul(X.e1nilinf, Y.e1nilinf)), mul(X.e2, Y.e2)), mul(X.e2nilinf, Y.e2nilinf)), mul(X.e3, Y.e3)), mul(X.e3nilinf, Y.e3nilinf)), mul(X.einf, Y.enil)), mul(X.enil, Y.einf)), mul(X.enilinf, Y.enilinf)), mul(X.scalar, Y.scalar)), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e12, Y.e12), mul(X.e123, Y.e123)), mul(X.e123nilinf, Y.e123nilinf)), mul(X.e12inf, Y.e12nil)), mul(X.e12nil, Y.e12inf)), mul(X.e12nilinf, Y.e12nilinf)), mul(X.e13, Y.e13)), mul(X.e13inf, Y.e13nil)), mul(X.e13nil, Y.e13inf)), mul(X.e13nilinf, Y.e13nilinf)), mul(X.e1inf, Y.e1nil)), mul(X.e1nil, Y.e1inf)), mul(X.e23, Y.e23)), mul(X.e23inf, Y.e23nil)), mul(X.e23nil, Y.e23inf)), mul(X.e23nilinf, Y.e23nilinf)), mul(X.e2inf, Y.e2nil)), mul(X.e2nil, Y.e2inf)), mul(X.e3inf, Y.e3nil)), mul(X.e3nil, Y.e3inf))), sub(add(add(add(add(add(add(add(mul(X.e1, Y.scalar), mul(X.e12, Y.e2)), mul(X.e12nilinf, Y.e2nilinf)), mul(X.e13, Y.e3)), mul(X.e13nilinf, Y.e3nilinf)), mul(X.e1inf, Y.enil)), mul(X.e1nil, Y.einf)), mul(X.e1nilinf, Y.enilinf)), add(add(add(add(add(add(add(mul(X.e123, Y.e23), mul(X.e123inf, Y.e23nil)), mul(X.e123nil, Y.e23inf)), mul(X.e123nilinf, Y.e23nilinf)), mul(X.e12inf, Y.e2nil)), mul(X.e12nil, Y.e2inf)), mul(X.e13inf, Y.e3nil)), mul(X.e13nil, Y.e3inf))), sub(add(add(add(add(add(add(add(add(add(add(add(mul(X.e123, Y.e13), mul(X.e123inf, Y.e13nil)), mul(X.e123nil, Y.e13inf)), mul(X.e123nilinf, Y.e13nilinf)), mul(X.e12inf, Y.e1nil)), mul(X.e12nil, Y.e1inf)), mul(X.e2, Y.scalar)), mul(X.e23, Y.e3)), mul(X.e23nilinf, Y.e3nilinf)), mul(X.e2inf, Y.enil)), mul(X.e2nil, Y.einf)), mul(X.e2nilinf, Y.enilinf)), add(add(add(mul(X.e12, Y.e1), mul(X.e12nilinf, Y.e1nilinf)), mul(X.e23inf, Y.e3nil)), mul(X.e23nil, Y.e3inf))), sub(add(add(add(add(add(add(add(mul(X.e13inf, Y.e1nil), mul(X.e13nil, Y.e1inf)), mul(X.e23inf, Y.e2nil)), mul(X.e23nil, Y.e2inf)), mul(X.e3, Y.scalar)), mul(X.e3inf, Y.enil)), mul(X.e3nil, Y.einf)), mul(X.e3nilinf, Y.enilinf)), add(add(add(add(add(add(add(mul(X.e123, Y.e12), mul(X.e123inf, Y.e12nil)), mul(X.e123nil, Y.e12inf)), mul(X.e123nilinf, Y.e12nilinf)), mul(X.e13, Y.e1)), mul(X.e13nilinf, Y.e1nilinf)), mul(X.e23, Y.e2)), mul(X.e23nilinf, Y.e2nilinf))), sub(add(add(add(add(add(mul(X.e123nil, Y.e123), mul(X.e1nilinf, Y.e1nil)), mul(X.e2nilinf, Y.e2nil)), mul(X.e3nilinf, Y.e3nil)), mul(X.enil, Y.scalar)), mul(X.enilinf, Y.enil)), add(add(add(add(add(add(add(add(add(mul(X.e123nilinf, Y.e123nil), mul(X.e12nil, Y.e12)), mul(X.e12nilinf, Y.e12nil)), mul(X.e13nil, Y.e13)), mul(X.e13nilinf, Y.e13nil)), mul(X.e1nil, Y.e1)), mul(X.e23nil, Y.e23)), mul(X.e23nilinf, Y.e23nil)), mul(X.e2nil, Y.e2)), mul(X.e3nil, Y.e3))), sub(add(add(add(add(add(mul(X.e123inf, Y.e123), mul(X.e123nilinf, Y.e123inf)), mul(X.e12nilinf, Y.e12inf)), mul(X.e13nilinf, Y.e13inf)), mul(X.e23nilinf, Y.e23inf)), mul(X.einf, Y.scalar)), add(add(add(add(add(add(add(add(add(mul(X.e12inf, Y.e12), mul(X.e13inf, Y.e13)), mul(X.e1inf, Y.e1)), mul(X.e1nilinf, Y.e1inf)), mul(X.e23inf, Y.e23)), mul(X.e2inf, Y.e2)), mul(X.e2nilinf, Y.e2inf)), mul(X.e3inf, Y.e3)), mul(X.e3nilinf, Y.e3inf)), mul(X.enilinf, Y.einf))), sub(add(add(add(add(add(mul(X.e12, Y.scalar), mul(X.e123, Y.e3)), mul(X.e123nilinf, Y.e3nilinf)), mul(X.e12inf, Y.enil)), mul(X.e12nil, Y.einf)), mul(X.e12nilinf, Y.enilinf)), add(mul(X.e123inf, Y.e3nil), mul(X.e123nil, Y.e3inf))), sub(add(add(add(add(add(mul(X.e123inf, Y.e2nil), mul(X.e123nil, Y.e2inf)), mul(X.e13, Y.scalar)), mul(X.e13inf, Y.enil)), mul(X.e13nil, Y.einf)), mul(X.e13nilinf, Y.enilinf)), add(mul(X.e123, Y.e2), mul(X.e123nilinf, Y.e2nilinf))), sub(add(add(add(mul(X.e12nilinf, Y.e2nil), mul(X.e13nilinf, Y.e3nil)), mul(X.e1nil, Y.scalar)), mul(X.e1nilinf, Y.enil)), add(add(add(mul(X.e123nil, Y.e23), mul(X.e123nilinf, Y.e23nil)), mul(X.e12nil, Y.e2)), mul(X.e13nil, Y.e3))), sub(add(mul(X.e123nilinf, Y.e23inf), mul(X.e1inf, Y.scalar)), add(add(add(add(add(mul(X.e123inf, Y.e23), mul(X.e12inf, Y.e2)), mul(X.e12nilinf, Y.e2inf)), mul(X.e13inf, Y.e3)), mul(X.e13nilinf, Y.e3inf)), mul(X.e1nilinf, Y.einf))), sub(add(add(add(add(add(mul(X.e123, Y.e1), mul(X.e123nilinf, Y.e1nilinf)), mul(X.e23, Y.scalar)), mul(X.e23inf, Y.enil)), mul(X.e23nil, Y.einf)), mul(X.e23nilinf, Y.enilinf)), add(mul(X.e123inf, Y.e1nil), mul(X.e123nil, Y.e1inf))), sub(add(add(add(add(add(mul(X.e123nil, Y.e13), mul(X.e123nilinf, Y.e13nil)), mul(X.e12nil, Y.e1)), mul(X.e23nilinf, Y.e3nil)), mul(X.e2nil, Y.scalar)), mul(X.e2nilinf, Y.enil)), add(mul(X.e12nilinf, Y.e1nil), mul(X.e23nil, Y.e3))), sub(add(add(add(mul(X.e123inf, Y.e13), mul(X.e12inf, Y.e1)), mul(X.e12nilinf, Y.e1inf)), mul(X.e2inf, Y.scalar)), add(add(add(mul(X.e123nilinf, Y.e13inf), mul(X.e23inf, Y.e3)), mul(X.e23nilinf, Y.e3inf)), mul(X.e2nilinf, Y.einf))), sub(add(add(add(mul(X.e13nil, Y.e1), mul(X.e23nil, Y.e2)), mul(X.e3nil, Y.scalar)), mul(X.e3nilinf, Y.enil)), add(add(add(mul(X.e123nil, Y.e12), mul(X.e123nilinf, Y.e12nil)), mul(X.e13nilinf, Y.e1nil)), mul(X.e23nilinf, Y.e2nil))), sub(add(add(add(add(add(mul(X.e123nilinf, Y.e12inf), mul(X.e13inf, Y.e1)), mul(X.e13nilinf, Y.e1inf)), mul(X.e23inf, Y.e2)), mul(X.e23nilinf, Y.e2inf)), mul(X.e3inf, Y.scalar)), add(mul(X.e123inf, Y.e12), mul(X.e3nilinf, Y.einf))), sub(add(add(add(mul(X.e1nilinf, Y.e1), mul(X.e2nilinf, Y.e2)), mul(X.e3nilinf, Y.e3)), mul(X.enilinf, Y.scalar)), add(add(add(mul(X.e123nilinf, Y.e123), mul(X.e12nilinf, Y.e12)), mul(X.e13nilinf, Y.e13)), mul(X.e23nilinf, Y.e23))), add(add(add(mul(X.e123, Y.scalar), mul(X.e123inf, Y.enil)), mul(X.e123nil, Y.einf)), mul(X.e123nilinf, Y.enilinf)), sub(add(add(mul(X.e123nilinf, Y.e3nil), mul(X.e12nil, Y.scalar)), mul(X.e12nilinf, Y.enil)), mul(X.e123nil, Y.e3)), sub(mul(X.e12inf, Y.scalar), add(add(mul(X.e123inf, Y.e3), mul(X.e123nilinf, Y.e3inf)), mul(X.e12nilinf, Y.einf))), sub(add(add(mul(X.e123nil, Y.e2), mul(X.e13nil, Y.scalar)), mul(X.e13nilinf, Y.enil)), mul(X.e123nilinf, Y.e2nil)), sub(add(add(mul(X.e123inf, Y.e2), mul(X.e123nilinf, Y.e2inf)), mul(X.e13inf, Y.scalar)), mul(X.e13nilinf, Y.einf)), sub(add(add(mul(X.e12nilinf, Y.e2), mul(X.e13nilinf, Y.e3)), mul(X.e1nilinf, Y.scalar)), mul(X.e123nilinf, Y.e23)), sub(add(add(mul(X.e123nilinf, Y.e1nil), mul(X.e23nil, Y.scalar)), mul(X.e23nilinf, Y.enil)), mul(X.e123nil, Y.e1)), sub(mul(X.e23inf, Y.scalar), add(add(mul(X.e123inf, Y.e1), mul(X.e123nilinf, Y.e1inf)), mul(X.e23nilinf, Y.einf))), sub(add(add(mul(X.e123nilinf, Y.e13), mul(X.e23nilinf, Y.e3)), mul(X.e2nilinf, Y.scalar)), mul(X.e12nilinf, Y.e1)), sub(mul(X.e3nilinf, Y.scalar), add(add(mul(X.e123nilinf, Y.e12), mul(X.e13nilinf, Y.e1)), mul(X.e23nilinf, Y.e2))), add(mul(X.e123nil, Y.scalar), mul(X.e123nilinf, Y.enil)), sub(mul(X.e123inf, Y.scalar), mul(X.e123nilinf, Y.einf)), add(mul(X.e123nilinf, Y.e3), mul(X.e12nilinf, Y.scalar)), sub(mul(X.e13nilinf, Y.scalar), mul(X.e123nilinf, Y.e2)), add(mul(X.e123nilinf, Y.e1), mul(X.e23nilinf, Y.scalar)), mul(X.e123nilinf, Y.scalar));
}

CGA3 reverse(CGA3 X){
    return CGA3(X.scalar, X.e1, X.e2, X.e3, X.enil, X.einf, mul(-1, X.e12), mul(-1, X.e13), mul(-1, X.e1nil), mul(-1, X.e1inf), mul(-1, X.e23), mul(-1, X.e2nil), mul(-1, X.e2inf), mul(-1, X.e3nil), mul(-1, X.e3inf), mul(-1, X.enilinf), mul(-1, X.e123), mul(-1, X.e12nil), mul(-1, X.e12inf), mul(-1, X.e13nil), mul(-1, X.e13inf), mul(-1, X.e1nilinf), mul(-1, X.e23nil), mul(-1, X.e23inf), mul(-1, X.e2nilinf), mul(-1, X.e3nilinf), X.e123nil, X.e123inf, X.e12nilinf, X.e13nilinf, X.e23nilinf, X.e123nilinf);
}

CGA3 fromArray(float X[32]){
    return CGA3(scalar_Dual(X[0]), scalar_Dual(X[1]), scalar_Dual(X[2]), scalar_Dual(X[3]), scalar_Dual(X[4]), scalar_Dual(X[5]), scalar_Dual(X[6]), scalar_Dual(X[7]), scalar_Dual(X[8]), scalar_Dual(X[9]), scalar_Dual(X[10]), scalar_Dual(X[11]), scalar_Dual(X[12]), scalar_Dual(X[13]), scalar_Dual(X[14]), scalar_Dual(X[15]), scalar_Dual(X[16]), scalar_Dual(X[17]), scalar_Dual(X[18]), scalar_Dual(X[19]), scalar_Dual(X[20]), scalar_Dual(X[21]), scalar_Dual(X[22]), scalar_Dual(X[23]), scalar_Dual(X[24]), scalar_Dual(X[25]), scalar_Dual(X[26]), scalar_Dual(X[27]), scalar_Dual(X[28]), scalar_Dual(X[29]), scalar_Dual(X[30]), scalar_Dual(X[31]));
}

CGA3 conjugate(CGA3 X){
    return reverse(involve(X));
}

CGA3 outer(CGA3 X, CGA3 Y, CGA3 Z){
    return outer(outer(X, Y), Z);
}

CGA3 INF(){
    return CGA3(ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ONE_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual);
}

CGA3 MNK(){
    return CGA3(ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ONE_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual);
}

CGA3 NIL(){
    return CGA3(ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ONE_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual, ZERO_Dual);
}

CGA3 point(CGA3 X){
    return CGA3(sub(add(add(add(mul(X.e1nil, X.e1nilinf), mul(X.e2nil, X.e2nilinf)), mul(X.e3nil, X.e3nilinf)), X.scalar), add(add(add(mul(X.e12, X.e12nil), mul(X.e123nil, X.e123nilinf)), mul(X.e13, X.e13nil)), mul(X.e23, X.e23nil))), sub(add(add(add(X.e1, mul(X.e12nilinf, X.e2nil)), mul(X.e13nilinf, X.e3nil)), mul(X.e1nilinf, X.enil)), add(add(add(mul(X.e123nil, X.e23), mul(X.e123nilinf, X.e23nil)), mul(X.e12nil, X.e2)), mul(X.e13nil, X.e3))), sub(add(add(add(add(add(mul(X.e1, X.e12nil), mul(X.e123nil, X.e13)), mul(X.e123nilinf, X.e13nil)), X.e2), mul(X.e23nilinf, X.e3nil)), mul(X.e2nilinf, X.enil)), add(mul(X.e12nilinf, X.e1nil), mul(X.e23nil, X.e3))), sub(add(add(add(mul(X.e1, X.e13nil), mul(X.e2, X.e23nil)), X.e3), mul(X.e3nilinf, X.enil)), add(add(add(mul(X.e12, X.e123nil), mul(X.e123nilinf, X.e12nil)), mul(X.e13nilinf, X.e1nil)), mul(X.e23nilinf, X.e2nil))), add(X.enil, ONE_Dual), sub(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(0.5, mul(X.e1, X.e1)), mul(X.e12, X.e12nilinf)), mul(X.e123, X.e123nilinf)), mul(X.e123inf, X.e123nil)), mul(X.e13, X.e13nilinf)), mul(0.5, mul(X.e1nilinf, X.e1nilinf))), mul(0.5, mul(X.e2, X.e2))), mul(X.e23, X.e23nilinf)), mul(0.5, mul(X.e2nilinf, X.e2nilinf))), mul(0.5, mul(X.e3, X.e3))), mul(0.5, mul(X.e3nilinf, X.e3nilinf))), mul(X.einf, X.enil)), X.einf), mul(0.5, mul(X.enilinf, X.enilinf))), add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(add(mul(X.e1, X.e1nilinf), mul(0.5, mul(X.e12, X.e12))), mul(0.5, mul(X.e123, X.e123))), mul(0.5, mul(X.e123nilinf, X.e123nilinf))), mul(X.e12inf, X.e12nil)), mul(0.5, mul(X.e12nilinf, X.e12nilinf))), mul(0.5, mul(X.e13, X.e13))), mul(X.e13inf, X.e13nil)), mul(0.5, mul(X.e13nilinf, X.e13nilinf))), mul(X.e1inf, X.e1nil)), mul(X.e2, X.e2nilinf)), mul(0.5, mul(X.e23, X.e23))), mul(X.e23inf, X.e23nil)), mul(0.5, mul(X.e23nilinf, X.e23nilinf))), mul(X.e2inf, X.e2nil)), mul(X.e3, X.e3nilinf)), mul(X.e3inf, X.e3nil))), add(X.e12, mul(X.e123nilinf, X.e3nil)), sub(X.e13, mul(X.e123nilinf, X.e2nil)), X.e1nil, sub(add(add(mul(X.e123nilinf, X.e23), X.e1inf), mul(X.e1nilinf, X.enilinf)), add(add(add(add(add(mul(X.e123, X.e23), mul(X.e123nilinf, X.e23nilinf)), mul(X.e12inf, X.e2nil)), mul(X.e12nil, X.e2inf)), mul(X.e13inf, X.e3nil)), mul(X.e13nil, X.e3inf))), add(mul(X.e123nilinf, X.e1nil), X.e23), X.e2nil, sub(add(add(add(add(add(mul(X.e123, X.e13), mul(X.e123nilinf, X.e13nilinf)), mul(X.e12inf, X.e1nil)), mul(X.e12nil, X.e1inf)), X.e2inf), mul(X.e2nilinf, X.enilinf)), add(add(mul(X.e123nilinf, X.e13), mul(X.e23inf, X.e3nil)), mul(X.e23nil, X.e3inf))), X.e3nil, sub(add(add(add(add(add(add(mul(X.e12, X.e123nilinf), mul(X.e13inf, X.e1nil)), mul(X.e13nil, X.e1inf)), mul(X.e23inf, X.e2nil)), mul(X.e23nil, X.e2inf)), X.e3inf), mul(X.e3nilinf, X.enilinf)), add(mul(X.e12, X.e123), mul(X.e123nilinf, X.e12nilinf))), sub(add(add(add(mul(X.e1nil, X.e1nilinf), mul(X.e2nil, X.e2nilinf)), mul(X.e3nil, X.e3nilinf)), X.enilinf), add(add(add(mul(X.e12, X.e12nil), mul(X.e123nil, X.e123nilinf)), mul(X.e13, X.e13nil)), mul(X.e23, X.e23nil))), add(X.e123, mul(X.e123nilinf, X.enil)), X.e12nil, sub(add(add(add(add(add(mul(X.e123, X.e3), mul(X.e123nilinf, X.e3nilinf)), mul(X.e12inf, X.enil)), X.e12inf), mul(X.e12nil, X.einf)), mul(X.e12nilinf, X.enilinf)), add(add(mul(X.e123inf, X.e3nil), mul(X.e123nil, X.e3inf)), mul(X.e123nilinf, X.e3))), X.e13nil, sub(add(add(add(add(add(add(mul(X.e123inf, X.e2nil), mul(X.e123nil, X.e2inf)), mul(X.e123nilinf, X.e2)), mul(X.e13inf, X.enil)), X.e13inf), mul(X.e13nil, X.einf)), mul(X.e13nilinf, X.enilinf)), add(mul(X.e123, X.e2), mul(X.e123nilinf, X.e2nilinf))), sub(add(add(add(mul(X.e12nilinf, X.e2nil), mul(X.e13nilinf, X.e3nil)), mul(X.e1nilinf, X.enil)), X.e1nilinf), add(add(add(mul(X.e123nil, X.e23), mul(X.e123nilinf, X.e23nil)), mul(X.e12nil, X.e2)), mul(X.e13nil, X.e3))), X.e23nil, sub(add(add(add(add(add(mul(X.e1, X.e123), mul(X.e123nilinf, X.e1nilinf)), mul(X.e23inf, X.enil)), X.e23inf), mul(X.e23nil, X.einf)), mul(X.e23nilinf, X.enilinf)), add(add(mul(X.e1, X.e123nilinf), mul(X.e123inf, X.e1nil)), mul(X.e123nil, X.e1inf))), sub(add(add(add(add(add(mul(X.e1, X.e12nil), mul(X.e123nil, X.e13)), mul(X.e123nilinf, X.e13nil)), mul(X.e23nilinf, X.e3nil)), mul(X.e2nilinf, X.enil)), X.e2nilinf), add(mul(X.e12nilinf, X.e1nil), mul(X.e23nil, X.e3))), sub(add(add(add(mul(X.e1, X.e13nil), mul(X.e2, X.e23nil)), mul(X.e3nilinf, X.enil)), X.e3nilinf), add(add(add(mul(X.e12, X.e123nil), mul(X.e123nilinf, X.e12nil)), mul(X.e13nilinf, X.e1nil)), mul(X.e23nilinf, X.e2nil))), X.e123nil, add(X.e123inf, mul(X.e123nilinf, X.enilinf)), add(mul(X.e123nilinf, X.e3nil), X.e12nilinf), sub(X.e13nilinf, mul(X.e123nilinf, X.e2nil)), add(mul(X.e123nilinf, X.e1nil), X.e23nilinf), add(mul(X.e123nilinf, X.enil), X.e123nilinf));
}

CGA3 inv(CGA3 X){
    return mul(inv(lcontract(X,conjugate(X)).scalar), conjugate(X));
}

CGA3 div(CGA3 X, CGA3 Y){
    return mul(X, inv(Y));
}

CGA3 dual(CGA3 X){
    return div(X, I_CGA3);
}

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

// CGA3 inv(CGA3 x) {
//   return mul(1.0/mul(x, reverse(x)).scalar, reverse(x));
// }

// CGA3 div(CGA3 a, CGA3 b) {
//   return mul(a, inv(b));
// }

CGA3 weight(vec3 w) {
  return inv(fromVec(normalize(w)));
}

CGA3 U(float u) {
  CGA3 U = scalar_CGA3(u);
  U.scalar.enil1 = 1.0;
  return U;
}

CGA3 V(float v) {
  CGA3 V = scalar_CGA3(v);
  V.scalar.enil2 = 1.0;
  return V;
}

CGA3 bilinearQuad(
  vec3 p0, vec3 p1, vec3 p2, vec3 p3,
  vec3 w0, vec3 w1, vec3 w2, vec3 w3,
  float u, float v) {

    CGA3 W0 = mul(sub(ONE_CGA3, U(u)), sub(ONE_CGA3, V(v)),  weight(w0)); // 0, 0
    CGA3 W1 = mul(U(u), sub(ONE_CGA3, V(v)),                weight(w1)); // 1, 0
    CGA3 W2 = mul(sub(ONE_CGA3, U(u)), V(v),                weight(w2)); // 0, 1
    CGA3 W3 = mul(U(u), V(v),                               weight(w3)); // 1, 1
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
    // return bottom;
}