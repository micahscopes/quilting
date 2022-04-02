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
