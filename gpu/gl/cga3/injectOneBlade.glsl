/* Inject array V at the indices for e1, e2, e3 in array U */
void injectOneBladeArray(inout Dual U[32], Dual V[3]){
    U[Idx_CGA3_e1] = V[0];
    U[Idx_CGA3_e2] = V[1];
    U[Idx_CGA3_e3] = V[2];
}

/* Inject array V into e1, e2, e3 of struct U */
CGA3 injectOneBlade(CGA3 U, Dual V[3]){
    Dual U_ary[32];
    toArray(U, U_ary);
    injectOneBladeArray(U_ary, V);
    return fromArray(U_ary);
}