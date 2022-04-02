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