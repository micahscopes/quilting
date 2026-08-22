#version 300 es
precision highp float;

// Pass 2: Edge coherence + canonical sort + atlas LUT lookup.
//
// Reads per-face LOD exponents from pass 1 (as texture), reads adjacency
// data (static texture), enforces shared-edge LOD agreement via max,
// then sorts to canonical form and looks up the atlas index.

// Pass 1 output: LOD exponents + visibility per face.
// Tiled 4096-wide RGBA32F: texel = (lod_a, lod_b, lod_c, visible).
uniform highp sampler2D u_pass1_lods;

// Adjacency: for each face × 3 edges, (neighbor_face, neighbor_lod_idx, 0, 0).
// Linear layout: texel at index (face_id * 3 + edge) in 4096-wide tiling.
// neighbor_face < 0 means boundary (no neighbor).
uniform highp sampler2D u_adjacency;

// Atlas LUT: exponent triple → atlas index (40×30 R8, same as pass 1 used to use)
uniform highp sampler2D u_atlas_lut;

uniform int u_num_faces;

// Transform feedback outputs: directly consumable by group_into_batches
out float out_canon_a;
out float out_canon_b;
out float out_canon_c;
out float out_perm_index;
out float out_parity;
out float out_atlas_index;

// Read pass 1 LODs for a face (tiled 4096-wide)
vec4 read_face(int face_id) {
    int tx = face_id % 4096;
    int ty = face_id / 4096;
    return texelFetch(u_pass1_lods, ivec2(tx, ty), 0);
}

// Read adjacency for face_id, edge (0-2).
// Returns vec2(neighbor_face, neighbor_lod_idx). neighbor_face < 0 = boundary.
vec2 read_adj(int face_id, int edge) {
    int idx = face_id * 3 + edge;
    int tx = idx % 4096;
    int ty = idx / 4096;
    return texelFetch(u_adjacency, ivec2(tx, ty), 0).xy;
}

void main() {
    int fi = gl_VertexID;
    if (fi >= u_num_faces) {
        out_canon_a = 1.0; out_canon_b = 1.0; out_canon_c = 1.0;
        out_perm_index = 0.0; out_parity = 1.0; out_atlas_index = -1.0;
        gl_Position = vec4(0.0);
        return;
    }

    vec4 face = read_face(fi);
    if (face.w < 0.5) {
        out_canon_a = 1.0; out_canon_b = 1.0; out_canon_c = 1.0;
        out_perm_index = 0.0; out_parity = 1.0; out_atlas_index = -1.0;
        gl_Position = vec4(0.0);
        return;
    }
    float ea = face.x;
    float eb = face.y;
    float ec = face.z;

    // Edge coherence: for each of this face's 3 edges, if there's a neighbor,
    // take max of our LOD exponent and the neighbor's corresponding exponent.
    for (int edge = 0; edge < 3; edge++) {
        vec2 adj = read_adj(fi, edge);
        int neighbor = int(adj.x);
        if (neighbor < 0) continue; // boundary edge

        int neighbor_lod_idx = int(adj.y);
        vec4 neighbor_face = read_face(neighbor);
        if (neighbor_face.w < 0.5) continue;

        float neighbor_exp = (neighbor_lod_idx == 0) ? neighbor_face.x
                           : (neighbor_lod_idx == 1) ? neighbor_face.y
                           : neighbor_face.z;

        if (edge == 0) ea = max(ea, neighbor_exp);
        else if (edge == 1) eb = max(eb, neighbor_exp);
        else ec = max(ec, neighbor_exp);
    }

    // Convert exponents back to integers for canonical sort
    int ia = int(ea + 0.5);
    int ib = int(eb + 0.5);
    int ic = int(ec + 0.5);

    // Canonical form: sort ascending, track S3 permutation
    int sa, sb, sc, perm;
    if (ia <= ib && ib <= ic)       { sa=ia; sb=ib; sc=ic; perm=0; }
    else if (ia <= ic && ic <= ib)  { sa=ia; sb=ic; sc=ib; perm=1; }
    else if (ib <= ia && ia <= ic)  { sa=ib; sb=ia; sc=ic; perm=2; }
    else if (ib <= ic && ic <= ia)  { sa=ib; sb=ic; sc=ia; perm=4; }
    else if (ic <= ia && ia <= ib)  { sa=ic; sb=ia; sc=ib; perm=3; }
    else                            { sa=ic; sb=ib; sc=ia; perm=5; }

    // Atlas LUT lookup
    int key = sa + sb * 10 + sc * 100;
    int lut_x = key % 40;
    int lut_y = key / 40;
    float atlas_index = texelFetch(u_atlas_lut, ivec2(lut_x, lut_y), 0).r * 255.0;

    // Parity: even permutations (identity, cycles) = +1, odd (transpositions) = -1
    // Perms 0,3,4 are even; 1,2,5 are odd
    float parity = (perm == 1 || perm == 2 || perm == 5) ? -1.0 : 1.0;

    // Output canonical LODs as actual values (2^exponent), not exponents
    out_canon_a = exp2(float(sa));
    out_canon_b = exp2(float(sb));
    out_canon_c = exp2(float(sc));
    out_perm_index = float(perm);
    out_parity = parity;
    out_atlas_index = atlas_index;

    gl_Position = vec4(0.0);
}
