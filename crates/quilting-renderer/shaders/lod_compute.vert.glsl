#version 300 es
precision highp float;

// Per-face input: 3 vertex indices (packed as ivec3 via float attrib)
layout(location = 0) in vec3 face_indices;

// Rest-pose vertex positions: sampled via texelFetch.
// For static models, this is the only position source.
// Layout: texel (vertex_index) = vec4(x, y, z, 0), packed into 4096-wide rows.
uniform highp sampler2D u_positions;
uniform int u_num_vertices;

// --- GPU animation support ---
// When u_num_joints > 0 or u_num_morph_targets > 0, the shader evaluates
// animated positions per-vertex before computing LOD. This matches the
// rendering vertex shader's animation pipeline (morph first, then skin).

uniform int u_num_joints;        // 0 = no skeletal animation
uniform int u_num_morph_targets; // 0 = no morph animation

// Skinning data: width = num_vertices, height = 2 (RGBA32F)
// Row 0: joint indices (as float), Row 1: joint weights
uniform highp sampler2D u_skinning;

// Joint matrices: width = 4, height = num_joints (RGBA32F)
// Each row = one joint's 4x4 skin matrix (world * inverse_bind), column-major.
// texelFetch(u_joints, ivec2(col, joint_idx), 0) = column `col` of the matrix
uniform highp sampler2D u_joints;

// Morph deltas: width = num_vertices, height = num_targets (RGBA32F)
// Each texel = (dx, dy, dz, 0) position delta
uniform highp sampler2D u_morph_deltas;

// Morph weights: width = num_morph_targets, height = 1 (R32F or RGBA32F)
// texelFetch(u_morph_wt, ivec2(target, 0), 0).r = blend weight for target
uniform highp sampler2D u_morph_wt;

// --- Möbius transform ---
// M(q) = (a*q + b) * (c*q + d)^{-1}, stored as 4 quaternions
uniform vec4 mob_a;
uniform vec4 mob_b;
uniform vec4 mob_c;
uniform vec4 mob_d;

uniform float density;
uniform float mesh_radius;
uniform float min_px;        // minimum pixels per subdivision (0 = no attenuation)
uniform float max_lod;       // maximum LOD the atlas supports (clamp, don't fall off)
uniform mat4 vp_matrix;      // view-projection matrix for screen-space projection
uniform float vp_width;      // viewport width in pixels
uniform float vp_height;     // viewport height in pixels
uniform highp sampler2D u_atlas_lut; // exponent triple → atlas index (40×30 R8)

// Transform feedback outputs
out float out_atlas_index;
out float out_perm_index;

// --- Quaternion math ---

vec4 qmul(vec4 a, vec4 b) {
    return vec4(
        a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
        a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
        a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
        a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x
    );
}

vec4 qinv(vec4 q) {
    float n = dot(q, q);
    return vec4(q.x, -q.yzw) / max(n, 1e-20);
}

vec3 mobius(vec3 p) {
    vec4 q = vec4(0.0, p);
    vec4 top = qmul(mob_a, q) + mob_b;
    vec4 bot = qmul(mob_c, q) + mob_d;
    vec4 result = qmul(top, qinv(bot));
    return result.yzw;
}

float snap_pow2(float v) {
    return exp2(round(log2(max(v, 1.0))));
}

// --- Position fetch with animation ---

vec3 fetch_rest_pos(int vertex_id) {
    int tx = vertex_id % 4096;
    int ty = vertex_id / 4096;
    return texelFetch(u_positions, ivec2(tx, ty), 0).xyz;
}

vec3 apply_morph(vec3 pos, int vertex_id) {
    if (u_num_morph_targets <= 0) return pos;
    vec3 result = pos;
    for (int t = 0; t < 64; t++) {
        if (t >= u_num_morph_targets) break;
        float w = texelFetch(u_morph_wt, ivec2(t, 0), 0).r;
        if (abs(w) < 1e-6) continue;
        vec3 delta = texelFetch(u_morph_deltas, ivec2(vertex_id, t), 0).xyz;
        result += w * delta;
    }
    return result;
}

vec3 skin_position(vec3 pos, int vertex_id) {
    if (u_num_joints <= 0) return pos;
    vec4 ji = texelFetch(u_skinning, ivec2(vertex_id, 0), 0);
    vec4 jw = texelFetch(u_skinning, ivec2(vertex_id, 1), 0);

    vec3 skinned = vec3(0.0);
    vec4 p4 = vec4(pos, 1.0);

    for (int k = 0; k < 4; k++) {
        float w = jw[k];
        if (w < 1e-6) continue;
        int idx = int(ji[k]);
        if (idx >= u_num_joints) continue;
        mat4 m = mat4(
            texelFetch(u_joints, ivec2(0, idx), 0),
            texelFetch(u_joints, ivec2(1, idx), 0),
            texelFetch(u_joints, ivec2(2, idx), 0),
            texelFetch(u_joints, ivec2(3, idx), 0)
        );
        skinned += w * (m * p4).xyz;
    }
    return skinned;
}

// Fetch rest-pose, apply morph targets, then skeletal skinning.
vec3 fetch_animated_pos(int vertex_id) {
    vec3 pos = fetch_rest_pos(vertex_id);
    pos = apply_morph(pos, vertex_id);
    pos = skin_position(pos, vertex_id);
    return pos;
}

// --- Main ---

void main() {
    float out_lod_a, out_lod_b, out_lod_c;

    // Fetch animated vertex positions
    vec3 p0 = fetch_animated_pos(int(face_indices.x));
    vec3 p1 = fetch_animated_pos(int(face_indices.y));
    vec3 p2 = fetch_animated_pos(int(face_indices.z));

    // Edge midpoints
    vec3 mid_a = (p1 + p2) * 0.5;
    vec3 mid_b = (p0 + p2) * 0.5;
    vec3 mid_c = (p0 + p1) * 0.5;

    // Apply Möbius transform
    vec3 d0 = mobius(p0);
    vec3 d1 = mobius(p1);
    vec3 d2 = mobius(p2);
    vec3 dm_a = mobius(mid_a);
    vec3 dm_b = mobius(mid_b);
    vec3 dm_c = mobius(mid_c);

    // Deformed medians: vertex-to-opposite-midpoint distance in post-Möbius space.
    // Captures both intrinsic geometry size and Möbius conformal stretch.
    float med_a = distance(d0, dm_a);
    float med_b = distance(d1, dm_b);
    float med_c = distance(d2, dm_c);

    float target_size = mesh_radius / density;

    // Uniform per-face LOD from the largest median.
    // Max prevents small-median sabotage on skinny triangles.
    // Uniform prevents anisotropic tessellation artifacts.
    float max_med = max(med_a, max(med_b, med_c)) / target_size;
    out_lod_a = clamp(snap_pow2(max_med), 2.0, max_lod);
    out_lod_b = out_lod_a;
    out_lod_c = out_lod_a;

    // Screen-space attenuation
    if (min_px > 0.0) {
        vec4 c0 = vp_matrix * vec4(d0, 1.0);
        vec4 c1 = vp_matrix * vec4(d1, 1.0);
        vec4 c2 = vp_matrix * vec4(d2, 1.0);
        vec2 s0 = (c0.xy / max(abs(c0.w), 0.001)) * vec2(vp_width, vp_height) * 0.5;
        vec2 s1 = (c1.xy / max(abs(c1.w), 0.001)) * vec2(vp_width, vp_height) * 0.5;
        vec2 s2 = (c2.xy / max(abs(c2.w), 0.001)) * vec2(vp_width, vp_height) * 0.5;

        float px_a = distance(s1, s2);
        float px_b = distance(s0, s2);
        float px_c = distance(s0, s1);

        if (px_a / out_lod_a < min_px) out_lod_a = clamp(snap_pow2(px_a / min_px), 2.0, max_lod);
        if (px_b / out_lod_b < min_px) out_lod_b = clamp(snap_pow2(px_b / min_px), 2.0, max_lod);
        if (px_c / out_lod_c < min_px) out_lod_c = clamp(snap_pow2(px_c / min_px), 2.0, max_lod);
    }

    // Canonical form + S3 permutation
    int ea = int(log2(out_lod_a));
    int eb = int(log2(out_lod_b));
    int ec = int(log2(out_lod_c));

    int sa, sb, sc, perm;
    if (ea <= eb && eb <= ec)       { sa=ea; sb=eb; sc=ec; perm=0; }
    else if (ea <= ec && ec <= eb)  { sa=ea; sb=ec; sc=eb; perm=1; }
    else if (eb <= ea && ea <= ec)  { sa=eb; sb=ea; sc=ec; perm=2; }
    else if (eb <= ec && ec <= ea)  { sa=eb; sb=ec; sc=ea; perm=4; }
    else if (ec <= ea && ea <= eb)  { sa=ec; sb=ea; sc=eb; perm=3; }
    else                            { sa=ec; sb=eb; sc=ea; perm=5; }

    // Atlas LUT lookup
    int key = sa + sb * 10 + sc * 100;
    int lut_x = key % 40;
    int lut_y = key / 40;
    out_atlas_index = texelFetch(u_atlas_lut, ivec2(lut_x, lut_y), 0).r * 255.0;
    out_perm_index = float(perm);

    gl_Position = vec4(0.0);
}
