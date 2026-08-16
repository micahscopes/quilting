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
uniform mat4 model_matrix;

uniform float density;
uniform float mesh_radius;
uniform float min_px;        // minimum pixels per subdivision (0 = no attenuation)
uniform float max_lod;       // maximum LOD the atlas supports (clamp, don't fall off)
uniform mat4 vp_matrix;      // view-projection matrix for screen-space projection
uniform float vp_width;      // viewport width in pixels
uniform float vp_height;     // viewport height in pixels
// Pass LOD exponents to fragment shader for FBO render (pass 2 reads as texture)
flat out vec3 v_lods;

uniform int u_fbo_width;
uniform int u_fbo_height;

// --- Quaternion math ---

vec4 qmul(vec4 a, vec4 b) {
    return vec4(
        a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
        a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
        a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
        a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x
    );
}

// Inverse with the Möbius-pole guard. Must match qinv in
// quilting-shaders/shaders/math/quaternion.wgsl and Quat::inv in
// quilting-core (SINGULARITY_NORM_SQ / SINGULARITY_SENTINEL): below the
// threshold the point is "very far away", not at the origin. The previous
// max(n, 1e-20) form returned 0 at the pole, collapsing straddling vertices
// to the world origin — the median then collapsed too and the most distorted
// face on screen got the LEAST tessellation.
vec4 qinv(vec4 q) {
    float n = dot(q, q);
    if (n < 1e-20) return vec4(1e10, 0.0, 0.0, 0.0);
    return vec4(q.x, -q.yzw) / n;
}

// Smallest |c*q + d|^2 seen by mobius() for the current face. The sentinel
// alone cannot keep the medians honest: when the pole sits on a sampled
// point, f32 rounding cancels the imaginary numerator together with the
// denominator (the point and the pole share the same f32 bits), so even
// sentinel * top lands at the origin. Track the denominator and saturate the
// LOD directly instead (see main below).
float min_bot_sq = 1e30;

vec3 mobius(vec3 p) {
    vec4 q = vec4(0.0, p);
    vec4 top = qmul(mob_a, q) + mob_b;
    vec4 bot = qmul(mob_c, q) + mob_d;
    min_bot_sq = min(min_bot_sq, dot(bot, bot));
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
    float lod_a, lod_b, lod_c;

    // Fetch animated vertex positions
    vec3 p0 = (model_matrix * vec4(fetch_animated_pos(int(face_indices.x)), 1.0)).xyz;
    vec3 p1 = (model_matrix * vec4(fetch_animated_pos(int(face_indices.y)), 1.0)).xyz;
    vec3 p2 = (model_matrix * vec4(fetch_animated_pos(int(face_indices.z)), 1.0)).xyz;

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
    lod_a = clamp(snap_pow2(max_med), 2.0, max_lod);
    lod_b = lod_a;
    lod_c = lod_a;

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

        if (px_a / lod_a < min_px) lod_a = clamp(snap_pow2(px_a / min_px), 2.0, max_lod);
        if (px_b / lod_b < min_px) lod_b = clamp(snap_pow2(px_b / min_px), 2.0, max_lod);
        if (px_c / lod_c < min_px) lod_c = clamp(snap_pow2(px_c / min_px), 2.0, max_lod);
    }

    // Pole proximity overrides everything: a face whose sampled denominator
    // vanishes is the most conformally stretched face in the frame, whatever
    // its (possibly origin-collapsed) medians or screen extents claim.
    // Threshold mirrors POLE_PROXIMITY_NORM_SQ in quilting-core, which
    // applies the same saturation in the CPU LOD path.
    if (min_bot_sq < 1e-8) {
        lod_a = max_lod;
        lod_b = max_lod;
        lod_c = max_lod;
    }

    // Output raw LOD exponents to fragment shader for FBO write
    v_lods = vec3(log2(lod_a), log2(lod_b), log2(lod_c));

    // Position this face's point at its corresponding pixel in the FBO
    int face_id = gl_VertexID;
    int px = face_id % u_fbo_width;
    int py = face_id / u_fbo_width;
    // Map pixel center to NDC: (px + 0.5) / width * 2 - 1
    float ndc_x = (float(px) + 0.5) / float(u_fbo_width) * 2.0 - 1.0;
    float ndc_y = (float(py) + 0.5) / float(u_fbo_height) * 2.0 - 1.0;
    gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);
    gl_PointSize = 1.0;
}
