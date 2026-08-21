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

// --- Conformal-dilation primitive (mirrors quilting-core::conformal_lod) ---
uniform vec4  u_pole;        // pole h = -c^-1 d, packed (w,x,y,z) like mob_*
uniform float u_mob_k;       // inversion power k (|F(x)-F(y)| = k|x-y|/(|x-h||y-h|))
uniform float u_c_norm_sq;   // |c|^2
uniform float u_has_pole;    // 1.0 if the transform has a finite pole

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

// Pure Möbius apply that does NOT touch the min_bot_sq accumulator. Used by the
// interior-floor image-point probe: its samples sit near the pole, so routing
// them through mobius() would poison the cull's sampled denominator.
vec3 mobius_pure(vec3 p) {
    vec4 q = vec4(0.0, p);
    vec4 top = qmul(mob_a, q) + mob_b;
    vec4 bot = qmul(mob_c, q) + mob_d;
    return qmul(top, qinv(bot)).yzw;
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
    // Skinning data is tiled 4096 vertices wide, with two rows per tile.
    int chunk = vertex_id / 4096;
    int col = vertex_id % 4096;
    vec4 ji = texelFetch(u_skinning, ivec2(col, chunk * 2), 0);
    vec4 jw = texelFetch(u_skinning, ivec2(col, chunk * 2 + 1), 0);

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

// Squared distance from p to segment [a,b].
float seg_dist_sq(vec3 p, vec3 a, vec3 b) {
    vec3 ab = b - a;
    float t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-30), 0.0, 1.0);
    vec3 q = a + t * ab;
    return dot(p - q, p - q);
}

// Closest point on triangle abc to p (Ericson §5.1.5). Branch order MUST match
// quilting-core::conformal_lod::closest_point_triangle so CPU/GPU LoD agree.
vec3 closest_point_triangle(vec3 p, vec3 a, vec3 b, vec3 c) {
    vec3 ab = b - a; vec3 ac = c - a; vec3 ap = p - a;
    float d1 = dot(ab, ap); float d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;
    vec3 bp = p - b;
    float d3 = dot(ab, bp); float d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return b;
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return a + ab * (d1 / (d1 - d3));
    vec3 cp = p - c;
    float d5 = dot(ab, cp); float d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return c;
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return a + ac * (d2 / (d2 - d6));
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    float denom = 1.0 / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

// Deformed world point -> screen pixels (matches the attenuation projection).
vec2 project_screen(vec3 world) {
    vec4 c = vp_matrix * vec4(world, 1.0);
    return (c.xy / max(c.w, 0.001)) * vec2(vp_width, vp_height) * 0.5;
}

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

    // Conformal primitive: exact closest approach of this face to the Möbius pole.
    // bot = c·q + d is affine ⇒ |bot|² convex ⇒ the max-dilation point (closest to
    // the pole) is closed-form. This is interior-complete: it catches the
    // barycentric-centre dilation the 6 rim samples miss (the spike-fan).
    // Mirrors quilting-core::conformal_lod.
    float min_bot_true = min_bot_sq;   // fallback: sampled min (affine/degenerate)
    vec3  x_star = mid_a;
    float dT2 = 0.0;
    bool  patch_valid = false;         // true only for a non-degenerate face with a pole
    if (u_has_pole > 0.5) {
        // Degenerate (needle / collinear) faces have no well-defined closest point;
        // closest_point_triangle would divide by ~0 and spray NaN through the pole
        // guard and floor. Skip them and fall back to the rim, mirroring
        // ConformalPatch::new's None guard.
        vec3 nrm = cross(p1 - p0, p2 - p0);
        if (dot(nrm, nrm) >= 1e-24) {
            vec3 hi = u_pole.yzw;
            float hw2 = u_pole.x * u_pole.x;
            x_star = closest_point_triangle(hi, p0, p1, p2);
            dT2 = hw2 + dot(x_star - hi, x_star - hi);
            min_bot_true = u_c_norm_sq * dT2;
            patch_valid = true;
        }
    }

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
        // Also project the deformed edge midpoints (already computed above).
        vec4 cma = vp_matrix * vec4(dm_a, 1.0);
        vec4 cmb = vp_matrix * vec4(dm_b, 1.0);
        vec4 cmc = vp_matrix * vec4(dm_c, 1.0);
        // Clamp w to the near side. A point behind the camera (w < 0) must not
        // mirror to a finite screen position (the old abs(w) did) — that faked a
        // small extent and over-attenuated straddling faces.
        vec2 s0 = (c0.xy / max(c0.w, 0.001)) * vec2(vp_width, vp_height) * 0.5;
        vec2 s1 = (c1.xy / max(c1.w, 0.001)) * vec2(vp_width, vp_height) * 0.5;
        vec2 s2 = (c2.xy / max(c2.w, 0.001)) * vec2(vp_width, vp_height) * 0.5;
        vec2 sma = (cma.xy / max(cma.w, 0.001)) * vec2(vp_width, vp_height) * 0.5;
        vec2 smb = (cmb.xy / max(cmb.w, 0.001)) * vec2(vp_width, vp_height) * 0.5;
        vec2 smc = (cmc.xy / max(cmc.w, 0.001)) * vec2(vp_width, vp_height) * 0.5;

        // Screen extent via the deformed midpoint (corner -> mid -> corner),
        // not corner-to-corner. Projective maps preserve collinearity, so for an
        // undeformed straight edge the midpoint projects onto the segment and
        // this sum equals the old chord exactly (normal scenes unchanged). For a
        // Möbius-deformed edge it lower-bounds the true screen arc and catches
        // the funnel excursion that corner-only sampling misses — which starved
        // giant pole-adjacent faces to LOD ~4 and produced the spike-fan.
        // Mirrors the CPU screen_arc_len sampling in evaluate.rs.
        float px_a = distance(s1, sma) + distance(sma, s2);
        float px_b = distance(s0, smb) + distance(smb, s2);
        float px_c = distance(s0, smc) + distance(smc, s1);

        // Interior floor — robust analytic form (mirrors conformal_edge_lods).
        // The LoD that makes the tessellated sub-edge at the max-dilation point x*
        // about min_px on screen is  λ*·ρ·L_max, where:
        //   • λ* = k / d_T²  is the CLOSED-FORM peak conformal scale. The old code
        //     estimated the peak by a per-edge boost and a finite-difference of F
        //     itself; both are near-singular next to the pole, so on animated
        //     (skinned) faces that graze the pole they exploded into the flickering
        //     high-LoD starbursts. This form never differences the singular map.
        //   • ρ is the projection Jacobian at the well-conditioned IMAGE point
        //     y* = F(x*) — a benign point, not the pole — by finite differences.
        // Applied to all three edges via max (the atlas grades interior density as
        // the geometric mean of the edges); far-from-pole faces have a tiny λ* so it
        // stays inert with no gate. If y* is behind the near plane the face wraps
        // past it — leave it to the rim + cull rather than emit a spurious spike.
        float px_int = 0.0;
        if (patch_valid && min_bot_true >= 1e-8) {
            float l_max = max(distance(p1, p2), max(distance(p0, p2), distance(p0, p1)));
            float lambda_star = u_mob_k / max(dT2, 1e-30);
            vec3 y_star = mobius_pure(x_star);
            vec4 cy = vp_matrix * vec4(y_star, 1.0);
            if (cy.w > 0.001) {
                vec2 s0 = (cy.xy / cy.w) * vec2(vp_width, vp_height) * 0.5;
                float eps = 1e-3;
                float rho = 0.0;
                vec3 dirs[4] = vec3[4](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0),
                    vec3(0.0, 0.0, 1.0), vec3(0.5774, 0.5774, 0.5774));
                for (int i = 0; i < 4; i++) {
                    vec4 cyp = vp_matrix * vec4(y_star + eps * dirs[i], 1.0);
                    if (cyp.w > 0.001) {
                        vec2 sp = (cyp.xy / cyp.w) * vec2(vp_width, vp_height) * 0.5;
                        rho = max(rho, distance(sp, s0) / eps);
                    }
                }
                px_int = lambda_star * rho * l_max;
            }
        }
        lod_a = clamp(snap_pow2(max(px_a, px_int) / min_px), 2.0, max_lod);
        lod_b = clamp(snap_pow2(max(px_b, px_int) / min_px), 2.0, max_lod);
        lod_c = clamp(snap_pow2(max(px_c, px_int) / min_px), 2.0, max_lod);
    }

    // Pole proximity overrides everything. For a valid patch min_bot_true is the
    // EXACT minimum |c·q+d|² over the solid triangle (closed form), so this fires
    // whenever the pole is genuinely inside/on the face — not only when a sample
    // lands on it. Gated on patch_valid so affine/degenerate faces (whose
    // min_bot_true is only the sampled rim min) don't false-fire.
    // Threshold mirrors POLE_PROXIMITY_NORM_SQ in quilting-core.
    if (patch_valid && min_bot_true < 1e-8) {
        lod_a = max_lod;
        lod_b = max_lod;
        lod_c = max_lod;
    }

    // Frustum cull. Geometric LOD is camera-independent, so a far-offscreen
    // (Möbius-inflated) face still demands — and builds — a huge patch it
    // contributes zero pixels to. Test the 6 already-deformed sample points
    // (corners + edge midpoints): if all lie past one frustum plane, or all sit
    // behind the camera, pin the face to MIN LOD so it is never tessellated.
    {
        vec3 pts[6] = vec3[6](d0, d1, d2, dm_a, dm_b, dm_c);
        bool allBehind = true, allR = true, allL = true, allT = true, allB = true, allFront = true;
        for (int i = 0; i < 6; i++) {
            vec4 cp = vp_matrix * vec4(pts[i], 1.0);
            allBehind = allBehind && (cp.w <= 0.0);
            allFront  = allFront  && (cp.w >  0.0);
            allR = allR && (cp.x >  cp.w);
            allL = allL && (cp.x < -cp.w);
            allT = allT && (cp.y >  cp.w);
            allB = allB && (cp.y < -cp.w);
        }
        // Only cull faces that are NOT conformally inflated. A near-pole patch
        // bulges far beyond these corner/midpoint samples (the inversion
        // spheroid), so its rendered surface can be on-screen even when every
        // sample is off — culling on the samples alone drops it. Keying on the
        // EXACT min_bot_true (not the sampled min_bot_sq, which can read large when
        // no sample happens to land near an interior pole) means a pole-inflated
        // face is never culled even if all six samples fall off-screen: large
        // min_bot_true ⇒ tame patch whose samples bound its true extent ⇒ safe.
        bool off = allBehind || (allFront && (allR || allL || allT || allB));
        if (off && min_bot_true > 0.25) {
            lod_a = 2.0; lod_b = 2.0; lod_c = 2.0;
        }
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
