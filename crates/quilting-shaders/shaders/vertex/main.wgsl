// Main vertex shader for QB surface rendering.
// Imports quilting modules for quaternion math, surface evaluation, and density viz.
// Compiles to GLSL ES 300 via naga for WebGL2.

#import quilting::math::quaternion::{qmul, qconj, qinv, q_to_point}
#import quilting::surface::qb_eval::{eval_qb_with_normal, QBResult}
#import quilting::viz::density::edge_density

struct Uniforms {
    mvp: mat4x4<f32>,
    mv: mat4x4<f32>,
    perm_parity: f32,
    perm_index: i32,
    use_qb: i32,
    _pad: f32,
    // Möbius transform: p' = (a*p + b) * (c*p + d)^{-1}
    mob_a: vec4<f32>,
    mob_b: vec4<f32>,
    mob_c: vec4<f32>,
    mob_d: vec4<f32>,
    camera_pos: vec4<f32>,   // world-space camera position (xyz, w unused)
}

@group(0) @binding(0)
var<uniform> u: Uniforms;

// Hard ceiling on evaluated positions, in model units (meshes are normalized
// to unit half-extent upstream). Near a Möbius pole |X| ~ 1/|bot| grows
// without bound and bottoms out at the qinv sentinel (~1e10) when the f32
// denominator cancels outright; coordinates that large wreck rasterizer and
// depth precision. 1e4 is far beyond anything distinguishable on screen at
// any usable zoom — sphere inversions that legitimately send geometry far
// away are untouched — while staying well inside f32 range for the mvp
// multiply. Direction is preserved, so clamped spikes still point the right
// way and shrink continuously as the pole recedes.
const POSITION_CLAMP: f32 = 1e4;

// Skeletal animation: skin matrices (joint_world * inverse_bind) uploaded per frame.
// num_joints = 0 means no skinning active.
const MAX_JOINTS: u32 = 128u;

struct JointMatrices {
    num_joints: i32,
    skin_tex_width: i32,
    num_morph_targets: i32,
    _jpad2: i32,
    matrices: array<mat4x4<f32>, 128>,
    // Morph weights packed as vec4s after the matrices.
    // morph_weights[i/4][i%4] = weight for target i
    // Max 64 morph targets (16 vec4s)
    morph_weights: array<vec4<f32>, 16>,
}

@group(0) @binding(1)
var<uniform> joints: JointMatrices;

// Per-vertex skinning data texture: width = num_verts, height = 2
// Row 0: joint indices (as f32), Row 1: joint weights
@group(0) @binding(2)
var skinning_tex: texture_2d<f32>;

// Morph target deltas texture: width = num_verts, height = num_targets
// Each texel = (dx, dy, dz, 0) position delta for that vertex+target
@group(0) @binding(3)
var morph_tex: texture_2d<f32>;

// Apply skeletal skinning to a position.
// vertex_idx indexes into the skinning texture.
// Apply morph target deltas to a position.
fn apply_morph(pos: vec3<f32>, vertex_idx: i32) -> vec3<f32> {
    if joints.num_morph_targets <= 0 { return pos; }
    var result = pos;
    let nt = joints.num_morph_targets;
    for (var t = 0; t < 64; t = t + 1) {
        if t >= nt { break; }
        let w = joints.morph_weights[t / 4][t % 4];
        if abs(w) < 1e-6 { continue; }
        let delta = textureLoad(morph_tex, vec2<i32>(vertex_idx, t), 0).xyz;
        result = result + w * delta;
    }
    return result;
}

fn skin_tex_lookup(vertex_idx: i32) -> array<vec4<f32>, 2> {
    // Tiled layout: width = skin_tex_width, rows alternate (indices, weights) per chunk
    let w = joints.skin_tex_width;
    let chunk = vertex_idx / w;
    let col = vertex_idx % w;
    var result: array<vec4<f32>, 2>;
    result[0] = textureLoad(skinning_tex, vec2<i32>(col, chunk * 2), 0);
    result[1] = textureLoad(skinning_tex, vec2<i32>(col, chunk * 2 + 1), 0);
    return result;
}

fn skin_position(pos: vec3<f32>, vertex_idx: i32) -> vec3<f32> {
    if joints.num_joints <= 0 { return pos; }

    let skin_data = skin_tex_lookup(vertex_idx);
    let ji = skin_data[0];
    let jw = skin_data[1];

    var skinned = vec3<f32>(0.0);
    let p4 = vec4<f32>(pos, 1.0);

    for (var k = 0u; k < 4u; k = k + 1u) {
        let w = jw[k];
        if w < 1e-6 { continue; }
        let idx = i32(ji[k]);
        if idx >= joints.num_joints { continue; }
        let m = joints.matrices[idx];
        skinned = skinned + w * (m * p4).xyz;
    }
    return skinned;
}

// Apply skeletal skinning to a normal using the cofactor (adjugate) of the 3x3.
// cofactor(M) = det(M) * inverse_transpose(M), which correctly handles non-uniform scale.
// For pure rotation (det=1), this is equivalent to M itself.
fn skin_normal(nrm: vec3<f32>, vertex_idx: i32) -> vec3<f32> {
    if joints.num_joints <= 0 { return nrm; }

    let skin_data = skin_tex_lookup(vertex_idx);
    let ji = skin_data[0];
    let jw = skin_data[1];

    var skinned = vec3<f32>(0.0);

    for (var k = 0u; k < 4u; k = k + 1u) {
        let w = jw[k];
        if w < 1e-6 { continue; }
        let idx = i32(ji[k]);
        if idx >= joints.num_joints { continue; }
        let m = joints.matrices[idx];
        let m3 = mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
        // Cofactor matrix: each row is cross product of the other two columns
        let cof = mat3x3<f32>(
            cross(m3[1], m3[2]),
            cross(m3[2], m3[0]),
            cross(m3[0], m3[1]),
        );
        skinned = skinned + w * (cof * nrm);
    }
    let l = length(skinned);
    if l > 1e-8 { return skinned / l; }
    return nrm;
}

struct VertexInput {
    @location(0) bary: vec3<f32>,
    // Per-instance QB control points (13 vec4s = 52 floats per instance)
    @location(1) p0: vec4<f32>,
    @location(2) p1: vec4<f32>,
    @location(3) p2: vec4<f32>,
    @location(4) w0: vec4<f32>,
    @location(5) w1: vec4<f32>,
    @location(6) w2: vec4<f32>,
    @location(7) lod_info: vec4<f32>,
    @location(8) vert_lod: vec4<f32>,
    @location(9) uv01: vec4<f32>,    // (u0, v0, u1, v1)
    @location(10) uv2_pad: vec4<f32>, // (u2, v2, 0, 0)
    @location(11) smooth_n0: vec4<f32>,  // (nx0, ny0, nz0, 0)
    @location(12) smooth_n1: vec4<f32>,  // (nx1, ny1, nz1, 0)
    @location(13) smooth_n2: vec4<f32>,  // (nx2, ny2, nz2, 0)
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal_vs: vec3<f32>,
    @location(1) density: f32,
    @location(2) tex_uv: vec2<f32>,
    @location(3) position_vs: vec3<f32>,
    @location(4) tangent_vs: vec3<f32>,
    @location(5) bitangent_vs: vec3<f32>,
    @location(6) normal_ws: vec3<f32>,
    @location(7) position_ws: vec3<f32>,
    @location(8) camera_pos_ws: vec3<f32>,
    @location(9) fade: f32,
    @location(10) tess_bary: vec3<f32>,
    @location(11) instance_id: f32,
    @location(12) mobius_stretch: f32,
}

// S3 permutation remapping: reorder bary coords so one tessellation
// buffer can serve all 6 permutations of a canonical LOD triple.
fn perm_bary(b: vec3<f32>, p: i32) -> vec3<f32> {
    switch p {
        case 1: { return vec3<f32>(b.x, b.z, b.y); }
        case 2: { return vec3<f32>(b.y, b.x, b.z); }
        case 3: { return vec3<f32>(b.y, b.z, b.x); }
        case 4: { return vec3<f32>(b.z, b.x, b.y); }
        case 5: { return vec3<f32>(b.z, b.y, b.x); }
        default: { return b; }
    }
}

// Flat evaluation: linear interpolation of imaginary parts
fn eval_flat(bary: vec3<f32>, p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>) -> vec3<f32> {
    return bary.x * p0.yzw + bary.y * p1.yzw + bary.z * p2.yzw;
}

// Fused Möbius + QB evaluation.
// The Möbius transform folds directly into the rational Bézier form:
//   top = Σ λᵢ (a*pᵢ+b)*wᵢ     (p'w' — inverse cancels algebraically)
//   bot = Σ λᵢ (c*pᵢ+d)*wᵢ     (conformal weight)
//   X = top * bot⁻¹              (only 1 inverse total)
struct MobiusQBResult {
    position: vec3<f32>,
    normal: vec3<f32>,
    fade: f32,
}

fn eval_mobius_qb(
    bary: vec3<f32>,
    p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>,
    w0: vec4<f32>, w1: vec4<f32>, w2: vec4<f32>,
    perm_parity: f32,
) -> MobiusQBResult {
    // Möbius-fused numerator: (a*pᵢ+b)*wᵢ
    let pw0 = qmul(qmul(u.mob_a, p0) + u.mob_b, w0);
    let pw1 = qmul(qmul(u.mob_a, p1) + u.mob_b, w1);
    let pw2 = qmul(qmul(u.mob_a, p2) + u.mob_b, w2);
    // Möbius-fused denominator: (c*pᵢ+d)*wᵢ
    let bw0 = qmul(qmul(u.mob_c, p0) + u.mob_d, w0);
    let bw1 = qmul(qmul(u.mob_c, p1) + u.mob_d, w1);
    let bw2 = qmul(qmul(u.mob_c, p2) + u.mob_d, w2);

    let top = bary.x * pw0 + bary.y * pw1 + bary.z * pw2;
    let bot = bary.x * bw0 + bary.y * bw1 + bary.z * bw2;
    let bi = qinv(bot);
    let X = qmul(top, bi);

    // Conformal fade: |bot|² → 0 near Möbius pole.
    //
    // Fade is deliberately a *shading* control (fragment alpha + discard),
    // not a geometry one. Collapsing or clipping faded vertices here would
    // pop whole patches that still contain valid visible geometry — a patch
    // can have one corner at the pole and the others in plain sight. Instead:
    // the LOD passes saturate tessellation near the pole (so every
    // sub-triangle touching it has all corners deep in the fade≈0 band and
    // gets discarded by the fragment shaders), and POSITION_CLAMP bounds
    // whatever the rasterizer sees. Sphere inversions that legitimately send
    // geometry far away keep rendering.
    let fade = smoothstep(0.0001, 0.001, dot(bot, bot));

    // Analytic normal via quotient rule: dXu = (dtop_u - X*dbot_u) * bot⁻¹.
    // Both tangents share the right factor bot⁻¹ = conj(bot)/|bot|². The
    // 1/|bot|² part is a positive real scalar common to both, so it scales
    // the cross product without turning it — and it is exactly what overflows
    // f32 near the pole: |dX| ~ 1/|bot|², so length(n) computes
    // dot(n, n) ~ 1/|bot|⁸, which hits +inf at |bot|² ≈ 2.6e-10 (ten orders
    // above the qinv guard), making n/inf = 0 and the view-space normal NaN.
    // Right-multiplying by conj(bot) instead keeps the exact normal direction
    // with every intermediate O(|X|·|bot|); normalize() absorbs the scale.
    let dtop_u = pw1 - pw0;
    let dbot_u = bw1 - bw0;
    let dtop_v = pw2 - pw0;
    let dbot_v = bw2 - bw0;
    let cb = qconj(bot);
    let dXu = qmul(dtop_u - qmul(X, dbot_u), cb);
    let dXv = qmul(dtop_v - qmul(X, dbot_v), cb);

    var n = cross(dXu.yzw, dXv.yzw);
    let nl = length(n);
    if nl > 1e-10 {
        n = n / nl;
    } else {
        n = vec3<f32>(0.0, 0.0, 1.0);
    }
    n = n * perm_parity;

    return MobiusQBResult(q_to_point(X), n, fade);
}

@vertex
fn vs_main(@builtin(instance_index) instance_idx: u32, in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Bary coords are already permuted by the WASM batch assembly.
    // Do NOT re-permute here — that was a double-application bug.
    let bary = in.bary;

    var pos: vec3<f32>;
    var nrm: vec3<f32>;
    out.fade = 1.0;

    // GPU animation: vertex indices packed in p.x, positions in p.yzw.
    // Apply morph targets first, then skeletal skinning, before Möbius/QB eval.
    // Animated positions become pure quaternion control points (w=0, xyz=position)
    // for the fused Möbius-QB surface evaluation.
    // Instance data: p.x = vertex_index, p.yzw = position.
    // Always construct pure imaginary quaternions (w=0) for Möbius math.
    let vi0 = i32(in.p0.x);
    let vi1 = i32(in.p1.x);
    let vi2 = i32(in.p2.x);
    var sp0: vec4<f32>;
    var sp1: vec4<f32>;
    var sp2: vec4<f32>;
    let has_gpu_anim = joints.num_joints > 0 || joints.num_morph_targets > 0;
    if has_gpu_anim {
        var pos0 = apply_morph(in.p0.yzw, vi0);
        var pos1 = apply_morph(in.p1.yzw, vi1);
        var pos2 = apply_morph(in.p2.yzw, vi2);
        pos0 = skin_position(pos0, vi0);
        pos1 = skin_position(pos1, vi1);
        pos2 = skin_position(pos2, vi2);
        sp0 = vec4<f32>(0.0, pos0);
        sp1 = vec4<f32>(0.0, pos1);
        sp2 = vec4<f32>(0.0, pos2);
    } else {
        sp0 = vec4<f32>(0.0, in.p0.yzw);
        sp1 = vec4<f32>(0.0, in.p1.yzw);
        sp2 = vec4<f32>(0.0, in.p2.yzw);
    }

    if u.use_qb == 1 {
        let result = eval_mobius_qb(
            bary,
            sp0, sp1, sp2,
            in.w0, in.w1, in.w2,
            u.perm_parity,
        );
        pos = result.position;
        nrm = result.normal;
        out.fade = result.fade;
    } else {
        // Flat path: apply Möbius to vertices directly
        let mp0 = qmul(qmul(u.mob_a, sp0) + u.mob_b, qinv(qmul(u.mob_c, sp0) + u.mob_d));
        let mp1 = qmul(qmul(u.mob_a, sp1) + u.mob_b, qinv(qmul(u.mob_c, sp1) + u.mob_d));
        let mp2 = qmul(qmul(u.mob_a, sp2) + u.mob_b, qinv(qmul(u.mob_c, sp2) + u.mob_d));
        pos = eval_flat(bary, mp0, mp1, mp2);
        // Normalize the edges before crossing: near a Möbius pole the
        // transformed corners reach sentinel scale (~1e10) and the raw cross
        // product overflows dot(n, n) inside normalize(). Unit edges give the
        // same direction at O(1) magnitude; degenerate edges fall back
        // instead of producing NaN.
        let fe1 = normalize(mp1.yzw - mp0.yzw);
        let fe2 = normalize(mp2.yzw - mp0.yzw);
        let fcross = cross(fe1, fe2);
        let flen = length(fcross);
        if flen > 1e-10 {
            nrm = fcross / flen;
        } else {
            nrm = vec3<f32>(0.0, 0.0, 1.0);
        }
        nrm = nrm * u.perm_parity;
    }

    // Backstop against Möbius-pole blow-ups; see POSITION_CLAMP.
    let pos_r = length(pos);
    if pos_r > POSITION_CLAMP {
        pos = pos * (POSITION_CLAMP / pos_r);
    }

    // Smooth normals: skin them if GPU skinning is active, then transform through Möbius.
    var sn0 = in.smooth_n0.xyz;
    var sn1 = in.smooth_n1.xyz;
    var sn2 = in.smooth_n2.xyz;
    if joints.num_joints > 0 {
        let vi0 = i32(in.p0.x);
        let vi1 = i32(in.p1.x);
        let vi2 = i32(in.p2.x);
        sn0 = skin_normal(sn0, vi0);
        sn1 = skin_normal(sn1, vi1);
        sn2 = skin_normal(sn2, vi2);
    }
    let has_smooth = dot(sn0, sn0) + dot(sn1, sn1) + dot(sn2, sn2) > 0.01;
    if has_smooth {
        // Threshold mirrors AFFINE_C_NORM_SQ in quilting-core: the CPU decides
        // whether to bake reflected smooth normals with the same predicate.
        let is_mobius = dot(u.mob_c, u.mob_c) > 0.001;
        if is_mobius {
            // Transform each vertex normal through the Möbius differential at that vertex
            let bot0 = qmul(u.mob_c, sp0) + u.mob_d;
            let M0 = qmul(qmul(u.mob_a, sp0) + u.mob_b, qinv(bot0));
            let a0 = u.mob_a - qmul(M0, u.mob_c);
            let rn0 = qmul(qmul(a0, vec4<f32>(0.0, sn0)), qinv(bot0)).yzw;

            let bot1 = qmul(u.mob_c, sp1) + u.mob_d;
            let M1 = qmul(qmul(u.mob_a, sp1) + u.mob_b, qinv(bot1));
            let a1 = u.mob_a - qmul(M1, u.mob_c);
            let rn1 = qmul(qmul(a1, vec4<f32>(0.0, sn1)), qinv(bot1)).yzw;

            let bot2 = qmul(u.mob_c, sp2) + u.mob_d;
            let M2 = qmul(qmul(u.mob_a, sp2) + u.mob_b, qinv(bot2));
            let a2 = u.mob_a - qmul(M2, u.mob_c);
            let rn2 = qmul(qmul(a2, vec4<f32>(0.0, sn2)), qinv(bot2)).yzw;

            // Each rnᵢ carries ~1/|botᵢ|² of conformal magnitude and reaches
            // ~1e20 near a Möbius pole, where dot() inside normalize() would
            // overflow to inf and NaN the normal. Rescale the blend by its
            // largest component first — a positive scalar, so direction and
            // blend weights are untouched. If the blend is degenerate, keep
            // the analytic normal already in nrm.
            let nsum = bary.x * rn0 + bary.y * rn1 + bary.z * rn2;
            let nmax = max(max(abs(nsum.x), abs(nsum.y)), abs(nsum.z));
            if nmax > 1e-20 {
                nrm = normalize(nsum / nmax);
            }
        } else {
            nrm = normalize(bary.x * sn0 + bary.y * sn1 + bary.z * sn2);
        }
    }

    out.normal_vs = normalize((u.mv * vec4<f32>(nrm, 0.0)).xyz);

    let d = edge_density(bary, in.lod_info.xyz);
    out.density = log2(max(d, 1.0)) / 10.0;

    let uv0 = in.uv01.xy;
    let uv1 = in.uv01.zw;
    let uv2 = in.uv2_pad.xy;
    out.tex_uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;

    // Tangent frame from Möbius-transformed positions
    let v0 = pos; // use evaluated position, not original
    // For tangent frame, use original UVs — they're invariant under Möbius
    let duv01 = uv1 - uv0;
    let duv02 = uv2 - uv0;
    let det = duv01.x * duv02.y - duv01.y * duv02.x;
    var tangent = vec3<f32>(1.0, 0.0, 0.0);
    var bitangent = vec3<f32>(0.0, 1.0, 0.0);
    if abs(det) > 1e-6 {
        // Use (possibly skinned) vertex positions for edge vectors
        let edge01 = sp1.yzw - sp0.yzw;
        let edge02 = sp2.yzw - sp0.yzw;
        let inv_det = 1.0 / det;
        tangent = normalize((edge01 * duv02.y - edge02 * duv01.y) * inv_det);
        bitangent = normalize((edge02 * duv01.x - edge01 * duv02.x) * inv_det);
    }
    out.tangent_vs = normalize((u.mv * vec4<f32>(tangent, 0.0)).xyz);
    out.bitangent_vs = normalize((u.mv * vec4<f32>(bitangent, 0.0)).xyz);

    out.normal_ws = nrm;
    out.position_ws = pos;
    out.camera_pos_ws = u.camera_pos.xyz;
    out.position_vs = (u.mv * vec4<f32>(pos, 1.0)).xyz;
    out.clip_pos = u.mvp * vec4<f32>(pos, 1.0);
    out.tess_bary = bary;
    out.instance_id = f32(instance_idx) + u._pad;

    // Möbius conformal stretch.
    let is_mobius = dot(u.mob_c, u.mob_c) > 0.001;
    if is_mobius {
        let b0 = qmul(u.mob_c, sp0) + u.mob_d;
        let b1 = qmul(u.mob_c, sp1) + u.mob_d;
        let b2 = qmul(u.mob_c, sp2) + u.mob_d;
        let s0 = 1.0 / max(dot(b0, b0), 0.001);
        let s1 = 1.0 / max(dot(b1, b1), 0.001);
        let s2 = 1.0 / max(dot(b2, b2), 0.001);
        let stretch = bary.x * s0 + bary.y * s1 + bary.z * s2;
        // Signed log2, mapped to [0,1] via sigmoid for smooth falloff (no hard cutoff).
        // 0.5 = no stretch, 0 = max squash, 1 = max expand.
        let log_s = log2(stretch);
        out.mobius_stretch = 1.0 / (1.0 + exp(-log_s * 0.25));
    } else {
        out.mobius_stretch = 0.5; // neutral = no Möbius
    }

    return out;
}
