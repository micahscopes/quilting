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

// Skeletal animation: skin matrices (joint_world * inverse_bind) uploaded per frame.
// num_joints = 0 means no skinning active.
const MAX_JOINTS: u32 = 128u;

struct JointMatrices {
    num_joints: i32,
    skin_tex_width: i32,
    _jpad1: i32,
    _jpad2: i32,
    matrices: array<mat4x4<f32>, 128>,
}

@group(0) @binding(1)
var<uniform> joints: JointMatrices;

// Per-vertex skinning data texture: width = num_verts, height = 2
// Row 0: joint indices (as f32), Row 1: joint weights
@group(0) @binding(2)
var skinning_tex: texture_2d<f32>;

// Apply skeletal skinning to a position.
// vertex_idx indexes into the skinning texture.
fn skin_tex_lookup(vertex_idx: i32) -> array<vec4<f32>, 2> {
    // Tiled texture: width = skin_tex_width, rows = chunks * 2
    // Row 2*chunk = joint indices, row 2*chunk+1 = joint weights
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

// Apply skeletal skinning to a normal (mat3 upper-left of joint matrix).
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
        skinned = skinned + w * (mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz) * nrm);
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

    // Conformal fade: |bot|² → 0 near Möbius pole
    let fade = smoothstep(0.0001, 0.001, dot(bot, bot));

    // Analytic normal via quotient rule
    let dtop_u = pw1 - pw0;
    let dbot_u = bw1 - bw0;
    let dtop_v = pw2 - pw0;
    let dbot_v = bw2 - bw0;
    let dXu = qmul(dtop_u - qmul(X, dbot_u), bi);
    let dXv = qmul(dtop_v - qmul(X, dbot_v), bi);

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

    // GPU skeletal skinning: vertex indices are packed in p.w (flat path only).
    // When joints.num_joints > 0, skin rest-pose positions before Möbius eval.
    var sp0 = in.p0;
    var sp1 = in.p1;
    var sp2 = in.p2;
    if joints.num_joints > 0 && u.use_qb == 0 {
        let vi0 = i32(in.p0.x);
        let vi1 = i32(in.p1.x);
        let vi2 = i32(in.p2.x);
        // Skin positions (rest pose is in p.yzw)
        let skinned0 = skin_position(in.p0.yzw, vi0);
        let skinned1 = skin_position(in.p1.yzw, vi1);
        let skinned2 = skin_position(in.p2.yzw, vi2);
        // Rebuild quaternion format (w=0 for flat path)
        sp0 = vec4<f32>(0.0, skinned0);
        sp1 = vec4<f32>(0.0, skinned1);
        sp2 = vec4<f32>(0.0, skinned2);
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
        nrm = normalize(cross(mp1.yzw - mp0.yzw, mp2.yzw - mp0.yzw));
        nrm = nrm * u.perm_parity;
    }

    // Smooth normals: skin them if GPU skinning is active, then transform through Möbius.
    var sn0 = in.smooth_n0.xyz;
    var sn1 = in.smooth_n1.xyz;
    var sn2 = in.smooth_n2.xyz;
    if joints.num_joints > 0 && u.use_qb == 0 {
        let vi0 = i32(in.p0.x);
        let vi1 = i32(in.p1.x);
        let vi2 = i32(in.p2.x);
        sn0 = skin_normal(sn0, vi0);
        sn1 = skin_normal(sn1, vi1);
        sn2 = skin_normal(sn2, vi2);
    }
    let has_smooth = dot(sn0, sn0) + dot(sn1, sn1) + dot(sn2, sn2) > 0.01;
    if has_smooth {
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

            nrm = normalize(bary.x * rn0 + bary.y * rn1 + bary.z * rn2);
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
    out.instance_id = f32(instance_idx) + u._pad;  // _pad = batch offset for pick pass, 0 normally
    return out;
}
