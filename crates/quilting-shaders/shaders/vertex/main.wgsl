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
}

@group(0) @binding(0)
var<uniform> u: Uniforms;

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

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let bary = perm_bary(in.bary, u.perm_index);

    var pos: vec3<f32>;
    var nrm: vec3<f32>;

    if u.use_qb == 1 {
        let result = eval_qb_with_normal(
            bary,
            in.p0, in.p1, in.p2,
            in.w0, in.w1, in.w2,
            u.perm_parity,
        );
        pos = result.position;
        nrm = result.normal;
    } else {
        pos = eval_flat(bary, in.p0, in.p1, in.p2);
        nrm = normalize(cross(in.p1.yzw - in.p0.yzw, in.p2.yzw - in.p0.yzw));
        nrm = nrm * u.perm_parity;
    }

    // Smooth normals with conformal transform support.
    // Interpolate glTF vertex normals, then rotate by the Möbius conformal
    // factor. Since Möbius is conformal, the Jacobian is a pure rotation
    // given by the weight quaternion q = w/|w| where w = c·p + d.
    // For identity transforms, w = (1,0,0,0), rotation is identity.
    let sn0 = in.smooth_n0.xyz;
    let sn1 = in.smooth_n1.xyz;
    let sn2 = in.smooth_n2.xyz;
    let has_smooth = dot(sn0, sn0) + dot(sn1, sn1) + dot(sn2, sn2) > 0.01;
    if has_smooth {
        let smooth_n = normalize(bary.x * sn0 + bary.y * sn1 + bary.z * sn2);
        // Interpolate weight quaternion and use as conformal rotation
        let w = normalize(bary.x * in.w0 + bary.y * in.w1 + bary.z * in.w2);
        // Rotate normal: n' = w * n * conj(w)
        let n_quat = vec4<f32>(0.0, smooth_n.x, smooth_n.y, smooth_n.z);
        let rotated = qmul(qmul(w, n_quat), qconj(w));
        nrm = normalize(rotated.yzw);
    }

    out.normal_vs = normalize((u.mv * vec4<f32>(nrm, 0.0)).xyz);

    // Density: edge-based interpolation, log-mapped to [0,1]
    let d = edge_density(bary, in.lod_info.xyz);
    out.density = log2(max(d, 1.0)) / 10.0;

    // Interpolate per-vertex UVs with barycentric coordinates
    let uv0 = in.uv01.xy;
    let uv1 = in.uv01.zw;
    let uv2 = in.uv2_pad.xy;
    out.tex_uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;

    // Compute per-face tangent frame from position and UV edge vectors.
    // This avoids needing tangent vertex attributes — works for all models.
    let v0 = in.p0.yzw; let v1 = in.p1.yzw; let v2 = in.p2.yzw;
    let edge01 = v1 - v0;
    let edge02 = v2 - v0;
    let duv01 = uv1 - uv0;
    let duv02 = uv2 - uv0;
    let det = duv01.x * duv02.y - duv01.y * duv02.x;
    var tangent = vec3<f32>(1.0, 0.0, 0.0);
    var bitangent = vec3<f32>(0.0, 1.0, 0.0);
    if abs(det) > 1e-6 {
        let inv_det = 1.0 / det;
        tangent = normalize((edge01 * duv02.y - edge02 * duv01.y) * inv_det);
        bitangent = normalize((edge02 * duv01.x - edge01 * duv02.x) * inv_det);
    }
    out.tangent_vs = normalize((u.mv * vec4<f32>(tangent, 0.0)).xyz);
    out.bitangent_vs = normalize((u.mv * vec4<f32>(bitangent, 0.0)).xyz);

    out.position_vs = (u.mv * vec4<f32>(pos, 1.0)).xyz;
    out.clip_pos = u.mvp * vec4<f32>(pos, 1.0);
    return out;
}
