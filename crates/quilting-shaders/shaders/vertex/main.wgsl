// Main vertex shader for QB surface rendering.
// Imports quilting modules for quaternion math, surface evaluation, and density viz.
// Compiles to GLSL ES 300 via naga for WebGL2.

#import quilting::math::quaternion::{qmul, qconj, qinv, q_to_point}
#import quilting::surface::qb_eval::eval_qb

struct Uniforms {
    mvp: mat4x4<f32>,
    mv: mat4x4<f32>,
    _reserved_winding_parity: f32,
    _reserved_perm_index: i32,
    use_qb: i32,
    _reserved_face_offset: f32,
    // Möbius transform: p' = (a*p + b) * (c*p + d)^{-1}
    mob_a: vec4<f32>,
    mob_b: vec4<f32>,
    mob_c: vec4<f32>,
    mob_d: vec4<f32>,
    camera_pos: vec4<f32>,   // world-space camera position (xyz, w unused)
    model: mat4x4<f32>,      // ordinary affine transform before Möbius
    normal_model: mat4x4<f32>,
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

// Immutable source-face records, packed as thirteen RGBA32F texels per face in
// the normative 52-float instance layout. Animated LOD updates stream only a
// topology record containing edge LODs, permutation, and source face ID.
@group(0) @binding(4)
var face_data_tex: texture_2d<f32>;

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
    var applied_weight = 0.0;
    let p4 = vec4<f32>(pos, 1.0);

    for (var k = 0u; k < 4u; k = k + 1u) {
        let w = jw[k];
        if w < 1e-6 { continue; }
        let idx = i32(ji[k]);
        if idx >= joints.num_joints { continue; }
        let m = joints.matrices[idx];
        applied_weight = applied_weight + w;
        skinned = skinned + w * (m * p4).xyz;
    }
    if applied_weight <= 1e-6 { return pos; }
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
    // Per-instance QB control data. lod_info.w carries the S3 permutation index
    // so all six orientations share one canonical tessellation buffer.
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
    // Posed ordinary-space point before the Möbius map. The focus sphere is
    // classified here so it is the same geometric sphere inversion consumes.
    @location(13) source_position_ws: vec3<f32>,
    /// Stable semantic glTF node. Kept flat so consolidated draw buckets can
    /// still support exact object selection without interpolating identities.
    @location(14) @interpolate(flat) node_id: f32,
}

// Backend-neutral prepared-patch record. Locations 0..12 are thirteen
// contiguous vec4s, exactly matching quilting_core::instance_layout's
// 52-float stride.
// WebGL2 writes it with transform feedback; WebGPU can produce the same logical
// record from a compute pass without changing render semantics.
struct PatchPrepareInput {
    @location(7) lod_info: vec4<f32>,
    @location(8) face_info: vec4<f32>,
}

struct PreparedPatchOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) p0: vec4<f32>,
    @location(1) p1: vec4<f32>,
    @location(2) p2: vec4<f32>,
    @location(3) w0: vec4<f32>,
    @location(4) w1: vec4<f32>,
    @location(5) w2: vec4<f32>,
    @location(6) lod_info: vec4<f32>,
    @location(7) vert_lod: vec4<f32>,
    @location(8) uv01: vec4<f32>,
    @location(9) uv2_prepare: vec4<f32>,
    @location(10) smooth_n0: vec4<f32>,
    @location(11) smooth_n1: vec4<f32>,
    @location(12) smooth_n2: vec4<f32>,
}

fn culled_vertex_output() -> VertexOutput {
    return VertexOutput(
        vec4<f32>(2.0, 2.0, 2.0, 1.0),
        vec3<f32>(0.0),
        0.0,
        vec2<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        0.0,
        vec3<f32>(0.0),
        0.0,
        0.5,
        vec3<f32>(0.0),
        0.0,
    );
}

fn finite_vec3(value: vec3<f32>) -> bool {
    // Comparisons with NaN are false; infinities exceed the finite guard.
    return all(abs(value) < vec3<f32>(1e30));
}

fn finite_vec4(value: vec4<f32>) -> bool {
    return all(abs(value) < vec4<f32>(1e30));
}

fn mobius_point(p: vec3<f32>) -> vec3<f32> {
    let q = vec4<f32>(0.0, p);
    return qmul(qmul(u.mob_a, q) + u.mob_b, qinv(qmul(u.mob_c, q) + u.mob_d)).yzw;
}

// Return true only when a world-space ball lies wholly outside one homogeneous
// clip plane. Invalid bounds survive, preserving conservative visibility.
fn sphere_outside_frustum(center: vec3<f32>, radius: f32) -> bool {
    if !finite_vec3(center) || !(radius >= 0.0 && radius < 1e30) {
        return false;
    }
    let clip = u.mvp * vec4<f32>(center, 1.0);
    let r0 = vec3<f32>(u.mvp[0][0], u.mvp[1][0], u.mvp[2][0]);
    let r1 = vec3<f32>(u.mvp[0][1], u.mvp[1][1], u.mvp[2][1]);
    let r2 = vec3<f32>(u.mvp[0][2], u.mvp[1][2], u.mvp[2][2]);
    let r3 = vec3<f32>(u.mvp[0][3], u.mvp[1][3], u.mvp[2][3]);
    return clip.w + clip.x < -radius * length(r3 + r0)
        || clip.w - clip.x < -radius * length(r3 - r0)
        || clip.w + clip.y < -radius * length(r3 + r1)
        || clip.w - clip.y < -radius * length(r3 - r1)
        || clip.w + clip.z < -radius * length(r3 + r2)
        || clip.w - clip.z < -radius * length(r3 - r2);
}

// Conservative bound on the complete Möbius image of a flat source patch. A
// source ball maps to a ball when the pole is outside it. Pole-containing,
// singular, and non-finite cases deliberately survive.
fn flat_patch_outside_frustum(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> bool {
    let center = (p0 + p1 + p2) / 3.0;
    let source_radius = sqrt(max(
        dot(p0 - center, p0 - center),
        max(dot(p1 - center, p1 - center), dot(p2 - center, p2 - center)),
    ));
    var direction = vec3<f32>(1.0, 0.0, 0.0);
    let c_norm_sq = dot(u.mob_c, u.mob_c);
    if c_norm_sq > 1e-20 {
        let pole = -qmul(qinv(u.mob_c), u.mob_d);
        let delta = center - pole.yzw;
        let pole_distance_sq = pole.x * pole.x + dot(delta, delta);
        let guarded_radius = 1.05 * source_radius;
        if pole_distance_sq <= guarded_radius * guarded_radius {
            return false;
        }
        let delta_len = length(delta);
        if delta_len <= 1e-20 {
            return false;
        }
        direction = delta / delta_len;
    }

    let plus = mobius_point(center + source_radius * direction);
    let minus = mobius_point(center - source_radius * direction);
    if !finite_vec3(plus) || !finite_vec3(minus) {
        return false;
    }
    let image_center = 0.5 * (plus + minus);
    let image_radius = 0.5 * distance(plus, minus) * 1.05 + 1e-6;
    return sphere_outside_frustum(image_center, image_radius);
}

// Exact Euclidean distance from the origin to a triangle embedded in R4.
// Checking its vertices, edges, and (when non-degenerate) interior gives the
// minimum over the closed barycentric simplex without assuming 3D geometry.
fn origin_to_quaternion_triangle(a: vec4<f32>, b: vec4<f32>, c: vec4<f32>) -> f32 {
    var distance_sq = min(dot(a, a), min(dot(b, b), dot(c, c)));

    let ab = b - a;
    let ab_len_sq = dot(ab, ab);
    if ab_len_sq > 1e-20 {
        let t = clamp(-dot(a, ab) / ab_len_sq, 0.0, 1.0);
        let closest = a + t * ab;
        distance_sq = min(distance_sq, dot(closest, closest));
    }

    let ac = c - a;
    let ac_len_sq = dot(ac, ac);
    if ac_len_sq > 1e-20 {
        let t = clamp(-dot(a, ac) / ac_len_sq, 0.0, 1.0);
        let closest = a + t * ac;
        distance_sq = min(distance_sq, dot(closest, closest));
    }

    let bc = c - b;
    let bc_len_sq = dot(bc, bc);
    if bc_len_sq > 1e-20 {
        let t = clamp(-dot(b, bc) / bc_len_sq, 0.0, 1.0);
        let closest = b + t * bc;
        distance_sq = min(distance_sq, dot(closest, closest));
    }

    let g00 = ab_len_sq;
    let g01 = dot(ab, ac);
    let g11 = ac_len_sq;
    let determinant = g00 * g11 - g01 * g01;
    if determinant > 1e-20 {
        let rhs0 = -dot(a, ab);
        let rhs1 = -dot(a, ac);
        let s = (rhs0 * g11 - g01 * rhs1) / determinant;
        let t = (g00 * rhs1 - g01 * rhs0) / determinant;
        if s >= 0.0 && t >= 0.0 && s + t <= 1.0 {
            let closest = a + s * ab + t * ac;
            distance_sq = min(distance_sq, dot(closest, closest));
        }
    }
    return sqrt(max(distance_sq, 0.0));
}

// Conservative bound for a genuine rational QB patch after the current
// Möbius transform. For barycentric lambda, the fused evaluator is N/D where
// N and D each range over a quaternion triangle. Triangle inequality bounds
// |N| by the largest numerator control norm; the exact distance from the
// denominator triangle to zero lower-bounds |D|. Therefore the whole patch is
// inside an origin-centred ball of radius max|N_i|/min|D|. A denominator
// triangle touching zero is a possible pole and deliberately remains visible.
fn rational_patch_outside_frustum(
    p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>,
    w0: vec4<f32>, w1: vec4<f32>, w2: vec4<f32>,
) -> bool {
    let qp0 = vec4<f32>(0.0, p0);
    let qp1 = vec4<f32>(0.0, p1);
    let qp2 = vec4<f32>(0.0, p2);
    let numerator0 = qmul(qmul(u.mob_a, qp0) + u.mob_b, w0);
    let numerator1 = qmul(qmul(u.mob_a, qp1) + u.mob_b, w1);
    let numerator2 = qmul(qmul(u.mob_a, qp2) + u.mob_b, w2);
    let denominator0 = qmul(qmul(u.mob_c, qp0) + u.mob_d, w0);
    let denominator1 = qmul(qmul(u.mob_c, qp1) + u.mob_d, w1);
    let denominator2 = qmul(qmul(u.mob_c, qp2) + u.mob_d, w2);
    if !finite_vec4(numerator0) || !finite_vec4(numerator1) || !finite_vec4(numerator2)
        || !finite_vec4(denominator0) || !finite_vec4(denominator1)
        || !finite_vec4(denominator2) {
        return false;
    }
    let denominator_distance = origin_to_quaternion_triangle(
        denominator0, denominator1, denominator2,
    );
    if !(denominator_distance > 1e-8) {
        return false;
    }
    let numerator_radius = max(
        length(numerator0),
        max(length(numerator1), length(numerator2)),
    );
    // Rendering clamps evaluated positions to POSITION_CLAMP, so the smaller
    // of the analytic radius and that clamp remains conservative.
    let radius = min(POSITION_CLAMP, numerator_radius / denominator_distance * 1.01 + 1e-6);
    return sphere_outside_frustum(vec3<f32>(0.0), radius);
}

fn patch_outside_frustum(
    p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>,
    w0: vec4<f32>, w1: vec4<f32>, w2: vec4<f32>,
) -> bool {
    let common_weight = length(w0) > 1e-10
        && all(w1 == w0)
        && all(w2 == w0);
    if u.use_qb != 1 || common_weight {
        return flat_patch_outside_frustum(p0, p1, p2);
    }
    return rational_patch_outside_frustum(p0, p1, p2, w0, w1, w2);
}

fn posed_position(control: vec4<f32>) -> vec3<f32> {
    let vertex_index = i32(control.x);
    var position = control.yzw;
    if joints.num_joints > 0 || joints.num_morph_targets > 0 {
        position = apply_morph(position, vertex_index);
        position = skin_position(position, vertex_index);
    }
    return (u.model * vec4<f32>(position, 1.0)).xyz;
}

fn face_data_load(face_index: i32, slot: i32) -> vec4<f32> {
    let dimensions = textureDimensions(face_data_tex, 0);
    let width = i32(dimensions.x);
    let texel = face_index * 13 + slot;
    return textureLoad(face_data_tex, vec2<i32>(texel % width, texel / width), 0);
}

@vertex
fn prepare_patches(in: PatchPrepareInput) -> PreparedPatchOutput {
    let face_index = max(i32(round(in.face_info.x)), 0);
    let source_p0 = face_data_load(face_index, 0);
    let source_p1 = face_data_load(face_index, 1);
    let source_p2 = face_data_load(face_index, 2);
    let p0 = vec4<f32>(source_p0.x, posed_position(source_p0));
    let p1 = vec4<f32>(source_p1.x, posed_position(source_p1));
    let p2 = vec4<f32>(source_p2.x, posed_position(source_p2));
    let source_w0 = face_data_load(face_index, 3);
    let source_w1 = face_data_load(face_index, 4);
    let source_w2 = face_data_load(face_index, 5);
    let visible = !patch_outside_frustum(
        p0.yzw, p1.yzw, p2.yzw,
        source_w0, source_w1, source_w2,
    );
    let source_uv2 = face_data_load(face_index, 9);
    let uv2_prepare = vec4<f32>(
        source_uv2.xy,
        select(0.0, 1.0, visible),
        1.0,
    );
    let vert_lod = vec4<f32>(in.face_info.yzw, f32(face_index));
    return PreparedPatchOutput(
        vec4<f32>(0.0),
        p0, p1, p2,
        source_w0, source_w1, source_w2,
        in.lod_info,
        vert_lod,
        face_data_load(face_index, 8),
        uv2_prepare,
        face_data_load(face_index, 10),
        face_data_load(face_index, 11),
        face_data_load(face_index, 12),
    );
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
    return MobiusQBResult(q_to_point(X), n, fade);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // The atlas stores only canonical tessellations. Map their barycentrics back
    // to this face using the permutation packed beside its edge LODs.
    let perm_index = clamp(i32(round(in.lod_info.w)), 0, 5);
    let bary = perm_bary(in.bary, perm_index);

    var pos: vec3<f32>;
    var nrm: vec3<f32>;
    out.fade = 1.0;

    let prepared = in.uv2_pad.w > 0.5;
    if prepared && in.uv2_pad.z < 0.5 {
        return culled_vertex_output();
    }

    // Prepared records already contain posed, affine-transformed control
    // points. The fallback path preserves rendering when preparation is absent.
    var sp0: vec4<f32>;
    var sp1: vec4<f32>;
    var sp2: vec4<f32>;
    if prepared {
        sp0 = vec4<f32>(0.0, in.p0.yzw);
        sp1 = vec4<f32>(0.0, in.p1.yzw);
        sp2 = vec4<f32>(0.0, in.p2.yzw);
    } else {
        sp0 = vec4<f32>(0.0, posed_position(in.p0));
        sp1 = vec4<f32>(0.0, posed_position(in.p1));
        sp2 = vec4<f32>(0.0, posed_position(in.p2));
        if patch_outside_frustum(
            sp0.yzw, sp1.yzw, sp2.yzw,
            in.w0, in.w1, in.w2,
        ) {
            return culled_vertex_output();
        }
    }

    if u.use_qb == 1 {
        out.source_position_ws = eval_qb(bary, sp0, sp1, sp2, in.w0, in.w1, in.w2);
    } else {
        out.source_position_ws = eval_flat(bary, sp0, sp1, sp2);
    }

    if u.use_qb == 1 {
        let result = eval_mobius_qb(
            bary,
            sp0, sp1, sp2,
            in.w0, in.w1, in.w2,
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
    sn0 = (u.normal_model * vec4<f32>(sn0, 0.0)).xyz;
    sn1 = (u.normal_model * vec4<f32>(sn1, 0.0)).xyz;
    sn2 = (u.normal_model * vec4<f32>(sn2, 0.0)).xyz;
    let has_smooth = dot(sn0, sn0) + dot(sn1, sn1) + dot(sn2, sn2) > 0.01;
    if has_smooth {
        // Transform every normal through the Möbius differential. The c=0
        // branch still contains rotations and negative scales; treating it as
        // "no transform" gives visibly incorrect lighting and parity.
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
    }

    out.normal_vs = normalize((u.mv * vec4<f32>(nrm, 0.0)).xyz);

    // Current resident per-vertex LODs are max-reconciled over the welded
    // topology. Log interpolation is therefore smooth inside a face and C0
    // continuous across every shared edge, independent of atlas permutation.
    let log_vertex_lod = log2(max(in.vert_lod.xyz, vec3<f32>(1.0)));
    out.density = dot(bary, log_vertex_lod) / 10.0;

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
    out.instance_id = in.vert_lod.w;
    out.node_id = in.smooth_n0.w;

    // Möbius conformal scale is |a - F(p)c| / |cp + d|. The historical
    // 1/|cp+d|² shortcut only holds for normalized inversive generators and
    // incorrectly reports every c=0 scale as neutral.
    let mb0 = qmul(u.mob_c, sp0) + u.mob_d;
    let mb1 = qmul(u.mob_c, sp1) + u.mob_d;
    let mb2 = qmul(u.mob_c, sp2) + u.mob_d;
    let mm0 = qmul(qmul(u.mob_a, sp0) + u.mob_b, qinv(mb0));
    let mm1 = qmul(qmul(u.mob_a, sp1) + u.mob_b, qinv(mb1));
    let mm2 = qmul(qmul(u.mob_a, sp2) + u.mob_b, qinv(mb2));
    let ma0 = u.mob_a - qmul(mm0, u.mob_c);
    let ma1 = u.mob_a - qmul(mm1, u.mob_c);
    let ma2 = u.mob_a - qmul(mm2, u.mob_c);
    let s0 = sqrt(max(dot(ma0, ma0), 1e-20) / max(dot(mb0, mb0), 1e-20));
    let s1 = sqrt(max(dot(ma1, ma1), 1e-20) / max(dot(mb1, mb1), 1e-20));
    let s2 = sqrt(max(dot(ma2, ma2), 1e-20) / max(dot(mb2, mb2), 1e-20));
    let stretch = bary.x * s0 + bary.y * s1 + bary.z * s2;
    // Signed log2, mapped to [0,1] via sigmoid for smooth falloff (no hard cutoff).
    // 0.5 = no stretch, 0 = max squash, 1 = max expand.
    let log_s = log2(max(stretch, 1e-20));
    // Stretch is a diagnostic view, so make a one-octave change plainly
    // visible while retaining a smooth, symmetric response around scale 1.
    out.mobius_stretch = 1.0 / (1.0 + exp(-log_s));

    return out;
}
