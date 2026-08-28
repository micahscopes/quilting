// Main vertex shader for QB surface rendering.
// Imports quilting modules for quaternion math, surface evaluation, and density viz.
// Compiles to GLSL ES 300 via naga for WebGL2.

#import quilting::math::quaternion::{qmul, qconj, qinv}
#import quilting::surface::patch_prepare::{PreparedPatchRecord, PosedPatchControls, prepare_patch_record}
#import quilting::surface::patch_render::{PatchRenderTransform, PatchSurfaceInput, POSITION_CLAMP, evaluate_patch_surface}

struct Uniforms {
    mvp: mat4x4<f32>,
    mv: mat4x4<f32>,
    _reserved_winding_parity: f32,
    suppress_source_roots: i32,
    use_qb: i32,
    _reserved_face_offset: f32,
    mob_a: vec4<f32>,
    mob_b: vec4<f32>,
    mob_c: vec4<f32>,
    mob_d: vec4<f32>,
    camera_pos: vec4<f32>,
    model: mat4x4<f32>,
    normal_model: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> u: Uniforms;

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

// Sparse renderer-owned mask for baseline source roots replaced by an
// adaptive overlay. Overlay batches disable the mask, so an affected face can
// still publish a root leaf when only its topology or corner density changed.
@group(0) @binding(5)
var suppressed_face_tex: texture_2d<f32>;

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
    // Separate camera-dependent stream. Unprepared/fallback VAOs leave this
    // attribute disabled at its WebGL default; it is consulted only when the
    // 52-float record carries the prepared flag.
    @location(14) prepared_visibility: f32,
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
    @location(1) leaf_meta: vec2<f32>,
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

// Camera-dependent classification consumes already-posed controls and writes
// one float per patch. Keeping it separate prevents a view change from
// rewriting the complete 52-float prepared record.
struct PatchVisibilityInput {
    @location(1) p0: vec4<f32>,
    @location(2) p1: vec4<f32>,
    @location(3) p2: vec4<f32>,
    @location(4) w0: vec4<f32>,
    @location(5) w1: vec4<f32>,
    @location(6) w2: vec4<f32>,
    @location(8) face_info: vec4<f32>,
}

struct PatchVisibilityOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) visibility: f32,
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
// Möbius transform. For barycentric lambda, the fused evaluator is q = N/D,
// where N and D each range over a quaternion triangle. For any quaternion c,
// q-c = (N-cD)D^-1. Choosing c at the barycentric center cancels the average
// residual, so translated patches retain local rather than origin-centered
// bounds. The exact distance from the denominator triangle to zero lower-
// bounds |D|. A denominator triangle touching zero is a possible pole and
// deliberately remains visible.
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
    let denominator_sum = denominator0 + denominator1 + denominator2;
    let denominator_sum_norm_sq = dot(denominator_sum, denominator_sum);
    if !(denominator_sum_norm_sq > 1e-16) {
        return false;
    }
    let numerator_sum = numerator0 + numerator1 + numerator2;
    let center_quaternion = qmul(numerator_sum, qconj(denominator_sum))
        / denominator_sum_norm_sq;
    if !finite_vec4(center_quaternion) {
        return false;
    }
    let residual_radius = max(
        length(numerator0 - qmul(center_quaternion, denominator0)),
        max(
            length(numerator1 - qmul(center_quaternion, denominator1)),
            length(numerator2 - qmul(center_quaternion, denominator2)),
        ),
    );
    let radius = residual_radius / denominator_distance * 1.01 + 1e-6;
    let center = center_quaternion.yzw;
    if !finite_vec3(center) || !(radius >= 0.0 && radius < 1e30) {
        return false;
    }

    // eval_mobius_qb later clamps positions radially into the origin-centered
    // POSITION_CLAMP ball. That operation cannot change this local bound only
    // when the complete analytic ball is already inside the clamp ball. For a
    // farther patch, retain the clamp ball itself rather than incorrectly
    // taking the smaller of two spheres with different centers.
    if length(center) + radius <= POSITION_CLAMP {
        return sphere_outside_frustum(center, radius);
    }
    return sphere_outside_frustum(vec3<f32>(0.0), POSITION_CLAMP);
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

fn posed_normal(normal: vec3<f32>, vertex_index: i32) -> vec3<f32> {
    var result = normal;
    if joints.num_joints > 0 {
        result = skin_normal(result, vertex_index);
    }
    return (u.normal_model * vec4<f32>(result, 0.0)).xyz;
}

@vertex
fn prepare_patches(in: PatchPrepareInput) -> PreparedPatchOutput {
    let face_index = max(i32(round(in.face_info.x)), 0);
    // Slots 6 and 7 are supplied by the topology stream, so there is no
    // reason to fetch their immutable source values from the face texture.
    let source = PreparedPatchRecord(
        face_data_load(face_index, 0),
        face_data_load(face_index, 1),
        face_data_load(face_index, 2),
        face_data_load(face_index, 3),
        face_data_load(face_index, 4),
        face_data_load(face_index, 5),
        vec4<f32>(0.0),
        vec4<f32>(0.0),
        face_data_load(face_index, 8),
        face_data_load(face_index, 9),
        face_data_load(face_index, 10),
        face_data_load(face_index, 11),
        face_data_load(face_index, 12),
    );
    let posed_p0 = posed_position(source.record_position_a);
    let posed_p1 = posed_position(source.record_position_b);
    let posed_p2 = posed_position(source.record_position_c);
    // Pose all prepared normals once per control. Adaptive restriction then
    // interpolates this posed field; root records also avoid repeating the
    // same work for every tessellation vertex during the draw pass.
    let posed_n0 = posed_normal(source.record_normal_a.xyz, i32(source.record_position_a.x));
    let posed_n1 = posed_normal(source.record_normal_b.xyz, i32(source.record_position_b.x));
    let posed_n2 = posed_normal(source.record_normal_c.xyz, i32(source.record_position_c.x));
    let prepared = prepare_patch_record(
        source,
        in.lod_info,
        in.face_info,
        in.leaf_meta,
        PosedPatchControls(
            posed_p0, posed_p1, posed_p2,
            posed_n0, posed_n1, posed_n2,
        ),
    );
    return PreparedPatchOutput(
        vec4<f32>(0.0),
        prepared.record_position_a,
        prepared.record_position_b,
        prepared.record_position_c,
        prepared.record_weight_a,
        prepared.record_weight_b,
        prepared.record_weight_c,
        prepared.record_lod_info,
        prepared.record_vertex_lod,
        prepared.record_uv_ab,
        prepared.record_uv_c_prepare,
        prepared.record_normal_a,
        prepared.record_normal_b,
        prepared.record_normal_c,
    );
}

@vertex
fn classify_patch_visibility(in: PatchVisibilityInput) -> PatchVisibilityOutput {
    var visible = !patch_outside_frustum(
        in.p0.yzw, in.p1.yzw, in.p2.yzw,
        in.w0, in.w1, in.w2,
    );
    if visible && u.suppress_source_roots != 0 && in.p0.x >= -0.5 {
        let face_id = max(i32(round(in.face_info.w)), 0);
        let mask_size = textureDimensions(suppressed_face_tex);
        let mask_width = max(i32(mask_size.x), 1);
        let mask_coord = vec2<i32>(face_id % mask_width, face_id / mask_width);
        visible = textureLoad(suppressed_face_tex, mask_coord, 0).x < 0.5;
    }
    return PatchVisibilityOutput(
        vec4<f32>(0.0),
        select(0.0, 1.0, visible),
    );
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let prepared = in.uv2_pad.w > 0.5;
    if prepared && in.prepared_visibility < 0.5 {
        return culled_vertex_output();
    }

    var control_a = in.p0;
    var control_b = in.p1;
    var control_c = in.p2;
    var normal_a = in.smooth_n0;
    var normal_b = in.smooth_n1;
    var normal_c = in.smooth_n2;
    if !prepared {
        control_a = vec4<f32>(in.p0.x, posed_position(in.p0));
        control_b = vec4<f32>(in.p1.x, posed_position(in.p1));
        control_c = vec4<f32>(in.p2.x, posed_position(in.p2));
        if patch_outside_frustum(
            control_a.yzw, control_b.yzw, control_c.yzw,
            in.w0, in.w1, in.w2,
        ) {
            return culled_vertex_output();
        }
        normal_a = vec4<f32>(
            posed_normal(in.smooth_n0.xyz, i32(in.p0.x)),
            in.smooth_n0.w,
        );
        normal_b = vec4<f32>(
            posed_normal(in.smooth_n1.xyz, i32(in.p1.x)),
            in.smooth_n1.w,
        );
        normal_c = vec4<f32>(
            posed_normal(in.smooth_n2.xyz, i32(in.p2.x)),
            in.smooth_n2.w,
        );
    }

    let surface = evaluate_patch_surface(
        PatchRenderTransform(
            u.mvp,
            u.mv,
            u.use_qb,
            u.mob_a,
            u.mob_b,
            u.mob_c,
            u.mob_d,
            u.camera_pos,
        ),
        PatchSurfaceInput(
            in.bary,
            control_a, control_b, control_c,
            in.w0, in.w1, in.w2,
            in.lod_info,
            in.vert_lod,
            in.uv01,
            in.uv2_pad,
            normal_a, normal_b, normal_c,
            select(0u, 1u, prepared),
        ),
    );
    return VertexOutput(
        surface.clip_pos,
        surface.normal_vs,
        surface.density,
        surface.tex_uv,
        surface.position_vs,
        surface.tangent_vs,
        surface.bitangent_vs,
        surface.normal_ws,
        surface.position_ws,
        surface.camera_pos_ws,
        surface.fade,
        surface.tess_bary,
        surface.instance_id,
        surface.mobius_stretch,
        surface.source_position_ws,
        surface.node_id,
    );
}
