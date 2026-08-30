// Main vertex shader for QB surface rendering.
// Imports quilting modules for quaternion math, surface evaluation, and density viz.
// Compiles to GLSL ES 300 via naga for WebGL2.

#import quilting::math::quaternion::{qmul, qinv}
#import quilting::surface::patch_prepare::{PreparedPatchRecord, PosedPatchControls, prepare_patch_record}
#import quilting::surface::patch_render::{PatchRenderTransform, PatchSurfaceInput, evaluate_patch_surface}
#import quilting::surface::patch_visibility::prepared_patch_outside_frustum

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

// Portable graphics namespace: the legacy combined view/entity draw packet is
// updated per batch and therefore occupies the entity/batch group.
@group(1) @binding(0)
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

// Pose and immutable source data occupy group 0.
@group(0) @binding(0)
var<uniform> joints: JointMatrices;

// Per-vertex skinning data texture: width = num_verts, height = 2
// Row 0: joint indices (as f32), Row 1: joint weights
@group(0) @binding(1)
var skinning_tex: texture_2d<f32>;

// Morph target deltas texture: width = num_verts, height = num_targets
// Each texel = (dx, dy, dz, 0) position delta for that vertex+target
@group(0) @binding(2)
var morph_tex: texture_2d<f32>;

// Immutable source-face records, packed as thirteen RGBA32F texels per face in
// the normative 52-float instance layout. Animated LOD updates stream only a
// topology record containing edge LODs, permutation, and source face ID.
@group(0) @binding(3)
var face_data_tex: texture_2d<f32>;

// Sparse renderer-owned mask for baseline source roots replaced by an
// adaptive overlay. Overlay batches disable the mask, so an affected face can
// still publish a root leaf when only its topology or corner density changed.
@group(0) @binding(4)
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

fn visibility_transform() -> PatchRenderTransform {
    return PatchRenderTransform(
        u.mvp,
        u.mv,
        u.use_qb,
        u.mob_a,
        u.mob_b,
        u.mob_c,
        u.mob_d,
        u.camera_pos,
    );
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
    var visible = !prepared_patch_outside_frustum(
        visibility_transform(),
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
        if prepared_patch_outside_frustum(
            visibility_transform(),
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
