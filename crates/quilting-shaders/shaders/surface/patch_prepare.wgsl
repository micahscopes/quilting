#define_import_path quilting::surface::patch_prepare

#import quilting::math::quaternion::{qmul, qinv, q_to_point}

// Canonical 52-float patch record shared by immutable source faces and posed
// render instances. Thirteen contiguous vec4s give this struct a 208-byte
// storage stride, exactly matching quilting_core::instance_layout::STRIDE.
// Camera-dependent visibility deliberately remains a separate stream.
struct PreparedPatchRecord {
    record_position_a: vec4<f32>,
    record_position_b: vec4<f32>,
    record_position_c: vec4<f32>,
    record_weight_a: vec4<f32>,
    record_weight_b: vec4<f32>,
    record_weight_c: vec4<f32>,
    record_lod_info: vec4<f32>,
    record_vertex_lod: vec4<f32>,
    record_uv_ab: vec4<f32>,
    record_uv_c_prepare: vec4<f32>,
    record_normal_a: vec4<f32>,
    record_normal_b: vec4<f32>,
    record_normal_c: vec4<f32>,
}

// Current-pose values are kept separate from the immutable source record.
// Every prepared record receives already-posed normals, avoiding repeated
// skinning and affine normal transforms at each tessellated render vertex.
struct PosedPatchControls {
    pose_position_a: vec3<f32>,
    pose_position_b: vec3<f32>,
    pose_position_c: vec3<f32>,
    pose_normal_a: vec3<f32>,
    pose_normal_b: vec3<f32>,
    pose_normal_c: vec3<f32>,
}

struct PatchDomain {
    domain_corner_a: vec3<f32>,
    domain_corner_b: vec3<f32>,
    domain_corner_c: vec3<f32>,
}

fn patch_leaf_depth(leaf_meta: vec2<f32>) -> i32 {
    return clamp(i32(round(leaf_meta.x)), 0, 12);
}

// Reconstruct the exact dyadic source-barycentric domain carried by the
// compact topology stream. Two child bits are appended per level; depth 12 is
// the largest path exactly representable by the WebGL2 stream's f32 lane.
fn dyadic_leaf_domain(leaf_meta: vec2<f32>) -> PatchDomain {
    let depth = patch_leaf_depth(leaf_meta);
    let path = u32(max(round(leaf_meta.y), 0.0));
    var c0 = vec3<f32>(1.0, 0.0, 0.0);
    var c1 = vec3<f32>(0.0, 1.0, 0.0);
    var c2 = vec3<f32>(0.0, 0.0, 1.0);
    for (var level = 0; level < 12; level = level + 1) {
        if level >= depth { break; }
        let shift = u32(2 * (depth - level - 1));
        let child = (path >> shift) & 3u;
        let c01 = 0.5 * (c0 + c1);
        let c02 = 0.5 * (c0 + c2);
        let c12 = 0.5 * (c1 + c2);
        if child == 0u {
            c1 = c01;
            c2 = c02;
        } else if child == 1u {
            c0 = c01;
            c2 = c12;
        } else if child == 2u {
            c0 = c02;
            c1 = c12;
        } else {
            c0 = c01;
            c1 = c12;
            c2 = c02;
        }
    }
    return PatchDomain(c0, c1, c2);
}

struct RestrictedControl {
    restricted_control_position: vec3<f32>,
    restricted_control_weight: vec4<f32>,
}

// Restrict numerator and denominator homogeneously. This is the degree-one
// rational QB restriction, so evaluating the child reproduces the source
// patch exactly rather than sampling a polynomial approximation.
fn restrict_control(
    source_bary: vec3<f32>,
    control_p0: vec3<f32>, control_p1: vec3<f32>, control_p2: vec3<f32>,
    control_w0: vec4<f32>, control_w1: vec4<f32>, control_w2: vec4<f32>,
) -> RestrictedControl {
    let numerator = source_bary.x * qmul(vec4<f32>(0.0, control_p0), control_w0)
        + source_bary.y * qmul(vec4<f32>(0.0, control_p1), control_w1)
        + source_bary.z * qmul(vec4<f32>(0.0, control_p2), control_w2);
    let weight = source_bary.x * control_w0
        + source_bary.y * control_w1
        + source_bary.z * control_w2;
    return RestrictedControl(q_to_point(qmul(numerator, qinv(weight))), weight);
}

// Pure assembly shared by the WebGL2 transform-feedback entry and WebGPU
// compute. Pose loading is backend-specific; topology interpretation,
// rational restriction, UV/normal restriction, adaptive tags, and the exact
// prepared-record field order are not.
fn prepare_patch_record(
    source: PreparedPatchRecord,
    lod_info: vec4<f32>,
    face_info: vec4<f32>,
    leaf_meta: vec2<f32>,
    posed: PosedPatchControls,
) -> PreparedPatchRecord {
    var result = source;
    result.record_position_a = vec4<f32>(source.record_position_a.x, posed.pose_position_a);
    result.record_position_b = vec4<f32>(source.record_position_b.x, posed.pose_position_b);
    result.record_position_c = vec4<f32>(source.record_position_c.x, posed.pose_position_c);
    result.record_lod_info = lod_info;
    result.record_vertex_lod = vec4<f32>(face_info.yzw, face_info.x);
    result.record_uv_c_prepare = vec4<f32>(source.record_uv_c_prepare.xy, 0.0, 1.0);
    result.record_normal_a = vec4<f32>(posed.pose_normal_a, source.record_normal_a.w);
    result.record_normal_b = vec4<f32>(posed.pose_normal_b, source.record_normal_b.w);
    result.record_normal_c = vec4<f32>(posed.pose_normal_c, source.record_normal_c.w);

    let leaf_depth = patch_leaf_depth(leaf_meta);
    if leaf_depth > 0 {
        let domain = dyadic_leaf_domain(leaf_meta);
        let r0 = restrict_control(
            domain.domain_corner_a,
            posed.pose_position_a, posed.pose_position_b, posed.pose_position_c,
            source.record_weight_a, source.record_weight_b, source.record_weight_c,
        );
        let r1 = restrict_control(
            domain.domain_corner_b,
            posed.pose_position_a, posed.pose_position_b, posed.pose_position_c,
            source.record_weight_a, source.record_weight_b, source.record_weight_c,
        );
        let r2 = restrict_control(
            domain.domain_corner_c,
            posed.pose_position_a, posed.pose_position_b, posed.pose_position_c,
            source.record_weight_a, source.record_weight_b, source.record_weight_c,
        );
        // A negative p0 tag distinguishes adaptive records without growing
        // the ABI. p1.x retains the dyadic path so the render pass can recover
        // source-face barycentrics for picking.
        result.record_position_a = vec4<f32>(
            -f32(leaf_depth + 1), r0.restricted_control_position,
        );
        result.record_position_b = vec4<f32>(
            leaf_meta.y, r1.restricted_control_position,
        );
        result.record_position_c = vec4<f32>(0.0, r2.restricted_control_position);
        result.record_weight_a = r0.restricted_control_weight;
        result.record_weight_b = r1.restricted_control_weight;
        result.record_weight_c = r2.restricted_control_weight;

        let source_uv0 = source.record_uv_ab.xy;
        let source_uv1 = source.record_uv_ab.zw;
        let source_uv2 = source.record_uv_c_prepare.xy;
        let child_uv0 = domain.domain_corner_a.x * source_uv0
            + domain.domain_corner_a.y * source_uv1
            + domain.domain_corner_a.z * source_uv2;
        let child_uv1 = domain.domain_corner_b.x * source_uv0
            + domain.domain_corner_b.y * source_uv1
            + domain.domain_corner_b.z * source_uv2;
        let child_uv2 = domain.domain_corner_c.x * source_uv0
            + domain.domain_corner_c.y * source_uv1
            + domain.domain_corner_c.z * source_uv2;
        result.record_uv_ab = vec4<f32>(child_uv0, child_uv1);
        result.record_uv_c_prepare = vec4<f32>(child_uv2, 0.0, 1.0);

        result.record_normal_a = vec4<f32>(
            domain.domain_corner_a.x * posed.pose_normal_a
                + domain.domain_corner_a.y * posed.pose_normal_b
                + domain.domain_corner_a.z * posed.pose_normal_c,
            source.record_normal_a.w,
        );
        result.record_normal_b = vec4<f32>(
            domain.domain_corner_b.x * posed.pose_normal_a
                + domain.domain_corner_b.y * posed.pose_normal_b
                + domain.domain_corner_b.z * posed.pose_normal_c,
            source.record_normal_b.w,
        );
        result.record_normal_c = vec4<f32>(
            domain.domain_corner_c.x * posed.pose_normal_a
                + domain.domain_corner_c.y * posed.pose_normal_b
                + domain.domain_corner_c.z * posed.pose_normal_c,
            source.record_normal_c.w,
        );
    }
    return result;
}
