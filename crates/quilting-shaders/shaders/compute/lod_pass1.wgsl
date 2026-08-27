#import quilting::math::quaternion::{qmul, qinv}
#import quilting::compute::lod_types::{LodFaceRecord, LodPass1Record, LodSubjectState, LodDispatchUniforms}
#import quilting::compute::pose::animated_position

// WebGPU form of the current WebGL2 LOD classifier's first pass. One compute
// invocation owns one source face and emits one 16-byte intermediate record.
// All indices are integers and all formerly tiled texture payloads are linear
// buffers; the geometric policy intentionally mirrors lod_compute.vert.glsl.

@group(0) @binding(0) var<uniform> dispatch: LodDispatchUniforms;
@group(0) @binding(1) var<storage, read> faces: array<LodFaceRecord>;
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
// Bindings 3..6 are the shared dynamic-pose residency imported above.
@group(0) @binding(7) var<storage, read> subject_states: array<LodSubjectState>;
@group(0) @binding(8) var<storage, read_write> pass1_records: array<LodPass1Record>;

struct MobiusPoint {
    point: vec3<f32>,
    bot_sq: f32,
}

fn baseline_subject() -> LodSubjectState {
    return LodSubjectState(
        dispatch.baseline_mob_a,
        dispatch.baseline_mob_b,
        dispatch.baseline_mob_c,
        dispatch.baseline_mob_d,
        dispatch.baseline_model,
        dispatch.baseline_pole,
        vec4<f32>(
            dispatch.baseline_conformal.x,
            dispatch.baseline_conformal.y,
            dispatch.baseline_conformal.z,
            1.0,
        ),
    );
}

fn subject_for_face(face: LodFaceRecord) -> LodSubjectState {
    var subject = baseline_subject();
    if dispatch.baseline_conformal.w > 0.5 {
        let candidate = subject_states[face.subject_index];
        if candidate.conformal.w > 0.5 {
            subject = candidate;
        }
    }
    return subject;
}

fn mobius(subject: LodSubjectState, point: vec3<f32>) -> MobiusPoint {
    let q = vec4<f32>(0.0, point);
    let top = qmul(subject.mob_a, q) + subject.mob_b;
    let bot = qmul(subject.mob_c, q) + subject.mob_d;
    return MobiusPoint(qmul(top, qinv(bot)).yzw, dot(bot, bot));
}

fn mobius_pure(subject: LodSubjectState, point: vec3<f32>) -> vec3<f32> {
    return mobius(subject, point).point;
}

fn fetch_animated_position(vertex: u32) -> vec3<f32> {
    return animated_position(
        positions[vertex].xyz,
        vertex,
        dispatch.counts.y,
        dispatch.counts.z,
        dispatch.counts.w,
    );
}

fn finite_vec3(value: vec3<f32>) -> bool {
    return all(value == value) && all(abs(value) <= vec3<f32>(3.402823e38));
}

fn sphere_outside_frustum(center: vec3<f32>, radius: f32) -> bool {
    if !finite_vec3(center) || radius != radius || abs(radius) > 3.402823e38 || radius < 0.0 {
        return false;
    }
    let view_projection = dispatch.view_projection;
    let clip = view_projection * vec4<f32>(center, 1.0);
    let row0 = vec3<f32>(view_projection[0].x, view_projection[1].x, view_projection[2].x);
    let row1 = vec3<f32>(view_projection[0].y, view_projection[1].y, view_projection[2].y);
    let row2 = vec3<f32>(view_projection[0].z, view_projection[1].z, view_projection[2].z);
    let row3 = vec3<f32>(view_projection[0].w, view_projection[1].w, view_projection[2].w);
    let guard = 1.25;
    let guarded_w = guard * clip.w;
    let guarded_row3 = guard * row3;
    return guarded_w + clip.x < -radius * length(guarded_row3 + row0)
        || guarded_w - clip.x < -radius * length(guarded_row3 - row0)
        || guarded_w + clip.y < -radius * length(guarded_row3 + row1)
        || guarded_w - clip.y < -radius * length(guarded_row3 - row1)
        || clip.w + clip.z < -radius * length(row3 + row2)
        || clip.w - clip.z < -radius * length(row3 - row2);
}

fn image_ball_outside_frustum(
    subject: LodSubjectState,
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>,
) -> bool {
    let center = (p0 + p1 + p2) / 3.0;
    let source_radius = sqrt(max(
        dot(p0 - center, p0 - center),
        max(dot(p1 - center, p1 - center), dot(p2 - center, p2 - center)),
    ));
    var direction = vec3<f32>(1.0, 0.0, 0.0);
    if subject.conformal.z > 0.5 {
        let delta = center - subject.pole.yzw;
        let pole_distance_sq = subject.pole.x * subject.pole.x + dot(delta, delta);
        let guarded_radius = 1.05 * source_radius;
        if pole_distance_sq <= guarded_radius * guarded_radius {
            return false;
        }
        let delta_length = length(delta);
        if delta_length <= 1e-20 {
            return false;
        }
        direction = delta / delta_length;
    }

    let plus = mobius_pure(subject, center + source_radius * direction);
    let minus = mobius_pure(subject, center - source_radius * direction);
    if !finite_vec3(plus) || !finite_vec3(minus) {
        return false;
    }
    let image_center = 0.5 * (plus + minus);
    let image_radius = 0.5 * distance(plus, minus) * 1.05 + 1e-6;
    return sphere_outside_frustum(image_center, image_radius);
}

fn closest_point_triangle(
    point: vec3<f32>,
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 { return a; }
    let bp = point - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 { return b; }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 { return a + ab * (d1 / (d1 - d3)); }
    let cp = point - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 { return c; }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 { return a + ac * (d2 / (d2 - d6)); }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denominator = 1.0 / (va + vb + vc);
    return a + ab * (vb * denominator) + ac * (vc * denominator);
}

fn snapped_power_of_two(value: f32) -> f32 {
    return exp2(round(log2(max(value, 1.0))));
}

fn floored_power_of_two(value: f32) -> f32 {
    return exp2(floor(log2(max(value, 1.0))));
}

fn screen_point(world: vec3<f32>) -> vec2<f32> {
    let clip = dispatch.view_projection * vec4<f32>(world, 1.0);
    return (clip.xy / max(clip.w, 0.001)) * dispatch.viewport.xy * 0.5;
}

@compute @workgroup_size(64)
fn classify_lod_pass1(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let face_index = invocation.x;
    if face_index >= dispatch.counts.x {
        return;
    }
    let face = faces[face_index];
    let subject = subject_for_face(face);
    let p0 = (subject.model * vec4<f32>(fetch_animated_position(face.vertex_indices.x), 1.0)).xyz;
    let p1 = (subject.model * vec4<f32>(fetch_animated_position(face.vertex_indices.y), 1.0)).xyz;
    let p2 = (subject.model * vec4<f32>(fetch_animated_position(face.vertex_indices.z), 1.0)).xyz;

    let min_px = dispatch.density_metrics.z;
    let max_lod = dispatch.density_metrics.w;
    if image_ball_outside_frustum(subject, p0, p1, p2) {
        let standby_lod = select(max_lod, min(2.0, max_lod), min_px > 0.0);
        let standby_exponent = log2(max(standby_lod, 1.0));
        pass1_records[face_index] = LodPass1Record(
            vec3<f32>(standby_exponent),
            0.0,
        );
        return;
    }

    let mid_a = (p1 + p2) * 0.5;
    let mid_b = (p0 + p2) * 0.5;
    let mid_c = (p0 + p1) * 0.5;
    let transformed0 = mobius(subject, p0);
    let transformed1 = mobius(subject, p1);
    let transformed2 = mobius(subject, p2);
    let transformed_mid_a = mobius(subject, mid_a);
    let transformed_mid_b = mobius(subject, mid_b);
    let transformed_mid_c = mobius(subject, mid_c);
    let d0 = transformed0.point;
    let d1 = transformed1.point;
    let d2 = transformed2.point;
    let dm_a = transformed_mid_a.point;
    let dm_b = transformed_mid_b.point;
    let dm_c = transformed_mid_c.point;

    let source_length_a = distance(p1, p2);
    let source_length_b = distance(p0, p2);
    let source_length_c = distance(p0, p1);
    let deformed_arc_a = distance(d1, dm_a) + distance(dm_a, d2);
    let deformed_arc_b = distance(d0, dm_b) + distance(dm_b, d2);
    let deformed_arc_c = distance(d0, dm_c) + distance(dm_c, d1);
    var min_bot_true = min(
        min(transformed0.bot_sq, min(transformed1.bot_sq, transformed2.bot_sq)),
        min(transformed_mid_a.bot_sq, min(transformed_mid_b.bot_sq, transformed_mid_c.bot_sq)),
    );
    var x_star = mid_a;
    var transformed_distance_sq = 0.0;
    var patch_valid = false;
    if subject.conformal.z > 0.5 {
        let normal = cross(p1 - p0, p2 - p0);
        if dot(normal, normal) >= 1e-24 {
            let pole_imaginary = subject.pole.yzw;
            x_star = closest_point_triangle(pole_imaginary, p0, p1, p2);
            transformed_distance_sq = subject.pole.x * subject.pole.x
                + dot(x_star - pole_imaginary, x_star - pole_imaginary);
            min_bot_true = subject.conformal.y * transformed_distance_sq;
            patch_valid = true;
        }
    }

    let target_size = dispatch.density_metrics.y / dispatch.density_metrics.x;
    let intrinsic_similarity = select(1.0, max(subject.conformal.x, 1e-12), subject.conformal.z > 0.5);
    var world_demand = vec3<f32>(
        deformed_arc_a,
        deformed_arc_b,
        deformed_arc_c,
    ) / (target_size * intrinsic_similarity);
    var lambda_star = 0.0;
    if patch_valid && min_bot_true < 1e-8 {
        world_demand = vec3<f32>(max_lod);
    } else if patch_valid {
        lambda_star = subject.conformal.x / max(transformed_distance_sq, 1e-30);
        let intrinsic_peak = lambda_star / (intrinsic_similarity * target_size);
        world_demand = max(
            world_demand,
            intrinsic_peak * vec3<f32>(source_length_a, source_length_b, source_length_c),
        );
    }
    var lod = clamp(
        vec3<f32>(
            snapped_power_of_two(world_demand.x),
            snapped_power_of_two(world_demand.y),
            snapped_power_of_two(world_demand.z),
        ),
        vec3<f32>(1.0),
        vec3<f32>(max_lod),
    );
    var adaptive_priority = 0.0;

    if min_px > 0.0 {
        let s0 = screen_point(d0);
        let s1 = screen_point(d1);
        let s2 = screen_point(d2);
        let screen_mid_a = screen_point(dm_a);
        let screen_mid_b = screen_point(dm_b);
        let screen_mid_c = screen_point(dm_c);
        let rim_extent = vec3<f32>(
            distance(s1, screen_mid_a) + distance(screen_mid_a, s2),
            distance(s0, screen_mid_b) + distance(screen_mid_b, s2),
            distance(s0, screen_mid_c) + distance(screen_mid_c, s1),
        );
        let max_screen_extent = length(dispatch.viewport.xy);
        let pole_extent = select(0.0, max_screen_extent, patch_valid && min_bot_true < 1e-8);
        var interior_extent = vec3<f32>(pole_extent);
        if patch_valid && min_bot_true >= 1e-8 {
            let y_star = mobius_pure(subject, x_star);
            let clip_star = dispatch.view_projection * vec4<f32>(y_star, 1.0);
            if clip_star.w > 0.001 {
                let star_screen = (clip_star.xy / clip_star.w) * dispatch.viewport.xy * 0.5;
                let epsilon = 1e-3;
                let directions = array<vec3<f32>, 4>(
                    vec3<f32>(1.0, 0.0, 0.0),
                    vec3<f32>(0.0, 1.0, 0.0),
                    vec3<f32>(0.0, 0.0, 1.0),
                    vec3<f32>(0.5774, 0.5774, 0.5774),
                );
                var projection_scale = 0.0;
                for (var direction = 0u; direction < 4u; direction++) {
                    let nearby_clip = dispatch.view_projection
                        * vec4<f32>(y_star + epsilon * directions[direction], 1.0);
                    if nearby_clip.w > 0.001 {
                        let nearby_screen = (nearby_clip.xy / nearby_clip.w)
                            * dispatch.viewport.xy * 0.5;
                        projection_scale = max(
                            projection_scale,
                            distance(nearby_screen, star_screen) / epsilon,
                        );
                    }
                }
                let projected_peak = lambda_star * projection_scale;
                interior_extent = min(
                    projected_peak * vec3<f32>(source_length_a, source_length_b, source_length_c),
                    vec3<f32>(max_screen_extent),
                );
            }
        }

        let metric_variation = max(
            max(
                interior_extent.x / max(rim_extent.x, 1.0),
                interior_extent.y / max(rim_extent.y, 1.0),
            ),
            interior_extent.z / max(rim_extent.z, 1.0),
        );
        let variation_octaves = log2(max(metric_variation, 1.0));
        var pole_octaves = 0.0;
        if patch_valid {
            let source_scale = max(source_length_a, max(source_length_b, source_length_c));
            let relative_pole_distance = sqrt(max(transformed_distance_sq, 1e-30))
                / max(source_scale, 1e-15);
            pole_octaves = max(0.0, -log2(max(relative_pole_distance, 1e-8)));
        }
        adaptive_priority = ceil(clamp(
            max(variation_octaves, pole_octaves) * 32.0,
            0.0,
            255.0,
        ));
        let bounded_extent = min(max(rim_extent, interior_extent), vec3<f32>(max_screen_extent));
        let capacity = clamp(
            vec3<f32>(
                floored_power_of_two(bounded_extent.x / min_px),
                floored_power_of_two(bounded_extent.y / min_px),
                floored_power_of_two(bounded_extent.z / min_px),
            ),
            vec3<f32>(1.0),
            vec3<f32>(max_lod),
        );
        lod = min(lod, capacity);
    }

    pass1_records[face_index] = LodPass1Record(
        vec3<f32>(log2(lod.x), log2(lod.y), log2(lod.z)),
        1.0 + adaptive_priority,
    );
}
