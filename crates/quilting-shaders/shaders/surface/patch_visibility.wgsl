#define_import_path quilting::surface::patch_visibility

#import quilting::math::quaternion::{qmul, qconj, qinv}
#import quilting::surface::patch_render::{PatchRenderTransform, POSITION_CLAMP}

fn visibility_finite_vec3(value: vec3<f32>) -> bool {
    // Comparisons with NaN are false; infinities exceed the finite guard.
    return all(abs(value) < vec3<f32>(1e30));
}

fn visibility_finite_vec4(value: vec4<f32>) -> bool {
    return all(abs(value) < vec4<f32>(1e30));
}

fn visibility_mobius_point(
    transform: PatchRenderTransform,
    point: vec3<f32>,
) -> vec3<f32> {
    let q = vec4<f32>(0.0, point);
    return qmul(
        qmul(transform.mob_a, q) + transform.mob_b,
        qinv(qmul(transform.mob_c, q) + transform.mob_d),
    ).yzw;
}

// Return true only when a world-space ball lies wholly outside one homogeneous
// clip plane. Invalid bounds survive, preserving conservative visibility.
fn visibility_sphere_outside_frustum(
    transform: PatchRenderTransform,
    center: vec3<f32>,
    radius: f32,
) -> bool {
    if !visibility_finite_vec3(center) || !(radius >= 0.0 && radius < 1e30) {
        return false;
    }
    let clip = transform.mvp * vec4<f32>(center, 1.0);
    let r0 = vec3<f32>(transform.mvp[0][0], transform.mvp[1][0], transform.mvp[2][0]);
    let r1 = vec3<f32>(transform.mvp[0][1], transform.mvp[1][1], transform.mvp[2][1]);
    let r2 = vec3<f32>(transform.mvp[0][2], transform.mvp[1][2], transform.mvp[2][2]);
    let r3 = vec3<f32>(transform.mvp[0][3], transform.mvp[1][3], transform.mvp[2][3]);
    return clip.w + clip.x < -radius * length(r3 + r0)
        || clip.w - clip.x < -radius * length(r3 - r0)
        || clip.w + clip.y < -radius * length(r3 + r1)
        || clip.w - clip.y < -radius * length(r3 - r1)
        || clip.w + clip.z < -radius * length(r3 + r2)
        || clip.w - clip.z < -radius * length(r3 - r2);
}

// Conservative bound on the complete Mobius image of a flat source patch. A
// source ball maps to a ball when the pole is outside it. Pole-containing,
// singular, and non-finite cases deliberately survive.
fn flat_patch_outside_frustum(
    transform: PatchRenderTransform,
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
    let c_norm_sq = dot(transform.mob_c, transform.mob_c);
    if c_norm_sq > 1e-20 {
        let pole = -qmul(qinv(transform.mob_c), transform.mob_d);
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

    let plus = visibility_mobius_point(transform, center + source_radius * direction);
    let minus = visibility_mobius_point(transform, center - source_radius * direction);
    if !visibility_finite_vec3(plus) || !visibility_finite_vec3(minus) {
        return false;
    }
    let image_center = 0.5 * (plus + minus);
    let image_radius = 0.5 * distance(plus, minus) * 1.05 + 1e-6;
    return visibility_sphere_outside_frustum(transform, image_center, image_radius);
}

// Exact Euclidean distance from the origin to a triangle embedded in R4.
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
// Mobius transform. For barycentric lambda, the fused evaluator is q = N/D,
// where N and D each range over a quaternion triangle.
fn rational_patch_outside_frustum(
    transform: PatchRenderTransform,
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>,
    w0: vec4<f32>,
    w1: vec4<f32>,
    w2: vec4<f32>,
) -> bool {
    let qp0 = vec4<f32>(0.0, p0);
    let qp1 = vec4<f32>(0.0, p1);
    let qp2 = vec4<f32>(0.0, p2);
    let numerator0 = qmul(qmul(transform.mob_a, qp0) + transform.mob_b, w0);
    let numerator1 = qmul(qmul(transform.mob_a, qp1) + transform.mob_b, w1);
    let numerator2 = qmul(qmul(transform.mob_a, qp2) + transform.mob_b, w2);
    let denominator0 = qmul(qmul(transform.mob_c, qp0) + transform.mob_d, w0);
    let denominator1 = qmul(qmul(transform.mob_c, qp1) + transform.mob_d, w1);
    let denominator2 = qmul(qmul(transform.mob_c, qp2) + transform.mob_d, w2);
    if !visibility_finite_vec4(numerator0)
        || !visibility_finite_vec4(numerator1)
        || !visibility_finite_vec4(numerator2)
        || !visibility_finite_vec4(denominator0)
        || !visibility_finite_vec4(denominator1)
        || !visibility_finite_vec4(denominator2) {
        return false;
    }
    let denominator_distance = origin_to_quaternion_triangle(
        denominator0,
        denominator1,
        denominator2,
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
    if !visibility_finite_vec4(center_quaternion) {
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
    if !visibility_finite_vec3(center) || !(radius >= 0.0 && radius < 1e30) {
        return false;
    }

    // Surface evaluation clamps positions into the origin-centered ball. That
    // preserves the local analytic bound only when the complete ball is
    // already inside the clamp ball.
    if length(center) + radius <= POSITION_CLAMP {
        return visibility_sphere_outside_frustum(transform, center, radius);
    }
    return visibility_sphere_outside_frustum(transform, vec3<f32>(0.0), POSITION_CLAMP);
}

fn prepared_patch_outside_frustum(
    transform: PatchRenderTransform,
    p0: vec3<f32>,
    p1: vec3<f32>,
    p2: vec3<f32>,
    w0: vec4<f32>,
    w1: vec4<f32>,
    w2: vec4<f32>,
) -> bool {
    let common_weight = length(w0) > 1e-10 && all(w1 == w0) && all(w2 == w0);
    if transform.use_qb != 1 || common_weight {
        return flat_patch_outside_frustum(transform, p0, p1, p2);
    }
    return rational_patch_outside_frustum(transform, p0, p1, p2, w0, w1, w2);
}
