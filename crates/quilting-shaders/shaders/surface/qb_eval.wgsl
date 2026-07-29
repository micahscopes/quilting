#define_import_path quilting::surface::qb_eval

#import quilting::math::quaternion::{qmul, qconj, qinv, q_to_point}

// QB triangle surface evaluation.
//
// X(bary) = (Σ λᵢ pᵢ wᵢ) / (Σ λᵢ wᵢ)
//
// where λᵢ are barycentric coordinates, pᵢ are position quaternions,
// wᵢ are weight quaternions. Returns a 3D point (imaginary part of X).

fn eval_qb(
    bary: vec3<f32>,
    p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>,
    w0: vec4<f32>, w1: vec4<f32>, w2: vec4<f32>,
) -> vec3<f32> {
    let top = bary.x * qmul(p0, w0) + bary.y * qmul(p1, w1) + bary.z * qmul(p2, w2);
    let bot = bary.x * w0 + bary.y * w1 + bary.z * w2;
    return q_to_point(qmul(top, qinv(bot)));
}

// QB evaluation with analytic normal via quotient rule.
//
// dX/du = (dtop_u - X * dbot_u) * bot⁻¹
// dX/dv = (dtop_v - X * dbot_v) * bot⁻¹
//
// Tangent directions: u = bary x→y, v = bary x→z.

struct QBResult {
    position: vec3<f32>,
    normal: vec3<f32>,
}

fn eval_qb_with_normal(
    bary: vec3<f32>,
    p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>,
    w0: vec4<f32>, w1: vec4<f32>, w2: vec4<f32>,
    perm_parity: f32,
) -> QBResult {
    let pw0 = qmul(p0, w0);
    let pw1 = qmul(p1, w1);
    let pw2 = qmul(p2, w2);

    let top = bary.x * pw0 + bary.y * pw1 + bary.z * pw2;
    let bot = bary.x * w0 + bary.y * w1 + bary.z * w2;
    let bi = qinv(bot);
    let X = qmul(top, bi);

    // Partial derivatives along bary tangent directions
    let dtop_u = pw1 - pw0;
    let dbot_u = w1 - w0;
    let dtop_v = pw2 - pw0;
    let dbot_v = w2 - w0;

    // Quotient rule, right-multiplied by conj(bot) instead of bot⁻¹: the
    // omitted 1/|bot|² is a positive real scalar common to both tangents, so
    // the normalized cross product is identical, but intermediates stay
    // bounded when |bot| gets small (the raw form overflows f32 through
    // dot(n, n) ~ 1/|bot|⁸; see eval_mobius_qb in vertex/main.wgsl).
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

    // Flip for odd permutations (reflections in S3)
    n = n * perm_parity;

    return QBResult(q_to_point(X), n);
}
