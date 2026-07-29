#define_import_path quilting::math::quaternion

// Quaternion: vec4(w, x, y, z) — scalar part in .x, vector part in .yzw
// This matches the Rust Quat layout and the instance data packing.

fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
        a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
        a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
        a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x
    );
}

fn qconj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(q.x, -q.y, -q.z, -q.w);
}

// Inverse, with a Möbius-pole guard. The threshold and sentinel must match
// SINGULARITY_NORM_SQ / SINGULARITY_SENTINEL in quilting-core's quaternion.rs —
// the CPU computes LODs and smooth normals for the same geometry this shader
// evaluates, so the two must agree on where the pole is. 1e-20 is about as low
// as f32 can go before dot(q, q) drifts into the subnormal range, and the 1e10
// sentinel squares to 1e20, far under f32's 3.4e38 ceiling.
fn qinv(q: vec4<f32>) -> vec4<f32> {
    let d = dot(q, q);
    if d < 1e-20 {
        return vec4<f32>(1e10, 0.0, 0.0, 0.0);
    }
    return qconj(q) / d;
}

fn qnorm(q: vec4<f32>) -> f32 {
    return length(q);
}

// Rotate a 3D vector by a unit quaternion: v' = q * v * q̄
fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = vec4<f32>(0.0, v.x, v.y, v.z);
    return qmul(qmul(q, qv), qconj(q)).yzw;
}

// Extract 3D position from quaternion (imaginary part)
fn q_to_point(q: vec4<f32>) -> vec3<f32> {
    return q.yzw;
}

// Pure imaginary quaternion from 3D point
fn point_to_q(p: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(0.0, p.x, p.y, p.z);
}
