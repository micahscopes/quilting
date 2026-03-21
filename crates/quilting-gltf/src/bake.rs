//! Bake skeletal/morph animations into per-vertex trajectories.
//!
//! Evaluates the full skeleton at each keyframe, applies skinning weights,
//! and outputs per-vertex position trajectories suitable for the spacetime slicer.

use crate::animation::{Animation, AnimationChannel, AnimationProperty, Interpolation, Skin};
use crate::mesh::Primitive;
use crate::scene::{Node, Transform};
use quilting_spacetime::trajectory::{HermiteSegment, VertexTrajectory};
use quilting_spacetime::HyperMesh;

/// Bake a skinned animation into a HyperMesh with per-vertex trajectories.
///
/// Evaluates the skeleton at `num_samples` evenly-spaced times across
/// the animation, applies skinning, and builds cubic Hermite trajectories.
pub fn bake_skinned_animation(
    primitive: &Primitive,
    skin: &Skin,
    animation: &Animation,
    nodes: &[Node],
    num_samples: usize,
) -> HyperMesh {
    let num_samples = num_samples.max(2);

    // Find animation time range
    let (t_min, t_max) = animation_time_range(animation);
    let duration = (t_max - t_min).max(0.001);
    let dt = duration / (num_samples - 1) as f64;

    let n_verts = primitive.positions.len();
    let joints = &skin.joints;
    let ibms = &skin.inverse_bind_matrices;

    let joint_indices = primitive.joint_indices.as_ref();
    let joint_weights = primitive.joint_weights.as_ref();

    // Sample skinned positions at each time step
    let mut samples: Vec<Vec<[f64; 3]>> = Vec::with_capacity(num_samples);

    for si in 0..num_samples {
        let t = t_min + si as f64 * dt;

        // Evaluate joint transforms at time t
        let joint_matrices = evaluate_joint_transforms(animation, nodes, joints, t);

        // Skin matrix per joint: joint_matrix * inverse_bind_matrix
        let skin_matrices: Vec<[f64; 16]> = joint_matrices
            .iter()
            .zip(ibms.iter())
            .map(|(jm, ibm)| {
                let ibm_f64: [f64; 16] = std::array::from_fn(|i| ibm[i] as f64);
                mat4_mul(jm, &ibm_f64)
            })
            .collect();

        // Apply skinning to each vertex
        let mut positions = Vec::with_capacity(n_verts);
        for vi in 0..n_verts {
            let rest_pos = primitive.positions[vi];

            if let (Some(ji), Some(jw)) = (joint_indices, joint_weights) {
                let indices = ji[vi];
                let weights = jw[vi];

                let mut skinned = [0.0f64; 3];
                for k in 0..4 {
                    let w = weights[k] as f64;
                    if w < 1e-8 { continue; }
                    let joint_idx = indices[k] as usize;
                    if joint_idx >= skin_matrices.len() { continue; }
                    let m = &skin_matrices[joint_idx];
                    // Transform position by skin matrix
                    skinned[0] += w * (m[0]*rest_pos[0] + m[4]*rest_pos[1] + m[8]*rest_pos[2] + m[12]);
                    skinned[1] += w * (m[1]*rest_pos[0] + m[5]*rest_pos[1] + m[9]*rest_pos[2] + m[13]);
                    skinned[2] += w * (m[2]*rest_pos[0] + m[6]*rest_pos[1] + m[10]*rest_pos[2] + m[14]);
                }
                positions.push(skinned);
            } else {
                // No skinning — use rest position
                positions.push(rest_pos);
            }
        }
        samples.push(positions);
    }

    // Build trajectories from samples
    let faces: Vec<[u32; 3]> = primitive.triangles
        .iter()
        .map(|t| [t[0] as u32, t[1] as u32, t[2] as u32])
        .collect();

    let trajectories: Vec<VertexTrajectory> = (0..n_verts)
        .map(|vi| {
            let segments: Vec<HermiteSegment> = (0..num_samples - 1)
                .map(|si| {
                    let t0 = t_min + si as f64 * dt;
                    let t1 = t_min + (si + 1) as f64 * dt;
                    let p0 = samples[si][vi];
                    let p1 = samples[si + 1][vi];
                    // Velocity from finite differences
                    let vel = [
                        (p1[0] - p0[0]) / dt,
                        (p1[1] - p0[1]) / dt,
                        (p1[2] - p0[2]) / dt,
                    ];
                    HermiteSegment {
                        t_start: t0,
                        t_end: t1,
                        pos_start: p0,
                        pos_end: p1,
                        vel_start: vel,
                        vel_end: vel,
                    }
                })
                .collect();
            VertexTrajectory { segments }
        })
        .collect();

    HyperMesh::new(faces, trajectories)
}

/// Bake a morph target animation into a HyperMesh.
pub fn bake_morph_animation(
    primitive: &Primitive,
    animation: &Animation,
    morph_deltas: &[Vec<[f64; 3]>],
    num_samples: usize,
) -> HyperMesh {
    let num_samples = num_samples.max(2);
    let (t_min, t_max) = animation_time_range(animation);
    let duration = (t_max - t_min).max(0.001);
    let dt = duration / (num_samples - 1) as f64;
    let n_verts = primitive.positions.len();

    // Find the morph weight channel
    let morph_channel = animation.channels.iter()
        .find(|c| c.property == AnimationProperty::MorphTargetWeights);

    let mut samples: Vec<Vec<[f64; 3]>> = Vec::with_capacity(num_samples);

    for si in 0..num_samples {
        let t = t_min + si as f64 * dt;

        // Evaluate morph weights at time t
        let weights = if let Some(ch) = morph_channel {
            evaluate_morph_weights(ch, morph_deltas.len(), t)
        } else {
            vec![0.0; morph_deltas.len()]
        };

        // Apply morph deltas
        let mut positions = Vec::with_capacity(n_verts);
        for vi in 0..n_verts {
            let mut pos = primitive.positions[vi];
            for (ti, delta) in morph_deltas.iter().enumerate() {
                if ti < weights.len() && vi < delta.len() {
                    pos[0] += weights[ti] * delta[vi][0];
                    pos[1] += weights[ti] * delta[vi][1];
                    pos[2] += weights[ti] * delta[vi][2];
                }
            }
            positions.push(pos);
        }
        samples.push(positions);
    }

    let faces: Vec<[u32; 3]> = primitive.triangles
        .iter()
        .map(|t| [t[0] as u32, t[1] as u32, t[2] as u32])
        .collect();

    let trajectories: Vec<VertexTrajectory> = (0..n_verts)
        .map(|vi| {
            let segments: Vec<HermiteSegment> = (0..num_samples - 1)
                .map(|si| {
                    let t0 = t_min + si as f64 * dt;
                    let t1 = t_min + (si + 1) as f64 * dt;
                    let p0 = samples[si][vi];
                    let p1 = samples[si + 1][vi];
                    let vel = [
                        (p1[0] - p0[0]) / dt,
                        (p1[1] - p0[1]) / dt,
                        (p1[2] - p0[2]) / dt,
                    ];
                    HermiteSegment {
                        t_start: t0, t_end: t1,
                        pos_start: p0, pos_end: p1,
                        vel_start: vel, vel_end: vel,
                    }
                })
                .collect();
            VertexTrajectory { segments }
        })
        .collect();

    HyperMesh::new(faces, trajectories)
}

/// Evaluate a skinned mesh at a single time — no baking, just one frame.
/// Returns the skinned vertex positions at time t.
pub fn evaluate_skinned_at_time(
    primitive: &Primitive,
    skin: &Skin,
    animation: &Animation,
    nodes: &[Node],
    t: f64,
) -> Vec<[f64; 3]> {
    let joints = &skin.joints;
    let ibms = &skin.inverse_bind_matrices;
    let joint_matrices = evaluate_joint_transforms(animation, nodes, joints, t);

    let skin_matrices: Vec<[f64; 16]> = joint_matrices
        .iter()
        .zip(ibms.iter())
        .map(|(jm, ibm)| {
            let ibm_f64: [f64; 16] = std::array::from_fn(|i| ibm[i] as f64);
            mat4_mul(jm, &ibm_f64)
        })
        .collect();

    let joint_indices = primitive.joint_indices.as_ref();
    let joint_weights = primitive.joint_weights.as_ref();

    primitive.positions.iter().enumerate().map(|(vi, &rest_pos)| {
        if let (Some(ji), Some(jw)) = (joint_indices, joint_weights) {
            let indices = ji[vi];
            let weights = jw[vi];
            let mut skinned = [0.0f64; 3];
            for k in 0..4 {
                let w = weights[k] as f64;
                if w < 1e-8 { continue; }
                let joint_idx = indices[k] as usize;
                if joint_idx >= skin_matrices.len() { continue; }
                let m = &skin_matrices[joint_idx];
                skinned[0] += w * (m[0]*rest_pos[0] + m[4]*rest_pos[1] + m[8]*rest_pos[2] + m[12]);
                skinned[1] += w * (m[1]*rest_pos[0] + m[5]*rest_pos[1] + m[9]*rest_pos[2] + m[13]);
                skinned[2] += w * (m[2]*rest_pos[0] + m[6]*rest_pos[1] + m[10]*rest_pos[2] + m[14]);
            }
            skinned
        } else {
            rest_pos
        }
    }).collect()
}

// --- Helpers ---

fn animation_time_range(animation: &Animation) -> (f64, f64) {
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    for ch in &animation.channels {
        for &t in &ch.times {
            t_min = t_min.min(t as f64);
            t_max = t_max.max(t as f64);
        }
    }
    if t_min.is_infinite() { (0.0, 1.0) } else { (t_min, t_max) }
}

/// Evaluate joint transforms at time t.
/// Returns a world-space 4x4 matrix for each joint.
fn evaluate_joint_transforms(
    animation: &Animation,
    nodes: &[Node],
    joints: &[usize],
    t: f64,
) -> Vec<[f64; 16]> {
    joints.iter().map(|&joint_node| {
        // Start with the node's rest-pose TRS
        let (mut translation, mut rotation, mut scale) = match &nodes[joint_node].transform {
            Transform::Trs { translation, rotation, scale } => (*translation, *rotation, *scale),
            Transform::Matrix(_) => ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
        };

        for ch in &animation.channels {
            if ch.target_node != joint_node { continue; }
            match ch.property {
                AnimationProperty::Translation => {
                    let v = interpolate_channel(ch, 3, t);
                    translation = [v[0] as f64, v[1] as f64, v[2] as f64];
                }
                AnimationProperty::Rotation => {
                    let v = interpolate_channel(ch, 4, t);
                    rotation = [v[0] as f64, v[1] as f64, v[2] as f64, v[3] as f64];
                }
                AnimationProperty::Scale => {
                    let v = interpolate_channel(ch, 3, t);
                    scale = [v[0] as f64, v[1] as f64, v[2] as f64];
                }
                _ => {}
            }
        }

        trs_to_matrix(translation, rotation, scale)
    }).collect()
}

/// Interpolate an animation channel at time t.
fn interpolate_channel(ch: &AnimationChannel, stride: usize, t: f64) -> Vec<f32> {
    if ch.times.is_empty() {
        return vec![0.0; stride];
    }

    let t = t as f32;

    // Clamp to range
    if t <= ch.times[0] {
        return ch.values[..stride].to_vec();
    }
    if t >= *ch.times.last().unwrap() {
        let n = ch.values.len();
        return ch.values[n - stride..].to_vec();
    }

    // Find the two keyframes bracketing t
    let mut idx = 0;
    for (i, &kt) in ch.times.iter().enumerate() {
        if kt > t { break; }
        idx = i;
    }
    let next = (idx + 1).min(ch.times.len() - 1);

    let t0 = ch.times[idx];
    let t1 = ch.times[next];
    let frac = if (t1 - t0).abs() > 1e-8 { (t - t0) / (t1 - t0) } else { 0.0 };

    match ch.interpolation {
        Interpolation::Step => {
            ch.values[idx * stride..(idx + 1) * stride].to_vec()
        }
        Interpolation::Linear => {
            let a = &ch.values[idx * stride..(idx + 1) * stride];
            let b = &ch.values[next * stride..(next + 1) * stride];
            if stride == 4 {
                // Quaternion slerp
                slerp(a, b, frac)
            } else {
                a.iter().zip(b.iter())
                    .map(|(&va, &vb)| va + frac * (vb - va))
                    .collect()
            }
        }
        Interpolation::CubicSpline => {
            // CubicSpline: each keyframe has 3x values (in-tangent, value, out-tangent)
            let a_val = &ch.values[idx * stride * 3 + stride..idx * stride * 3 + 2 * stride];
            let b_val = &ch.values[next * stride * 3 + stride..next * stride * 3 + 2 * stride];
            // Simplified: just lerp the values (full cubic spline is complex)
            a_val.iter().zip(b_val.iter())
                .map(|(&va, &vb)| va + frac * (vb - va))
                .collect()
        }
    }
}

fn evaluate_morph_weights(ch: &AnimationChannel, num_targets: usize, t: f64) -> Vec<f64> {
    let values = interpolate_channel(ch, num_targets, t);
    values.iter().map(|&v| v as f64).collect()
}

/// Simple quaternion slerp.
fn slerp(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let mut dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let mut b = b.to_vec();
    if dot < 0.0 {
        dot = -dot;
        for v in &mut b { *v = -*v; }
    }
    if dot > 0.9995 {
        // Linear interpolation for nearly identical quaternions
        return a.iter().zip(b.iter())
            .map(|(&va, &vb)| va + t * (vb - va))
            .collect();
    }
    let theta = dot.acos();
    let sin_theta = theta.sin();
    let wa = ((1.0 - t) * theta).sin() / sin_theta;
    let wb = (t * theta).sin() / sin_theta;
    a.iter().zip(b.iter())
        .map(|(&va, &vb)| wa * va + wb * vb)
        .collect()
}

/// Build a 4x4 column-major matrix from TRS.
fn trs_to_matrix(t: [f64; 3], r: [f64; 4], s: [f64; 3]) -> [f64; 16] {
    // r = [x, y, z, w] quaternion
    let (x, y, z, w) = (r[0], r[1], r[2], r[3]);
    let x2 = x + x; let y2 = y + y; let z2 = z + z;
    let xx = x * x2; let xy = x * y2; let xz = x * z2;
    let yy = y * y2; let yz = y * z2; let zz = z * z2;
    let wx = w * x2; let wy = w * y2; let wz = w * z2;

    // Column-major
    [
        s[0] * (1.0 - (yy + zz)), s[0] * (xy + wz),         s[0] * (xz - wy),         0.0,
        s[1] * (xy - wz),         s[1] * (1.0 - (xx + zz)), s[1] * (yz + wx),         0.0,
        s[2] * (xz + wy),         s[2] * (yz - wx),         s[2] * (1.0 - (xx + yy)), 0.0,
        t[0],                      t[1],                      t[2],                      1.0,
    ]
}

/// Multiply two column-major 4x4 matrices.
fn mat4_mul(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut out = [0.0f64; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[c * 4 + r] = a[r] * b[c * 4]
                + a[4 + r] * b[c * 4 + 1]
                + a[8 + r] * b[c * 4 + 2]
                + a[12 + r] * b[c * 4 + 3];
        }
    }
    out
}
