//! Animation data extraction from glTF.
//!
//! Handles skeletal animation (joint hierarchies, inverse bind matrices, skins)
//! and morph target animations (blend weight keyframes + morph target deltas).

use gltf::buffer;
use crate::GltfError;

/// A complete animation clip.
#[derive(Debug, Clone)]
pub struct Animation {
    pub name: Option<String>,
    pub channels: Vec<AnimationChannel>,
}

/// One channel of an animation — targets a single property of a single node.
#[derive(Debug, Clone)]
pub struct AnimationChannel {
    /// Index of the target node in GltfScene::nodes.
    pub target_node: usize,
    /// Which property is being animated.
    pub property: AnimationProperty,
    /// Interpolation method.
    pub interpolation: Interpolation,
    /// Keyframe times in seconds.
    pub times: Vec<f32>,
    /// Keyframe values. Layout depends on property:
    /// - Translation: 3 floats per keyframe
    /// - Rotation: 4 floats per keyframe (xyzw quaternion)
    /// - Scale: 3 floats per keyframe
    /// - MorphTargetWeights: N floats per keyframe (N = number of morph targets)
    ///
    /// For CubicSpline: each keyframe has 3x the values (in-tangent, value, out-tangent).
    pub values: Vec<f32>,
}

/// Animated property type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationProperty {
    Translation,
    Rotation,
    Scale,
    MorphTargetWeights,
}

/// Keyframe interpolation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Linear,
    Step,
    CubicSpline,
}



/// Skinning data (joint hierarchy + inverse bind matrices).
#[derive(Debug, Clone)]
pub struct Skin {
    pub name: Option<String>,
    /// Indices of joint nodes in GltfScene::nodes.
    pub joints: Vec<usize>,
    /// Inverse bind matrices, one per joint. Column-major 4x4.
    pub inverse_bind_matrices: Vec<[f32; 16]>,
    /// Index of the skeleton root node, if specified.
    pub skeleton_root: Option<usize>,
}

/// Extract animation data from a glTF animation.
pub fn extract_animation(
    anim: &gltf::Animation<'_>,
    buffers: &[buffer::Data],
) -> Result<Animation, GltfError> {
    let mut channels = Vec::new();

    for channel in anim.channels() {
        let target = channel.target();
        let target_node = target.node().index();

        let property = match target.property() {
            gltf::animation::Property::Translation => AnimationProperty::Translation,
            gltf::animation::Property::Rotation => AnimationProperty::Rotation,
            gltf::animation::Property::Scale => AnimationProperty::Scale,
            gltf::animation::Property::MorphTargetWeights => {
                AnimationProperty::MorphTargetWeights
            }
        };

        let sampler = channel.sampler();
        let interpolation = match sampler.interpolation() {
            gltf::animation::Interpolation::Linear => Interpolation::Linear,
            gltf::animation::Interpolation::Step => Interpolation::Step,
            gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
        };

        let reader = channel.reader(|buf| Some(&buffers[buf.index()]));

        let times: Vec<f32> = reader
            .read_inputs()
            .ok_or_else(|| GltfError::MissingData("animation channel missing timestamps".into()))?
            .collect();

        let outputs = reader
            .read_outputs()
            .ok_or_else(|| GltfError::MissingData("animation channel missing values".into()))?;

        let values: Vec<f32> = read_outputs_to_f32(outputs);

        channels.push(AnimationChannel {
            target_node,
            property,
            interpolation,
            times,
            values,
        });
    }

    Ok(Animation {
        name: anim.name().map(|s| s.to_string()),
        channels,
    })
}

/// Extract skin data from a glTF skin.
pub fn extract_skin(
    skin: &gltf::Skin<'_>,
    buffers: &[buffer::Data],
) -> Result<Skin, GltfError> {
    let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();

    let reader = skin.reader(|buf| Some(&buffers[buf.index()]));

    let inverse_bind_matrices: Vec<[f32; 16]> = if let Some(ibm_iter) = reader.read_inverse_bind_matrices() {
        ibm_iter
            .map(|mat| {
                // gltf crate returns [[f32; 4]; 4] (column-major), flatten to [f32; 16].
                let mut flat = [0.0f32; 16];
                for col in 0..4 {
                    for row in 0..4 {
                        flat[col * 4 + row] = mat[col][row];
                    }
                }
                flat
            })
            .collect()
    } else {
        // If no IBM accessor, use identity matrices per spec.
        vec![[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]; joints.len()]
    };

    let skeleton_root = skin.skeleton().map(|n| n.index());

    Ok(Skin {
        name: skin.name().map(|s| s.to_string()),
        joints,
        inverse_bind_matrices,
        skeleton_root,
    })
}

/// Convert animation output samples to a flat Vec<f32>, regardless of the underlying type.
fn read_outputs_to_f32(outputs: gltf::animation::util::ReadOutputs<'_>) -> Vec<f32> {
    use gltf::animation::util::ReadOutputs;
    match outputs {
        ReadOutputs::Translations(iter) => iter.flat_map(|v| v.into_iter()).collect(),
        ReadOutputs::Rotations(rotations) => {
            rotations.into_f32().flat_map(|v| v.into_iter()).collect()
        }
        ReadOutputs::Scales(iter) => iter.flat_map(|v| v.into_iter()).collect(),
        ReadOutputs::MorphTargetWeights(weights) => weights.into_f32().collect(),
    }
}


/// Number of f32 values per keyframe for a given property.
fn property_stride(prop: AnimationProperty) -> usize {
    match prop {
        AnimationProperty::Translation => 3,
        AnimationProperty::Rotation => 4,
        AnimationProperty::Scale => 3,
        // MorphTargetWeights stride is dynamic (depends on target count).
        AnimationProperty::MorphTargetWeights => 1,
    }
}

// --- Animation evaluation helpers ---
// These were previously in bake.rs but are needed by evaluator.rs and wasm bindings.

/// Get the time range of an animation clip.
pub fn animation_time_range(animation: &Animation) -> (f64, f64) {
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
/// Returns a WORLD-space 4x4 matrix for each joint by walking the full
/// node hierarchy with animated local transforms applied.
pub fn evaluate_joint_transforms(
    animation: &Animation,
    nodes: &[crate::scene::Node],
    joints: &[usize],
    t: f64,
) -> Vec<[f64; 16]> {
    use crate::scene::{Transform, Scene, compute_world_transforms};

    // Build animated local transforms for all nodes
    let mut animated_nodes = nodes.to_vec();
    for ch in &animation.channels {
        let ni = ch.target_node;
        if ni >= animated_nodes.len() { continue; }
        let (mut translation, mut rotation, mut scale) = match &animated_nodes[ni].transform {
            Transform::Trs { translation, rotation, scale } => (*translation, *rotation, *scale),
            Transform::Matrix(_) => ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
        };
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
            _ => continue,
        }
        animated_nodes[ni].transform = Transform::Trs { translation, rotation, scale };
    }

    let root_nodes: Vec<usize> = (0..nodes.len())
        .filter(|&i| !nodes.iter().any(|n| n.children.contains(&i)))
        .collect();
    let scene = Scene { name: None, root_nodes };
    let world = compute_world_transforms(&animated_nodes, &scene);

    joints.iter().map(|&ji| world[ji]).collect()
}

/// Interpolate an animation channel at time t.
pub fn interpolate_channel(ch: &AnimationChannel, stride: usize, t: f64) -> Vec<f32> {
    if ch.times.is_empty() {
        return vec![0.0; stride];
    }

    let t = t as f32;

    if t <= ch.times[0] {
        return ch.values[..stride].to_vec();
    }
    if t >= *ch.times.last().unwrap() {
        let n = ch.values.len();
        return ch.values[n - stride..].to_vec();
    }

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
                slerp(a, b, frac)
            } else {
                a.iter().zip(b.iter())
                    .map(|(&va, &vb)| va + frac * (vb - va))
                    .collect()
            }
        }
        Interpolation::CubicSpline => {
            let a_val = &ch.values[idx * stride * 3 + stride..idx * stride * 3 + 2 * stride];
            let b_val = &ch.values[next * stride * 3 + stride..next * stride * 3 + 2 * stride];
            a_val.iter().zip(b_val.iter())
                .map(|(&va, &vb)| va + frac * (vb - va))
                .collect()
        }
    }
}

/// Evaluate morph target weights at time t.
pub fn evaluate_morph_weights(ch: &AnimationChannel, num_targets: usize, t: f64) -> Vec<f64> {
    let values = interpolate_channel(ch, num_targets, t);
    values.iter().map(|&v| v as f64).collect()
}

/// Quaternion slerp.
pub fn slerp(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let mut dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let mut b = b.to_vec();
    if dot < 0.0 {
        dot = -dot;
        for v in &mut b { *v = -*v; }
    }
    if dot > 0.9995 {
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

/// Multiply two column-major 4x4 matrices.
pub fn mat4_mul(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
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

/// Evaluate a skinned mesh at a single time — no baking, just one frame.
/// Returns the skinned vertex positions at time t.
pub fn evaluate_skinned_at_time(
    primitive: &crate::mesh::Primitive,
    skin: &Skin,
    animation: &Animation,
    nodes: &[crate::scene::Node],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn animation_time_range_basic() {
        let anim = Animation {
            name: Some("test".into()),
            channels: vec![
                AnimationChannel {
                    target_node: 0,
                    property: AnimationProperty::Translation,
                    interpolation: Interpolation::Linear,
                    times: vec![0.5, 2.0],
                    values: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                },
            ],
        };
        let (t_min, t_max) = animation_time_range(&anim);
        assert!((t_min - 0.5).abs() < 1e-6);
        assert!((t_max - 2.0).abs() < 1e-6);
    }

    #[test]
    fn interpolate_linear_translation() {
        let ch = AnimationChannel {
            target_node: 0,
            property: AnimationProperty::Translation,
            interpolation: Interpolation::Linear,
            times: vec![0.0, 1.0],
            values: vec![0.0, 0.0, 0.0, 2.0, 4.0, 6.0],
        };
        let v = interpolate_channel(&ch, 3, 0.5);
        assert!((v[0] - 1.0).abs() < 1e-4);
        assert!((v[1] - 2.0).abs() < 1e-4);
        assert!((v[2] - 3.0).abs() < 1e-4);
    }
}
