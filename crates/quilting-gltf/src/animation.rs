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

/// A single keyframe: time + value.
#[derive(Debug, Clone)]
pub struct Keyframe {
    pub time: f32,
    pub value: Vec<f32>,
}

/// Per-joint transform trajectory for spacetime effects.
#[derive(Debug, Clone)]
pub struct JointTrajectory {
    /// Index of the joint node in GltfScene::nodes.
    pub node_index: usize,
    /// Keyframes for translation (if animated).
    pub translations: Option<Vec<Keyframe>>,
    /// Keyframes for rotation as [x,y,z,w] quaternions (if animated).
    pub rotations: Option<Vec<Keyframe>>,
    /// Keyframes for scale (if animated).
    pub scales: Option<Vec<Keyframe>>,
}

/// Per-vertex position trajectory for morph target playback.
#[derive(Debug, Clone)]
pub struct VertexTrajectory {
    /// Blend weight keyframes — each value is a Vec of weights, one per morph target.
    pub weight_keyframes: Vec<Keyframe>,
    /// Morph target position deltas. morph_deltas[target_index][vertex_index] = position delta.
    pub morph_deltas: Vec<Vec<[f64; 3]>>,
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

/// Build per-joint trajectories from an animation, for spacetime FX.
///
/// Groups the animation channels by target node and separates TRS channels
/// into a per-joint structure.
pub fn build_joint_trajectories(animation: &Animation) -> Vec<JointTrajectory> {
    // Collect all unique target nodes.
    let mut node_set: Vec<usize> = animation
        .channels
        .iter()
        .filter(|c| c.property != AnimationProperty::MorphTargetWeights)
        .map(|c| c.target_node)
        .collect();
    node_set.sort_unstable();
    node_set.dedup();

    node_set
        .iter()
        .map(|&node_index| {
            let mut traj = JointTrajectory {
                node_index,
                translations: None,
                rotations: None,
                scales: None,
            };

            for ch in &animation.channels {
                if ch.target_node != node_index {
                    continue;
                }
                let stride = property_stride(ch.property);
                let keyframes: Vec<Keyframe> = ch
                    .times
                    .iter()
                    .enumerate()
                    .map(|(i, &t)| {
                        let start = i * stride;
                        let end = start + stride;
                        Keyframe {
                            time: t,
                            value: ch.values[start..end].to_vec(),
                        }
                    })
                    .collect();

                match ch.property {
                    AnimationProperty::Translation => traj.translations = Some(keyframes),
                    AnimationProperty::Rotation => traj.rotations = Some(keyframes),
                    AnimationProperty::Scale => traj.scales = Some(keyframes),
                    AnimationProperty::MorphTargetWeights => {}
                }
            }

            traj
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_trajectories_groups_by_node() {
        let anim = Animation {
            name: Some("walk".into()),
            channels: vec![
                AnimationChannel {
                    target_node: 0,
                    property: AnimationProperty::Translation,
                    interpolation: Interpolation::Linear,
                    times: vec![0.0, 1.0],
                    values: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                },
                AnimationChannel {
                    target_node: 0,
                    property: AnimationProperty::Rotation,
                    interpolation: Interpolation::Linear,
                    times: vec![0.0, 1.0],
                    values: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.707, 0.0, 0.707],
                },
                AnimationChannel {
                    target_node: 1,
                    property: AnimationProperty::Scale,
                    interpolation: Interpolation::Step,
                    times: vec![0.0],
                    values: vec![1.0, 1.0, 1.0],
                },
            ],
        };

        let trajectories = build_joint_trajectories(&anim);
        assert_eq!(trajectories.len(), 2);

        // Node 0 should have translation + rotation.
        let t0 = &trajectories[0];
        assert_eq!(t0.node_index, 0);
        assert!(t0.translations.is_some());
        assert!(t0.rotations.is_some());
        assert!(t0.scales.is_none());
        assert_eq!(t0.translations.as_ref().unwrap().len(), 2);

        // Node 1 should have scale only.
        let t1 = &trajectories[1];
        assert_eq!(t1.node_index, 1);
        assert!(t1.translations.is_none());
        assert!(t1.scales.is_some());
        assert_eq!(t1.scales.as_ref().unwrap().len(), 1);
    }
}
