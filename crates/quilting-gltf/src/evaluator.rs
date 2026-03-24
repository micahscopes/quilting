//! Lightweight per-frame animation evaluator for GPU skinning.
//!
//! Replaces the prebake path for animated models. Evaluates joint matrices
//! and morph weights at arbitrary time t, returning data ready for GPU upload.

use crate::animation::{Animation, AnimationProperty, Skin, animation_time_range, evaluate_joint_transforms, evaluate_morph_weights, mat4_mul};
use crate::scene::Node;

/// Metadata about a single animation clip.
#[derive(Debug, Clone)]
pub struct AnimationInfo {
    pub index: usize,
    pub name: Option<String>,
    pub duration: f64,
    pub t_min: f64,
    pub t_max: f64,
}

/// Per-frame pose output, ready for GPU upload.
#[derive(Debug, Clone)]
pub struct AnimationPose {
    /// Skin matrices (joint_world * inverse_bind) as column-major f32.
    /// Length: num_joints * 16. Empty if no skin.
    pub joint_matrices: Vec<f32>,
    /// Morph target blend weights. Empty if no morph targets.
    pub morph_weights: Vec<f32>,
    /// Number of joints (joint_matrices.len() / 16).
    pub num_joints: usize,
}

/// Evaluates animation at arbitrary time for GPU skinning.
///
/// Holds cloned animation/skin/node data so it can live in WASM state
/// independently of the original GltfScene.
pub struct AnimationEvaluator {
    animation: Animation,
    skin: Option<Skin>,
    nodes: Vec<Node>,
    t_min: f64,
    t_max: f64,
    num_morph_targets: usize,
}

impl AnimationEvaluator {
    /// Create a new evaluator from animation data.
    ///
    /// `skin` is None for morph-only animations.
    /// `num_morph_targets` is 0 if no morph targets are present.
    pub fn new(
        animation: Animation,
        skin: Option<Skin>,
        nodes: Vec<Node>,
        num_morph_targets: usize,
    ) -> Self {
        let (t_min, t_max) = animation_time_range(&animation);
        Self {
            animation,
            skin,
            nodes,
            t_min,
            t_max,
            num_morph_targets,
        }
    }

    /// Evaluate the animation at time `t`, returning joint matrices and morph weights.
    ///
    /// Time is wrapped to the animation range for looping.
    /// Joint matrices are skin matrices (world * inverse_bind), column-major f32.
    pub fn evaluate(&self, t: f64) -> AnimationPose {
        let t = self.wrap_time(t);

        let (joint_matrices, num_joints) = if let Some(ref skin) = self.skin {
            let joints = &skin.joints;
            let ibms = &skin.inverse_bind_matrices;
            let world_mats = evaluate_joint_transforms(&self.animation, &self.nodes, joints, t);

            // Compute skin matrices: world * inverse_bind, convert to f32
            let mut matrices = Vec::with_capacity(world_mats.len() * 16);
            for (jm, ibm) in world_mats.iter().zip(ibms.iter()) {
                let ibm_f64: [f64; 16] = std::array::from_fn(|i| ibm[i] as f64);
                let skin_mat = mat4_mul(jm, &ibm_f64);
                for &v in &skin_mat {
                    matrices.push(v as f32);
                }
            }
            (matrices, world_mats.len())
        } else {
            (Vec::new(), 0)
        };

        let morph_weights = if self.num_morph_targets > 0 {
            // Find the morph weight channel
            if let Some(ch) = self.animation.channels.iter()
                .find(|c| c.property == AnimationProperty::MorphTargetWeights)
            {
                let weights_f64 = evaluate_morph_weights(ch, self.num_morph_targets, t);
                weights_f64.iter().map(|&w| w as f32).collect()
            } else {
                vec![0.0f32; self.num_morph_targets]
            }
        } else {
            Vec::new()
        };

        AnimationPose {
            joint_matrices,
            morph_weights,
            num_joints,
        }
    }

    /// Animation duration in seconds.
    pub fn duration(&self) -> f64 {
        self.t_max - self.t_min
    }

    /// Animation name.
    pub fn name(&self) -> Option<&str> {
        self.animation.name.as_deref()
    }

    /// Number of joints (0 if no skin).
    pub fn num_joints(&self) -> usize {
        self.skin.as_ref().map_or(0, |s| s.joints.len())
    }

    /// Number of morph targets.
    pub fn num_morph_targets(&self) -> usize {
        self.num_morph_targets
    }

    /// Time range (t_min, t_max).
    pub fn time_range(&self) -> (f64, f64) {
        (self.t_min, self.t_max)
    }

    /// Wrap time to the animation range for looping.
    fn wrap_time(&self, t: f64) -> f64 {
        let duration = self.duration();
        if duration <= 0.0 {
            return self.t_min;
        }
        let offset = t - self.t_min;
        let wrapped = offset.rem_euclid(duration);
        self.t_min + wrapped
    }
}

/// Get metadata for all animations in a scene.
pub fn list_animations(
    animations: &[Animation],
) -> Vec<AnimationInfo> {
    animations.iter().enumerate().map(|(i, anim)| {
        let (t_min, t_max) = animation_time_range(anim);
        AnimationInfo {
            index: i,
            name: anim.name.clone(),
            duration: t_max - t_min,
            t_min,
            t_max,
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{AnimationChannel, Interpolation};
    use crate::scene::Transform;

    fn make_test_nodes() -> Vec<Node> {
        vec![
            Node {
                name: Some("root".into()),
                transform: Transform::Trs {
                    translation: [0.0, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
                children: vec![1],
                mesh: None,
                skin: None,
            },
            Node {
                name: Some("bone".into()),
                transform: Transform::Trs {
                    translation: [0.0, 1.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
                children: vec![],
                mesh: Some(0),
                skin: Some(0),
            },
        ]
    }

    fn identity_mat() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    #[test]
    fn evaluator_identity_pose() {
        let anim = Animation {
            name: Some("idle".into()),
            channels: vec![
                AnimationChannel {
                    target_node: 0,
                    property: AnimationProperty::Translation,
                    interpolation: Interpolation::Linear,
                    times: vec![0.0, 1.0],
                    values: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
            ],
        };

        let skin = Skin {
            name: None,
            joints: vec![0],
            inverse_bind_matrices: vec![identity_mat()],
            skeleton_root: Some(0),
        };

        let eval = AnimationEvaluator::new(anim, Some(skin), make_test_nodes(), 0);
        assert_eq!(eval.duration(), 1.0);
        assert_eq!(eval.num_joints(), 1);
        assert_eq!(eval.name(), Some("idle"));

        let pose = eval.evaluate(0.0);
        assert_eq!(pose.num_joints, 1);
        assert_eq!(pose.joint_matrices.len(), 16);
        assert!(pose.morph_weights.is_empty());
    }

    #[test]
    fn evaluator_time_wrapping() {
        let anim = Animation {
            name: None,
            channels: vec![
                AnimationChannel {
                    target_node: 0,
                    property: AnimationProperty::Translation,
                    interpolation: Interpolation::Linear,
                    times: vec![0.0, 2.0],
                    values: vec![0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
                },
            ],
        };

        let skin = Skin {
            name: None,
            joints: vec![0],
            inverse_bind_matrices: vec![identity_mat()],
            skeleton_root: Some(0),
        };

        let eval = AnimationEvaluator::new(anim, Some(skin), make_test_nodes(), 0);

        // t=1.0 should give translation (2, 0, 0) — halfway
        let pose_mid = eval.evaluate(1.0);
        // Translation goes into column 3 (indices 12,13,14) of the joint matrix
        assert!((pose_mid.joint_matrices[12] - 2.0).abs() < 1e-4, "expected ~2.0, got {}", pose_mid.joint_matrices[12]);

        // t=3.0 wraps to t=1.0, same result
        let pose_wrapped = eval.evaluate(3.0);
        assert!((pose_wrapped.joint_matrices[12] - 2.0).abs() < 1e-4, "expected ~2.0, got {}", pose_wrapped.joint_matrices[12]);

        // t=-1.0 wraps to t=1.0 (rem_euclid)
        let pose_neg = eval.evaluate(-1.0);
        assert!((pose_neg.joint_matrices[12] - 2.0).abs() < 1e-4, "expected ~2.0, got {}", pose_neg.joint_matrices[12]);
    }

    #[test]
    fn evaluator_morph_weights() {
        let anim = Animation {
            name: Some("blink".into()),
            channels: vec![
                AnimationChannel {
                    target_node: 0,
                    property: AnimationProperty::MorphTargetWeights,
                    interpolation: Interpolation::Linear,
                    times: vec![0.0, 1.0],
                    values: vec![0.0, 0.0, 1.0, 0.5],
                },
            ],
        };

        let eval = AnimationEvaluator::new(anim, None, make_test_nodes(), 2);
        assert_eq!(eval.num_morph_targets(), 2);
        assert_eq!(eval.num_joints(), 0);

        let pose = eval.evaluate(0.5);
        assert!(pose.joint_matrices.is_empty());
        assert_eq!(pose.morph_weights.len(), 2);
        // Halfway: [0.5, 0.25]
        assert!((pose.morph_weights[0] - 0.5).abs() < 1e-4);
        assert!((pose.morph_weights[1] - 0.25).abs() < 1e-4);
    }

    #[test]
    fn list_animations_metadata() {
        let anims = vec![
            Animation {
                name: Some("walk".into()),
                channels: vec![AnimationChannel {
                    target_node: 0,
                    property: AnimationProperty::Translation,
                    interpolation: Interpolation::Linear,
                    times: vec![0.0, 2.0],
                    values: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                }],
            },
            Animation {
                name: None,
                channels: vec![AnimationChannel {
                    target_node: 0,
                    property: AnimationProperty::Rotation,
                    interpolation: Interpolation::Step,
                    times: vec![0.0, 0.5, 1.5],
                    values: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.707, 0.0, 0.707, 0.0, 0.0, 0.0, 1.0],
                }],
            },
        ];

        let info = list_animations(&anims);
        assert_eq!(info.len(), 2);

        assert_eq!(info[0].index, 0);
        assert_eq!(info[0].name.as_deref(), Some("walk"));
        assert!((info[0].duration - 2.0).abs() < 1e-6);

        assert_eq!(info[1].index, 1);
        assert!(info[1].name.is_none());
        assert!((info[1].duration - 1.5).abs() < 1e-6);
    }
}
