use super::{
    surface_walk::smooth_surface_direction, CameraBasis, CameraError, CameraRig,
    SphereReflectionState,
};
use quilting_core::Quat;
use serde::{Deserialize, Serialize};

const EPSILON: f64 = 1.0e-12;

/// Curves have explicit names so authored presentations and input replays do
/// not depend on a browser's animation implementation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionEasing {
    Linear,
    SmoothStep,
    #[default]
    SmootherStep,
}

impl TransitionEasing {
    pub fn sample(self, linear: f64) -> f64 {
        let t = linear.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::SmoothStep => t * t * (3.0 - 2.0 * t),
            Self::SmootherStep => t * t * t * (t * (t * 6.0 - 15.0) + 10.0),
        }
    }
}

/// A deterministic transition between complete semantic camera states.
/// Positions and lens angle are linear, positive scale-like quantities are
/// logarithmic, and orientation follows the shortest quaternion arc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraTransition {
    pub start: CameraRig,
    pub target: CameraRig,
    pub elapsed_seconds: f64,
    pub duration_seconds: f64,
    pub easing: TransitionEasing,
}

/// A camera target sampled from an attached surface in the active output
/// chart. The unit normal is kept beside the camera because it controls the
/// direction of the re-anchor hop, but it is not part of the camera pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceAnchorTarget {
    pub camera: CameraRig,
    pub normal: [f64; 3],
}

impl SurfaceAnchorTarget {
    pub fn new(mut camera: CameraRig, normal: [f64; 3]) -> Result<Self, CameraError> {
        camera.validate()?;
        camera.semantic_target = None;
        let normal = normalize3(normal).ok_or(CameraError::DegenerateBasis)?;
        Ok(Self { camera, normal })
    }
}

/// Deterministic camera glide used when a surface walker is attached to a new
/// address. The destination may be refreshed while an animated surface moves;
/// elapsed time, the starting pose, and hop height remain stable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceAnchorTransition {
    pub start: CameraRig,
    pub target: SurfaceAnchorTarget,
    pub elapsed_seconds: f64,
    pub duration_seconds: f64,
    pub easing: TransitionEasing,
    pub hop_height: f64,
}

impl SurfaceAnchorTransition {
    pub fn new(
        mut start: CameraRig,
        target: SurfaceAnchorTarget,
        scene_radius: f64,
        duration_seconds: f64,
        easing: TransitionEasing,
    ) -> Result<Self, CameraError> {
        start.validate()?;
        if !scene_radius.is_finite()
            || scene_radius <= 0.0
            || !duration_seconds.is_finite()
            || duration_seconds <= 0.0
        {
            return Err(CameraError::InvalidTransition);
        }
        let target = SurfaceAnchorTarget::new(target.camera, target.normal)?;
        start.semantic_target = None;
        let hop_height = (distance3(start.eye, target.camera.eye) * 0.22).min(scene_radius * 0.12);
        Ok(Self {
            start,
            target,
            elapsed_seconds: 0.0,
            duration_seconds,
            easing,
            hop_height,
        })
    }

    /// Refresh the animated destination without restarting the glide.
    pub fn retarget(&mut self, target: SurfaceAnchorTarget) -> Result<(), CameraError> {
        self.target = SurfaceAnchorTarget::new(target.camera, target.normal)?;
        Ok(())
    }

    pub fn advance(&mut self, delta_seconds: f64, camera: &mut CameraRig) -> bool {
        if !delta_seconds.is_finite() || delta_seconds <= 0.0 {
            return false;
        }
        self.elapsed_seconds += delta_seconds;
        let linear = transition_progress(self.elapsed_seconds, self.duration_seconds);
        if linear >= 1.0 {
            self.elapsed_seconds = self.duration_seconds;
        }
        *camera = self.sample(linear);
        linear >= 1.0
    }

    pub fn sample(&self, linear: f64) -> CameraRig {
        if linear >= 1.0 {
            return self.target.camera;
        }
        if linear <= 0.0 {
            let mut start = self.start;
            // The incumbent applies scale-relative clipping immediately;
            // only eye/orientation glide to the new contact.
            start.control_distance = self.target.camera.control_distance;
            start.lens = self.target.camera.lens;
            return start;
        }
        let t = self.easing.sample(linear);
        let start_basis = self.start.basis();
        let target_basis = self.target.camera.basis();
        let basis = smooth_surface_direction(start_basis.forward, target_basis.forward, t)
            .and_then(|forward| {
                smooth_surface_direction(start_basis.up, target_basis.up, t).and_then(|up_hint| {
                    let right = normalize3(cross3(forward, up_hint))?;
                    let up = normalize3(cross3(right, forward))?;
                    Some(CameraBasis { right, up, forward })
                })
            })
            .unwrap_or(target_basis);
        let hop = (std::f64::consts::PI * t).sin() * self.hop_height;
        let eye = add3(
            lerp3(self.start.eye, self.target.camera.eye, t),
            scale3(self.target.normal, hop),
        );
        CameraRig::new(
            eye,
            basis,
            self.target.camera.control_distance,
            None,
            self.target.camera.lens,
        )
        .unwrap_or(self.target.camera)
    }
}

impl CameraTransition {
    pub fn new(
        mut start: CameraRig,
        target: CameraRig,
        duration_seconds: f64,
        easing: TransitionEasing,
    ) -> Result<Self, CameraError> {
        start.validate()?;
        target.validate()?;
        if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
            return Err(CameraError::InvalidTransition);
        }
        if target.semantic_target.is_none() {
            // The pose is unchanged; only the representation of its sight line
            // switches from a finite point to the orientation's tangent.
            start.semantic_target = None;
        }
        Ok(Self {
            start,
            target,
            elapsed_seconds: 0.0,
            duration_seconds,
            easing,
        })
    }

    pub fn advance(&mut self, delta_seconds: f64, camera: &mut CameraRig) -> bool {
        if !delta_seconds.is_finite() || delta_seconds <= 0.0 {
            return false;
        }
        self.elapsed_seconds += delta_seconds;
        let linear = transition_progress(self.elapsed_seconds, self.duration_seconds);
        if linear >= 1.0 {
            self.elapsed_seconds = self.duration_seconds;
        }
        *camera = self.sample(linear);
        linear >= 1.0
    }

    pub fn sample(&self, linear: f64) -> CameraRig {
        if linear >= 1.0 {
            return self.target;
        }
        if linear <= 0.0 {
            return self.start;
        }
        let t = self.easing.sample(linear);
        let start_target = self.start.view_target();
        let target_target = self.target.view_target();
        let mut camera = interpolate_camera(self.start, self.target, t);
        camera.semantic_target = self
            .target
            .semantic_target
            .is_some()
            .then(|| lerp3(start_target, target_target, t));
        camera
    }

    /// Keep both endpoints in the same output chart when an active inversion
    /// sphere moves during the transition.
    pub fn transport_between_reflections(
        &mut self,
        previous: SphereReflectionState,
        next: SphereReflectionState,
    ) -> Result<(), CameraError> {
        let mut start = self.start;
        let mut target = self.target;
        start.transport_between_reflections(previous, next)?;
        target.transport_between_reflections(previous, next)?;
        self.start = start;
        self.target = target;
        Ok(())
    }
}

fn interpolate_camera(start: CameraRig, target: CameraRig, t: f64) -> CameraRig {
    CameraRig {
        eye: lerp3(start.eye, target.eye, t),
        orientation: quaternion_slerp(start.orientation, target.orientation, t),
        control_distance: log_lerp(start.control_distance, target.control_distance, t),
        semantic_target: None,
        lens: super::PerspectiveLens {
            vertical_fov_radians: start.lens.vertical_fov_radians
                + (target.lens.vertical_fov_radians - start.lens.vertical_fov_radians) * t,
            near: log_lerp(start.lens.near, target.lens.near, t),
            far: log_lerp(start.lens.far, target.lens.far, t),
        },
    }
}

fn transition_progress(elapsed_seconds: f64, duration_seconds: f64) -> f64 {
    let linear = (elapsed_seconds / duration_seconds).clamp(0.0, 1.0);
    if linear >= 1.0 - EPSILON {
        1.0
    } else {
        linear
    }
}

fn lerp3(start: [f64; 3], target: [f64; 3], t: f64) -> [f64; 3] {
    std::array::from_fn(|axis| start[axis] + (target[axis] - start[axis]) * t)
}

fn log_lerp(start: f64, target: f64, t: f64) -> f64 {
    (start.ln() + (target.ln() - start.ln()) * t).exp()
}

fn add3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] + right[axis])
}

fn scale3(value: [f64; 3], amount: f64) -> [f64; 3] {
    std::array::from_fn(|axis| value[axis] * amount)
}

fn cross3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn distance3(left: [f64; 3], right: [f64; 3]) -> f64 {
    (left
        .into_iter()
        .zip(right)
        .map(|(left, right)| (right - left).powi(2))
        .sum::<f64>())
    .sqrt()
}

fn normalize3(value: [f64; 3]) -> Option<[f64; 3]> {
    if value.into_iter().any(|component| !component.is_finite()) {
        return None;
    }
    let length = value
        .into_iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    (length > EPSILON).then(|| scale3(value, length.recip()))
}

fn quaternion_slerp(start: Quat, target: Quat, t: f64) -> Quat {
    let start = start / start.norm();
    let mut target = target / target.norm();
    let mut cosine = start.dot(target);
    if cosine < 0.0 {
        target = -target;
        cosine = -cosine;
    }
    if cosine > 1.0 - 1.0e-8 {
        return normalize_quaternion(start.lerp(target, t));
    }
    let angle = cosine.clamp(-1.0, 1.0).acos();
    let denominator = angle.sin();
    if denominator.abs() <= EPSILON {
        return start;
    }
    normalize_quaternion(
        start * (((1.0 - t) * angle).sin() / denominator)
            + target * ((t * angle).sin() / denominator),
    )
}

fn normalize_quaternion(value: Quat) -> Quat {
    let mut normalized = value / value.norm();
    if normalized.w < 0.0 {
        normalized = -normalized;
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CameraBasis, PerspectiveLens};

    #[test]
    fn camera_transition_uses_shortest_rotation_and_log_distance() {
        let start = CameraRig::default();
        let target = CameraRig::new(
            [2.0, 4.0, 6.0],
            CameraBasis {
                right: [-1.0, 0.0, 0.0],
                up: [0.0, 1.0, 0.0],
                forward: [0.0, 0.0, 1.0],
            },
            12.0,
            Some([2.0, 4.0, 18.0]),
            PerspectiveLens::default(),
        )
        .unwrap();
        let transition =
            CameraTransition::new(start, target, 2.0, TransitionEasing::Linear).unwrap();
        let midpoint = transition.sample(0.5);
        assert_eq!(midpoint.eye, [1.0, 2.0, 4.5]);
        assert!((midpoint.control_distance - 6.0).abs() < 1.0e-10);
        assert!((midpoint.orientation.norm() - 1.0).abs() < 1.0e-10);
        assert_eq!(transition.sample(1.0), target);
    }

    #[test]
    fn target_free_transition_does_not_retain_a_finite_start_target() {
        let start = CameraRig {
            semantic_target: Some([0.0, 0.0, 0.0]),
            ..CameraRig::default()
        };
        let target = CameraRig {
            eye: [1.0, 2.0, 4.0],
            semantic_target: None,
            ..CameraRig::default()
        };
        let transition =
            CameraTransition::new(start, target, 1.0, TransitionEasing::Linear).unwrap();

        assert_eq!(transition.sample(0.0).semantic_target, None);
        assert_eq!(transition.sample(0.5).semantic_target, None);
        assert_eq!(transition.sample(1.0).semantic_target, None);
    }

    #[test]
    fn surface_anchor_glide_matches_the_browser_endpoints_and_hop() {
        let start = CameraRig {
            eye: [0.0, 0.0, 0.0],
            ..CameraRig::default()
        };
        let target = SurfaceAnchorTarget::new(
            CameraRig {
                eye: [10.0, 0.0, 0.0],
                ..CameraRig::default()
            },
            [0.0, 2.0, 0.0],
        )
        .unwrap();
        let transition =
            SurfaceAnchorTransition::new(start, target, 100.0, 1.0, TransitionEasing::SmootherStep)
                .unwrap();

        assert_eq!(transition.sample(0.0).eye, [0.0, 0.0, 0.0]);
        assert_eq!(transition.sample(0.5).eye, [5.0, 2.2, 0.0]);
        assert!((transition.sample(1.0).eye[0] - 10.0).abs() < 1.0e-12);
        assert!(transition.sample(1.0).eye[1].abs() < 1.0e-12);
    }

    #[test]
    fn surface_anchor_retarget_preserves_start_time_and_hop_height() {
        let start = CameraRig::default();
        let first = SurfaceAnchorTarget::new(
            CameraRig {
                eye: [3.0, 0.0, 3.0],
                ..start
            },
            [0.0, 1.0, 0.0],
        )
        .unwrap();
        let mut transition =
            SurfaceAnchorTransition::new(start, first, 10.0, 2.0, TransitionEasing::Linear)
                .unwrap();
        let hop_height = transition.hop_height;
        let second = SurfaceAnchorTarget::new(
            CameraRig {
                eye: [5.0, 0.0, 3.0],
                ..start
            },
            [0.0, 0.0, 2.0],
        )
        .unwrap();
        transition.retarget(second).unwrap();

        assert_eq!(transition.elapsed_seconds, 0.0);
        assert_eq!(transition.hop_height, hop_height);
        let midpoint = transition.sample(0.5);
        assert!((midpoint.eye[0] - 2.5).abs() < 1.0e-12);
        assert!((midpoint.eye[2] - (3.0 + hop_height)).abs() < 1.0e-12);
    }

    #[test]
    fn surface_anchor_matches_incumbent_basis_and_applies_walk_lens_immediately() {
        let start = CameraRig::default();
        let target_camera = CameraRig::new(
            [2.0, 0.0, 3.0],
            CameraBasis {
                right: [0.0, 0.0, -1.0],
                up: [1.0, 0.0, 0.0],
                forward: [0.0, 1.0, 0.0],
            },
            7.0,
            None,
            PerspectiveLens {
                vertical_fov_radians: 1.2,
                near: 0.000_4,
                far: 20_000.0,
            },
        )
        .unwrap();
        let transition = SurfaceAnchorTransition::new(
            start,
            SurfaceAnchorTarget::new(target_camera, [0.0, 1.0, 0.0]).unwrap(),
            10.0,
            1.0,
            TransitionEasing::Linear,
        )
        .unwrap();

        let initial = transition.sample(0.0);
        assert_eq!(initial.eye, start.eye);
        assert_eq!(initial.control_distance, target_camera.control_distance);
        assert_eq!(initial.lens, target_camera.lens);

        let midpoint = transition.sample(0.5);
        let basis = midpoint.basis();
        let expected_forward = [0.0, 2.0_f64.sqrt().recip(), -2.0_f64.sqrt().recip()];
        let expected_right = [
            3.0_f64.sqrt().recip(),
            -3.0_f64.sqrt().recip(),
            -3.0_f64.sqrt().recip(),
        ];
        let expected_up = [
            (2.0 / 3.0_f64).sqrt(),
            6.0_f64.sqrt().recip(),
            6.0_f64.sqrt().recip(),
        ];
        for axis in 0..3 {
            assert!((basis.forward[axis] - expected_forward[axis]).abs() < 1.0e-12);
            assert!((basis.right[axis] - expected_right[axis]).abs() < 1.0e-12);
            assert!((basis.up[axis] - expected_up[axis]).abs() < 1.0e-12);
        }
        assert_eq!(midpoint.control_distance, target_camera.control_distance);
        assert_eq!(midpoint.lens, target_camera.lens);
    }
}
