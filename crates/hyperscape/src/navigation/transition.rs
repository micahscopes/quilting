use super::{CameraError, CameraRig, SphereReflectionState};
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

impl CameraTransition {
    pub fn new(
        start: CameraRig,
        target: CameraRig,
        duration_seconds: f64,
        easing: TransitionEasing,
    ) -> Result<Self, CameraError> {
        start.validate()?;
        target.validate()?;
        if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
            return Err(CameraError::InvalidTransition);
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
        let linear = (self.elapsed_seconds / self.duration_seconds).clamp(0.0, 1.0);
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
        CameraRig {
            eye: lerp3(self.start.eye, self.target.eye, t),
            orientation: quaternion_slerp(self.start.orientation, self.target.orientation, t),
            control_distance: log_lerp(
                self.start.control_distance,
                self.target.control_distance,
                t,
            ),
            semantic_target: (self.start.semantic_target.is_some()
                || self.target.semantic_target.is_some())
            .then(|| lerp3(start_target, target_target, t)),
            lens: super::PerspectiveLens {
                vertical_fov_radians: self.start.lens.vertical_fov_radians
                    + (self.target.lens.vertical_fov_radians
                        - self.start.lens.vertical_fov_radians)
                        * t,
                near: log_lerp(self.start.lens.near, self.target.lens.near, t),
                far: log_lerp(self.start.lens.far, self.target.lens.far, t),
            },
        }
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

fn lerp3(start: [f64; 3], target: [f64; 3], t: f64) -> [f64; 3] {
    std::array::from_fn(|axis| start[axis] + (target[axis] - start[axis]) * t)
}

fn log_lerp(start: f64, target: f64, t: f64) -> f64 {
    (start.ln() + (target.ln() - start.ln()) * t).exp()
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
}
