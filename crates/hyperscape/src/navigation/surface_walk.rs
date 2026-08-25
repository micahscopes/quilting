use super::{CameraBasis, CameraError, CameraRig, SphereReflectionState};
use crate::SurfaceContact;
use std::error::Error;
use std::fmt;

const EPSILON: f64 = 1.0e-12;
const SURFACE_FRAME_EPSILON: f64 = 1.0e-8;

/// Device-independent walking controls.
///
/// The octave values use 100 steps per doubling so a URL slider, SpaceMouse,
/// keyboard, replay, or Blender adapter can all emit the same semantic state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceWalkControls {
    pub base_radii_per_second: f64,
    pub base_eye_height: f64,
    pub speed_octave_steps: f64,
    pub body_scale_octave_steps: f64,
    pub eye_height_octave_steps: f64,
    pub smoothing_seconds: f64,
    pub tangent_pull_fraction: f64,
    pub fast_multiplier: f64,
    pub default_near: f64,
    pub minimum_near: f64,
    pub near_eye_fraction: f64,
}

impl Default for SurfaceWalkControls {
    fn default() -> Self {
        Self {
            base_radii_per_second: 0.2,
            base_eye_height: 0.035,
            speed_octave_steps: 0.0,
            body_scale_octave_steps: 0.0,
            eye_height_octave_steps: 0.0,
            smoothing_seconds: 0.18,
            tangent_pull_fraction: 0.7,
            fast_multiplier: 3.0,
            default_near: 0.01,
            minimum_near: 1.0e-7,
            near_eye_fraction: 0.08,
        }
    }
}

impl SurfaceWalkControls {
    pub fn metrics(
        self,
        scene_radius: f64,
        fast: bool,
    ) -> Result<SurfaceWalkMetrics, SurfaceWalkError> {
        if !self.is_finite() {
            return Err(SurfaceWalkError::NonFinite);
        }
        if scene_radius <= 0.0
            || self.base_radii_per_second < 0.0
            || self.base_eye_height <= 0.0
            || self.smoothing_seconds < 0.0
            || !(0.0..=1.0).contains(&self.tangent_pull_fraction)
            || self.fast_multiplier <= 0.0
            || self.default_near <= 0.0
            || self.minimum_near <= 0.0
            || self.minimum_near > self.default_near
            || self.near_eye_fraction <= 0.0
        {
            return Err(SurfaceWalkError::InvalidControls);
        }
        let body_scale = octave_scale(self.body_scale_octave_steps)?;
        let pace_scale = octave_scale(self.speed_octave_steps)?;
        let height_scale = octave_scale(self.eye_height_octave_steps)?;
        let fast_scale = if fast { self.fast_multiplier } else { 1.0 };
        let radii_per_second = self.base_radii_per_second * pace_scale;
        let speed = scene_radius * radii_per_second * body_scale * fast_scale;
        let eye_height = self.base_eye_height * body_scale * height_scale;
        if !radii_per_second.is_finite()
            || !speed.is_finite()
            || !eye_height.is_finite()
            || eye_height <= 0.0
        {
            return Err(SurfaceWalkError::InvalidControls);
        }
        let near = scale_relative_near_plane(
            eye_height,
            self.default_near,
            self.minimum_near,
            self.near_eye_fraction,
        )
        .ok_or(SurfaceWalkError::InvalidControls)?;
        Ok(SurfaceWalkMetrics {
            body_scale,
            radii_per_second,
            speed,
            eye_height,
            near,
        })
    }

    fn is_finite(self) -> bool {
        [
            self.base_radii_per_second,
            self.base_eye_height,
            self.speed_octave_steps,
            self.body_scale_octave_steps,
            self.eye_height_octave_steps,
            self.smoothing_seconds,
            self.tangent_pull_fraction,
            self.fast_multiplier,
            self.default_near,
            self.minimum_near,
            self.near_eye_fraction,
        ]
        .into_iter()
        .all(f64::is_finite)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceWalkMetrics {
    pub body_scale: f64,
    pub radii_per_second: f64,
    pub speed: f64,
    pub eye_height: f64,
    pub near: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SurfaceWalkInput {
    pub forward_axis: f64,
    pub right_axis: f64,
    pub fast: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceWalkMotion {
    pub desired_output_velocity: [f64; 3],
    pub tangent_forward: [f64; 3],
    pub tangent_right: [f64; 3],
    pub metrics: SurfaceWalkMetrics,
}

/// Geometry-independent contact consumed by the semantic view controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceWalkContactFrame {
    pub output_position: [f64; 3],
    pub output_normal: [f64; 3],
}

impl From<&SurfaceContact> for SurfaceWalkContactFrame {
    fn from(value: &SurfaceContact) -> Self {
        Self {
            output_position: value.output_position,
            output_normal: value.output_normal,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceRelativeView {
    pub tangent: [f64; 3],
    pub pitch_radians: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceWalkFrame {
    pub camera: CameraRig,
    pub filtered_position: [f64; 3],
    pub filtered_normal: [f64; 3],
    pub tangent_forward: Option<[f64; 3]>,
    pub relative_pitch_radians: Option<f64>,
    pub metrics: SurfaceWalkMetrics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceWalkError {
    NonFinite,
    InvalidControls,
    InvalidDelta,
    DegenerateFrame,
    Camera(CameraError),
}

impl fmt::Display for SurfaceWalkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => formatter.write_str("surface-walk values must be finite"),
            Self::InvalidControls => formatter.write_str("surface-walk controls are invalid"),
            Self::InvalidDelta => formatter.write_str("surface-walk delta must be nonnegative"),
            Self::DegenerateFrame => {
                formatter.write_str("surface-walk tangent frame is degenerate")
            }
            Self::Camera(error) => write!(formatter, "surface-walk camera is invalid: {error}"),
        }
    }
}

impl Error for SurfaceWalkError {}

impl From<CameraError> for SurfaceWalkError {
    fn from(value: CameraError) -> Self {
        Self::Camera(value)
    }
}

/// Stateful camera follower for an animated surface attachment.
///
/// Surface evaluation/topology remains in [`crate::SurfaceWalker`]. This
/// controller owns only the semantic locomotion and view response that should
/// be identical in a browser, native game, replay, or Blender preview.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SurfaceWalkController {
    state: Option<SurfaceWalkState>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SurfaceWalkState {
    filtered_position: [f64; 3],
    filtered_normal: [f64; 3],
    tangent_forward: Option<[f64; 3]>,
    relative_pitch_radians: Option<f64>,
}

impl SurfaceWalkController {
    pub fn is_active(&self) -> bool {
        self.state.is_some()
    }

    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Carry retained contact filtering and heading through an exact
    /// spherical-reflection chart change. Relative pitch is intrinsic to the
    /// surface frame and therefore remains unchanged. A pole rejects the
    /// whole operation without mutating the follower.
    pub fn transport_between_reflections(
        &mut self,
        previous: SphereReflectionState,
        next: SphereReflectionState,
    ) -> Result<bool, CameraError> {
        let Some(state) = self.state else {
            return Ok(false);
        };
        if previous == next {
            return Ok(false);
        }
        let tangent = state.tangent_forward.unwrap_or(state.filtered_normal);
        let transported = previous.transport_point_and_directions(
            next,
            state.filtered_position,
            [state.filtered_normal, tangent],
        )?;
        let filtered_normal =
            normalize_surface(transported.directions[0]).ok_or(CameraError::DegenerateBasis)?;
        let tangent_forward = state
            .tangent_forward
            .map(|_| {
                project_surface_tangent(transported.directions[1], filtered_normal)
                    .ok_or(CameraError::DegenerateBasis)
            })
            .transpose()?;
        self.state = Some(SurfaceWalkState {
            filtered_position: transported.point,
            filtered_normal,
            tangent_forward,
            relative_pitch_radians: state.relative_pitch_radians,
        });
        Ok(true)
    }

    /// Map semantic forward/right input into displayed-chart velocity.
    /// Diagonal and combined-device input is length-limited exactly once.
    pub fn plan_motion(
        &self,
        camera: &CameraRig,
        output_normal: [f64; 3],
        scene_radius: f64,
        controls: SurfaceWalkControls,
        input: SurfaceWalkInput,
    ) -> Result<SurfaceWalkMotion, SurfaceWalkError> {
        camera.validate()?;
        if !finite3(output_normal)
            || !input.forward_axis.is_finite()
            || !input.right_axis.is_finite()
        {
            return Err(SurfaceWalkError::NonFinite);
        }
        let normal = normalize_surface(output_normal).ok_or(SurfaceWalkError::DegenerateFrame)?;
        let metrics = controls.metrics(scene_radius, input.fast)?;
        let basis = camera.basis();
        let forward = project_surface_tangent(basis.forward, normal)
            .or_else(|| project_surface_tangent(basis.up, normal))
            .ok_or(SurfaceWalkError::DegenerateFrame)?;
        let right = project_surface_tangent(basis.right, normal)
            .or_else(|| normalize_surface(cross(forward, normal)))
            .ok_or(SurfaceWalkError::DegenerateFrame)?;
        let input_length = input.forward_axis.hypot(input.right_axis).max(1.0);
        let velocity = scale(
            add(
                scale(forward, input.forward_axis),
                scale(right, input.right_axis),
            ),
            metrics.speed / input_length,
        );
        if !finite3(velocity) {
            return Err(SurfaceWalkError::InvalidControls);
        }
        Ok(SurfaceWalkMotion {
            desired_output_velocity: velocity,
            tangent_forward: forward,
            tangent_right: right,
            metrics,
        })
    }

    /// Follow the latest animated contact and produce a complete camera rig.
    ///
    /// Set `capture_relative_view` when a user pitch/yaw input changed the
    /// incoming camera. `orient = false` updates eye/contact filtering while
    /// retaining orientation, which is useful during conformal frame edits.
    /// Invalid contact/view data is rejected without mutating this response
    /// state; the attachment owner must then explicitly detach/reset or retry.
    /// The future composed walker action will make that lifecycle atomic.
    #[allow(clippy::too_many_arguments)]
    pub fn follow_contact(
        &mut self,
        camera: &CameraRig,
        contact: &SurfaceContact,
        scene_radius: f64,
        controls: SurfaceWalkControls,
        delta_seconds: f64,
        orient: bool,
        capture_relative_view: bool,
    ) -> Result<SurfaceWalkFrame, SurfaceWalkError> {
        self.follow_frame(
            camera,
            SurfaceWalkContactFrame::from(contact),
            scene_radius,
            controls,
            delta_seconds,
            orient,
            capture_relative_view,
        )
    }

    /// Geometry-independent form used by replay and renderer adapters that
    /// have already projected a stable attachment into the output chart.
    #[allow(clippy::too_many_arguments)]
    pub fn follow_frame(
        &mut self,
        camera: &CameraRig,
        contact: SurfaceWalkContactFrame,
        scene_radius: f64,
        controls: SurfaceWalkControls,
        delta_seconds: f64,
        orient: bool,
        capture_relative_view: bool,
    ) -> Result<SurfaceWalkFrame, SurfaceWalkError> {
        camera.validate()?;
        if !delta_seconds.is_finite() {
            return Err(SurfaceWalkError::NonFinite);
        }
        if delta_seconds < 0.0 {
            return Err(SurfaceWalkError::InvalidDelta);
        }
        if !finite3(contact.output_position) || !finite3(contact.output_normal) {
            return Err(SurfaceWalkError::NonFinite);
        }
        let target_normal =
            normalize_surface(contact.output_normal).ok_or(SurfaceWalkError::DegenerateFrame)?;
        let metrics = controls.metrics(scene_radius, false)?;
        let prior = self.state;
        let filter_amount =
            if prior.is_none() || delta_seconds <= 0.0 || controls.smoothing_seconds <= 0.0 {
                1.0
            } else {
                1.0 - (-delta_seconds / controls.smoothing_seconds).exp()
            };
        let filtered_position = prior.map_or(contact.output_position, |state| {
            lerp(
                state.filtered_position,
                contact.output_position,
                filter_amount,
            )
        });
        let filtered_normal = prior
            .and_then(|state| {
                smooth_surface_direction(state.filtered_normal, target_normal, filter_amount)
            })
            .unwrap_or(target_normal);
        let eye = add(
            filtered_position,
            scale(filtered_normal, metrics.eye_height),
        );

        let mut tangent_forward = prior.and_then(|state| state.tangent_forward);
        let mut relative_pitch_radians = prior.and_then(|state| state.relative_pitch_radians);
        let basis = camera.basis();
        let final_basis = if orient {
            let capture = prior.is_none()
                || capture_relative_view
                || relative_pitch_radians.is_none()
                || tangent_forward.is_none();
            let relative_view = capture.then(|| {
                decompose_surface_relative_forward(
                    basis.forward,
                    filtered_normal,
                    tangent_forward.or(Some(basis.up)),
                )
            });
            let relative_view = relative_view.flatten();
            let tangent = relative_view
                .map(|view| view.tangent)
                .or_else(|| {
                    tangent_forward
                        .and_then(|value| project_surface_tangent(value, filtered_normal))
                })
                .or_else(|| project_surface_tangent(basis.up, filtered_normal))
                .or_else(|| normalize_surface(cross(filtered_normal, [1.0, 0.0, 0.0])))
                .or_else(|| normalize_surface(cross(filtered_normal, [0.0, 0.0, 1.0])))
                .ok_or(SurfaceWalkError::DegenerateFrame)?;
            if let Some(relative_view) = relative_view {
                relative_pitch_radians = Some(relative_view.pitch_radians);
            }
            let target_forward = compose_surface_relative_forward(
                tangent,
                filtered_normal,
                relative_pitch_radians.unwrap_or(0.0),
            )
            .unwrap_or(tangent);
            let pull_amount = if prior.is_none() || delta_seconds <= 0.0 {
                1.0
            } else {
                1.0 - (-(controls.tangent_pull_fraction * 8.0) * delta_seconds).exp()
            };
            let forward = smooth_surface_direction(basis.forward, target_forward, pull_amount)
                .ok_or(SurfaceWalkError::DegenerateFrame)?;
            let right = normalize_surface(cross(forward, filtered_normal))
                .ok_or(SurfaceWalkError::DegenerateFrame)?;
            let up = normalize_surface(cross(right, forward))
                .ok_or(SurfaceWalkError::DegenerateFrame)?;
            tangent_forward = Some(tangent);
            CameraBasis { right, up, forward }
        } else {
            basis
        };

        let mut lens = camera.lens;
        lens.near = metrics.near;
        let camera = CameraRig::new(eye, final_basis, camera.control_distance, None, lens)?;
        let next_state = SurfaceWalkState {
            filtered_position,
            filtered_normal,
            tangent_forward,
            relative_pitch_radians,
        };
        let frame = SurfaceWalkFrame {
            camera,
            filtered_position,
            filtered_normal,
            tangent_forward,
            relative_pitch_radians,
            metrics,
        };
        self.state = Some(next_state);
        Ok(frame)
    }
}

/// Split a view direction into a tangent heading and pitch about a surface.
pub fn decompose_surface_relative_forward(
    forward: [f64; 3],
    normal: [f64; 3],
    tangent_hint: Option<[f64; 3]>,
) -> Option<SurfaceRelativeView> {
    let view = normalize(forward)?;
    let normal = normalize(normal)?;
    let pitch_sine = dot(view, normal).clamp(-1.0, 1.0);
    let tangent = project_tangent(view, normal)
        .or_else(|| tangent_hint.and_then(|hint| project_tangent(hint, normal)))?;
    Some(SurfaceRelativeView {
        tangent,
        pitch_radians: pitch_sine.asin(),
    })
}

/// Rebuild a view direction at the same pitch in a new surface frame.
pub fn compose_surface_relative_forward(
    tangent: [f64; 3],
    normal: [f64; 3],
    pitch_radians: f64,
) -> Option<[f64; 3]> {
    if !pitch_radians.is_finite() {
        return None;
    }
    let normal = normalize(normal)?;
    let heading = project_tangent(tangent, normal)?;
    let pitch = pitch_radians.clamp(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
    normalize(add(scale(heading, pitch.cos()), scale(normal, pitch.sin())))
}

/// Keep the near plane below the eye-to-surface offset at every body scale.
pub fn scale_relative_near_plane(
    eye_height: f64,
    default_near: f64,
    minimum_near: f64,
    eye_fraction: f64,
) -> Option<f64> {
    if !eye_height.is_finite()
        || !default_near.is_finite()
        || !minimum_near.is_finite()
        || !eye_fraction.is_finite()
        || default_near <= 0.0
        || minimum_near <= 0.0
        || eye_fraction <= 0.0
    {
        return None;
    }
    if eye_height == 0.0 {
        return Some(default_near);
    }
    Some(
        (eye_height.abs() * eye_fraction)
            .min(default_near)
            .max(minimum_near),
    )
}

fn octave_scale(steps: f64) -> Result<f64, SurfaceWalkError> {
    let value = (steps / 100.0).exp2();
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(SurfaceWalkError::InvalidControls)
    }
}

pub(super) fn smooth_surface_direction(
    from: [f64; 3],
    to: [f64; 3],
    amount: f64,
) -> Option<[f64; 3]> {
    let from = normalize_surface(from)?;
    let to = normalize_surface(to)?;
    if amount >= 1.0 {
        return Some(to);
    }
    if amount <= 0.0 {
        return Some(from);
    }
    let cosine = dot(from, to).clamp(-1.0, 1.0);
    if cosine > 0.9995 {
        return normalize_surface(lerp(from, to, amount)).or(Some(to));
    }
    if cosine < -0.9995 {
        let orthogonal = normalize_surface(cross(from, [1.0, 0.0, 0.0]))
            .or_else(|| normalize_surface(cross(from, [0.0, 1.0, 0.0])))?;
        return normalize_surface(add(
            scale(from, (std::f64::consts::PI * amount).cos()),
            scale(orthogonal, (std::f64::consts::PI * amount).sin()),
        ))
        .or(Some(to));
    }
    let angle = cosine.acos();
    let sine = angle.sin();
    if sine.abs() < EPSILON {
        return Some(to);
    }
    normalize_surface(add(
        scale(from, ((1.0 - amount) * angle).sin() / sine),
        scale(to, (amount * angle).sin() / sine),
    ))
    .or(Some(to))
}

fn project_tangent(value: [f64; 3], normal: [f64; 3]) -> Option<[f64; 3]> {
    normalize(sub(value, scale(normal, dot(value, normal))))
}

fn project_surface_tangent(value: [f64; 3], normal: [f64; 3]) -> Option<[f64; 3]> {
    normalize_surface(sub(value, scale(normal, dot(value, normal))))
}

fn finite3(value: [f64; 3]) -> bool {
    value.into_iter().all(f64::is_finite)
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize(value: [f64; 3]) -> Option<[f64; 3]> {
    let length = dot(value, value).sqrt();
    (length > EPSILON && length.is_finite()).then(|| scale(value, 1.0 / length))
}

fn normalize_surface(value: [f64; 3]) -> Option<[f64; 3]> {
    let length = dot(value, value).sqrt();
    (length > SURFACE_FRAME_EPSILON && length.is_finite()).then(|| scale(value, 1.0 / length))
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(value: [f64; 3], factor: f64) -> [f64; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn lerp(left: [f64; 3], right: [f64; 3], amount: f64) -> [f64; 3] {
    [
        left[0] + (right[0] - left[0]) * amount,
        left[1] + (right[1] - left[1]) * amount,
        left[2] + (right[2] - left[2]) * amount,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FocusSphere, StableEntityId, SurfaceAddress};
    use std::f64::consts::FRAC_PI_6;
    use uuid::Uuid;

    fn camera() -> CameraRig {
        CameraRig::new(
            [0.0, 1.0, 3.0],
            CameraBasis::CANONICAL,
            3.0,
            None,
            super::super::PerspectiveLens::default(),
        )
        .unwrap()
    }

    fn contact(position: [f64; 3], normal: [f64; 3]) -> SurfaceContact {
        SurfaceContact {
            address: SurfaceAddress::new(StableEntityId(Uuid::from_u128(1)), 0, [1.0, 0.0, 0.0])
                .unwrap(),
            output_position: position,
            output_normal: normal,
            surface_velocity: [0.0; 3],
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "{actual} != {expected}"
        );
    }

    fn assert_vec_close(actual: [f64; 3], expected: [f64; 3]) {
        for axis in 0..3 {
            assert_close(actual[axis], expected[axis]);
        }
    }

    #[test]
    fn scene_scale_controls_speed_height_and_near_plane() {
        let controls = SurfaceWalkControls {
            speed_octave_steps: 100.0,
            body_scale_octave_steps: -300.0,
            eye_height_octave_steps: 100.0,
            ..SurfaceWalkControls::default()
        };
        let metrics = controls.metrics(2.0, false).unwrap();
        assert_close(metrics.body_scale, 0.125);
        assert_close(metrics.radii_per_second, 0.4);
        assert_close(metrics.speed, 0.1);
        assert_close(metrics.eye_height, 0.00875);
        assert_close(metrics.near, 0.0007);
        assert_close(controls.metrics(2.0, true).unwrap().speed, 0.3);
    }

    #[test]
    fn motion_is_tangent_and_diagonal_input_is_limited_once() {
        let controller = SurfaceWalkController::default();
        let motion = controller
            .plan_motion(
                &camera(),
                [0.0, 1.0, 0.0],
                5.0,
                SurfaceWalkControls::default(),
                SurfaceWalkInput {
                    forward_axis: 1.0,
                    right_axis: 1.0,
                    fast: false,
                },
            )
            .unwrap();
        assert_close(
            motion
                .desired_output_velocity
                .into_iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt(),
            1.0,
        );
        assert_close(dot(motion.desired_output_velocity, [0.0, 1.0, 0.0]), 0.0);
        assert_vec_close(motion.tangent_forward, [0.0, 0.0, -1.0]);
        assert_vec_close(motion.tangent_right, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn follower_retains_surface_relative_pitch_as_normal_changes() {
        let mut controller = SurfaceWalkController::default();
        let pitched_forward =
            compose_surface_relative_forward([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], -FRAC_PI_6)
                .unwrap();
        let pitched = CameraRig::new(
            [0.0, 1.0, 3.0],
            CameraBasis::from_forward_up(pitched_forward, [0.0, 1.0, 0.0]).unwrap(),
            3.0,
            None,
            super::super::PerspectiveLens::default(),
        )
        .unwrap();
        let first = controller
            .follow_contact(
                &pitched,
                &contact([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                2.0,
                SurfaceWalkControls::default(),
                0.0,
                true,
                false,
            )
            .unwrap();
        assert_close(first.relative_pitch_radians.unwrap(), -FRAC_PI_6);

        let controls = SurfaceWalkControls {
            smoothing_seconds: 0.0,
            tangent_pull_fraction: 1.0,
            ..SurfaceWalkControls::default()
        };
        let second = controller
            .follow_contact(
                &first.camera,
                &contact([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
                2.0,
                controls,
                10.0,
                true,
                false,
            )
            .unwrap();
        let decomposed = decompose_surface_relative_forward(
            second.camera.basis().forward,
            second.filtered_normal,
            second.tangent_forward,
        )
        .unwrap();
        assert_close(decomposed.pitch_radians, -FRAC_PI_6);
        assert_vec_close(second.camera.eye, [1.0, 0.0, 0.035]);
    }

    #[test]
    fn contact_filter_uses_time_constant_and_tiny_scale_updates_lens() {
        let mut controller = SurfaceWalkController::default();
        let controls = SurfaceWalkControls {
            body_scale_octave_steps: -800.0,
            smoothing_seconds: 0.5,
            ..SurfaceWalkControls::default()
        };
        let first = controller
            .follow_contact(
                &camera(),
                &contact([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                1.0,
                controls,
                0.0,
                true,
                false,
            )
            .unwrap();
        let next = controller
            .follow_contact(
                &first.camera,
                &contact([10.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                1.0,
                controls,
                0.5,
                true,
                false,
            )
            .unwrap();
        assert_close(next.filtered_position[0], 10.0 * (1.0 - (-1.0f64).exp()));
        assert_close(next.metrics.eye_height, 0.035 / 256.0);
        assert_close(next.camera.lens.near, 0.0000109375);
    }

    #[test]
    fn follower_state_transports_between_reflection_charts_without_resetting_pitch() {
        let mut controller = SurfaceWalkController::default();
        controller
            .follow_contact(
                &camera(),
                &contact([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                1.0,
                SurfaceWalkControls {
                    smoothing_seconds: 1.0,
                    ..SurfaceWalkControls::default()
                },
                0.0,
                true,
                false,
            )
            .unwrap();
        let original = controller.clone();
        let inversion = SphereReflectionState::Sphere(FocusSphere::new([0.0; 3], 1.0).unwrap());

        assert!(controller
            .transport_between_reflections(SphereReflectionState::Identity, inversion)
            .unwrap());
        let transported = controller.state.unwrap();
        assert_vec_close(transported.filtered_position, [0.5, 0.0, 0.0]);
        assert_vec_close(transported.filtered_normal, [-1.0, 0.0, 0.0]);
        assert_vec_close(transported.tangent_forward.unwrap(), [0.0, 0.0, -1.0]);
        assert_eq!(
            transported.relative_pitch_radians,
            original.state.unwrap().relative_pitch_radians,
        );

        assert!(controller
            .transport_between_reflections(inversion, SphereReflectionState::Identity)
            .unwrap());
        let round_trip = controller.state.unwrap();
        let original = original.state.unwrap();
        assert_vec_close(round_trip.filtered_position, original.filtered_position);
        assert_vec_close(round_trip.filtered_normal, original.filtered_normal);
        assert_vec_close(
            round_trip.tangent_forward.unwrap(),
            original.tangent_forward.unwrap(),
        );
        assert_eq!(
            round_trip.relative_pitch_radians,
            original.relative_pitch_radians
        );
    }

    #[test]
    fn follower_reflection_pole_rejection_is_atomic() {
        let mut controller = SurfaceWalkController::default();
        controller
            .follow_contact(
                &camera(),
                &contact([0.0; 3], [0.0, 1.0, 0.0]),
                1.0,
                SurfaceWalkControls::default(),
                0.0,
                true,
                false,
            )
            .unwrap();
        let before = controller.clone();
        let inversion = SphereReflectionState::Sphere(FocusSphere::new([0.0; 3], 1.0).unwrap());
        let result =
            controller.transport_between_reflections(SphereReflectionState::Identity, inversion);
        assert_eq!(result, Err(CameraError::ReflectionPole),);
        assert_eq!(controller, before);
    }

    #[test]
    fn invalid_follow_is_atomic_and_reset_discards_filter_history() {
        let mut controller = SurfaceWalkController::default();
        let first = controller
            .follow_contact(
                &camera(),
                &contact([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                1.0,
                SurfaceWalkControls::default(),
                0.0,
                true,
                false,
            )
            .unwrap();
        let before = controller.clone();
        let error = controller.follow_contact(
            &first.camera,
            &contact([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            1.0,
            SurfaceWalkControls::default(),
            1.0 / 60.0,
            true,
            false,
        );
        assert_eq!(error, Err(SurfaceWalkError::DegenerateFrame));
        assert_eq!(controller, before);
        controller.reset();
        assert!(!controller.is_active());
    }

    #[test]
    fn metrics_reject_nonfinite_derived_values() {
        let controls = SurfaceWalkControls {
            base_radii_per_second: f64::MAX,
            speed_octave_steps: 100.0,
            ..SurfaceWalkControls::default()
        };
        assert_eq!(
            controls.metrics(1.0e-308, false),
            Err(SurfaceWalkError::InvalidControls)
        );

        let controls = SurfaceWalkControls {
            body_scale_octave_steps: 1.0e6,
            ..SurfaceWalkControls::default()
        };
        assert_eq!(
            controls.metrics(1.0, false),
            Err(SurfaceWalkError::InvalidControls)
        );
    }
}
