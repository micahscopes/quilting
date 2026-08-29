use super::FocusSphere;
use bevy_ecs::prelude::{Component, Resource};
use quilting_core::Quat;
use std::error::Error;
use std::fmt;

const EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraError {
    NonFinite,
    DegenerateBasis,
    InvalidLens,
    InvalidControlDistance,
    InvalidFraming,
    InvalidTransition,
    ReflectionPole,
}

impl fmt::Display for CameraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NonFinite => "camera values must be finite",
            Self::DegenerateBasis => "camera basis must contain independent forward and up axes",
            Self::InvalidLens => "camera lens values are invalid",
            Self::InvalidControlDistance => "camera control distance must be finite and positive",
            Self::InvalidFraming => {
                "camera framing radius, aspect, field of view, and margin are invalid"
            }
            Self::InvalidTransition => "camera transition duration must be finite and positive",
            Self::ReflectionPole => "camera transport reached a spherical-reflection pole",
        })
    }
}

impl Error for CameraError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveLens {
    pub vertical_fov_radians: f64,
    pub near: f64,
    pub far: f64,
}

impl Default for PerspectiveLens {
    fn default() -> Self {
        Self {
            vertical_fov_radians: std::f64::consts::FRAC_PI_3,
            near: 0.01,
            far: 10_000.0,
        }
    }
}

impl PerspectiveLens {
    pub fn validate(self) -> Result<Self, CameraError> {
        if !self.vertical_fov_radians.is_finite()
            || !self.near.is_finite()
            || !self.far.is_finite()
            || self.vertical_fov_radians <= 0.0
            || self.vertical_fov_radians >= std::f64::consts::PI
            || self.near <= 0.0
            || self.far <= self.near
        {
            return Err(CameraError::InvalidLens);
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraBasis {
    pub right: [f64; 3],
    pub up: [f64; 3],
    pub forward: [f64; 3],
}

impl CameraBasis {
    pub const CANONICAL: Self = Self {
        right: [1.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        forward: [0.0, 0.0, -1.0],
    };

    pub fn from_forward_up(forward: [f64; 3], up: [f64; 3]) -> Result<Self, CameraError> {
        let forward = normalize(forward).ok_or(CameraError::DegenerateBasis)?;
        let right = normalize(cross(forward, up)).ok_or(CameraError::DegenerateBasis)?;
        let up = cross(right, forward);
        Ok(Self { right, up, forward })
    }

    pub fn as_array(self) -> [f64; 9] {
        [
            self.right[0],
            self.right[1],
            self.right[2],
            self.up[0],
            self.up[1],
            self.up[2],
            self.forward[0],
            self.forward[1],
            self.forward[2],
        ]
    }

    pub(crate) fn orientation(self) -> Result<Quat, CameraError> {
        // Rotation matrix columns are right, up, and camera-local +Z. The
        // camera looks down local -Z, so the final column is -forward.
        let m00 = self.right[0];
        let m01 = self.up[0];
        let m02 = -self.forward[0];
        let m10 = self.right[1];
        let m11 = self.up[1];
        let m12 = -self.forward[1];
        let m20 = self.right[2];
        let m21 = self.up[2];
        let m22 = -self.forward[2];
        let trace = m00 + m11 + m22;
        let q = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            Quat::new(0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
        } else if m00 > m11 && m00 > m22 {
            let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
            Quat::new((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
        } else if m11 > m22 {
            let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
            Quat::new((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
        } else {
            let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
            Quat::new((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
        };
        normalize_quaternion(q).ok_or(CameraError::DegenerateBasis)
    }
}

/// Rust-authoritative six-degree-of-freedom camera state.
///
/// `orientation` rotates canonical camera axes into the active Euclidean
/// chart. `control_distance` is the scale used by orbit/reframe controls; it is
/// not a lens zoom. A semantic target is transported as a point. Without one,
/// conformal edits transport the sight tangent at the eye.
#[derive(Component, Resource, Debug, Clone, Copy, PartialEq)]
pub struct CameraRig {
    pub eye: [f64; 3],
    pub orientation: Quat,
    pub control_distance: f64,
    pub semantic_target: Option<[f64; 3]>,
    pub lens: PerspectiveLens,
}

impl Default for CameraRig {
    fn default() -> Self {
        Self {
            eye: [0.0, 0.0, 3.0],
            orientation: Quat::ONE,
            control_distance: 3.0,
            semantic_target: None,
            lens: PerspectiveLens::default(),
        }
    }
}

impl CameraRig {
    pub fn new(
        eye: [f64; 3],
        basis: CameraBasis,
        control_distance: f64,
        semantic_target: Option<[f64; 3]>,
        lens: PerspectiveLens,
    ) -> Result<Self, CameraError> {
        if !finite3(eye) || semantic_target.is_some_and(|target| !finite3(target)) {
            return Err(CameraError::NonFinite);
        }
        if !control_distance.is_finite() || control_distance <= 0.0 {
            return Err(CameraError::InvalidControlDistance);
        }
        Ok(Self {
            eye,
            orientation: basis.orientation()?,
            control_distance,
            semantic_target,
            lens: lens.validate()?,
        })
    }

    /// Validate state received through an interchange or replay boundary.
    pub fn validate(&self) -> Result<(), CameraError> {
        if !finite3(self.eye) || self.semantic_target.is_some_and(|target| !finite3(target)) {
            return Err(CameraError::NonFinite);
        }
        if !self.control_distance.is_finite() || self.control_distance <= 0.0 {
            return Err(CameraError::InvalidControlDistance);
        }
        let orientation_norm = self.orientation.norm();
        if !orientation_norm.is_finite() || orientation_norm <= EPSILON {
            return Err(CameraError::DegenerateBasis);
        }
        self.lens.validate()?;
        Ok(())
    }

    pub fn basis(&self) -> CameraBasis {
        let q = normalize_quaternion(self.orientation).unwrap_or(Quat::ONE);
        CameraBasis {
            right: rotate(q, [1.0, 0.0, 0.0]),
            up: rotate(q, [0.0, 1.0, 0.0]),
            forward: rotate(q, [0.0, 0.0, -1.0]),
        }
    }

    pub fn view_target(&self) -> [f64; 3] {
        self.semantic_target
            .unwrap_or_else(|| add(self.eye, scale(self.basis().forward, self.control_distance)))
    }

    /// Preserve orientation, lens, and control distance while making one
    /// finite output-chart point the explicit camera target.
    pub fn targeted_at(&self, target: [f64; 3]) -> Result<Self, CameraError> {
        if !finite3(target) {
            return Err(CameraError::NonFinite);
        }
        let basis = self.basis();
        CameraRig::new(
            sub(target, scale(basis.forward, self.control_distance)),
            basis,
            self.control_distance,
            Some(target),
            self.lens,
        )
    }

    /// Build a point-target camera that contains an output-chart sphere in the
    /// narrower perspective axis while preserving orientation and lens.
    pub fn framed_sphere_target(
        &self,
        center: [f64; 3],
        radius: f64,
        viewport_aspect: f64,
        margin: f64,
    ) -> Result<Self, CameraError> {
        if !finite3(center) {
            return Err(CameraError::NonFinite);
        }
        let control_distance = framed_sphere_distance(
            radius,
            viewport_aspect,
            self.lens.vertical_fov_radians,
            margin,
        )?
        .clamp(0.1, 100.0);
        let basis = self.basis();
        CameraRig::new(
            sub(center, scale(basis.forward, control_distance)),
            basis,
            control_distance,
            Some(center),
            self.lens,
        )
    }

    pub fn translate_local(&mut self, delta: [f64; 3]) -> Result<(), CameraError> {
        if !finite3(delta) {
            return Err(CameraError::NonFinite);
        }
        let basis = self.basis();
        let world = add(
            add(scale(basis.right, delta[0]), scale(basis.up, delta[1])),
            scale(basis.forward, delta[2]),
        );
        self.eye = add(self.eye, world);
        if let Some(target) = self.semantic_target.as_mut() {
            *target = add(*target, world);
        }
        Ok(())
    }

    /// Rotate around the eye in the current rolled local frame.
    pub fn rotate_local(&mut self, rotation: [f64; 3]) -> Result<(), CameraError> {
        if !finite3(rotation) {
            return Err(CameraError::NonFinite);
        }
        let basis = self.basis();
        let world_axis = add(
            add(
                scale(basis.right, rotation[0]),
                scale(basis.up, rotation[1]),
            ),
            scale(basis.forward, rotation[2]),
        );
        self.rotate_about_world_axis(world_axis, length(rotation))?;
        if self.semantic_target.is_some() {
            self.semantic_target = Some(add(
                self.eye,
                scale(self.basis().forward, self.control_distance),
            ));
        }
        Ok(())
    }

    pub fn level_horizon(&mut self) -> Result<(), CameraError> {
        let basis = self.basis();
        let mut right = [-basis.forward[2], 0.0, basis.forward[0]];
        if length(right) < 1.0e-6 {
            right = [basis.right[0], 0.0, basis.right[2]];
        }
        if length(right) < 1.0e-6 {
            right = [1.0, 0.0, 0.0];
        }
        right = normalize(right).ok_or(CameraError::DegenerateBasis)?;
        let up = cross(right, basis.forward);
        self.orientation = CameraBasis {
            right,
            up,
            forward: basis.forward,
        }
        .orientation()?;
        Ok(())
    }

    pub fn rotate_horizon_locked(&mut self, pitch: f64, yaw: f64) -> Result<(), CameraError> {
        if !pitch.is_finite() || !yaw.is_finite() {
            return Err(CameraError::NonFinite);
        }
        self.level_horizon()?;
        self.rotate_about_world_axis(self.basis().right, pitch)?;
        self.rotate_about_world_axis([0.0, 1.0, 0.0], yaw)?;
        self.level_horizon()?;
        if self.semantic_target.is_some() {
            self.semantic_target = Some(add(
                self.eye,
                scale(self.basis().forward, self.control_distance),
            ));
        }
        Ok(())
    }

    pub fn orbit(
        &mut self,
        rotation: [f64; 3],
        pan: [f64; 2],
        dolly_log: f64,
        horizon_locked: bool,
    ) -> Result<(), CameraError> {
        if !finite3(rotation)
            || pan.into_iter().any(|value| !value.is_finite())
            || !dolly_log.is_finite()
        {
            return Err(CameraError::NonFinite);
        }
        let pivot = self.view_target();
        if horizon_locked {
            self.rotate_horizon_locked(rotation[0], rotation[1])?;
        } else {
            self.rotate_local(rotation)?;
        }
        let basis = self.basis();
        let pan_world = add(scale(basis.right, pan[0]), scale(basis.up, pan[1]));
        let pivot = add(pivot, pan_world);
        self.control_distance = (self.control_distance * dolly_log.exp()).clamp(0.1, 100.0);
        self.eye = sub(pivot, scale(basis.forward, self.control_distance));
        if self.semantic_target.is_some() {
            self.semantic_target = Some(pivot);
        }
        Ok(())
    }

    /// Orbit around the current explicit or implicit view target using the
    /// established pointer-camera convention: world-up yaw first, then pitch
    /// around the newly rotated camera right axis. Pan is expressed in the
    /// resulting screen plane and dolly is logarithmic.
    pub fn apply_turntable(&mut self, frame: TurntableFrame) -> Result<(), CameraError> {
        frame.validate()?;
        let pivot = self.view_target();
        self.rotate_about_world_axis([0.0, 1.0, 0.0], frame.yaw)?;
        self.rotate_about_world_axis(self.basis().right, frame.pitch)?;
        let basis = self.basis();
        let pan_world = add(
            scale(basis.right, frame.pan[0]),
            scale(basis.up, frame.pan[1]),
        );
        let pivot = add(pivot, pan_world);
        self.control_distance = (self.control_distance * frame.dolly_log.exp()).clamp(0.1, 100.0);
        self.eye = sub(pivot, scale(basis.forward, self.control_distance));
        if self.semantic_target.is_some() {
            self.semantic_target = Some(pivot);
        }
        Ok(())
    }

    pub fn move_drone(&mut self, delta: [f64; 3]) -> Result<(), CameraError> {
        if !finite3(delta) {
            return Err(CameraError::NonFinite);
        }
        let basis = self.basis();
        let right = horizontal_or(basis.right, [1.0, 0.0, 0.0]);
        let forward = horizontal_or(basis.forward, [-basis.up[0], 0.0, -basis.up[2]]);
        let world = add(
            add(scale(right, delta[0]), [0.0, delta[1], 0.0]),
            scale(forward, delta[2]),
        );
        self.eye = add(self.eye, world);
        if let Some(target) = self.semantic_target.as_mut() {
            *target = add(*target, world);
        }
        Ok(())
    }

    pub fn apply_navigation(
        &mut self,
        preset: NavigationPreset,
        frame: NavigationFrame,
    ) -> Result<(), CameraError> {
        frame.validate()?;
        match preset {
            NavigationPreset::Hyperscope => {
                // Established Hyperscope behavior: move in the current frame,
                // then rotate around the newly translated eye.
                self.translate_local(frame.translation)?;
                self.rotate_local(frame.rotation)?;
            }
            NavigationPreset::Object => {
                self.orbit(
                    frame.rotation,
                    [frame.translation[0], frame.translation[1]],
                    frame.dolly_log,
                    frame.horizon_locked,
                )?;
            }
            NavigationPreset::Fly => {
                if frame.horizon_locked {
                    self.rotate_horizon_locked(frame.rotation[0], frame.rotation[1])?;
                } else {
                    self.rotate_local(frame.rotation)?;
                }
                self.translate_local(frame.translation)?;
            }
            NavigationPreset::Drone => {
                self.rotate_horizon_locked(frame.rotation[0], frame.rotation[1])?;
                self.move_drone(frame.translation)?;
            }
        }
        Ok(())
    }

    /// Transport the camera through `next ∘ previous⁻¹`. Sphere reflections
    /// are self-inverse, so this is two direct reflection evaluations.
    pub fn transport_between_reflections(
        &mut self,
        previous: SphereReflectionState,
        next: SphereReflectionState,
    ) -> Result<f64, CameraError> {
        if previous == next {
            return Ok(1.0);
        }
        let basis = self.basis();
        let remap =
            previous.transport_point_and_directions(next, self.eye, [basis.up, basis.forward])?;
        let local_scale = remap.local_scale;

        let mapped_target = self
            .semantic_target
            .map(|target| {
                previous
                    .transport_point_and_directions(next, target, [])
                    .map(|transport| transport.point)
            })
            .transpose()?;

        let (forward, up, control_distance) = if let Some(target) = mapped_target {
            let forward = sub(target, remap.point);
            let distance = length(forward);
            if !distance.is_finite() || distance <= EPSILON {
                return Err(CameraError::DegenerateBasis);
            }
            let up = transport_up_along_sightline(basis.forward, forward, basis.up)
                .ok_or(CameraError::DegenerateBasis)?;
            (forward, up, distance)
        } else {
            (
                remap.directions[1],
                remap.directions[0],
                self.control_distance * local_scale,
            )
        };
        let basis = CameraBasis::from_forward_up(forward, up)?;
        self.eye = remap.point;
        self.orientation = basis.orientation()?;
        self.control_distance = control_distance;
        self.semantic_target = mapped_target;
        Ok(local_scale)
    }

    fn rotate_about_world_axis(
        &mut self,
        world_axis: [f64; 3],
        angle: f64,
    ) -> Result<(), CameraError> {
        if !finite3(world_axis) || !angle.is_finite() {
            return Err(CameraError::NonFinite);
        }
        if angle.abs() <= EPSILON || length(world_axis) <= EPSILON {
            return Ok(());
        }
        let axis = normalize(world_axis).ok_or(CameraError::DegenerateBasis)?;
        let half = angle * 0.5;
        let delta = Quat::new(
            half.cos(),
            axis[0] * half.sin(),
            axis[1] * half.sin(),
            axis[2] * half.sin(),
        );
        self.orientation =
            normalize_quaternion(delta * self.orientation).ok_or(CameraError::DegenerateBasis)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationPreset {
    Hyperscope,
    Object,
    Fly,
    Drone,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationFrame {
    /// World-distance deltas expressed as camera-local right/up/forward.
    pub translation: [f64; 3],
    /// Radian deltas around camera-local right/up/forward.
    pub rotation: [f64; 3],
    /// Logarithmic orbit-distance delta used by object mode.
    pub dolly_log: f64,
    pub horizon_locked: bool,
}

impl Default for NavigationFrame {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [0.0; 3],
            dolly_log: 0.0,
            horizon_locked: false,
        }
    }
}

impl NavigationFrame {
    fn validate(self) -> Result<(), CameraError> {
        if !finite3(self.translation) || !finite3(self.rotation) || !self.dolly_log.is_finite() {
            return Err(CameraError::NonFinite);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TurntableFrame {
    /// World-distance deltas in the post-rotation screen right/up plane.
    pub pan: [f64; 2],
    /// Radians around the post-yaw camera right axis.
    pub pitch: f64,
    /// Radians around Euclidean world up.
    pub yaw: f64,
    /// Logarithmic control-distance delta.
    pub dolly_log: f64,
}

impl TurntableFrame {
    fn validate(self) -> Result<(), CameraError> {
        if self.pan.into_iter().any(|value| !value.is_finite())
            || !self.pitch.is_finite()
            || !self.yaw.is_finite()
            || !self.dolly_log.is_finite()
        {
            return Err(CameraError::NonFinite);
        }
        Ok(())
    }
}

/// Browser pointer gestures admitted by the turntable camera boundary.
///
/// The resulting [`TurntableFrame`] is device-neutral; this enum only freezes
/// the incumbent pixel-delta response while browser authority is migrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointerTurntableGesture {
    Orbit,
    Pan,
    Wheel,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointerTurntableInput {
    /// Browser movement deltas in CSS pixels. For wheel input only Y is used.
    pub delta: [f64; 2],
    pub gesture: PointerTurntableGesture,
    /// Current Euclidean control distance used to make pan scale-relative.
    pub control_distance: f64,
}

/// Convert the incumbent browser mouse/trackpad response into one semantic
/// turntable frame. Keeping this mapping in Rust makes the JS, shadow, and
/// authoritative routes compare the same constants and sign conventions.
pub fn map_pointer_turntable(input: PointerTurntableInput) -> Result<TurntableFrame, CameraError> {
    if input.delta.into_iter().any(|value| !value.is_finite()) {
        return Err(CameraError::NonFinite);
    }
    if !input.control_distance.is_finite() || input.control_distance <= 0.0 {
        return Err(CameraError::InvalidControlDistance);
    }
    Ok(match input.gesture {
        PointerTurntableGesture::Orbit => TurntableFrame {
            pitch: -input.delta[1] * 0.005,
            yaw: input.delta[0] * 0.005,
            ..TurntableFrame::default()
        },
        PointerTurntableGesture::Pan => TurntableFrame {
            pan: [
                -input.delta[0] * 0.003 * input.control_distance,
                -input.delta[1] * 0.003 * input.control_distance,
            ],
            ..TurntableFrame::default()
        },
        PointerTurntableGesture::Wheel => TurntableFrame {
            dolly_log: if input.delta[1] > 0.0 {
                1.1_f64.ln()
            } else {
                0.9_f64.ln()
            },
            ..TurntableFrame::default()
        },
    })
}

/// Perspective distance that contains a sphere in the narrower viewport axis.
/// This is the backend-neutral form of the established Hyperscope framing
/// rule and deliberately operates in the sphere's current output chart.
pub fn framed_sphere_distance(
    radius: f64,
    viewport_aspect: f64,
    vertical_fov_radians: f64,
    margin: f64,
) -> Result<f64, CameraError> {
    if !radius.is_finite()
        || !viewport_aspect.is_finite()
        || !vertical_fov_radians.is_finite()
        || !margin.is_finite()
        || radius <= 0.0
        || viewport_aspect <= 0.0
        || vertical_fov_radians <= 0.0
        || vertical_fov_radians >= std::f64::consts::PI
        || margin < 1.0
    {
        return Err(CameraError::InvalidFraming);
    }
    let half_y = vertical_fov_radians * 0.5;
    let half_x = (half_y.tan() * viewport_aspect).atan();
    let distance = radius * margin / half_x.min(half_y).sin();
    if !distance.is_finite() || distance <= 0.0 {
        return Err(CameraError::InvalidFraming);
    }
    Ok(distance)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationAxes {
    pub translation: [f64; 3],
    pub rotation: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpaceMouseMapping {
    pub preset: NavigationPreset,
    pub swap_yz: bool,
    pub invert_pan: u8,
    pub invert_rotate: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaceMouseInputError {
    NonFinite,
    AxisOutOfRange,
    NegativeResponse,
    InvalidInversionMask,
}

impl fmt::Display for SpaceMouseInputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NonFinite => "SpaceMouse input and derived response must remain finite",
            Self::AxisOutOfRange => "normalized SpaceMouse axes must be within [-1, 1]",
            Self::NegativeResponse => {
                "SpaceMouse delta, gesture speed, and gains must be nonnegative"
            }
            Self::InvalidInversionMask => {
                "SpaceMouse inversion masks may contain only the X, Y, and Z bits"
            }
        })
    }
}

impl Error for SpaceMouseInputError {}

/// One filtered, normalized six-axis sample plus the response parameters needed
/// to turn it into a device-neutral navigation frame.
///
/// WebHID permission, report-layout decoding, dead-zone shaping, and temporal
/// smoothing remain platform work. The browser registers `linear_speed` once
/// when a translation gesture starts and keeps it fixed until that gesture
/// ends. Preset policy, Blender normalization, response integration, inversion
/// masks, object-mode dolly, and horizon policy are deterministic Rust
/// semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpaceMouseCameraInput {
    pub normalized_axes: [f64; 6],
    pub mapping: SpaceMouseMapping,
    pub delta_seconds: f64,
    /// World units per second at a move gain of one. Frozen for the gesture.
    pub registered_linear_speed: f64,
    /// User move sensitivity, expressed as the browser setting divided by 100.
    pub move_gain: f64,
    /// User rotation sensitivity, expressed as the setting divided by 100.
    pub rotate_gain: f64,
    /// User-requested horizon lock. Drone always locks; Hyperscope never does.
    pub horizon_lock_requested: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MappedSpaceMouseFrame {
    pub preset: NavigationPreset,
    pub frame: NavigationFrame,
}

impl Default for SpaceMouseMapping {
    fn default() -> Self {
        Self {
            preset: NavigationPreset::Hyperscope,
            swap_yz: false,
            invert_pan: 0b010,
            invert_rotate: 0b001,
        }
    }
}

/// Normalize the c63a device axes into right/up/forward and pitch/yaw/roll.
/// This is a direct Rust counterpart of the browser behavior oracle.
pub fn map_space_mouse_axes(raw: [f64; 6], mapping: SpaceMouseMapping) -> NavigationAxes {
    let blender = mapping.preset != NavigationPreset::Hyperscope;
    let right = raw[0];
    let mut up = if blender { -raw[2] } else { raw[1] };
    let mut forward = if blender { raw[1] } else { raw[2] };
    let pitch = if blender { -raw[3] } else { raw[3] };
    let mut yaw = if blender { raw[5] } else { raw[4] };
    let mut roll = if blender { -raw[4] } else { raw[5] };
    if mapping.swap_yz {
        (up, forward) = (-forward, up);
        (yaw, roll) = (-roll, yaw);
    }
    let mode_sign = if mapping.preset == NavigationPreset::Object {
        -1.0
    } else {
        1.0
    };
    NavigationAxes {
        translation: [
            right * mode_sign * mask_sign(mapping.invert_pan, 0),
            up * mode_sign * mask_sign(mapping.invert_pan, 1),
            forward * mode_sign * mask_sign(mapping.invert_pan, 2),
        ],
        rotation: [
            pitch * mode_sign * mask_sign(mapping.invert_rotate, 0),
            yaw * mode_sign * mask_sign(mapping.invert_rotate, 1),
            roll * mode_sign * mask_sign(mapping.invert_rotate, 2),
        ],
    }
}

/// Convert a filtered SpaceMouse sample into the same semantic frame consumed
/// by mouse, keyboard, replay, and future game/XR adapters.
pub fn map_space_mouse_camera(
    input: SpaceMouseCameraInput,
) -> Result<MappedSpaceMouseFrame, SpaceMouseInputError> {
    if input
        .normalized_axes
        .into_iter()
        .any(|value| !value.is_finite())
        || !input.delta_seconds.is_finite()
        || !input.registered_linear_speed.is_finite()
        || !input.move_gain.is_finite()
        || !input.rotate_gain.is_finite()
    {
        return Err(SpaceMouseInputError::NonFinite);
    }
    if input
        .normalized_axes
        .into_iter()
        .any(|value| !(-1.0..=1.0).contains(&value))
    {
        return Err(SpaceMouseInputError::AxisOutOfRange);
    }
    if input.delta_seconds < 0.0
        || input.registered_linear_speed < 0.0
        || input.move_gain < 0.0
        || input.rotate_gain < 0.0
    {
        return Err(SpaceMouseInputError::NegativeResponse);
    }
    if input.mapping.invert_pan & !0b111 != 0 || input.mapping.invert_rotate & !0b111 != 0 {
        return Err(SpaceMouseInputError::InvalidInversionMask);
    }

    let mapped = map_space_mouse_axes(input.normalized_axes, input.mapping);
    let translation_scale = input.registered_linear_speed * input.move_gain * input.delta_seconds;
    let rotation_scale = 1.5 * input.rotate_gain * input.delta_seconds;
    let object_dolly_scale = if input.mapping.preset == NavigationPreset::Object {
        1.5 * input.move_gain * input.delta_seconds
    } else {
        0.0
    };
    if !translation_scale.is_finite()
        || !rotation_scale.is_finite()
        || !object_dolly_scale.is_finite()
    {
        return Err(SpaceMouseInputError::NonFinite);
    }
    let mut translation = mapped.translation.map(|axis| axis * translation_scale);
    let dolly_log = if input.mapping.preset == NavigationPreset::Object {
        translation[2] = 0.0;
        mapped.translation[2] * object_dolly_scale
    } else {
        0.0
    };
    let horizon_locked = match input.mapping.preset {
        NavigationPreset::Hyperscope => false,
        NavigationPreset::Drone => true,
        NavigationPreset::Object | NavigationPreset::Fly => input.horizon_lock_requested,
    };
    let frame = NavigationFrame {
        translation,
        rotation: mapped.rotation.map(|axis| axis * rotation_scale),
        dolly_log,
        horizon_locked,
    };
    frame
        .validate()
        .map_err(|_| SpaceMouseInputError::NonFinite)?;
    Ok(MappedSpaceMouseFrame {
        preset: input.mapping.preset,
        frame,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SphereReflectionState {
    Identity,
    Sphere(FocusSphere),
}

/// A point and tangent directions transported through
/// `next ∘ previous⁻¹` between Euclidean and spherical-reflection
/// charts. Directions carry only the orthogonal conformal differential; the
/// positive local scale is reported separately.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReflectionTransport<const N: usize> {
    pub point: [f64; 3],
    pub directions: [[f64; 3]; N],
    pub local_scale: f64,
}

impl SphereReflectionState {
    pub fn orientation_sign(self) -> i8 {
        match self {
            Self::Identity => 1,
            Self::Sphere(_) => -1,
        }
    }

    /// Transport a point and any tangent directions through the exact chart
    /// change. The operation stages both reflection evaluations before
    /// returning, so a pole never produces a partially transported frame.
    pub fn transport_point_and_directions<const N: usize>(
        self,
        next: Self,
        point: [f64; 3],
        directions: [[f64; 3]; N],
    ) -> Result<ReflectionTransport<N>, CameraError> {
        if !finite3(point) || directions.iter().any(|direction| !finite3(*direction)) {
            return Err(CameraError::NonFinite);
        }
        if self == next {
            return Ok(ReflectionTransport {
                point,
                directions,
                local_scale: 1.0,
            });
        }
        let unmap = reflect_point_and_directions(point, directions, self)?;
        let remap = reflect_point_and_directions(unmap.point, unmap.directions, next)?;
        let local_scale = unmap.scale * remap.scale;
        if !local_scale.is_finite() || local_scale <= 0.0 {
            return Err(CameraError::ReflectionPole);
        }
        Ok(ReflectionTransport {
            point: remap.point,
            directions: remap.directions,
            local_scale,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct Reflected<const N: usize> {
    point: [f64; 3],
    directions: [[f64; 3]; N],
    scale: f64,
}

fn reflect_point_and_directions<const N: usize>(
    point: [f64; 3],
    directions: [[f64; 3]; N],
    state: SphereReflectionState,
) -> Result<Reflected<N>, CameraError> {
    if !finite3(point) || directions.iter().any(|direction| !finite3(*direction)) {
        return Err(CameraError::NonFinite);
    }
    let SphereReflectionState::Sphere(sphere) = state else {
        return Ok(Reflected {
            point,
            directions,
            scale: 1.0,
        });
    };
    let delta = sub(point, sphere.center);
    let distance_squared = dot(delta, delta);
    if !distance_squared.is_finite() || distance_squared <= EPSILON {
        return Err(CameraError::ReflectionPole);
    }
    let normal = scale(delta, distance_squared.sqrt().recip());
    let local_scale = sphere.radius * sphere.radius / distance_squared;
    Ok(Reflected {
        point: add(sphere.center, scale(delta, local_scale)),
        directions: directions
            .map(|direction| sub(direction, scale(normal, 2.0 * dot(direction, normal)))),
        scale: local_scale,
    })
}

fn transport_up_along_sightline(
    previous_forward: [f64; 3],
    next_forward: [f64; 3],
    previous_up: [f64; 3],
) -> Option<[f64; 3]> {
    let from = normalize(previous_forward)?;
    let to = normalize(next_forward)?;
    let cosine = dot(from, to).clamp(-1.0, 1.0);
    if cosine > 1.0 - EPSILON || cosine < -1.0 + EPSILON {
        return Some(previous_up);
    }
    let axis = cross(from, to);
    let first = cross(axis, previous_up);
    let second = cross(axis, first);
    Some(add(
        add(previous_up, first),
        scale(second, 1.0 / (1.0 + cosine)),
    ))
}

fn rotate(q: Quat, vector: [f64; 3]) -> [f64; 3] {
    (q * Quat::from_point(vector[0], vector[1], vector[2]) * q.conj()).to_point()
}

fn normalize_quaternion(q: Quat) -> Option<Quat> {
    let norm = q.norm();
    (norm.is_finite() && norm > EPSILON).then(|| {
        let mut q = q / norm;
        if q.w < 0.0 {
            q = -q;
        }
        q
    })
}

fn mask_sign(mask: u8, axis: u8) -> f64 {
    if mask & (1 << axis) == 0 {
        1.0
    } else {
        -1.0
    }
}

fn horizontal_or(vector: [f64; 3], fallback: [f64; 3]) -> [f64; 3] {
    normalize([vector[0], 0.0, vector[2]])
        .or_else(|| normalize(fallback))
        .unwrap_or([0.0, 0.0, -1.0])
}

fn finite3(value: [f64; 3]) -> bool {
    value.into_iter().all(f64::is_finite)
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] + right[axis])
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn scale(value: [f64; 3], factor: f64) -> [f64; 3] {
    value.map(|coordinate| coordinate * factor)
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn length(value: [f64; 3]) -> f64 {
    dot(value, value).sqrt()
}

fn normalize(value: [f64; 3]) -> Option<[f64; 3]> {
    let length = length(value);
    (length.is_finite() && length > EPSILON).then(|| scale(value, length.recip()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1.0e-10;

    fn assert_point_close(actual: [f64; 3], expected: [f64; 3]) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < EPS,
                "axis {axis}: actual={actual:?}, expected={expected:?}"
            );
        }
    }

    #[test]
    fn canonical_orientation_round_trips_through_basis() {
        let rig = CameraRig::default();
        assert_eq!(rig.basis(), CameraBasis::CANONICAL);
        let rebuilt =
            CameraRig::new(rig.eye, rig.basis(), rig.control_distance, None, rig.lens).unwrap();
        assert_eq!(rebuilt.orientation, Quat::ONE);
    }

    #[test]
    fn local_pitch_and_yaw_follow_accumulated_roll() {
        let rolled = CameraBasis {
            right: [0.0, -1.0, 0.0],
            up: [1.0, 0.0, 0.0],
            forward: [0.0, 0.0, -1.0],
        };
        let mut pitch = CameraRig::new(
            [0.0, 0.0, 3.0],
            rolled,
            3.0,
            None,
            PerspectiveLens::default(),
        )
        .unwrap();
        pitch.rotate_local([0.2, 0.0, 0.0]).unwrap();
        assert!(pitch.basis().forward[0].abs() > 0.1);

        let mut yaw = CameraRig::new(
            [0.0, 0.0, 3.0],
            rolled,
            3.0,
            None,
            PerspectiveLens::default(),
        )
        .unwrap();
        yaw.rotate_local([0.0, 0.2, 0.0]).unwrap();
        assert!(yaw.basis().forward[1].abs() > 0.1);
    }

    #[test]
    fn navigation_presets_keep_translation_and_orbit_semantics_distinct() {
        let mut fly = CameraRig::default();
        fly.apply_navigation(
            NavigationPreset::Hyperscope,
            NavigationFrame {
                translation: [1.0, 2.0, 3.0],
                ..NavigationFrame::default()
            },
        )
        .unwrap();
        assert_point_close(fly.eye, [1.0, 2.0, 0.0]);

        let mut orbit = CameraRig::default();
        let pivot = orbit.view_target();
        orbit
            .apply_navigation(
                NavigationPreset::Object,
                NavigationFrame {
                    rotation: [0.0, 0.5, 0.0],
                    dolly_log: 0.25,
                    ..NavigationFrame::default()
                },
            )
            .unwrap();
        assert_point_close(orbit.view_target(), pivot);
        assert!((orbit.control_distance - 3.0 * 0.25_f64.exp()).abs() < EPS);
    }

    #[test]
    fn turntable_uses_world_yaw_then_rolled_pitch_and_post_rotation_pan() {
        let rolled = CameraBasis {
            right: [0.0, -1.0, 0.0],
            up: [1.0, 0.0, 0.0],
            forward: [0.0, 0.0, -1.0],
        };
        for semantic_target in [None, Some([0.0, 0.0, 0.0])] {
            let mut camera = CameraRig::new(
                [0.0, 0.0, 3.0],
                rolled,
                3.0,
                semantic_target,
                PerspectiveLens::default(),
            )
            .unwrap();
            let original_pivot = camera.view_target();
            camera
                .apply_turntable(TurntableFrame {
                    pan: [0.4, -0.2],
                    pitch: -0.1,
                    yaw: 0.25,
                    dolly_log: 0.3,
                })
                .unwrap();

            let basis = camera.basis();
            let expected_pivot = add(
                original_pivot,
                add(scale(basis.right, 0.4), scale(basis.up, -0.2)),
            );
            let expected_distance = 3.0 * 0.3_f64.exp();
            assert_point_close(camera.view_target(), expected_pivot);
            assert_point_close(
                camera.eye,
                sub(expected_pivot, scale(basis.forward, expected_distance)),
            );
            assert!((camera.control_distance - expected_distance).abs() < EPS);
            assert_eq!(camera.semantic_target.is_some(), semantic_target.is_some());
            // A rolled camera's pitch axis is not silently replaced by world X.
            assert!(basis.forward[0].abs() > 0.1);
        }
    }

    #[test]
    fn pointer_turntable_mapping_freezes_browser_response_and_signs() {
        assert_eq!(
            map_pointer_turntable(PointerTurntableInput {
                delta: [12.0, -8.0],
                gesture: PointerTurntableGesture::Orbit,
                control_distance: 4.0,
            })
            .unwrap(),
            TurntableFrame {
                pitch: 0.04,
                yaw: 0.06,
                ..TurntableFrame::default()
            }
        );
        let pan = map_pointer_turntable(PointerTurntableInput {
            delta: [12.0, -8.0],
            gesture: PointerTurntableGesture::Pan,
            control_distance: 4.0,
        })
        .unwrap();
        assert!((pan.pan[0] + 0.144).abs() < EPSILON);
        assert!((pan.pan[1] - 0.096).abs() < EPSILON);
        assert_eq!(pan.pitch, 0.0);
        assert_eq!(pan.yaw, 0.0);
        assert_eq!(pan.dolly_log, 0.0);
        for (delta_y, factor) in [(1.0, 1.1_f64), (-1.0, 0.9_f64), (0.0, 0.9_f64)] {
            let frame = map_pointer_turntable(PointerTurntableInput {
                delta: [99.0, delta_y],
                gesture: PointerTurntableGesture::Wheel,
                control_distance: 4.0,
            })
            .unwrap();
            assert!((frame.dolly_log - factor.ln()).abs() < EPSILON);
        }
    }

    #[test]
    fn pointer_turntable_mapping_rejects_invalid_boundary_values() {
        for input in [
            PointerTurntableInput {
                delta: [f64::NAN, 0.0],
                gesture: PointerTurntableGesture::Orbit,
                control_distance: 1.0,
            },
            PointerTurntableInput {
                delta: [0.0, f64::INFINITY],
                gesture: PointerTurntableGesture::Pan,
                control_distance: 1.0,
            },
        ] {
            assert_eq!(map_pointer_turntable(input), Err(CameraError::NonFinite));
        }
        for control_distance in [0.0, -1.0, f64::NAN] {
            assert_eq!(
                map_pointer_turntable(PointerTurntableInput {
                    delta: [0.0, 0.0],
                    gesture: PointerTurntableGesture::Wheel,
                    control_distance,
                }),
                Err(CameraError::InvalidControlDistance)
            );
        }
    }

    #[test]
    fn sphere_framing_matches_the_narrower_perspective_axis() {
        let radius = 2.0;
        let fov = std::f64::consts::FRAC_PI_3;
        let landscape = framed_sphere_distance(radius, 16.0 / 9.0, fov, 1.15).unwrap();
        let portrait = framed_sphere_distance(radius, 0.5, fov, 1.15).unwrap();
        let expected_landscape = radius * 1.15 / (fov * 0.5).sin();
        assert!((landscape - expected_landscape).abs() < EPSILON);
        assert!(portrait > landscape);

        let camera = CameraRig::new(
            [3.0, -1.0, 4.0],
            CameraBasis::from_forward_up([-0.5, 0.25, -1.0], [0.2, 1.0, 0.1]).unwrap(),
            4.0,
            None,
            PerspectiveLens {
                vertical_fov_radians: fov,
                ..PerspectiveLens::default()
            },
        )
        .unwrap();
        let target = camera
            .framed_sphere_target([2.0, 3.0, -4.0], radius, 16.0 / 9.0, 1.15)
            .unwrap();
        assert_eq!(target.orientation, camera.orientation);
        assert_eq!(target.lens, camera.lens);
        assert_eq!(target.semantic_target, Some([2.0, 3.0, -4.0]));
        assert!((target.control_distance - landscape).abs() < EPSILON);
        assert_point_close(target.view_target(), [2.0, 3.0, -4.0]);

        let aimed = camera.targeted_at([2.0, 3.0, -4.0]).unwrap();
        assert_eq!(aimed.orientation, camera.orientation);
        assert_eq!(aimed.lens, camera.lens);
        assert_eq!(aimed.control_distance, camera.control_distance);
        assert_eq!(aimed.semantic_target, Some([2.0, 3.0, -4.0]));
        assert_point_close(aimed.view_target(), [2.0, 3.0, -4.0]);
        assert_eq!(
            camera.targeted_at([f64::NAN, 0.0, 0.0]),
            Err(CameraError::NonFinite)
        );
    }

    #[test]
    fn sphere_framing_rejects_invalid_geometry_and_projection() {
        for arguments in [
            [0.0, 1.0, 1.0, 1.15],
            [1.0, 0.0, 1.0, 1.15],
            [1.0, 1.0, 0.0, 1.15],
            [1.0, 1.0, std::f64::consts::PI, 1.15],
            [1.0, 1.0, 1.0, 0.99],
            [f64::NAN, 1.0, 1.0, 1.15],
        ] {
            assert_eq!(
                framed_sphere_distance(arguments[0], arguments[1], arguments[2], arguments[3]),
                Err(CameraError::InvalidFraming)
            );
        }
    }

    #[test]
    fn horizon_locked_object_mode_suppresses_roll() {
        let mut orbit = CameraRig::default();
        orbit
            .apply_navigation(
                NavigationPreset::Object,
                NavigationFrame {
                    rotation: [0.0, 0.0, 0.5],
                    horizon_locked: true,
                    ..NavigationFrame::default()
                },
            )
            .unwrap();
        assert_point_close(orbit.basis().right, [1.0, 0.0, 0.0]);
        assert_point_close(orbit.basis().up, [0.0, 1.0, 0.0]);
        assert_point_close(orbit.basis().forward, [0.0, 0.0, -1.0]);
    }

    #[test]
    fn sphere_transport_matches_browser_behavior_oracle_and_round_trips() {
        let mut camera = CameraRig::new(
            [2.0, 0.0, 0.0],
            CameraBasis {
                right: [0.0, 0.0, 1.0],
                up: [0.0, 1.0, 0.0],
                forward: [1.0, 0.0, 0.0],
            },
            2.0,
            Some([4.0, 0.0, 0.0]),
            PerspectiveLens::default(),
        )
        .unwrap();
        let original = camera;
        let inversion = SphereReflectionState::Sphere(FocusSphere::new([0.0; 3], 1.0).unwrap());
        let local_scale = camera
            .transport_between_reflections(SphereReflectionState::Identity, inversion)
            .unwrap();
        assert_point_close(camera.eye, [0.5, 0.0, 0.0]);
        assert_point_close(camera.semantic_target.unwrap(), [0.25, 0.0, 0.0]);
        assert_point_close(camera.basis().forward, [-1.0, 0.0, 0.0]);
        assert_point_close(camera.basis().up, [0.0, 1.0, 0.0]);
        assert!((local_scale - 0.25).abs() < EPS);
        assert!((camera.control_distance - 0.25).abs() < EPS);

        camera
            .transport_between_reflections(inversion, SphereReflectionState::Identity)
            .unwrap();
        assert_point_close(camera.eye, original.eye);
        assert_point_close(
            camera.semantic_target.unwrap(),
            original.semantic_target.unwrap(),
        );
        assert_point_close(camera.basis().right, original.basis().right);
        assert_point_close(camera.basis().up, original.basis().up);
        assert_point_close(camera.basis().forward, original.basis().forward);
        assert!((camera.control_distance - original.control_distance).abs() < EPS);
    }

    #[test]
    fn target_free_transport_preserves_a_sight_tangent() {
        let mut camera = CameraRig::new(
            [2.0, 0.0, 0.0],
            CameraBasis {
                right: [0.0, 0.0, -1.0],
                up: [0.0, 1.0, 0.0],
                forward: [-1.0, 0.0, 0.0],
            },
            2.0,
            None,
            PerspectiveLens::default(),
        )
        .unwrap();
        camera
            .transport_between_reflections(
                SphereReflectionState::Identity,
                SphereReflectionState::Sphere(FocusSphere::new([0.0; 3], 1.0).unwrap()),
            )
            .unwrap();
        assert_point_close(camera.eye, [0.5, 0.0, 0.0]);
        assert_point_close(camera.basis().forward, [1.0, 0.0, 0.0]);
        assert!((camera.control_distance - 0.5).abs() < EPS);
        assert_eq!(camera.semantic_target, None);
        assert_point_close(camera.view_target(), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn spacemouse_mapping_matches_existing_hyperscope_and_blender_defaults() {
        let raw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            map_space_mouse_axes(raw, SpaceMouseMapping::default()),
            NavigationAxes {
                translation: [1.0, -2.0, 3.0],
                rotation: [-4.0, 5.0, 6.0],
            }
        );
        assert_eq!(
            map_space_mouse_axes(
                raw,
                SpaceMouseMapping {
                    preset: NavigationPreset::Fly,
                    swap_yz: false,
                    invert_pan: 0,
                    invert_rotate: 0,
                },
            ),
            NavigationAxes {
                translation: [1.0, -3.0, 2.0],
                rotation: [-4.0, 6.0, -5.0],
            }
        );
        assert_eq!(
            map_space_mouse_axes(
                raw,
                SpaceMouseMapping {
                    preset: NavigationPreset::Object,
                    swap_yz: false,
                    invert_pan: 0,
                    invert_rotate: 0,
                },
            ),
            NavigationAxes {
                translation: [-1.0, 3.0, -2.0],
                rotation: [4.0, -6.0, 5.0],
            }
        );
    }

    #[test]
    fn spacemouse_camera_conversion_matches_browser_response_and_policy() {
        let normalized_axes = [0.25, -0.5, 0.75, -1.0, 0.125, -0.25];
        let hyperscope = map_space_mouse_camera(SpaceMouseCameraInput {
            normalized_axes,
            mapping: SpaceMouseMapping::default(),
            delta_seconds: 0.25,
            registered_linear_speed: 2.0,
            move_gain: 0.5,
            rotate_gain: 4.0 / 3.0,
            horizon_lock_requested: true,
        })
        .unwrap();
        assert_eq!(hyperscope.preset, NavigationPreset::Hyperscope);
        assert_eq!(
            hyperscope.frame,
            NavigationFrame {
                translation: [0.0625, 0.125, 0.1875],
                rotation: [0.5, 0.0625, -0.125],
                dolly_log: 0.0,
                horizon_locked: false,
            }
        );

        let object = map_space_mouse_camera(SpaceMouseCameraInput {
            normalized_axes,
            mapping: SpaceMouseMapping {
                preset: NavigationPreset::Object,
                swap_yz: false,
                invert_pan: 0,
                invert_rotate: 0,
            },
            delta_seconds: 0.5,
            registered_linear_speed: 4.0,
            move_gain: 0.5,
            rotate_gain: 2.0 / 3.0,
            horizon_lock_requested: true,
        })
        .unwrap();
        assert_eq!(object.preset, NavigationPreset::Object);
        assert_eq!(
            object.frame,
            NavigationFrame {
                translation: [-0.25, 0.75, 0.0],
                rotation: [-0.5, 0.125, 0.0625],
                dolly_log: 0.1875,
                horizon_locked: true,
            }
        );

        let swapped = map_space_mouse_camera(SpaceMouseCameraInput {
            normalized_axes,
            mapping: SpaceMouseMapping {
                preset: NavigationPreset::Fly,
                swap_yz: true,
                invert_pan: 0b101,
                invert_rotate: 0b010,
            },
            delta_seconds: 0.5,
            registered_linear_speed: 2.0,
            move_gain: 1.0,
            rotate_gain: 4.0 / 3.0,
            horizon_lock_requested: true,
        })
        .unwrap();
        assert_eq!(swapped.frame.translation, [-0.25, 0.5, 0.75]);
        assert_eq!(swapped.frame.rotation, [1.0, -0.125, -0.25]);
        assert!(swapped.frame.horizon_locked);

        let drone = map_space_mouse_camera(SpaceMouseCameraInput {
            normalized_axes,
            mapping: SpaceMouseMapping {
                preset: NavigationPreset::Drone,
                swap_yz: false,
                invert_pan: 0,
                invert_rotate: 0,
            },
            delta_seconds: 0.0,
            registered_linear_speed: 0.0,
            move_gain: 0.0,
            rotate_gain: 0.0,
            horizon_lock_requested: false,
        })
        .unwrap();
        assert!(drone.frame.horizon_locked);
        assert_eq!(drone.frame.dolly_log, 0.0);
    }

    #[test]
    fn spacemouse_camera_conversion_rejects_invalid_boundary_values() {
        let valid = SpaceMouseCameraInput {
            normalized_axes: [0.0; 6],
            mapping: SpaceMouseMapping::default(),
            delta_seconds: 1.0,
            registered_linear_speed: 1.0,
            move_gain: 1.0,
            rotate_gain: 1.0,
            horizon_lock_requested: false,
        };
        assert_eq!(
            map_space_mouse_camera(SpaceMouseCameraInput {
                normalized_axes: [f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0],
                ..valid
            }),
            Err(SpaceMouseInputError::NonFinite)
        );
        assert_eq!(
            map_space_mouse_camera(SpaceMouseCameraInput {
                normalized_axes: [1.000_001, 0.0, 0.0, 0.0, 0.0, 0.0],
                ..valid
            }),
            Err(SpaceMouseInputError::AxisOutOfRange)
        );
        assert_eq!(
            map_space_mouse_camera(SpaceMouseCameraInput {
                rotate_gain: -1.0,
                ..valid
            }),
            Err(SpaceMouseInputError::NegativeResponse)
        );
        for overflowing in [
            SpaceMouseCameraInput {
                registered_linear_speed: f64::MAX,
                move_gain: 2.0,
                ..valid
            },
            SpaceMouseCameraInput {
                rotate_gain: f64::MAX,
                ..valid
            },
            SpaceMouseCameraInput {
                mapping: SpaceMouseMapping {
                    preset: NavigationPreset::Object,
                    ..valid.mapping
                },
                registered_linear_speed: 0.0,
                move_gain: f64::MAX,
                ..valid
            },
        ] {
            assert_eq!(
                map_space_mouse_camera(overflowing),
                Err(SpaceMouseInputError::NonFinite)
            );
        }
        assert_eq!(
            map_space_mouse_camera(SpaceMouseCameraInput {
                mapping: SpaceMouseMapping {
                    invert_pan: 0b1000,
                    ..valid.mapping
                },
                ..valid
            }),
            Err(SpaceMouseInputError::InvalidInversionMask)
        );
    }
}
