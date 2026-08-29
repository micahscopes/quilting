use super::{CameraBasis, CameraError, CameraRig, PerspectiveLens, SphereReflectionState};
use crate::{SurfaceAddressError, SurfaceAttachment, SurfaceSample};
use std::error::Error;
use std::fmt;

const EPSILON: f64 = 1.0e-12;

/// Orthonormal material frame evaluated at one stable source-surface address.
///
/// The frame follows the posed QB differential rather than a tessellated
/// output triangle. Its origin therefore remains the same material point while
/// animation, skinning, ordinary model transforms, and conformal transforms
/// move the displayed surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceTangentFrame {
    pub origin: [f64; 3],
    pub tangent_right: [f64; 3],
    pub normal: [f64; 3],
    pub tangent_forward: [f64; 3],
}

impl SurfaceTangentFrame {
    pub fn from_sample(
        attachment: SurfaceAttachment,
        sample: SurfaceSample,
    ) -> Result<Self, SurfaceAnchorError> {
        if !finite3(sample.output_position) {
            return Err(SurfaceAnchorError::NonFinite);
        }
        let normal = scale(
            sample.normal().ok_or(SurfaceAnchorError::DegenerateFrame)?,
            f64::from(attachment.normal_sign),
        );
        let tangent_forward =
            project_tangent(sample.tangent_u, normal).ok_or(SurfaceAnchorError::DegenerateFrame)?;
        Self::from_origin_forward_normal(sample.output_position, tangent_forward, normal)
    }

    pub fn basis(self) -> CameraBasis {
        CameraBasis {
            right: self.tangent_right,
            up: self.normal,
            forward: self.tangent_forward,
        }
    }

    /// Rotate the material heading within the tangent plane while preserving
    /// the selected normal side and origin.
    pub fn with_heading(self, heading_radians: f64) -> Result<Self, SurfaceAnchorError> {
        if !heading_radians.is_finite() {
            return Err(SurfaceAnchorError::NonFinite);
        }
        let (sin, cos) = heading_radians.sin_cos();
        let tangent_forward = add(
            scale(self.tangent_forward, cos),
            scale(self.tangent_right, sin),
        );
        Self::from_origin_forward_normal(self.origin, tangent_forward, self.normal)
    }

    pub fn world_point(self, local: [f64; 3]) -> Option<[f64; 3]> {
        self.world_direction(local)
            .map(|direction| add(self.origin, direction))
    }

    pub fn local_point(self, world: [f64; 3]) -> Option<[f64; 3]> {
        finite3(world)
            .then(|| sub(world, self.origin))
            .and_then(|direction| self.local_direction(direction))
    }

    pub fn world_direction(self, local: [f64; 3]) -> Option<[f64; 3]> {
        finite3(local).then(|| {
            add(
                add(
                    scale(self.tangent_right, local[0]),
                    scale(self.normal, local[1]),
                ),
                scale(self.tangent_forward, local[2]),
            )
        })
    }

    pub fn local_direction(self, world: [f64; 3]) -> Option<[f64; 3]> {
        finite3(world).then(|| {
            [
                dot(world, self.tangent_right),
                dot(world, self.normal),
                dot(world, self.tangent_forward),
            ]
        })
    }

    fn from_origin_forward_normal(
        origin: [f64; 3],
        tangent_forward: [f64; 3],
        normal: [f64; 3],
    ) -> Result<Self, SurfaceAnchorError> {
        if !finite3(origin) {
            return Err(SurfaceAnchorError::NonFinite);
        }
        let normal = normalize(normal).ok_or(SurfaceAnchorError::DegenerateFrame)?;
        let tangent_forward =
            project_tangent(tangent_forward, normal).ok_or(SurfaceAnchorError::DegenerateFrame)?;
        let tangent_right =
            normalize(cross(tangent_forward, normal)).ok_or(SurfaceAnchorError::DegenerateFrame)?;
        let tangent_forward =
            normalize(cross(normal, tangent_right)).ok_or(SurfaceAnchorError::DegenerateFrame)?;
        Ok(Self {
            origin,
            tangent_right,
            normal,
            tangent_forward,
        })
    }
}

/// Complete camera state expressed in an attached material frame.
///
/// Translation and orientation inputs may freely edit this relative camera;
/// re-evaluating the surface frame then carries those edits with animation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceRelativeCamera {
    pub eye_offset: [f64; 3],
    pub forward: [f64; 3],
    pub up: [f64; 3],
    pub control_distance: f64,
    pub semantic_target_offset: Option<[f64; 3]>,
    pub lens: PerspectiveLens,
}

impl SurfaceRelativeCamera {
    pub fn capture(
        frame: SurfaceTangentFrame,
        camera: CameraRig,
    ) -> Result<Self, SurfaceAnchorError> {
        camera.validate()?;
        let basis = camera.basis();
        let eye_offset = frame
            .local_point(camera.eye)
            .ok_or(SurfaceAnchorError::NonFinite)?;
        let forward = frame
            .local_direction(basis.forward)
            .ok_or(SurfaceAnchorError::NonFinite)?;
        let up = frame
            .local_direction(basis.up)
            .ok_or(SurfaceAnchorError::NonFinite)?;
        CameraBasis::from_forward_up(forward, up)?;
        let semantic_target_offset = camera
            .semantic_target
            .map(|target| {
                frame
                    .local_point(target)
                    .ok_or(SurfaceAnchorError::NonFinite)
            })
            .transpose()?;
        Ok(Self {
            eye_offset,
            forward,
            up,
            control_distance: camera.control_distance,
            semantic_target_offset,
            lens: camera.lens,
        })
    }

    pub fn resolve(self, frame: SurfaceTangentFrame) -> Result<CameraRig, SurfaceAnchorError> {
        let eye = frame
            .world_point(self.eye_offset)
            .ok_or(SurfaceAnchorError::NonFinite)?;
        let forward = frame
            .world_direction(self.forward)
            .ok_or(SurfaceAnchorError::NonFinite)?;
        let up = frame
            .world_direction(self.up)
            .ok_or(SurfaceAnchorError::NonFinite)?;
        let semantic_target = self
            .semantic_target_offset
            .map(|target| {
                frame
                    .world_point(target)
                    .ok_or(SurfaceAnchorError::NonFinite)
            })
            .transpose()?;
        CameraRig::new(
            eye,
            CameraBasis::from_forward_up(forward, up)?,
            self.control_distance,
            semantic_target,
            self.lens,
        )
        .map_err(Into::into)
    }
}

/// A fixed material-point attachment for ordinary orbit/fly/drone navigation.
///
/// Walking uses the same [`SurfaceAttachment`] but advances its address through
/// the surface metric. This anchor keeps the address fixed and carries a full
/// camera pose relative to the animated material frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimatedSurfaceAnchor {
    attachment: SurfaceAttachment,
    frame: SurfaceTangentFrame,
    relative_camera: SurfaceRelativeCamera,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceAnchoredCameraFrame {
    pub attachment: SurfaceAttachment,
    pub surface_frame: SurfaceTangentFrame,
    pub camera: CameraRig,
    pub surface_velocity: [f64; 3],
}

impl AnimatedSurfaceAnchor {
    pub fn attach(
        attachment: SurfaceAttachment,
        sample: SurfaceSample,
        camera: CameraRig,
    ) -> Result<Self, SurfaceAnchorError> {
        let attachment =
            SurfaceAttachment::with_normal_sign(attachment.address, attachment.normal_sign)?;
        let frame = SurfaceTangentFrame::from_sample(attachment, sample)?;
        let relative_camera = SurfaceRelativeCamera::capture(frame, camera)?;
        Ok(Self {
            attachment,
            frame,
            relative_camera,
        })
    }

    pub fn attachment(self) -> SurfaceAttachment {
        self.attachment
    }

    pub fn frame(self) -> SurfaceTangentFrame {
        self.frame
    }

    pub fn relative_camera(self) -> SurfaceRelativeCamera {
        self.relative_camera
    }

    /// Capture an explicit navigation edit in the previously evaluated
    /// material frame. Call this before consuming the next animation sample.
    pub fn recapture_camera(&mut self, camera: CameraRig) -> Result<(), SurfaceAnchorError> {
        let relative_camera = SurfaceRelativeCamera::capture(self.frame, camera)?;
        self.relative_camera = relative_camera;
        Ok(())
    }

    /// Follow the same source address into its latest posed output frame.
    /// Invalid samples reject atomically and preserve the previous anchor.
    pub fn follow_sample(
        &mut self,
        sample: SurfaceSample,
    ) -> Result<SurfaceAnchoredCameraFrame, SurfaceAnchorError> {
        let frame = SurfaceTangentFrame::from_sample(self.attachment, sample)?;
        let camera = self.relative_camera.resolve(frame)?;
        self.frame = frame;
        Ok(SurfaceAnchoredCameraFrame {
            attachment: self.attachment,
            surface_frame: frame,
            camera,
            surface_velocity: sample.surface_velocity,
        })
    }

    /// Rebase cached output-chart state after the caller has transported the
    /// live camera through a conformal chart change. Recapturing the exact
    /// transported camera avoids treating a finite camera offset as an
    /// infinitesimal vector under a nonlinear reflection.
    pub fn rebase_after_reflection_transport(
        &mut self,
        previous: SphereReflectionState,
        next: SphereReflectionState,
        transported_camera: CameraRig,
    ) -> Result<bool, SurfaceAnchorError> {
        if previous == next {
            return Ok(false);
        }
        let transported = previous.transport_point_and_directions(
            next,
            self.frame.origin,
            [self.frame.tangent_forward, self.frame.normal],
        )?;
        let mut attachment = self.attachment;
        if previous.orientation_sign() != next.orientation_sign() {
            attachment.normal_sign = -attachment.normal_sign;
        }
        let frame = SurfaceTangentFrame::from_origin_forward_normal(
            transported.point,
            transported.directions[0],
            transported.directions[1],
        )?;
        let relative_camera = SurfaceRelativeCamera::capture(frame, transported_camera)?;
        self.attachment = attachment;
        self.frame = frame;
        self.relative_camera = relative_camera;
        Ok(true)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceAnchorError {
    NonFinite,
    InvalidAttachment,
    DegenerateFrame,
    Camera(CameraError),
}

impl fmt::Display for SurfaceAnchorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => formatter.write_str("surface-anchor values must be finite"),
            Self::InvalidAttachment => formatter.write_str("surface attachment is invalid"),
            Self::DegenerateFrame => formatter.write_str("surface tangent frame is degenerate"),
            Self::Camera(error) => write!(formatter, "surface-relative camera is invalid: {error}"),
        }
    }
}

impl Error for SurfaceAnchorError {}

impl From<SurfaceAddressError> for SurfaceAnchorError {
    fn from(_: SurfaceAddressError) -> Self {
        Self::InvalidAttachment
    }
}

impl From<CameraError> for SurfaceAnchorError {
    fn from(value: CameraError) -> Self {
        Self::Camera(value)
    }
}

fn project_tangent(direction: [f64; 3], normal: [f64; 3]) -> Option<[f64; 3]> {
    finite3(direction)
        .then(|| sub(direction, scale(normal, dot(direction, normal))))
        .and_then(normalize)
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
    (length.is_finite() && length > EPSILON).then(|| scale(value, length.recip()))
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] + right[axis])
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn scale(value: [f64; 3], amount: f64) -> [f64; 3] {
    value.map(|component| component * amount)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FocusSphere, StableEntityId, SurfaceAddress};
    use uuid::Uuid;

    fn attachment() -> SurfaceAttachment {
        SurfaceAttachment::new(
            SurfaceAddress::new(StableEntityId(Uuid::from_u128(1)), 7, [0.5, 0.25, 0.25]).unwrap(),
        )
        .unwrap()
    }

    fn sample(origin: [f64; 3], tangent_forward: [f64; 3], tangent_v: [f64; 3]) -> SurfaceSample {
        SurfaceSample {
            output_position: origin,
            tangent_u: tangent_forward,
            tangent_v,
            surface_velocity: [0.25, 0.0, -0.5],
        }
    }

    fn camera(
        eye: [f64; 3],
        forward: [f64; 3],
        up: [f64; 3],
        target: Option<[f64; 3]>,
    ) -> CameraRig {
        CameraRig::new(
            eye,
            CameraBasis::from_forward_up(forward, up).unwrap(),
            4.0,
            target,
            PerspectiveLens::default(),
        )
        .unwrap()
    }

    fn assert_vec_close(actual: [f64; 3], expected: [f64; 3]) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < 1.0e-10,
                "{actual:?} != {expected:?}"
            );
        }
    }

    #[test]
    fn fixed_address_carries_camera_with_animated_material_frame() {
        let first = sample([0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        let initial = camera(
            [1.0, 2.0, 3.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            Some([5.0, 2.0, 3.0]),
        );
        let mut anchor = AnimatedSurfaceAnchor::attach(attachment(), first, initial).unwrap();
        let address = anchor.attachment().address;

        let moved = sample([10.0, 5.0, -2.0], [0.0, 0.0, -2.0], [-2.0, 0.0, 0.0]);
        let followed = anchor.follow_sample(moved).unwrap();

        assert_eq!(followed.attachment.address, address);
        assert_vec_close(followed.camera.eye, [13.0, 7.0, -3.0]);
        assert_vec_close(followed.camera.basis().forward, [0.0, 0.0, -1.0]);
        assert_vec_close(followed.camera.basis().up, [0.0, 1.0, 0.0]);
        assert_vec_close(followed.camera.semantic_target.unwrap(), [13.0, 7.0, -7.0]);
        assert_eq!(followed.surface_velocity, moved.surface_velocity);
    }

    #[test]
    fn explicit_camera_edit_is_recaptured_before_the_next_pose() {
        let first = sample([0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        let mut anchor = AnimatedSurfaceAnchor::attach(
            attachment(),
            first,
            camera([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], None),
        )
        .unwrap();
        let edited = camera([4.0, 3.0, 2.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], None);
        anchor.recapture_camera(edited).unwrap();

        let moved = sample([10.0, 5.0, -2.0], [0.0, 0.0, -2.0], [-2.0, 0.0, 0.0]);
        let followed = anchor.follow_sample(moved).unwrap();
        assert_vec_close(followed.camera.eye, [12.0, 8.0, -6.0]);
        assert_vec_close(followed.camera.basis().forward, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn discontinuous_pose_uses_authored_material_tangent_without_shortest_path_guessing() {
        let first = sample([0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        let mut anchor = AnimatedSurfaceAnchor::attach(
            attachment(),
            first,
            camera([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], None),
        )
        .unwrap();

        let half_turn = sample([0.0; 3], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let followed = anchor.follow_sample(half_turn).unwrap();
        assert_vec_close(followed.camera.basis().forward, [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn invalid_pose_sample_does_not_mutate_anchor() {
        let first = sample([0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        let mut anchor = AnimatedSurfaceAnchor::attach(
            attachment(),
            first,
            camera([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], None),
        )
        .unwrap();
        let before = anchor;
        let invalid = sample([f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert_eq!(
            anchor.follow_sample(invalid),
            Err(SurfaceAnchorError::NonFinite)
        );
        assert_eq!(anchor, before);
    }

    #[test]
    fn reflection_rebase_preserves_exact_transported_camera_and_source_address() {
        let first = sample([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]);
        let initial = camera([1.0, 1.0, 1.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], None);
        let mut anchor = AnimatedSurfaceAnchor::attach(attachment(), first, initial).unwrap();
        let address = anchor.attachment().address;
        let previous = SphereReflectionState::Identity;
        let next = SphereReflectionState::Sphere(FocusSphere::new([0.0, 0.0, 0.0], 2.0).unwrap());
        let mut transported_camera = initial;
        transported_camera
            .transport_between_reflections(previous, next)
            .unwrap();

        assert!(anchor
            .rebase_after_reflection_transport(previous, next, transported_camera)
            .unwrap());
        assert_eq!(anchor.attachment().address, address);
        assert_eq!(anchor.attachment().normal_sign, -1);
        let resolved = anchor.relative_camera().resolve(anchor.frame()).unwrap();
        assert_vec_close(resolved.eye, transported_camera.eye);
        assert_vec_close(resolved.basis().forward, transported_camera.basis().forward);
    }
}
