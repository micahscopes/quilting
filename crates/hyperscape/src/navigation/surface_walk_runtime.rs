use super::{
    CameraError, CameraRig, PerspectiveLens, SphereReflectionState, SurfaceAnchorTarget,
    SurfaceAnchorTransition, SurfaceWalkController, SurfaceWalkControls, SurfaceWalkError,
    SurfaceWalkFrame, SurfaceWalkInput, SurfaceWalkMotion, TransitionEasing,
};
use crate::{
    StableEntityId, SurfaceAddress, SurfaceAdvance, SurfaceAttachment, SurfaceDetachReason,
    SurfaceField, SurfaceWalker, SurfaceWalkerStatus,
};
use bevy_ecs::prelude::Resource;
use std::error::Error;
use std::fmt;

/// Complete semantic request for attaching to a stable source-surface address.
///
/// A missing `normal_sign` selects the side containing the current camera. An
/// explicit sign is useful when restoring a durable attachment whose physical
/// side is already known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceWalkAttachRequest {
    pub address: SurfaceAddress,
    pub normal_sign: Option<i8>,
    pub scene_radius: f64,
    pub controls: SurfaceWalkControls,
    pub transition_duration_seconds: f64,
    pub transition_easing: TransitionEasing,
}

/// One device-neutral walking frame. Raw keyboard, HID, gamepad, and network
/// events are mapped into this request before they reach Hyperscape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceWalkStepRequest {
    pub delta_seconds: f64,
    pub scene_radius: f64,
    pub controls: SurfaceWalkControls,
    pub input: SurfaceWalkInput,
    pub orient: bool,
    /// Recapture the user's heading/pitch after an explicit camera edit. This
    /// also cancels an in-flight anchor glide, matching the browser oracle.
    pub capture_relative_view: bool,
}

/// Broad-phase recovery followed by exact attachment and the same camera-side
/// selection used for an ordinary pointer attachment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceWalkRecoveryRequest {
    pub entity: StableEntityId,
    pub output_position: [f64; 3],
    pub maximum_distance: f64,
    pub scene_radius: f64,
    pub controls: SurfaceWalkControls,
    pub transition_duration_seconds: f64,
    pub transition_easing: TransitionEasing,
}

/// Atomic result of an attachment or walking frame.
///
/// `target_frame` is the contact-following destination before an optional
/// re-anchor glide. `camera` is the camera to render now. A detached result has
/// neither motion nor target frame and preserves the incoming camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceWalkUpdate {
    pub advance: SurfaceAdvance,
    pub motion: Option<SurfaceWalkMotion>,
    pub target_frame: Option<SurfaceWalkFrame>,
    pub camera: CameraRig,
    pub anchor_transition_remaining_seconds: Option<f64>,
}

impl SurfaceWalkUpdate {
    pub fn status(self) -> SurfaceWalkerStatus {
        self.advance.status
    }

    pub fn is_attached(self) -> bool {
        self.status() == SurfaceWalkerStatus::Attached
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceWalkRuntimeError {
    InvalidAttachment,
    RecoveryUnavailable,
    SampleUnavailable,
    Walk(SurfaceWalkError),
    Transition(CameraError),
}

impl fmt::Display for SurfaceWalkRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAttachment => formatter.write_str("surface attachment is invalid"),
            Self::RecoveryUnavailable => formatter.write_str("surface recovery found no address"),
            Self::SampleUnavailable => formatter.write_str("surface sample is unavailable"),
            Self::Walk(error) => write!(formatter, "{error}"),
            Self::Transition(error) => write!(formatter, "surface transition is invalid: {error}"),
        }
    }
}

impl Error for SurfaceWalkRuntimeError {}

impl From<SurfaceWalkError> for SurfaceWalkRuntimeError {
    fn from(value: SurfaceWalkError) -> Self {
        Self::Walk(value)
    }
}

/// Single owner for topology attachment, metric locomotion, animated contact
/// response, body/eye scale, and the surface re-anchor camera transition.
///
/// The runtime uses copy-on-commit internally. A malformed semantic request or
/// invalid transition leaves both this state and the camera untouched. A
/// topology failure is different: it commits a coordinated detach, resets the
/// contact response, and cancels the transition in the same operation.
#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct SurfaceWalkRuntime {
    walker: SurfaceWalker,
    controller: SurfaceWalkController,
    anchor_transition: Option<SurfaceAnchorTransition>,
}

/// Observable result of carrying an attached walker into a new reflection
/// chart. The stable source address never changes during this operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SurfaceWalkReflectionTransport {
    pub attached: bool,
    pub follower_transported: bool,
    pub normal_side_flipped: bool,
    pub anchor_transition_cancelled: bool,
}

impl SurfaceWalkRuntime {
    pub fn attachment(&self) -> Option<SurfaceAttachment> {
        self.walker.attachment()
    }

    pub fn last_detach_reason(&self) -> Option<SurfaceDetachReason> {
        self.walker.last_detach_reason()
    }

    pub fn is_active(&self) -> bool {
        self.attachment().is_some()
    }

    pub fn anchor_transition(&self) -> Option<SurfaceAnchorTransition> {
        self.anchor_transition
    }

    /// Keep an in-flight re-anchor glide on the same projection contract as
    /// the live camera. Surface walking may still refine the near plane from
    /// eye height on its next exact contact step.
    pub fn set_perspective_lens(&mut self, lens: PerspectiveLens) {
        if let Some(transition) = self.anchor_transition.as_mut() {
            transition.start.lens = lens;
            transition.target.camera.lens = lens;
        }
    }

    pub fn detach(&mut self, reason: SurfaceDetachReason) {
        self.walker.detach(reason);
        self.controller.reset();
        self.anchor_transition = None;
    }

    /// Transport retained displayed-chart state without changing the stable
    /// source address. This matches the current browser oracle by cancelling
    /// a re-anchor glide authored in the old chart; the next posed sample
    /// retargets the attached camera from the transported follower state.
    /// All changes are staged so a reflection pole cannot split topology,
    /// follower, side, and transition state.
    pub fn transport_between_reflections(
        &mut self,
        previous: SphereReflectionState,
        next: SphereReflectionState,
    ) -> Result<SurfaceWalkReflectionTransport, CameraError> {
        let attached = self.is_active();
        if previous == next {
            return Ok(SurfaceWalkReflectionTransport {
                attached,
                ..SurfaceWalkReflectionTransport::default()
            });
        }
        let mut candidate = self.clone();
        let follower_transported = candidate
            .controller
            .transport_between_reflections(previous, next)?;
        let normal_side_flipped =
            attached && previous.orientation_sign() != next.orientation_sign();
        if normal_side_flipped {
            candidate.walker.flip_normal_side();
        }
        let anchor_transition_cancelled = candidate.anchor_transition.take().is_some();
        *self = candidate;
        Ok(SurfaceWalkReflectionTransport {
            attached,
            follower_transported,
            normal_side_flipped,
            anchor_transition_cancelled,
        })
    }

    /// Preserve the physical side across an orientation-reversing chart edit.
    /// Filter history and the old-chart transition are deliberately discarded;
    /// the next posed sample establishes a fresh output-chart response.
    pub fn flip_normal_side(&mut self) -> bool {
        if self.attachment().is_none() {
            return false;
        }
        self.walker.flip_normal_side();
        self.controller.reset();
        self.anchor_transition = None;
        true
    }

    pub fn cancel_anchor_transition(&mut self) -> bool {
        self.anchor_transition.take().is_some()
    }

    pub fn begin_anchor_transition(
        &mut self,
        camera: &mut CameraRig,
        target: SurfaceAnchorTarget,
        scene_radius: f64,
        duration_seconds: f64,
        easing: TransitionEasing,
    ) -> Result<(), CameraError> {
        if !duration_seconds.is_finite() || duration_seconds < 0.0 {
            return Err(CameraError::InvalidTransition);
        }
        let target = SurfaceAnchorTarget::new(target.camera, target.normal)?;
        if duration_seconds == 0.0 {
            self.anchor_transition = None;
            *camera = target.camera;
        } else {
            let transition = SurfaceAnchorTransition::new(
                *camera,
                target,
                scene_radius,
                duration_seconds,
                easing,
            )?;
            *camera = transition.start;
            self.anchor_transition = Some(transition);
        }
        Ok(())
    }

    pub fn update_anchor_target(&mut self, target: SurfaceAnchorTarget) -> Result<(), CameraError> {
        self.anchor_transition
            .as_mut()
            .ok_or(CameraError::InvalidTransition)?
            .retarget(target)
    }

    pub fn advance_anchor_transition(
        &mut self,
        delta_seconds: f64,
        camera: &mut CameraRig,
    ) -> bool {
        let Some(transition) = self.anchor_transition.as_mut() else {
            return false;
        };
        let completed = transition.advance(delta_seconds, camera);
        if completed {
            self.anchor_transition = None;
        }
        completed
    }

    pub fn attach<F: SurfaceField>(
        &mut self,
        camera: &mut CameraRig,
        request: SurfaceWalkAttachRequest,
        field: &mut F,
    ) -> Result<SurfaceWalkUpdate, SurfaceWalkRuntimeError> {
        camera.validate().map_err(SurfaceWalkError::from)?;
        request.controls.metrics(request.scene_radius, false)?;
        if !request.transition_duration_seconds.is_finite()
            || request.transition_duration_seconds < 0.0
            || request
                .normal_sign
                .is_some_and(|sign| !matches!(sign, -1 | 1))
        {
            return Err(SurfaceWalkRuntimeError::InvalidAttachment);
        }
        let address = SurfaceAddress::new(
            request.address.entity,
            request.address.face,
            request.address.barycentric,
        )
        .map_err(|_| SurfaceWalkRuntimeError::InvalidAttachment)?;
        let sample = field
            .sample(address)
            .ok_or(SurfaceWalkRuntimeError::SampleUnavailable)?;
        let raw_normal = sample
            .normal()
            .ok_or(SurfaceWalkRuntimeError::SampleUnavailable)?;
        let normal_sign = request.normal_sign.unwrap_or_else(|| {
            if dot(sub(camera.eye, sample.output_position), raw_normal) < 0.0 {
                -1
            } else {
                1
            }
        });
        request.controls.metrics(request.scene_radius, false)?;
        let attachment = SurfaceAttachment::with_normal_sign(address, normal_sign)
            .map_err(|_| SurfaceWalkRuntimeError::InvalidAttachment)?;

        let mut candidate = self.clone();
        candidate.walker.attach(attachment);
        candidate.controller.reset();
        candidate.anchor_transition = None;
        let start_camera = *camera;
        let mut target_camera = start_camera;
        let mut update = candidate.compute_step(
            &mut target_camera,
            SurfaceWalkStepRequest {
                delta_seconds: 0.0,
                scene_radius: request.scene_radius,
                controls: request.controls,
                input: SurfaceWalkInput::default(),
                orient: true,
                capture_relative_view: false,
            },
            field,
        )?;
        if !update.is_attached() {
            return Err(SurfaceWalkRuntimeError::SampleUnavailable);
        }
        let target_frame = update
            .target_frame
            .ok_or(SurfaceWalkRuntimeError::SampleUnavailable)?;
        let mut display_camera = start_camera;
        candidate
            .begin_anchor_transition(
                &mut display_camera,
                SurfaceAnchorTarget::new(target_frame.camera, target_frame.filtered_normal)
                    .map_err(SurfaceWalkRuntimeError::Transition)?,
                request.scene_radius,
                request.transition_duration_seconds,
                request.transition_easing,
            )
            .map_err(SurfaceWalkRuntimeError::Transition)?;
        update.camera = display_camera;
        update.anchor_transition_remaining_seconds = candidate.transition_remaining_seconds();
        *self = candidate;
        *camera = display_camera;
        Ok(update)
    }

    pub fn recover<F: SurfaceField>(
        &mut self,
        camera: &mut CameraRig,
        request: SurfaceWalkRecoveryRequest,
        field: &mut F,
    ) -> Result<SurfaceWalkUpdate, SurfaceWalkRuntimeError> {
        if !finite3(request.output_position)
            || !request.maximum_distance.is_finite()
            || request.maximum_distance <= 0.0
        {
            return Err(SurfaceWalkRuntimeError::InvalidAttachment);
        }
        let address = field
            .recover(
                request.entity,
                request.output_position,
                request.maximum_distance,
            )
            .ok_or(SurfaceWalkRuntimeError::RecoveryUnavailable)?;
        self.attach(
            camera,
            SurfaceWalkAttachRequest {
                address,
                normal_sign: None,
                scene_radius: request.scene_radius,
                controls: request.controls,
                transition_duration_seconds: request.transition_duration_seconds,
                transition_easing: request.transition_easing,
            },
            field,
        )
    }

    pub fn step<F: SurfaceField>(
        &mut self,
        camera: &mut CameraRig,
        request: SurfaceWalkStepRequest,
        field: &mut F,
    ) -> Result<SurfaceWalkUpdate, SurfaceWalkRuntimeError> {
        camera.validate().map_err(SurfaceWalkError::from)?;
        request
            .controls
            .metrics(request.scene_radius, request.input.fast)?;
        if !request.delta_seconds.is_finite() || request.delta_seconds < 0.0 {
            return Err(SurfaceWalkError::InvalidDelta.into());
        }
        if !request.input.forward_axis.is_finite() || !request.input.right_axis.is_finite() {
            return Err(SurfaceWalkError::NonFinite.into());
        }

        let mut candidate = self.clone();
        let mut candidate_camera = *camera;
        if request.capture_relative_view || !request.orient {
            candidate.cancel_anchor_transition();
        }
        let mut update = candidate.compute_step(&mut candidate_camera, request, field)?;
        if update.is_attached() {
            let target_frame = update
                .target_frame
                .ok_or(SurfaceWalkRuntimeError::SampleUnavailable)?;
            if candidate.anchor_transition.is_some() {
                candidate
                    .update_anchor_target(
                        SurfaceAnchorTarget::new(target_frame.camera, target_frame.filtered_normal)
                            .map_err(SurfaceWalkRuntimeError::Transition)?,
                    )
                    .map_err(SurfaceWalkRuntimeError::Transition)?;
                candidate.advance_anchor_transition(request.delta_seconds, &mut candidate_camera);
            } else {
                candidate_camera = target_frame.camera;
            }
            update.camera = candidate_camera;
            update.anchor_transition_remaining_seconds = candidate.transition_remaining_seconds();
        }
        *self = candidate;
        *camera = update.camera;
        Ok(update)
    }

    fn compute_step<F: SurfaceField>(
        &mut self,
        camera: &mut CameraRig,
        request: SurfaceWalkStepRequest,
        field: &mut F,
    ) -> Result<SurfaceWalkUpdate, SurfaceWalkRuntimeError> {
        request
            .controls
            .metrics(request.scene_radius, request.input.fast)?;
        if self.walker.attachment().is_none() {
            let reason = self
                .walker
                .last_detach_reason()
                .unwrap_or(SurfaceDetachReason::SampleUnavailable);
            return Ok(self.coordinated_detach(*camera, reason, detached_advance(reason)));
        }
        let current = self.walker.advance(0.0, [0.0; 3], field);
        let Some(current_contact) = current.contact else {
            let reason = detached_reason(current.status);
            return Ok(self.coordinated_detach(*camera, reason, current));
        };
        let motion = self.controller.plan_motion(
            camera,
            current_contact.output_normal,
            request.scene_radius,
            request.controls,
            request.input,
        )?;
        let desired_absolute_velocity = add(
            motion.desired_output_velocity,
            current_contact.surface_velocity,
        );
        let advance = self
            .walker
            .advance(request.delta_seconds, desired_absolute_velocity, field);
        let Some(contact) = advance.contact else {
            let reason = detached_reason(advance.status);
            return Ok(self.coordinated_detach(*camera, reason, advance));
        };
        let mut frame = self.controller.follow_contact(
            camera,
            &contact,
            request.scene_radius,
            request.controls,
            request.delta_seconds,
            request.orient,
            request.capture_relative_view,
        )?;
        frame.metrics = motion.metrics;
        *camera = frame.camera;
        Ok(SurfaceWalkUpdate {
            advance,
            motion: Some(motion),
            target_frame: Some(frame),
            camera: frame.camera,
            anchor_transition_remaining_seconds: self.transition_remaining_seconds(),
        })
    }

    fn coordinated_detach(
        &mut self,
        camera: CameraRig,
        reason: SurfaceDetachReason,
        advance: SurfaceAdvance,
    ) -> SurfaceWalkUpdate {
        self.detach(reason);
        SurfaceWalkUpdate {
            advance,
            motion: None,
            target_frame: None,
            camera,
            anchor_transition_remaining_seconds: None,
        }
    }

    fn transition_remaining_seconds(&self) -> Option<f64> {
        self.anchor_transition
            .map(|transition| (transition.duration_seconds - transition.elapsed_seconds).max(0.0))
    }
}

fn detached_reason(status: SurfaceWalkerStatus) -> SurfaceDetachReason {
    match status {
        SurfaceWalkerStatus::Detached(reason) => reason,
        SurfaceWalkerStatus::Attached => SurfaceDetachReason::SampleUnavailable,
    }
}

fn detached_advance(reason: SurfaceDetachReason) -> SurfaceAdvance {
    SurfaceAdvance {
        status: SurfaceWalkerStatus::Detached(reason),
        contact: None,
        projected_output_velocity: [0.0; 3],
        condition_number: f64::INFINITY,
        substeps: 0,
        edge_crossings: 0,
    }
}

fn finite3(value: [f64; 3]) -> bool {
    value.into_iter().all(f64::is_finite)
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CameraBasis, FocusSphere, NavigationAction, NavigationController, PerspectiveLens,
        SurfaceSample, TriangleAdjacency,
    };
    use uuid::Uuid;

    #[derive(Clone)]
    struct PlanarField {
        positions: Vec<[f64; 3]>,
        faces: Vec<[usize; 3]>,
        adjacency: TriangleAdjacency,
        surface_velocity: [f64; 3],
        recover_address: Option<SurfaceAddress>,
    }

    impl SurfaceField for PlanarField {
        fn sample(&mut self, address: SurfaceAddress) -> Option<SurfaceSample> {
            let face = *self.faces.get(address.face as usize)?;
            let p0 = self.positions[face[0]];
            let p1 = self.positions[face[1]];
            let p2 = self.positions[face[2]];
            Some(SurfaceSample {
                output_position: add(
                    add(
                        scale(p0, address.barycentric[0]),
                        scale(p1, address.barycentric[1]),
                    ),
                    scale(p2, address.barycentric[2]),
                ),
                tangent_u: sub(p1, p0),
                tangent_v: sub(p2, p0),
                surface_velocity: self.surface_velocity,
            })
        }

        fn cross_edge(
            &mut self,
            address_on_edge: SurfaceAddress,
            opposite_corner: usize,
        ) -> Option<SurfaceAddress> {
            self.adjacency.cross_edge(address_on_edge, opposite_corner)
        }

        fn recover(
            &mut self,
            _entity: StableEntityId,
            _output_position: [f64; 3],
            _maximum_distance: f64,
        ) -> Option<SurfaceAddress> {
            self.recover_address
        }
    }

    fn scale(value: [f64; 3], factor: f64) -> [f64; 3] {
        [value[0] * factor, value[1] * factor, value[2] * factor]
    }

    fn entity() -> StableEntityId {
        StableEntityId(Uuid::from_u128(1))
    }

    fn address(barycentric: [f64; 3]) -> SurfaceAddress {
        SurfaceAddress::new(entity(), 0, barycentric).unwrap()
    }

    fn field() -> PlanarField {
        PlanarField {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
            ],
            faces: vec![[0, 1, 2], [1, 3, 2]],
            adjacency: TriangleAdjacency::new(vec![[0, 1, 2], [1, 3, 2]]),
            surface_velocity: [0.0; 3],
            recover_address: None,
        }
    }

    fn camera(eye: [f64; 3]) -> CameraRig {
        CameraRig::new(
            eye,
            CameraBasis::CANONICAL,
            3.0,
            None,
            PerspectiveLens::default(),
        )
        .unwrap()
    }

    fn attach_request() -> SurfaceWalkAttachRequest {
        SurfaceWalkAttachRequest {
            address: address([0.6, 0.2, 0.2]),
            normal_sign: None,
            scene_radius: 1.0,
            controls: SurfaceWalkControls {
                smoothing_seconds: 0.0,
                tangent_pull_fraction: 1.0,
                ..SurfaceWalkControls::default()
            },
            transition_duration_seconds: 1.0,
            transition_easing: TransitionEasing::SmootherStep,
        }
    }

    fn step_request(delta_seconds: f64, forward_axis: f64) -> SurfaceWalkStepRequest {
        SurfaceWalkStepRequest {
            delta_seconds,
            scene_radius: 1.0,
            controls: SurfaceWalkControls {
                smoothing_seconds: 0.0,
                tangent_pull_fraction: 1.0,
                ..SurfaceWalkControls::default()
            },
            input: SurfaceWalkInput {
                forward_axis,
                right_axis: 0.0,
                fast: false,
            },
            orient: true,
            capture_relative_view: false,
        }
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn attach_selects_camera_side_and_starts_one_owned_transition() {
        let mut runtime = SurfaceWalkRuntime::default();
        let mut field = field();
        let mut camera = camera([0.2, -2.0, -0.2]);
        let start = camera;
        let update = runtime
            .attach(&mut camera, attach_request(), &mut field)
            .unwrap();

        assert!(update.is_attached());
        assert_eq!(runtime.attachment().unwrap().normal_sign, -1);
        assert_eq!(camera, start);
        assert_eq!(update.camera, start);
        assert!(update.target_frame.unwrap().camera.eye[1] < 0.0);
        close(update.anchor_transition_remaining_seconds.unwrap(), 1.0);
    }

    #[test]
    fn attached_walker_rejects_point_target_mode_and_stays_free_on_the_next_step() {
        let mut controller = NavigationController {
            camera: camera([0.2, 2.0, -0.2]),
            ..NavigationController::default()
        };
        let mut field = field();
        let mut request = attach_request();
        request.transition_duration_seconds = 0.0;
        controller
            .surface_walk
            .attach(&mut controller.camera, request, &mut field)
            .unwrap();
        let before_camera = controller.camera;
        let before_attachment = controller.surface_walk.attachment();

        controller
            .push(NavigationAction::SetSemanticTargetEnabled(true))
            .unwrap();
        controller.tick(0.0).unwrap();

        assert_eq!(controller.camera, before_camera);
        assert_eq!(controller.camera.semantic_target, None);
        assert_eq!(controller.surface_walk.attachment(), before_attachment);
        assert!(controller.diagnostics.0[0]
            .contains("point-target camera transport is unavailable while surface walking"));

        controller
            .surface_walk
            .step(&mut controller.camera, step_request(0.1, 0.0), &mut field)
            .unwrap();
        assert_eq!(controller.camera.semantic_target, None);
        assert_eq!(controller.surface_walk.attachment(), before_attachment);
    }

    #[test]
    fn step_combines_surface_velocity_and_scale_intent_atomically() {
        let mut runtime = SurfaceWalkRuntime::default();
        let mut field = field();
        field.surface_velocity = [0.4, -0.2, 0.0];
        let mut camera = camera([0.2, 2.0, -0.2]);
        let mut request = attach_request();
        request.transition_duration_seconds = 0.0;
        runtime.attach(&mut camera, request, &mut field).unwrap();
        let before = runtime.attachment().unwrap().address.barycentric;

        let mut step = step_request(0.5, 0.0);
        step.controls.body_scale_octave_steps = -100.0;
        step.controls.eye_height_octave_steps = 100.0;
        let update = runtime.step(&mut camera, step, &mut field).unwrap();

        assert_eq!(runtime.attachment().unwrap().address.barycentric, before);
        assert_eq!(
            update.advance.projected_output_velocity,
            field.surface_velocity
        );
        close(update.target_frame.unwrap().camera.lens.near, 0.0028);
    }

    #[test]
    fn boundary_failure_detaches_topology_response_and_transition_together() {
        let mut runtime = SurfaceWalkRuntime::default();
        let mut field = field();
        field.faces.truncate(1);
        field.adjacency = TriangleAdjacency::new(vec![[0, 1, 2]]);
        let mut camera = camera([0.8, 2.0, -0.1]);
        let mut request = attach_request();
        request.address = address([0.1, 0.8, 0.1]);
        runtime.attach(&mut camera, request, &mut field).unwrap();

        let update = runtime
            .step(&mut camera, step_request(3.0, 1.0), &mut field)
            .unwrap();
        assert_eq!(
            update.status(),
            SurfaceWalkerStatus::Detached(SurfaceDetachReason::Boundary)
        );
        assert!(!runtime.is_active());
        assert!(runtime.anchor_transition().is_none());
        assert_eq!(
            runtime.last_detach_reason(),
            Some(SurfaceDetachReason::Boundary)
        );
    }

    #[test]
    fn invalid_step_preserves_complete_runtime_and_camera() {
        let mut runtime = SurfaceWalkRuntime::default();
        let mut field = field();
        let mut camera = camera([0.2, 2.0, -0.2]);
        runtime
            .attach(&mut camera, attach_request(), &mut field)
            .unwrap();
        let before_runtime = runtime.clone();
        let before_camera = camera;
        let mut invalid = step_request(1.0 / 60.0, 0.0);
        invalid.controls.body_scale_octave_steps = 1.0e9;

        assert!(runtime.step(&mut camera, invalid, &mut field).is_err());
        assert_eq!(runtime, before_runtime);
        assert_eq!(camera, before_camera);
    }

    #[test]
    fn recovery_uses_exact_attach_path_and_camera_side() {
        let mut runtime = SurfaceWalkRuntime::default();
        let mut field = field();
        field.recover_address = Some(address([0.5, 0.25, 0.25]));
        let mut camera = camera([0.25, -1.0, -0.25]);
        let request = SurfaceWalkRecoveryRequest {
            entity: entity(),
            output_position: [0.25, 0.1, -0.25],
            maximum_distance: 0.5,
            scene_radius: 1.0,
            controls: SurfaceWalkControls::default(),
            transition_duration_seconds: 0.0,
            transition_easing: TransitionEasing::SmootherStep,
        };

        let update = runtime.recover(&mut camera, request, &mut field).unwrap();
        assert!(update.is_attached());
        assert_eq!(runtime.attachment().unwrap().normal_sign, -1);
        assert_eq!(
            runtime.attachment().unwrap().address,
            field.recover_address.unwrap()
        );
    }

    #[test]
    fn capture_cancels_anchor_glide_without_losing_attachment() {
        let mut runtime = SurfaceWalkRuntime::default();
        let mut field = field();
        let mut camera = camera([0.2, 2.0, -0.2]);
        runtime
            .attach(&mut camera, attach_request(), &mut field)
            .unwrap();
        assert!(runtime.anchor_transition().is_some());

        let mut step = step_request(1.0 / 60.0, 0.0);
        step.capture_relative_view = true;
        let update = runtime.step(&mut camera, step, &mut field).unwrap();
        assert!(update.is_attached());
        assert!(runtime.anchor_transition().is_none());
        assert_eq!(camera, update.target_frame.unwrap().camera);
    }

    #[test]
    fn static_surface_locomotion_is_cadence_independent() {
        fn run(steps: &[f64]) -> (SurfaceAttachment, CameraRig) {
            let mut runtime = SurfaceWalkRuntime::default();
            let mut field = field();
            let mut camera = camera([0.2, 2.0, -0.2]);
            let mut attach = attach_request();
            attach.transition_duration_seconds = 0.0;
            runtime.attach(&mut camera, attach, &mut field).unwrap();
            for delta_seconds in steps {
                runtime
                    .step(&mut camera, step_request(*delta_seconds, 1.0), &mut field)
                    .unwrap();
            }
            (runtime.attachment().unwrap(), camera)
        }

        let single = run(&[0.5]);
        let partitioned = run(&[1.0 / 120.0; 60]);
        assert_eq!(single.0.address.face, partitioned.0.address.face);
        for axis in 0..3 {
            close(
                single.0.address.barycentric[axis],
                partitioned.0.address.barycentric[axis],
            );
            close(single.1.eye[axis], partitioned.1.eye[axis]);
        }
        let single_basis = single.1.basis();
        let partitioned_basis = partitioned.1.basis();
        for (single, partitioned) in [
            (single_basis.right, partitioned_basis.right),
            (single_basis.up, partitioned_basis.up),
            (single_basis.forward, partitioned_basis.forward),
        ] {
            for axis in 0..3 {
                close(single[axis], partitioned[axis]);
            }
        }
    }

    #[test]
    fn reflection_transport_preserves_address_and_flips_side_once() {
        let mut runtime = SurfaceWalkRuntime::default();
        let mut field = field();
        let mut camera = camera([0.2, 2.0, -0.2]);
        runtime
            .attach(&mut camera, attach_request(), &mut field)
            .unwrap();
        let before = runtime.attachment().unwrap();
        assert!(runtime.anchor_transition().is_some());

        let inversion =
            SphereReflectionState::Sphere(FocusSphere::new([0.0, 0.0, 0.0], 2.0).unwrap());
        let outcome = runtime
            .transport_between_reflections(SphereReflectionState::Identity, inversion)
            .unwrap();
        let after = runtime.attachment().unwrap();
        assert_eq!(after.address, before.address);
        assert_eq!(after.normal_sign, -before.normal_sign);
        assert_eq!(
            outcome,
            SurfaceWalkReflectionTransport {
                attached: true,
                follower_transported: true,
                normal_side_flipped: true,
                anchor_transition_cancelled: true,
            }
        );
        assert!(runtime.anchor_transition().is_none());

        let same = runtime
            .transport_between_reflections(inversion, inversion)
            .unwrap();
        assert_eq!(runtime.attachment().unwrap(), after);
        assert_eq!(
            same,
            SurfaceWalkReflectionTransport {
                attached: true,
                ..SurfaceWalkReflectionTransport::default()
            }
        );
    }

    #[test]
    fn sphere_to_sphere_transport_keeps_the_physical_side() {
        let mut runtime = SurfaceWalkRuntime::default();
        let mut field = field();
        let mut camera = camera([0.2, 2.0, -0.2]);
        runtime
            .attach(&mut camera, attach_request(), &mut field)
            .unwrap();
        let before = runtime.attachment().unwrap();
        let first = SphereReflectionState::Sphere(FocusSphere::new([0.0, 0.0, 0.0], 2.0).unwrap());
        let second =
            SphereReflectionState::Sphere(FocusSphere::new([0.25, 0.0, 0.0], 3.0).unwrap());

        let outcome = runtime
            .transport_between_reflections(first, second)
            .unwrap();
        let after = runtime.attachment().unwrap();
        assert_eq!(after, before);
        assert!(outcome.follower_transported);
        assert!(!outcome.normal_side_flipped);
        assert!(outcome.anchor_transition_cancelled);
    }

    #[test]
    fn reflection_pole_preserves_the_complete_surface_runtime() {
        let mut runtime = SurfaceWalkRuntime::default();
        let mut field = field();
        let mut camera = camera([0.2, 2.0, -0.2]);
        let mut request = attach_request();
        request.address = address([1.0, 0.0, 0.0]);
        runtime.attach(&mut camera, request, &mut field).unwrap();
        let before = runtime.clone();
        let inversion =
            SphereReflectionState::Sphere(FocusSphere::new([0.0, 0.0, 0.0], 1.0).unwrap());

        assert_eq!(
            runtime.transport_between_reflections(SphereReflectionState::Identity, inversion),
            Err(CameraError::ReflectionPole)
        );
        assert_eq!(runtime, before);
    }
}
