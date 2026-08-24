use super::{
    CameraRig, CameraTransition, FocusNavigation, FocusSphere, NavigationFrame, NavigationPreset,
    SphereReflectionState, SurfaceAnchorTarget, SurfaceWalkRuntime, TransitionEasing,
};
use crate::{HyperscapeDiagnostics, StableEntityId};
use bevy_ecs::prelude::{Res, ResMut, Resource};
use bevy_time::{Time, Virtual};
use std::collections::VecDeque;

/// Device-neutral edits accepted by camera, selection, and focus state.
/// Browser input adapters emit these values; they do not integrate a second
/// copy of the camera or inversion sphere.
#[derive(Debug, Clone, PartialEq)]
pub enum NavigationAction {
    SetPreset(NavigationPreset),
    ApplyFrame(NavigationFrame),
    SetCamera(CameraRig),
    TransitionCamera {
        target: CameraRig,
        duration_seconds: f64,
        easing: TransitionEasing,
    },
    /// Begin a minimum-jerk glide to a newly attached surface address.
    BeginSurfaceAnchorTransition {
        target: SurfaceAnchorTarget,
        scene_radius: f64,
        duration_seconds: f64,
        easing: TransitionEasing,
    },
    /// Refresh the destination pose while its animated surface moves.
    UpdateSurfaceAnchorTarget(SurfaceAnchorTarget),
    CancelSurfaceAnchorTransition,
    AnchorFocus {
        entity: StableEntityId,
        source_bound: FocusSphere,
        margin: f64,
        duration_seconds: f64,
        easing: TransitionEasing,
    },
    DetachFocus,
    /// Replace detached focus geometry without changing focus/inversion mode.
    SetFreeFocusSphere(FocusSphere),
    TransitionFreeFocusSphere {
        target: FocusSphere,
        duration_seconds: f64,
        easing: TransitionEasing,
    },
    TranslateFocus([f64; 3]),
    ScaleFocusLog(f64),
    SetFocusEnabled(bool),
    SetFocusField {
        coordinate: f64,
        angular_aperture: f64,
    },
    SetInversionEnabled(bool),
    ToggleInversion,
}

/// Sequence and virtual-clock time make input recordings independent of DOM
/// event ordering and render-frame cadence.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledNavigationAction {
    pub sequence: u64,
    pub at_seconds: f64,
    pub action: NavigationAction,
}

#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct NavigationActionQueue {
    next_sequence: u64,
    pending: VecDeque<ScheduledNavigationAction>,
}

/// Standalone owner for the same navigation state used by the ECS plugin.
/// WASM, tests, tools, and future non-Bevy consumers use this instead of
/// constructing a parallel integration loop.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NavigationController {
    pub queue: NavigationActionQueue,
    pub runtime: NavigationRuntime,
    pub surface_walk: SurfaceWalkRuntime,
    pub camera: CameraRig,
    pub focus: FocusNavigation,
    pub diagnostics: HyperscapeDiagnostics,
}

impl NavigationController {
    pub fn elapsed_seconds(&self) -> f64 {
        self.runtime.integrated_until_seconds
    }

    pub fn push(&mut self, action: NavigationAction) -> Result<u64, &'static str> {
        self.queue.push(self.elapsed_seconds(), action)
    }

    /// Sequence that will be assigned by the next locally-authored action.
    ///
    /// Adapters that share this controller with presentation or replay must
    /// never maintain a parallel counter: every producer advances this one
    /// queue authority.
    pub fn next_sequence(&self) -> u64 {
        self.queue.next_sequence
    }

    pub fn schedule(
        &mut self,
        at_seconds: f64,
        action: NavigationAction,
    ) -> Result<u64, &'static str> {
        self.queue.push(at_seconds, action)
    }

    pub fn tick(&mut self, delta_seconds: f64) -> Result<(), &'static str> {
        if !delta_seconds.is_finite() || delta_seconds < 0.0 {
            return Err("navigation tick must be finite and nonnegative");
        }
        self.advance_to(self.elapsed_seconds() + delta_seconds)
    }

    pub fn advance_to(&mut self, elapsed_seconds: f64) -> Result<(), &'static str> {
        if !elapsed_seconds.is_finite() || elapsed_seconds < self.elapsed_seconds() {
            return Err("navigation time must be finite and monotonic");
        }
        integrate_navigation_to(
            elapsed_seconds,
            &mut self.queue,
            &mut self.runtime,
            &mut self.surface_walk,
            &mut self.camera,
            &mut self.focus,
            &mut self.diagnostics,
        );
        Ok(())
    }
}

impl NavigationActionQueue {
    pub fn push(&mut self, at_seconds: f64, action: NavigationAction) -> Result<u64, &'static str> {
        let sequence = self.next_sequence;
        self.insert(ScheduledNavigationAction {
            sequence,
            at_seconds,
            action,
        })?;
        Ok(sequence)
    }

    pub fn insert(&mut self, scheduled: ScheduledNavigationAction) -> Result<(), &'static str> {
        if !scheduled.at_seconds.is_finite() || scheduled.at_seconds < 0.0 {
            return Err("navigation action time must be finite and nonnegative");
        }
        if self
            .pending
            .iter()
            .any(|queued| queued.sequence == scheduled.sequence)
        {
            return Err("navigation action sequence is already queued");
        }
        let position = self
            .pending
            .iter()
            .position(|queued| queued.sequence > scheduled.sequence)
            .unwrap_or(self.pending.len());
        if position
            .checked_sub(1)
            .and_then(|previous| self.pending.get(previous))
            .is_some_and(|previous| previous.at_seconds > scheduled.at_seconds)
            || self
                .pending
                .get(position)
                .is_some_and(|next| next.at_seconds < scheduled.at_seconds)
        {
            return Err("navigation action times must be nondecreasing by sequence");
        }
        self.next_sequence = self.next_sequence.max(scheduled.sequence.saturating_add(1));
        self.pending.insert(position, scheduled);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    fn pop_due(&mut self, elapsed_seconds: f64) -> Option<ScheduledNavigationAction> {
        (self.pending.front()?.at_seconds <= elapsed_seconds)
            .then(|| self.pending.pop_front())
            .flatten()
    }
}

/// Integration state that is not part of a camera pose itself.
#[derive(Resource, Debug, Clone, PartialEq)]
pub struct NavigationRuntime {
    pub preset: NavigationPreset,
    pub reflection: SphereReflectionState,
    pub camera_transition: Option<CameraTransition>,
    pub integrated_until_seconds: f64,
    pub last_applied_sequence: Option<u64>,
}

impl Default for NavigationRuntime {
    fn default() -> Self {
        Self {
            preset: NavigationPreset::Hyperscope,
            reflection: SphereReflectionState::Identity,
            camera_transition: None,
            integrated_until_seconds: 0.0,
            last_applied_sequence: None,
        }
    }
}

pub(crate) fn apply_navigation_actions(
    time: Res<Time<Virtual>>,
    mut queue: ResMut<NavigationActionQueue>,
    mut runtime: ResMut<NavigationRuntime>,
    mut surface_walk: ResMut<SurfaceWalkRuntime>,
    mut camera: ResMut<CameraRig>,
    mut focus: ResMut<FocusNavigation>,
    mut diagnostics: ResMut<HyperscapeDiagnostics>,
) {
    integrate_navigation_to(
        time.elapsed_secs_f64(),
        &mut queue,
        &mut runtime,
        &mut surface_walk,
        &mut camera,
        &mut focus,
        &mut diagnostics,
    );
}

fn integrate_navigation_to(
    now: f64,
    queue: &mut NavigationActionQueue,
    runtime: &mut NavigationRuntime,
    surface_walk: &mut SurfaceWalkRuntime,
    camera: &mut CameraRig,
    focus: &mut FocusNavigation,
    diagnostics: &mut HyperscapeDiagnostics,
) {
    if !runtime.integrated_until_seconds.is_finite() || runtime.integrated_until_seconds > now {
        runtime.integrated_until_seconds = now;
    }

    while let Some(scheduled) = queue.pop_due(now) {
        if runtime
            .last_applied_sequence
            .is_some_and(|last| scheduled.sequence <= last)
        {
            diagnostics.0.push(format!(
                "ignored replayed navigation action {} after sequence {:?}",
                scheduled.sequence, runtime.last_applied_sequence
            ));
            continue;
        }
        let action_time = scheduled
            .at_seconds
            .clamp(runtime.integrated_until_seconds, now);
        advance_navigation(
            action_time - runtime.integrated_until_seconds,
            runtime,
            surface_walk,
            camera,
            focus,
            diagnostics,
        );
        runtime.integrated_until_seconds = action_time;
        if let Err(error) = apply_action(scheduled.action, runtime, surface_walk, camera, focus) {
            diagnostics.0.push(format!(
                "navigation action {} failed: {error}",
                scheduled.sequence
            ));
        }
        runtime.last_applied_sequence = Some(scheduled.sequence);
        reconcile_reflection(runtime, surface_walk, camera, focus, diagnostics);
    }

    advance_navigation(
        now - runtime.integrated_until_seconds,
        runtime,
        surface_walk,
        camera,
        focus,
        diagnostics,
    );
    runtime.integrated_until_seconds = now;
}

fn apply_action(
    action: NavigationAction,
    runtime: &mut NavigationRuntime,
    surface_walk: &mut SurfaceWalkRuntime,
    camera: &mut CameraRig,
    focus: &mut FocusNavigation,
) -> Result<(), String> {
    match action {
        NavigationAction::SetPreset(preset) => runtime.preset = preset,
        NavigationAction::ApplyFrame(frame) => {
            runtime.camera_transition = None;
            surface_walk.cancel_anchor_transition();
            camera
                .apply_navigation(runtime.preset, frame)
                .map_err(|error| error.to_string())?;
        }
        NavigationAction::SetCamera(target) => {
            target.validate().map_err(|error| error.to_string())?;
            runtime.camera_transition = None;
            surface_walk.cancel_anchor_transition();
            *camera = target;
        }
        NavigationAction::TransitionCamera {
            target,
            duration_seconds,
            easing,
        } => {
            target.validate().map_err(|error| error.to_string())?;
            if !duration_seconds.is_finite() || duration_seconds < 0.0 {
                return Err("camera transition duration must be finite and nonnegative".into());
            }
            if duration_seconds == 0.0 {
                runtime.camera_transition = None;
                surface_walk.cancel_anchor_transition();
                *camera = target;
            } else {
                let transition = CameraTransition::new(*camera, target, duration_seconds, easing)
                    .map_err(|error| error.to_string())?;
                // `CameraTransition::new` may intentionally convert the start
                // from a finite semantic target to an equivalent free tangent.
                // Make the live state match before a same-timestamp reflection.
                *camera = transition.start;
                runtime.camera_transition = Some(transition);
                surface_walk.cancel_anchor_transition();
            }
        }
        NavigationAction::BeginSurfaceAnchorTransition {
            target,
            scene_radius,
            duration_seconds,
            easing,
        } => {
            if !duration_seconds.is_finite() || duration_seconds < 0.0 {
                return Err(
                    "surface anchor transition duration must be finite and nonnegative".into(),
                );
            }
            let target = SurfaceAnchorTarget::new(target.camera, target.normal)
                .map_err(|error| error.to_string())?;
            runtime.camera_transition = None;
            surface_walk
                .begin_anchor_transition(camera, target, scene_radius, duration_seconds, easing)
                .map_err(|error| error.to_string())?;
        }
        NavigationAction::UpdateSurfaceAnchorTarget(target) => {
            surface_walk
                .update_anchor_target(target)
                .map_err(|_| "surface anchor transition is not active".to_owned())?;
        }
        NavigationAction::CancelSurfaceAnchorTransition => {
            surface_walk.cancel_anchor_transition();
        }
        NavigationAction::AnchorFocus {
            entity,
            source_bound,
            margin,
            duration_seconds,
            easing,
        } => focus
            .anchor_to_with_easing(entity, source_bound, margin, duration_seconds, easing)
            .map_err(str::to_owned)?,
        NavigationAction::DetachFocus => focus.detach(),
        NavigationAction::SetFreeFocusSphere(sphere) => {
            let sphere = FocusSphere::new(sphere.center, sphere.radius).map_err(str::to_owned)?;
            focus.detach();
            focus.sphere = sphere;
        }
        NavigationAction::TransitionFreeFocusSphere {
            target,
            duration_seconds,
            easing,
        } => focus
            .transition_free_to(target, duration_seconds, easing)
            .map_err(str::to_owned)?,
        NavigationAction::TranslateFocus(delta) => {
            if !focus.translate_free(delta) {
                return Err("focus sphere cannot be translated while anchored".into());
            }
        }
        NavigationAction::ScaleFocusLog(log_delta) => {
            if !log_delta.is_finite() || !focus.scale_radius(log_delta.exp()) {
                return Err("focus scale must be finite".into());
            }
        }
        NavigationAction::SetFocusEnabled(enabled) => focus.focus_enabled = enabled,
        NavigationAction::SetFocusField {
            coordinate,
            angular_aperture,
        } => {
            if !coordinate.is_finite()
                || !(0.0..=1.0).contains(&coordinate)
                || !angular_aperture.is_finite()
                || angular_aperture <= 0.0
            {
                return Err(
                    "focus coordinate must be in [0,1] and aperture must be positive".into(),
                );
            }
            focus.focus_coordinate = coordinate;
            focus.angular_aperture = angular_aperture;
        }
        NavigationAction::SetInversionEnabled(enabled) => focus.inversion_enabled = enabled,
        NavigationAction::ToggleInversion => {
            focus.toggle_inversion();
        }
    }
    Ok(())
}

fn advance_navigation(
    delta_seconds: f64,
    runtime: &mut NavigationRuntime,
    surface_walk: &mut SurfaceWalkRuntime,
    camera: &mut CameraRig,
    focus: &mut FocusNavigation,
    diagnostics: &mut HyperscapeDiagnostics,
) {
    if !delta_seconds.is_finite() || delta_seconds <= 0.0 {
        return;
    }
    if surface_walk.anchor_transition().is_some() {
        surface_walk.advance_anchor_transition(delta_seconds, camera);
    } else if let Some(transition) = runtime.camera_transition.as_mut() {
        if transition.advance(delta_seconds, camera) {
            runtime.camera_transition = None;
        }
    }
    focus.advance(delta_seconds);
    reconcile_reflection(runtime, surface_walk, camera, focus, diagnostics);
}

fn reconcile_reflection(
    runtime: &mut NavigationRuntime,
    surface_walk: &mut SurfaceWalkRuntime,
    camera: &mut CameraRig,
    focus: &FocusNavigation,
    diagnostics: &mut HyperscapeDiagnostics,
) {
    let desired = if focus.inversion_enabled {
        SphereReflectionState::Sphere(focus.sphere)
    } else {
        SphereReflectionState::Identity
    };
    if desired == runtime.reflection {
        return;
    }
    // A surface target is expressed in the prior displayed chart. The browser
    // behavior oracle cancels re-anchor glides on chart edits and asks the
    // attached walker for a fresh frame; preserve that safety rule here.
    surface_walk.cancel_anchor_transition();
    let mut transported_camera = *camera;
    let mut transported_transition = runtime.camera_transition;
    let result = transported_camera
        .transport_between_reflections(runtime.reflection, desired)
        .and_then(|_| {
            if let Some(transition) = transported_transition.as_mut() {
                transition.transport_between_reflections(runtime.reflection, desired)?;
            }
            Ok(())
        });
    match result {
        Ok(()) => {
            *camera = transported_camera;
            runtime.camera_transition = transported_transition;
            runtime.reflection = desired;
        }
        Err(error) => diagnostics.0.push(format!(
            "could not transport camera with inversion sphere: {error}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HyperscapePlugin;
    use bevy_app::App;
    use std::time::Duration;

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins(HyperscapePlugin)
            .insert_resource(Time::<Virtual>::from_max_delta(Duration::MAX));
        app
    }

    fn tick(app: &mut App, seconds: f64) {
        app.world_mut()
            .resource_mut::<Time<Virtual>>()
            .advance_by(Duration::from_secs_f64(seconds));
        app.update();
    }

    #[test]
    fn queue_orders_inserted_sequences_and_defers_future_actions() {
        let mut queue = NavigationActionQueue::default();
        queue
            .insert(ScheduledNavigationAction {
                sequence: 2,
                at_seconds: 2.0,
                action: NavigationAction::SetPreset(NavigationPreset::Drone),
            })
            .unwrap();
        queue
            .insert(ScheduledNavigationAction {
                sequence: 1,
                at_seconds: 1.0,
                action: NavigationAction::SetPreset(NavigationPreset::Fly),
            })
            .unwrap();
        assert_eq!(queue.pop_due(1.5).unwrap().sequence, 1);
        assert!(queue.pop_due(1.5).is_none());
        assert_eq!(queue.pop_due(2.0).unwrap().sequence, 2);
    }

    #[test]
    fn standalone_controller_and_ecs_use_the_same_integration_path() {
        let frame = NavigationFrame {
            translation: [1.0, 2.0, 3.0],
            rotation: [0.1, -0.2, 0.3],
            ..NavigationFrame::default()
        };
        let mut controller = NavigationController::default();
        controller
            .push(NavigationAction::ApplyFrame(frame))
            .unwrap();
        controller.tick(0.25).unwrap();

        let mut app = test_app();
        app.world_mut()
            .resource_mut::<NavigationActionQueue>()
            .push(0.0, NavigationAction::ApplyFrame(frame))
            .unwrap();
        tick(&mut app, 0.25);

        assert_eq!(controller.camera, *app.world().resource::<CameraRig>());
        assert_eq!(controller.focus, *app.world().resource::<FocusNavigation>());
        assert_eq!(
            controller.runtime,
            *app.world().resource::<NavigationRuntime>()
        );
    }

    #[test]
    fn scheduled_transition_is_independent_of_render_tick_partition() {
        fn run(steps: &[f64]) -> CameraRig {
            let mut app = test_app();
            let target = CameraRig {
                eye: [4.0, 2.0, -1.0],
                control_distance: 12.0,
                ..CameraRig::default()
            };
            app.world_mut()
                .resource_mut::<NavigationActionQueue>()
                .push(
                    0.25,
                    NavigationAction::TransitionCamera {
                        target,
                        duration_seconds: 1.0,
                        easing: TransitionEasing::SmootherStep,
                    },
                )
                .unwrap();
            for seconds in steps {
                tick(&mut app, *seconds);
            }
            *app.world().resource::<CameraRig>()
        }

        let single = run(&[1.25]);
        let partitioned = run(&[0.1; 12].into_iter().chain([0.05]).collect::<Vec<_>>());
        assert_eq!(single, partitioned);
        assert_eq!(single.eye, [4.0, 2.0, -1.0]);
    }

    #[test]
    fn toggling_inversion_transports_camera_at_the_same_action_time() {
        let mut app = test_app();
        {
            let mut focus = app.world_mut().resource_mut::<FocusNavigation>();
            focus.sphere = FocusSphere::new([0.0; 3], 1.0).unwrap();
        }
        {
            let mut camera = app.world_mut().resource_mut::<CameraRig>();
            camera.eye = [2.0, 0.0, 0.0];
            camera.control_distance = 2.0;
        }
        app.world_mut()
            .resource_mut::<NavigationActionQueue>()
            .push(0.0, NavigationAction::ToggleInversion)
            .unwrap();
        tick(&mut app, 0.0);
        assert_eq!(app.world().resource::<CameraRig>().eye, [0.5, 0.0, 0.0]);
        assert_eq!(
            app.world().resource::<NavigationRuntime>().reflection,
            SphereReflectionState::Sphere(FocusSphere::new([0.0; 3], 1.0).unwrap())
        );
    }

    #[test]
    fn animated_inversion_sphere_transports_camera_for_every_step() {
        let mut app = test_app();
        {
            let mut focus = app.world_mut().resource_mut::<FocusNavigation>();
            focus.sphere = FocusSphere::new([0.0; 3], 1.0).unwrap();
            focus.inversion_enabled = true;
        }
        {
            let mut camera = app.world_mut().resource_mut::<CameraRig>();
            camera.eye = [2.0, 0.0, 0.0];
            camera.control_distance = 2.0;
        }
        app.world_mut()
            .resource_mut::<NavigationActionQueue>()
            .push(
                0.0,
                NavigationAction::AnchorFocus {
                    entity: StableEntityId(uuid::Uuid::from_u128(7)),
                    source_bound: FocusSphere::new([1.0, 0.0, 0.0], 1.0).unwrap(),
                    margin: 1.0,
                    duration_seconds: 1.0,
                    easing: TransitionEasing::Linear,
                },
            )
            .unwrap();

        tick(&mut app, 0.5);
        assert!((app.world().resource::<CameraRig>().eye[0] - 7.0 / 6.0).abs() < 1.0e-10);
        tick(&mut app, 0.5);
        assert_eq!(app.world().resource::<CameraRig>().eye, [2.0, 0.0, 0.0]);
    }

    #[test]
    fn surface_anchor_transition_is_cadence_independent() {
        fn run(steps: &[f64]) -> CameraRig {
            let mut controller = NavigationController::default();
            let target = SurfaceAnchorTarget::new(
                CameraRig {
                    eye: [4.0, 0.0, 3.0],
                    ..CameraRig::default()
                },
                [0.0, 1.0, 0.0],
            )
            .unwrap();
            controller
                .push(NavigationAction::BeginSurfaceAnchorTransition {
                    target,
                    scene_radius: 10.0,
                    duration_seconds: 1.0,
                    easing: TransitionEasing::SmootherStep,
                })
                .unwrap();
            for seconds in steps {
                controller.tick(*seconds).unwrap();
            }
            controller.camera
        }

        let single = run(&[1.0]);
        let partitioned = run(&[0.1; 10]);
        assert_eq!(single, partitioned);
        assert_eq!(single.eye, [4.0, 0.0, 3.0]);
    }

    #[test]
    fn animated_surface_target_updates_without_restarting_glide() {
        let mut controller = NavigationController::default();
        let target = |eye| {
            SurfaceAnchorTarget::new(
                CameraRig {
                    eye,
                    ..CameraRig::default()
                },
                [0.0, 1.0, 0.0],
            )
            .unwrap()
        };
        controller
            .push(NavigationAction::BeginSurfaceAnchorTransition {
                target: target([2.0, 0.0, 3.0]),
                scene_radius: 10.0,
                duration_seconds: 1.0,
                easing: TransitionEasing::Linear,
            })
            .unwrap();
        controller.tick(0.5).unwrap();
        let elapsed = controller
            .surface_walk
            .anchor_transition()
            .unwrap()
            .elapsed_seconds;
        controller
            .push(NavigationAction::UpdateSurfaceAnchorTarget(target([
                4.0, 0.0, 3.0,
            ])))
            .unwrap();
        controller.tick(0.0).unwrap();

        assert_eq!(
            controller
                .surface_walk
                .anchor_transition()
                .unwrap()
                .elapsed_seconds,
            elapsed
        );
        controller.tick(0.5).unwrap();
        assert_eq!(controller.camera.eye, [4.0, 0.0, 3.0]);
    }

    #[test]
    fn manual_navigation_and_reflection_cancel_surface_anchor_glides() {
        let mut controller = NavigationController::default();
        let target = SurfaceAnchorTarget::new(CameraRig::default(), [0.0, 1.0, 0.0]).unwrap();
        let begin = NavigationAction::BeginSurfaceAnchorTransition {
            target,
            scene_radius: 1.0,
            duration_seconds: 1.0,
            easing: TransitionEasing::Linear,
        };
        controller.push(begin.clone()).unwrap();
        controller.tick(0.0).unwrap();
        controller
            .push(NavigationAction::ApplyFrame(NavigationFrame::default()))
            .unwrap();
        controller.tick(0.0).unwrap();
        assert!(controller.surface_walk.anchor_transition().is_none());

        controller.push(begin).unwrap();
        controller.tick(0.0).unwrap();
        controller.push(NavigationAction::ToggleInversion).unwrap();
        controller.tick(0.0).unwrap();
        assert!(controller.surface_walk.anchor_transition().is_none());
    }
}
