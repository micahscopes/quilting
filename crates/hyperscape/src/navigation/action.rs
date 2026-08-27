use super::{
    CameraRig, CameraTransition, FocusNavigation, FocusSphere, NavigationFrame, NavigationPreset,
    PerspectiveLens, SphereReflectionState, SurfaceAnchorTarget, SurfaceWalkRuntime,
    TransitionEasing, TurntableFrame,
};
use crate::HyperscapeDiagnostics;
use bevy_ecs::prelude::{Res, ResMut, Resource};
use bevy_time::{Time, Virtual};
use hyperscape_protocol::AssetEntityId;
use std::collections::VecDeque;

/// Device-neutral edits accepted by camera, selection, and focus state.
/// Browser input adapters emit these values; they do not integrate a second
/// copy of the camera or inversion sphere.
#[derive(Debug, Clone, PartialEq)]
pub enum NavigationAction {
    SetPreset(NavigationPreset),
    ApplyFrame(NavigationFrame),
    /// Apply one device-independent camera sample together with the control
    /// policy that gives it meaning. SpaceMouse, keyboard, gamepad, and XR
    /// adapters can use this boundary without racing separate preset or point-
    /// target actions. Validation, transition cancellation, and camera
    /// integration either all commit or all roll back.
    ApplyCameraIntent {
        preset: NavigationPreset,
        semantic_target_enabled: bool,
        frame: NavigationFrame,
    },
    /// Apply one pointer/trackpad-style orbit sample. Target policy, world-up
    /// yaw, rolled local pitch, screen-plane pan, dolly, and transition
    /// cancellation are one transaction.
    ApplyTurntableIntent {
        semantic_target_enabled: bool,
        frame: TurntableFrame,
    },
    SetCamera(CameraRig),
    /// Replace projection parameters without introducing lens zoom into the
    /// camera's Euclidean/conformal control distance.
    SetPerspectiveLens(PerspectiveLens),
    /// Choose whether conformal chart changes transport the current finite
    /// view target or the free sight tangent at the camera eye. Enabling this
    /// mode captures `CameraRig::view_target`, so representation changes never
    /// introduce a latent orientation jump. A genuinely different point is
    /// introduced by a complete `SetCamera`/`TransitionCamera` target.
    SetSemanticTargetEnabled(bool),
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
        identity: AssetEntityId,
        source_bound: FocusSphere,
        source_pivot: [f64; 3],
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
    /// Replace the geometric field parameters and, when supplied, its enabled
    /// state as one navigation transaction. `None` preserves the existing
    /// enabled state for older adapters and replay fixtures; interactive views
    /// should provide `Some` so a rejected field edit cannot partially toggle
    /// focus.
    SetFocusField {
        enabled: Option<bool>,
        coordinate: f64,
        angular_aperture: f64,
    },
    SetInversionEnabled(bool),
    ToggleInversion,
    /// Restart an anchored object's focus fit and toggle spherical inversion
    /// as one conformal transaction. A detached sphere is only toggled. This
    /// is the device-neutral meaning of the selection inversion gesture; the
    /// camera, any active transition, the surface follower, focus geometry,
    /// and reflection chart either all cross together or all roll back.
    RefitFocusAndToggleInversion {
        duration_seconds: f64,
        easing: TransitionEasing,
    },
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
    /// Last successfully integrated manual edit of finite-target versus
    /// free-tangent camera semantics. Presentation reconciliation observes
    /// this fence so a future or rejected action cannot preempt authored aim.
    pub last_semantic_target_policy_sequence: Option<u64>,
}

impl Default for NavigationRuntime {
    fn default() -> Self {
        Self {
            preset: NavigationPreset::Hyperscope,
            reflection: SphereReflectionState::Identity,
            camera_transition: None,
            integrated_until_seconds: 0.0,
            last_applied_sequence: None,
            last_semantic_target_policy_sequence: None,
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
        // A semantic action and the conformal transport it requests are one
        // transaction. In particular, a focus/inversion edit must not leave
        // the desired sphere in a new chart when the camera, an in-flight
        // camera transition, or the surface follower reaches that chart's
        // pole. Keep queue time/sequence outside the staged state so rejected
        // input is consumed exactly once without splitting navigation state.
        let previous_runtime = runtime.clone();
        let previous_surface_walk = surface_walk.clone();
        let previous_camera = *camera;
        let previous_focus = focus.clone();
        let changes_semantic_target_policy = matches!(
            &scheduled.action,
            NavigationAction::SetSemanticTargetEnabled(_)
                | NavigationAction::ApplyCameraIntent { .. }
                | NavigationAction::ApplyTurntableIntent { .. }
        );
        let result =
            apply_action(scheduled.action, runtime, surface_walk, camera, focus).and_then(|()| {
                reconcile_reflection(runtime, surface_walk, camera, focus)
                    .map_err(|error| error.to_string())
            });
        if let Err(error) = result {
            *runtime = previous_runtime;
            *surface_walk = previous_surface_walk;
            *camera = previous_camera;
            *focus = previous_focus;
            diagnostics.0.push(format!(
                "navigation action {} failed: {error}",
                scheduled.sequence
            ));
        } else if changes_semantic_target_policy {
            runtime.last_semantic_target_policy_sequence = Some(scheduled.sequence);
        }
        runtime.last_applied_sequence = Some(scheduled.sequence);
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
        NavigationAction::ApplyCameraIntent {
            preset,
            semantic_target_enabled,
            frame,
        } => {
            runtime.preset = preset;
            apply_semantic_target_policy(semantic_target_enabled, runtime, surface_walk, camera)?;
            runtime.camera_transition = None;
            surface_walk.cancel_anchor_transition();
            camera
                .apply_navigation(preset, frame)
                .map_err(|error| error.to_string())?;
        }
        NavigationAction::ApplyTurntableIntent {
            semantic_target_enabled,
            frame,
        } => {
            apply_semantic_target_policy(semantic_target_enabled, runtime, surface_walk, camera)?;
            runtime.camera_transition = None;
            surface_walk.cancel_anchor_transition();
            camera
                .apply_turntable(frame)
                .map_err(|error| error.to_string())?;
        }
        NavigationAction::SetCamera(target) => {
            target.validate().map_err(|error| error.to_string())?;
            runtime.camera_transition = None;
            surface_walk.cancel_anchor_transition();
            *camera = target;
        }
        NavigationAction::SetPerspectiveLens(lens) => {
            let lens = lens.validate().map_err(|error| error.to_string())?;
            camera.lens = lens;
            if let Some(transition) = runtime.camera_transition.as_mut() {
                transition.start.lens = lens;
                transition.target.lens = lens;
            }
            surface_walk.set_perspective_lens(lens);
        }
        NavigationAction::SetSemanticTargetEnabled(enabled) => {
            apply_semantic_target_policy(enabled, runtime, surface_walk, camera)?;
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
            identity,
            source_bound,
            source_pivot,
            margin,
            duration_seconds,
            easing,
        } => focus
            .anchor_to_pivot_with_easing(
                identity,
                source_bound,
                source_pivot,
                margin,
                duration_seconds,
                easing,
            )
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
            enabled,
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
            if let Some(enabled) = enabled {
                focus.focus_enabled = enabled;
            }
            focus.focus_coordinate = coordinate;
            focus.angular_aperture = angular_aperture;
        }
        NavigationAction::SetInversionEnabled(enabled) => focus.inversion_enabled = enabled,
        NavigationAction::ToggleInversion => {
            focus.toggle_inversion();
        }
        NavigationAction::RefitFocusAndToggleInversion {
            duration_seconds,
            easing,
        } => {
            focus
                .refit_anchor(duration_seconds, easing)
                .map_err(str::to_owned)?;
            focus.toggle_inversion();
        }
    }
    Ok(())
}

fn apply_semantic_target_policy(
    enabled: bool,
    runtime: &mut NavigationRuntime,
    surface_walk: &SurfaceWalkRuntime,
    camera: &mut CameraRig,
) -> Result<(), String> {
    if enabled && (surface_walk.is_active() || surface_walk.anchor_transition().is_some()) {
        return Err("point-target camera transport is unavailable while surface walking".into());
    }
    if enabled && camera.semantic_target.is_none() {
        let current_target = camera.view_target();
        camera.semantic_target = Some(current_target);
        if let Some(transition) = runtime.camera_transition.as_mut() {
            let target_target = transition.target.view_target();
            let linear = (transition.elapsed_seconds / transition.duration_seconds).clamp(0.0, 1.0);
            let sampled = transition.easing.sample(linear);
            let remaining = 1.0 - sampled;
            let virtual_start = if remaining > 1.0e-12 {
                std::array::from_fn(|axis| {
                    (current_target[axis] - sampled * target_target[axis]) / remaining
                })
            } else {
                current_target
            };
            if virtual_start.iter().any(|value| !value.is_finite()) {
                return Err("semantic target transition became non-finite".into());
            }
            // Preserve the existing eye/orientation/lens path while choosing
            // the unique finite-target start whose sample at the current clock
            // equals the live view target.
            transition.start.semantic_target = Some(virtual_start);
            transition.target.semantic_target = Some(target_target);
        }
    } else if !enabled {
        camera.semantic_target = None;
        if let Some(transition) = runtime.camera_transition.as_mut() {
            transition.start.semantic_target = None;
            transition.target.semantic_target = None;
        }
    }
    camera.validate().map_err(|error| error.to_string())
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
        // A moving enabled inversion sphere invalidates an old-chart surface
        // anchor. Advance and reconcile that chart first so cancellation does
        // not consume one render-partition-dependent slice of the glide.
        focus.advance(delta_seconds);
        reconcile_reflection_or_stop_focus_transition(
            runtime,
            surface_walk,
            camera,
            focus,
            diagnostics,
        );
        if surface_walk.anchor_transition().is_some() {
            surface_walk.advance_anchor_transition(delta_seconds, camera);
        }
    } else if let Some(transition) = runtime.camera_transition.as_mut() {
        if transition.advance(delta_seconds, camera) {
            runtime.camera_transition = None;
        }
        focus.advance(delta_seconds);
        reconcile_reflection_or_stop_focus_transition(
            runtime,
            surface_walk,
            camera,
            focus,
            diagnostics,
        );
    } else {
        focus.advance(delta_seconds);
        reconcile_reflection_or_stop_focus_transition(
            runtime,
            surface_walk,
            camera,
            focus,
            diagnostics,
        );
    }
}

fn reconcile_reflection_or_stop_focus_transition(
    runtime: &mut NavigationRuntime,
    surface_walk: &mut SurfaceWalkRuntime,
    camera: &mut CameraRig,
    focus: &mut FocusNavigation,
    diagnostics: &mut HyperscapeDiagnostics,
) {
    if let Err(error) = reconcile_reflection(runtime, surface_walk, camera, focus) {
        // Reflection transport itself is staged. The sphere has already
        // advanced, however, so restore it to the reflection that remains
        // authoritative and stop retrying the same inaccessible transition
        // every frame. Direct actions use the stronger whole-action rollback
        // in `integrate_navigation_to` above.
        if let SphereReflectionState::Sphere(sphere) = runtime.reflection {
            focus.sphere = sphere;
        }
        focus.transition = None;
        diagnostics.0.push(format!(
            "could not advance inversion sphere across a reflection pole: {error}"
        ));
    }
}

fn reconcile_reflection(
    runtime: &mut NavigationRuntime,
    surface_walk: &mut SurfaceWalkRuntime,
    camera: &mut CameraRig,
    focus: &FocusNavigation,
) -> Result<(), super::CameraError> {
    let desired = if focus.inversion_enabled {
        SphereReflectionState::Sphere(focus.sphere)
    } else {
        SphereReflectionState::Identity
    };
    if desired == runtime.reflection {
        return Ok(());
    }
    let mut transported_camera = *camera;
    let mut transported_transition = runtime.camera_transition;
    let mut transported_surface_walk = surface_walk.clone();
    let result = transported_camera
        .transport_between_reflections(runtime.reflection, desired)
        .and_then(|_| {
            if let Some(transition) = transported_transition.as_mut() {
                transition.transport_between_reflections(runtime.reflection, desired)?;
            }
            transported_surface_walk.transport_between_reflections(runtime.reflection, desired)?;
            Ok(())
        });
    result?;
    *camera = transported_camera;
    runtime.camera_transition = transported_transition;
    *surface_walk = transported_surface_walk;
    runtime.reflection = desired;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CameraBasis, HyperscapePlugin};
    use bevy_app::App;
    use hyperscape_protocol::{AssetId, EntityId};
    use std::time::Duration;

    fn selection_identity(entity: u128) -> AssetEntityId {
        AssetEntityId::new(
            AssetId::from_u128(0x6000).unwrap(),
            EntityId::from_u128(entity).unwrap(),
        )
        .unwrap()
    }

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
    fn perspective_lens_edit_survives_active_camera_and_surface_glides() {
        let lens = PerspectiveLens {
            vertical_fov_radians: 1.25,
            near: 0.002,
            far: 25_000.0,
        };
        let mut camera_glide = NavigationController::default();
        camera_glide
            .push(NavigationAction::TransitionCamera {
                target: CameraRig {
                    eye: [4.0, 1.0, 2.0],
                    ..CameraRig::default()
                },
                duration_seconds: 1.0,
                easing: TransitionEasing::Linear,
            })
            .unwrap();
        camera_glide.tick(0.25).unwrap();
        camera_glide
            .push(NavigationAction::SetPerspectiveLens(lens))
            .unwrap();
        camera_glide.tick(0.0).unwrap();
        assert_eq!(camera_glide.camera.lens, lens);
        let transition = camera_glide.runtime.camera_transition.unwrap();
        assert_eq!(transition.start.lens, lens);
        assert_eq!(transition.target.lens, lens);
        camera_glide.tick(0.75).unwrap();
        assert_eq!(camera_glide.camera.lens, lens);

        let mut surface_glide = NavigationController::default();
        surface_glide
            .push(NavigationAction::BeginSurfaceAnchorTransition {
                target: SurfaceAnchorTarget::new(
                    CameraRig {
                        eye: [2.0, 1.0, 3.0],
                        ..CameraRig::default()
                    },
                    [0.0, 1.0, 0.0],
                )
                .unwrap(),
                scene_radius: 10.0,
                duration_seconds: 1.0,
                easing: TransitionEasing::Linear,
            })
            .unwrap();
        surface_glide.tick(0.25).unwrap();
        surface_glide
            .push(NavigationAction::SetPerspectiveLens(lens))
            .unwrap();
        surface_glide.tick(0.0).unwrap();
        let transition = surface_glide.surface_walk.anchor_transition().unwrap();
        assert_eq!(surface_glide.camera.lens, lens);
        assert_eq!(transition.start.lens, lens);
        assert_eq!(transition.target.camera.lens, lens);
        surface_glide.tick(0.75).unwrap();
        assert_eq!(surface_glide.camera.lens, lens);
    }

    #[test]
    fn invalid_lens_is_consumed_without_mutating_navigation() {
        let mut controller = NavigationController::default();
        let before = controller.clone();
        controller
            .push(NavigationAction::SetPerspectiveLens(PerspectiveLens {
                vertical_fov_radians: f64::NAN,
                ..PerspectiveLens::default()
            }))
            .unwrap();
        controller.tick(0.0).unwrap();

        assert_eq!(controller.camera, before.camera);
        assert_eq!(controller.focus, before.focus);
        assert_eq!(controller.runtime.reflection, before.runtime.reflection);
        assert_eq!(controller.runtime.last_applied_sequence, Some(0));
        assert_eq!(controller.diagnostics.0.len(), 1);
        assert!(controller.diagnostics.0[0].contains("camera lens values are invalid"));
    }

    #[test]
    fn semantic_target_mode_is_explicit_and_continuous_during_an_active_glide() {
        let mut controller = NavigationController::default();
        let target = CameraRig::new(
            [1.0, 2.0, 4.0],
            CameraBasis::from_forward_up([-1.0, -0.25, -1.0], [0.0, 1.0, 0.0]).unwrap(),
            9.0,
            None,
            PerspectiveLens::default(),
        )
        .unwrap();
        controller
            .push(NavigationAction::TransitionCamera {
                target,
                duration_seconds: 1.0,
                easing: TransitionEasing::Linear,
            })
            .unwrap();
        controller.tick(0.25).unwrap();
        let settled_pose = controller.camera;
        controller
            .push(NavigationAction::SetSemanticTargetEnabled(true))
            .unwrap();
        controller.tick(0.0).unwrap();

        assert_eq!(controller.camera.eye, settled_pose.eye);
        assert_eq!(controller.camera.orientation, settled_pose.orientation);
        assert_eq!(
            controller.camera.semantic_target,
            Some(settled_pose.view_target())
        );
        let transition = controller.runtime.camera_transition.unwrap();
        assert!(transition.start.semantic_target.is_some());
        assert!(transition.target.semantic_target.is_some());
        let resampled = transition.sample(transition.elapsed_seconds / transition.duration_seconds);
        for (actual, expected) in resampled
            .semantic_target
            .unwrap()
            .into_iter()
            .zip(settled_pose.view_target())
        {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        assert_eq!(
            controller.runtime.last_semantic_target_policy_sequence,
            Some(1)
        );

        controller
            .push(NavigationAction::SetSemanticTargetEnabled(false))
            .unwrap();
        controller.tick(0.0).unwrap();
        assert_eq!(controller.camera.semantic_target, None);
        let transition = controller.runtime.camera_transition.unwrap();
        assert_eq!(transition.start.semantic_target, None);
        assert_eq!(transition.target.semantic_target, None);
        assert_eq!(
            controller.runtime.last_semantic_target_policy_sequence,
            Some(2)
        );
        controller.tick(0.75).unwrap();
        assert_eq!(controller.camera.eye, target.eye);
        assert_eq!(controller.camera.orientation, target.orientation);
        assert_eq!(controller.camera.control_distance, target.control_distance);
        assert_eq!(controller.camera.semantic_target, None);
    }

    #[test]
    fn camera_intent_commits_preset_target_policy_and_frame_atomically() {
        let mut controller = NavigationController::default();
        controller
            .push(NavigationAction::ApplyCameraIntent {
                preset: NavigationPreset::Object,
                semantic_target_enabled: true,
                frame: NavigationFrame {
                    translation: [0.2, -0.1, 0.0],
                    rotation: [0.03, -0.04, 0.0],
                    dolly_log: 0.1,
                    horizon_locked: false,
                },
            })
            .unwrap();
        controller.tick(0.0).unwrap();

        assert_eq!(controller.runtime.preset, NavigationPreset::Object);
        assert!(controller.camera.semantic_target.is_some());
        assert_eq!(
            controller.runtime.last_semantic_target_policy_sequence,
            Some(0)
        );
        assert_ne!(controller.camera.eye, CameraRig::default().eye);
        assert_ne!(
            controller.camera.control_distance,
            CameraRig::default().control_distance
        );

        let accepted_runtime = controller.runtime.clone();
        let accepted_camera = controller.camera;
        let accepted_walk = controller.surface_walk.clone();
        controller
            .push(NavigationAction::ApplyCameraIntent {
                preset: NavigationPreset::Fly,
                semantic_target_enabled: false,
                frame: NavigationFrame {
                    translation: [f64::NAN, 0.0, 0.0],
                    ..NavigationFrame::default()
                },
            })
            .unwrap();
        controller.tick(0.0).unwrap();

        assert_eq!(controller.runtime.preset, accepted_runtime.preset);
        assert_eq!(
            controller.runtime.last_semantic_target_policy_sequence,
            Some(0)
        );
        assert_eq!(controller.camera, accepted_camera);
        assert_eq!(controller.surface_walk, accepted_walk);
        assert!(controller
            .diagnostics
            .0
            .last()
            .is_some_and(|message| message.contains("finite")));
    }

    #[test]
    fn turntable_intent_commits_target_policy_and_pose_or_rolls_back() {
        let mut controller = NavigationController::default();
        controller
            .push(NavigationAction::ApplyTurntableIntent {
                semantic_target_enabled: true,
                frame: TurntableFrame {
                    pan: [0.2, -0.1],
                    pitch: 0.03,
                    yaw: -0.04,
                    dolly_log: 0.1,
                },
            })
            .unwrap();
        controller.tick(0.0).unwrap();

        assert!(controller.camera.semantic_target.is_some());
        assert_eq!(
            controller.runtime.last_semantic_target_policy_sequence,
            Some(0)
        );
        assert_ne!(controller.camera.eye, CameraRig::default().eye);
        let accepted_runtime = controller.runtime.clone();
        let accepted_camera = controller.camera;
        let accepted_walk = controller.surface_walk.clone();

        controller
            .push(NavigationAction::ApplyTurntableIntent {
                semantic_target_enabled: false,
                frame: TurntableFrame {
                    pan: [f64::NAN, 0.0],
                    ..TurntableFrame::default()
                },
            })
            .unwrap();
        controller.tick(0.0).unwrap();

        assert_eq!(controller.runtime.preset, accepted_runtime.preset);
        assert_eq!(
            controller.runtime.last_semantic_target_policy_sequence,
            Some(0)
        );
        assert_eq!(controller.camera, accepted_camera);
        assert_eq!(controller.surface_walk, accepted_walk);
        assert!(controller
            .diagnostics
            .0
            .last()
            .is_some_and(|message| message.contains("finite")));
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
    fn scheduled_anchor_focus_commits_identity_bound_pivot_and_transition_together() {
        let mut controller = NavigationController::default();
        let identity = selection_identity(17);
        let source_bound = FocusSphere::new([1.0, -2.0, 0.5], 0.75).unwrap();
        let source_pivot = [1.2, -1.8, 0.4];
        controller
            .push(NavigationAction::AnchorFocus {
                identity,
                source_bound,
                source_pivot,
                margin: 1.1,
                duration_seconds: 0.7,
                easing: TransitionEasing::SmoothStep,
            })
            .unwrap();

        controller.tick(0.0).unwrap();

        let anchor = controller.focus.anchor.unwrap();
        assert_eq!(anchor.identity, identity);
        assert_eq!(anchor.source_bound, source_bound);
        assert_eq!(anchor.source_pivot, source_pivot);
        assert_eq!(anchor.margin, 1.1);
        assert_eq!(controller.focus.transition.unwrap().duration_seconds, 0.7);
        assert!(controller.focus.focus_enabled);
        assert_eq!(controller.runtime.last_applied_sequence, Some(0));
    }

    #[test]
    fn complete_focus_field_edit_commits_or_rolls_back_as_one_action() {
        let mut controller = NavigationController::default();
        controller
            .push(NavigationAction::SetFocusField {
                enabled: Some(true),
                coordinate: 0.375,
                angular_aperture: 0.08,
            })
            .unwrap();
        controller.tick(0.0).unwrap();

        assert!(controller.focus.focus_enabled);
        assert_eq!(controller.focus.focus_coordinate, 0.375);
        assert_eq!(controller.focus.angular_aperture, 0.08);

        let accepted = controller.focus.clone();
        controller
            .push(NavigationAction::SetFocusField {
                enabled: Some(false),
                coordinate: f64::NAN,
                angular_aperture: 0.12,
            })
            .unwrap();
        controller.tick(0.0).unwrap();

        assert_eq!(controller.focus, accepted);
        assert!(controller
            .diagnostics
            .0
            .last()
            .is_some_and(|message| message.contains("focus coordinate")));
    }

    #[test]
    fn selected_inversion_gesture_refits_and_crosses_chart_atomically() {
        let mut controller = NavigationController::default();
        let identity = selection_identity(18);
        let source_bound = FocusSphere::new([1.0, 0.5, 0.0], 0.75).unwrap();
        let source_pivot = [1.2, 0.4, 0.1];
        controller
            .push(NavigationAction::AnchorFocus {
                identity,
                source_bound,
                source_pivot,
                margin: 1.1,
                duration_seconds: 0.8,
                easing: TransitionEasing::Linear,
            })
            .unwrap();
        controller.tick(0.3).unwrap();
        let live_sphere = controller.focus.sphere;
        let camera_before_toggle = controller.camera;

        controller
            .push(NavigationAction::RefitFocusAndToggleInversion {
                duration_seconds: 1.25,
                easing: TransitionEasing::SmootherStep,
            })
            .unwrap();
        controller.tick(0.0).unwrap();

        let transition = controller.focus.transition.unwrap();
        assert_eq!(controller.focus.anchor.unwrap().identity, identity);
        assert_eq!(transition.start, live_sphere);
        assert_eq!(transition.target.center, source_bound.center);
        assert_eq!(transition.target.radius, source_bound.radius * 1.1);
        assert_eq!(transition.duration_seconds, 1.25);
        assert!(controller.focus.inversion_enabled);
        assert_eq!(
            controller.runtime.reflection,
            SphereReflectionState::Sphere(live_sphere)
        );
        assert_ne!(controller.camera.eye, camera_before_toggle.eye);
        assert_eq!(controller.runtime.last_applied_sequence, Some(1));
    }

    #[test]
    fn selected_inversion_pole_rejects_refit_toggle_as_one_action() {
        let mut controller = NavigationController::default();
        controller
            .push(NavigationAction::AnchorFocus {
                identity: selection_identity(19),
                source_bound: FocusSphere::new(controller.camera.eye, 2.0).unwrap(),
                source_pivot: controller.camera.eye,
                margin: 1.1,
                duration_seconds: 0.0,
                easing: TransitionEasing::SmootherStep,
            })
            .unwrap();
        controller.tick(0.0).unwrap();
        let before_camera = controller.camera;
        let before_focus = controller.focus.clone();
        let before_reflection = controller.runtime.reflection;

        controller
            .push(NavigationAction::RefitFocusAndToggleInversion {
                duration_seconds: 0.7,
                easing: TransitionEasing::SmootherStep,
            })
            .unwrap();
        controller.tick(0.0).unwrap();

        assert_eq!(controller.camera, before_camera);
        assert_eq!(controller.focus, before_focus);
        assert_eq!(controller.runtime.reflection, before_reflection);
        assert_eq!(controller.runtime.last_applied_sequence, Some(1));
        assert!(controller.diagnostics.0[0]
            .contains("camera transport reached a spherical-reflection pole"));
    }

    #[test]
    fn invalid_anchor_pivot_is_consumed_without_partial_selection() {
        let mut controller = NavigationController::default();
        let before = controller.focus.clone();
        controller
            .push(NavigationAction::AnchorFocus {
                identity: selection_identity(17),
                source_bound: FocusSphere::new([0.0; 3], 1.0).unwrap(),
                source_pivot: [f64::INFINITY, 0.0, 0.0],
                margin: 1.1,
                duration_seconds: 0.7,
                easing: TransitionEasing::SmootherStep,
            })
            .unwrap();

        controller.tick(0.0).unwrap();

        assert_eq!(controller.focus, before);
        assert_eq!(controller.runtime.last_applied_sequence, Some(0));
        assert!(controller.diagnostics.0[0].contains("source pivot must be finite"));
    }

    #[test]
    fn inversion_pole_rejects_the_complete_navigation_action() {
        let mut controller = NavigationController::default();
        controller.focus.sphere = FocusSphere::new([0.0, 0.0, 3.0], 2.0).unwrap();
        let before_camera = controller.camera;
        let before_focus = controller.focus.clone();
        let before_reflection = controller.runtime.reflection;
        let before_surface_walk = controller.surface_walk.clone();

        controller
            .push(NavigationAction::SetInversionEnabled(true))
            .unwrap();
        controller.tick(0.0).unwrap();

        assert_eq!(controller.camera, before_camera);
        assert_eq!(controller.focus, before_focus);
        assert_eq!(controller.runtime.reflection, before_reflection);
        assert_eq!(controller.surface_walk, before_surface_walk);
        assert_eq!(controller.runtime.last_applied_sequence, Some(0));
        assert_eq!(controller.queue.len(), 0);
        assert_eq!(controller.diagnostics.0.len(), 1);
        assert!(controller.diagnostics.0[0].contains(
            "navigation action 0 failed: camera transport reached a spherical-reflection pole"
        ));
    }

    #[test]
    fn animated_inversion_pole_stops_at_the_last_coherent_sphere() {
        let mut controller = NavigationController::default();
        controller.camera.eye = [0.0, 0.0, 3.0];
        controller.focus.sphere = FocusSphere::new([0.0, 0.0, 0.0], 1.0).unwrap();
        controller
            .push(NavigationAction::SetInversionEnabled(true))
            .unwrap();
        controller.tick(0.0).unwrap();
        let coherent_sphere = controller.focus.sphere;
        let coherent_reflection = controller.runtime.reflection;

        controller
            .push(NavigationAction::TransitionFreeFocusSphere {
                target: FocusSphere::new([0.0, 0.0, 3.0], 1.0).unwrap(),
                duration_seconds: 1.0,
                easing: TransitionEasing::Linear,
            })
            .unwrap();
        controller.tick(1.0).unwrap();

        assert_eq!(controller.focus.sphere, coherent_sphere);
        assert_eq!(controller.runtime.reflection, coherent_reflection);
        assert!(controller.focus.transition.is_none());
        assert_eq!(controller.camera.eye, [0.0, 0.0, 1.0 / 3.0]);
        assert!(controller.diagnostics.0.iter().any(|message| message
            .contains("could not advance inversion sphere across a reflection pole")));
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
                    identity: selection_identity(7),
                    source_bound: FocusSphere::new([1.0, 0.0, 0.0], 1.0).unwrap(),
                    source_pivot: [1.0, 0.0, 0.0],
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
    fn animated_reflection_cancels_surface_anchor_before_any_partitioned_progress() {
        fn run(steps: &[f64]) -> NavigationController {
            let mut controller = NavigationController::default();
            controller
                .push(NavigationAction::SetInversionEnabled(true))
                .unwrap();
            controller.tick(0.0).unwrap();
            let target = SurfaceAnchorTarget::new(
                CameraRig {
                    eye: [4.0, 1.0, 3.0],
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
                    easing: TransitionEasing::Linear,
                })
                .unwrap();
            controller
                .push(NavigationAction::TransitionFreeFocusSphere {
                    target: FocusSphere::new([0.75, 0.25, -0.5], 3.0).unwrap(),
                    duration_seconds: 1.0,
                    easing: TransitionEasing::Linear,
                })
                .unwrap();
            controller.tick(0.0).unwrap();
            assert!(controller.surface_walk.anchor_transition().is_some());
            for seconds in steps {
                controller.tick(*seconds).unwrap();
            }
            controller
        }

        let single = run(&[0.5]);
        let partitioned = run(&[0.1; 5]);
        for (single, partitioned) in single
            .camera
            .eye
            .into_iter()
            .chain(single.camera.basis().right)
            .chain(single.camera.basis().up)
            .chain(single.camera.basis().forward)
            .chain([single.camera.control_distance])
            .zip(
                partitioned
                    .camera
                    .eye
                    .into_iter()
                    .chain(partitioned.camera.basis().right)
                    .chain(partitioned.camera.basis().up)
                    .chain(partitioned.camera.basis().forward)
                    .chain([partitioned.camera.control_distance]),
            )
        {
            assert!((single - partitioned).abs() < 1.0e-12);
        }
        assert_eq!(single.camera.lens, partitioned.camera.lens);
        assert_eq!(
            single.camera.semantic_target,
            partitioned.camera.semantic_target
        );
        assert_eq!(single.focus, partitioned.focus);
        assert_eq!(single.runtime.reflection, partitioned.runtime.reflection);
        assert_eq!(single.surface_walk, partitioned.surface_walk);
        assert!(single.surface_walk.anchor_transition().is_none());
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
