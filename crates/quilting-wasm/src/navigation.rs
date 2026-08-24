use hyperscape::{
    CameraBasis, CameraRig, FocusNavigation, FocusSphere, NavigationAction, NavigationController,
    NavigationFrame, NavigationPreset, PerspectiveLens, PresentationRuntime, PresentationSnapshot,
    SphereReflectionState, StableEntityId, SurfaceAnchorTarget, TransitionEasing,
};
use hyperscope_app::SelectedFocusSnapshot;
use serde::Serialize;
use uuid::Uuid;
use wasm_bindgen::prelude::*;

/// Thin WASM adapter for Rust-authoritative Hyperscope navigation.
///
/// The browser is responsible for acquiring devices and scaling raw reports.
/// This class owns integration, camera/focus transitions, inversion transport,
/// and replay time. During migration it can run as a shadow oracle before its
/// snapshots become the renderer's authoritative camera state.
#[wasm_bindgen]
pub struct HyperscopeNavigation {
    controller: NavigationController,
    presentation: Option<PresentationRuntime>,
    active_presentation: Option<PresentationSnapshot>,
}

#[wasm_bindgen]
impl HyperscopeNavigation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            controller: NavigationController::default(),
            presentation: None,
            active_presentation: None,
        }
    }

    /// Bootstrap from an already-consistent browser state. This is the only
    /// direct state synchronization method; subsequent edits are actions.
    #[wasm_bindgen(js_name = synchronizeState)]
    #[allow(clippy::too_many_arguments)]
    pub fn synchronize_state(
        &mut self,
        eye: &[f64],
        forward: &[f64],
        up: &[f64],
        control_distance: f64,
        semantic_target: &[f64],
        vertical_fov_radians: f64,
        near: f64,
        far: f64,
        focus_center: &[f64],
        focus_radius: f64,
        focus_enabled: bool,
        inversion_enabled: bool,
        focus_coordinate: f64,
        angular_aperture: f64,
    ) -> Result<(), JsValue> {
        let (camera, focus) = synchronized_navigation_state(
            eye,
            forward,
            up,
            control_distance,
            semantic_target,
            vertical_fov_radians,
            near,
            far,
            focus_center,
            focus_radius,
            focus_enabled,
            inversion_enabled,
            focus_coordinate,
            angular_aperture,
        )?;

        // Re-synchronization replaces settled pose/focus and resets queued
        // sequence authority, but it must not rewind the shared virtual clock.
        // AppStore has the same clock-preserving contract.
        let elapsed_seconds = self.controller.elapsed_seconds();
        let mut controller = NavigationController::default();
        controller.advance_to(elapsed_seconds).map_err(js_error)?;
        controller.camera = camera;
        controller.focus = focus;
        controller.runtime.reflection = if inversion_enabled {
            SphereReflectionState::Sphere(controller.focus.sphere)
        } else {
            SphereReflectionState::Identity
        };
        self.controller = controller;
        Ok(())
    }

    #[wasm_bindgen(js_name = setPreset)]
    pub fn set_preset(&mut self, preset: &str) -> Result<u64, JsValue> {
        let preset = parse_preset(preset)?;
        self.push(NavigationAction::SetPreset(preset))
    }

    #[wasm_bindgen(js_name = setPerspectiveLens)]
    pub fn set_perspective_lens(
        &mut self,
        vertical_fov_radians: f64,
        near: f64,
        far: f64,
    ) -> Result<u64, JsValue> {
        self.push(NavigationAction::SetPerspectiveLens(perspective_lens(
            vertical_fov_radians,
            near,
            far,
        )?))
    }

    #[wasm_bindgen(js_name = setSemanticTargetEnabled)]
    pub fn set_semantic_target_enabled(&mut self, enabled: bool) -> Result<u64, JsValue> {
        self.push(NavigationAction::SetSemanticTargetEnabled(enabled))
    }

    #[wasm_bindgen(js_name = applyFrame)]
    pub fn apply_frame(
        &mut self,
        translation: &[f64],
        rotation: &[f64],
        dolly_log: f64,
        horizon_locked: bool,
    ) -> Result<u64, JsValue> {
        self.push(NavigationAction::ApplyFrame(NavigationFrame {
            translation: vector3(translation, "camera translation")?,
            rotation: vector3(rotation, "camera rotation")?,
            dolly_log,
            horizon_locked,
        }))
    }

    #[wasm_bindgen(js_name = transitionCamera)]
    #[allow(clippy::too_many_arguments)]
    pub fn transition_camera(
        &mut self,
        eye: &[f64],
        forward: &[f64],
        up: &[f64],
        control_distance: f64,
        semantic_target: &[f64],
        duration_seconds: f64,
        easing: &str,
    ) -> Result<u64, JsValue> {
        let target = CameraRig::new(
            vector3(eye, "camera eye")?,
            CameraBasis::from_forward_up(
                vector3(forward, "camera forward")?,
                vector3(up, "camera up")?,
            )
            .map_err(js_error)?,
            control_distance,
            optional_vector3(semantic_target, "camera target")?,
            self.controller.camera.lens,
        )
        .map_err(js_error)?;
        self.push(NavigationAction::TransitionCamera {
            target,
            duration_seconds,
            easing: parse_easing(easing)?,
        })
    }

    #[wasm_bindgen(js_name = beginSurfaceAnchorTransition)]
    #[allow(clippy::too_many_arguments)]
    pub fn begin_surface_anchor_transition(
        &mut self,
        eye: &[f64],
        forward: &[f64],
        up: &[f64],
        control_distance: f64,
        normal: &[f64],
        scene_radius: f64,
        duration_seconds: f64,
        easing: &str,
    ) -> Result<u64, JsValue> {
        let target = self.surface_anchor_target(eye, forward, up, control_distance, normal)?;
        self.push(NavigationAction::BeginSurfaceAnchorTransition {
            target,
            scene_radius,
            duration_seconds,
            easing: parse_easing(easing)?,
        })
    }

    #[wasm_bindgen(js_name = updateSurfaceAnchorTarget)]
    pub fn update_surface_anchor_target(
        &mut self,
        eye: &[f64],
        forward: &[f64],
        up: &[f64],
        control_distance: f64,
        normal: &[f64],
    ) -> Result<u64, JsValue> {
        let target = self.surface_anchor_target(eye, forward, up, control_distance, normal)?;
        self.push(NavigationAction::UpdateSurfaceAnchorTarget(target))
    }

    #[wasm_bindgen(js_name = cancelSurfaceAnchorTransition)]
    pub fn cancel_surface_anchor_transition(&mut self) -> Result<u64, JsValue> {
        self.push(NavigationAction::CancelSurfaceAnchorTransition)
    }

    #[wasm_bindgen(js_name = setFreeFocusSphere)]
    pub fn set_free_focus_sphere(&mut self, center: &[f64], radius: f64) -> Result<u64, JsValue> {
        let sphere =
            FocusSphere::new(vector3(center, "focus center")?, radius).map_err(js_error)?;
        self.push(NavigationAction::SetFreeFocusSphere(sphere))
    }

    #[wasm_bindgen(js_name = anchorFocus)]
    #[allow(clippy::too_many_arguments)]
    pub fn anchor_focus(
        &mut self,
        entity: &str,
        source_bound_center: &[f64],
        source_bound_radius: f64,
        source_pivot: &[f64],
        margin: f64,
        duration_seconds: f64,
        easing: &str,
    ) -> Result<u64, JsValue> {
        self.push(NavigationAction::AnchorFocus {
            entity: stable_entity_id(entity)?,
            source_bound: FocusSphere::new(
                vector3(source_bound_center, "focus source-bound center")?,
                source_bound_radius,
            )
            .map_err(js_error)?,
            source_pivot: vector3(source_pivot, "focus source pivot")?,
            margin,
            duration_seconds,
            easing: parse_easing(easing)?,
        })
    }

    #[wasm_bindgen(js_name = detachFocus)]
    pub fn detach_focus(&mut self) -> Result<u64, JsValue> {
        self.push(NavigationAction::DetachFocus)
    }

    #[wasm_bindgen(js_name = translateFocus)]
    pub fn translate_focus(&mut self, delta: &[f64]) -> Result<u64, JsValue> {
        self.push(NavigationAction::TranslateFocus(vector3(
            delta,
            "focus translation",
        )?))
    }

    #[wasm_bindgen(js_name = scaleFocusLog)]
    pub fn scale_focus_log(&mut self, log_delta: f64) -> Result<u64, JsValue> {
        self.push(NavigationAction::ScaleFocusLog(log_delta))
    }

    #[wasm_bindgen(js_name = setFocusEnabled)]
    pub fn set_focus_enabled(&mut self, enabled: bool) -> Result<u64, JsValue> {
        self.push(NavigationAction::SetFocusEnabled(enabled))
    }

    #[wasm_bindgen(js_name = setFocusField)]
    pub fn set_focus_field(
        &mut self,
        coordinate: f64,
        angular_aperture: f64,
    ) -> Result<u64, JsValue> {
        self.push(NavigationAction::SetFocusField {
            coordinate,
            angular_aperture,
        })
    }

    #[wasm_bindgen(js_name = setInversionEnabled)]
    pub fn set_inversion_enabled(&mut self, enabled: bool) -> Result<u64, JsValue> {
        self.push(NavigationAction::SetInversionEnabled(enabled))
    }

    #[wasm_bindgen(js_name = toggleInversion)]
    pub fn toggle_inversion(&mut self) -> Result<u64, JsValue> {
        self.push(NavigationAction::ToggleInversion)
    }

    pub fn tick(&mut self, delta_seconds: f64) -> Result<JsValue, JsValue> {
        if let Some(presentation) = self.presentation.as_mut() {
            presentation
                .tick_navigation(&mut self.controller, delta_seconds)
                .map_err(js_error)?;
        } else {
            self.controller.tick(delta_seconds).map_err(js_error)?;
        }
        self.snapshot()
    }

    pub fn snapshot(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&NavigationSnapshot::from(&self.controller))
            .map_err(|error| JsValue::from_str(&error.to_string()))
    }

    /// Validate and retain a complete presentation document. Loading data does
    /// not activate a cue or mutate the camera.
    #[wasm_bindgen(js_name = loadPresentation)]
    pub fn load_presentation(&mut self, json: &str) -> Result<JsValue, JsValue> {
        let presentation = PresentationRuntime::from_json(json).map_err(js_error)?;
        let normalized = serde_wasm_bindgen::to_value(presentation.presentation())
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        self.presentation = Some(presentation);
        self.active_presentation = None;
        Ok(normalized)
    }

    #[wasm_bindgen(js_name = startPresentation)]
    pub fn start_presentation(&mut self) -> Result<JsValue, JsValue> {
        self.activate_presentation(|presentation, navigation| {
            presentation.activate_index(0, navigation)
        })
    }

    #[wasm_bindgen(js_name = advancePresentation)]
    pub fn advance_presentation(&mut self) -> Result<JsValue, JsValue> {
        self.activate_presentation(PresentationRuntime::advance)
    }

    #[wasm_bindgen(js_name = reversePresentation)]
    pub fn reverse_presentation(&mut self) -> Result<JsValue, JsValue> {
        self.activate_presentation(PresentationRuntime::reverse)
    }

    #[wasm_bindgen(js_name = jumpToPresentationCue)]
    pub fn jump_to_presentation_cue(&mut self, cue_id: &str) -> Result<JsValue, JsValue> {
        let cue_id = Uuid::parse_str(cue_id).map_err(|error| {
            JsValue::from_str(&format!("invalid presentation cue UUID: {error}"))
        })?;
        self.activate_presentation(|presentation, navigation| {
            presentation.jump_to_cue(cue_id, navigation)
        })
    }

    #[wasm_bindgen(js_name = presentationSnapshot)]
    pub fn presentation_snapshot(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.active_presentation)
            .map_err(|error| JsValue::from_str(&error.to_string()))
    }

    #[wasm_bindgen(js_name = clearPresentation)]
    pub fn clear_presentation(&mut self) {
        self.presentation = None;
        self.active_presentation = None;
    }

    #[wasm_bindgen(js_name = clearDiagnostics)]
    pub fn clear_diagnostics(&mut self) {
        self.controller.diagnostics.0.clear();
    }
}

impl Default for HyperscopeNavigation {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperscopeNavigation {
    fn push(&mut self, action: NavigationAction) -> Result<u64, JsValue> {
        self.controller.push(action).map_err(js_error)
    }

    fn surface_anchor_target(
        &self,
        eye: &[f64],
        forward: &[f64],
        up: &[f64],
        control_distance: f64,
        normal: &[f64],
    ) -> Result<SurfaceAnchorTarget, JsValue> {
        let camera = CameraRig::new(
            vector3(eye, "surface anchor eye")?,
            CameraBasis::from_forward_up(
                vector3(forward, "surface anchor forward")?,
                vector3(up, "surface anchor up")?,
            )
            .map_err(js_error)?,
            control_distance,
            None,
            self.controller.camera.lens,
        )
        .map_err(js_error)?;
        SurfaceAnchorTarget::new(camera, vector3(normal, "surface anchor normal")?)
            .map_err(js_error)
    }

    fn activate_presentation(
        &mut self,
        activate: impl FnOnce(
            &mut PresentationRuntime,
            &mut NavigationController,
        ) -> Result<PresentationSnapshot, hyperscape::PresentationError>,
    ) -> Result<JsValue, JsValue> {
        let presentation = self
            .presentation
            .as_mut()
            .ok_or_else(|| JsValue::from_str("no presentation is loaded"))?;
        let snapshot = activate(presentation, &mut self.controller).map_err(js_error)?;
        let value = serde_wasm_bindgen::to_value(&snapshot)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        self.active_presentation = Some(snapshot);
        Ok(value)
    }
}

#[derive(Serialize)]
struct NavigationSnapshot<'a> {
    elapsed_seconds: f64,
    preset: &'static str,
    pending_actions: usize,
    last_applied_sequence: Option<u64>,
    reflection: &'static str,
    camera: CameraSnapshot,
    focus: FocusSnapshot,
    selected_focus: Option<SelectedFocusJsSnapshot>,
    diagnostics: &'a [String],
}

impl<'a> From<&'a NavigationController> for NavigationSnapshot<'a> {
    fn from(controller: &'a NavigationController) -> Self {
        let basis = controller.camera.basis();
        let camera_transition_remaining = controller
            .runtime
            .camera_transition
            .map(|transition| (transition.duration_seconds - transition.elapsed_seconds).max(0.0));
        let surface_anchor_transition_remaining = controller
            .surface_walk
            .anchor_transition()
            .map(|transition| (transition.duration_seconds - transition.elapsed_seconds).max(0.0));
        let surface_anchor_hop_height = controller
            .surface_walk
            .anchor_transition()
            .map(|transition| transition.hop_height);
        let focus_transition_remaining = controller
            .focus
            .transition
            .map(|transition| (transition.duration_seconds - transition.elapsed_seconds).max(0.0));
        Self {
            elapsed_seconds: controller.elapsed_seconds(),
            preset: preset_name(controller.runtime.preset),
            pending_actions: controller.queue.len(),
            last_applied_sequence: controller.runtime.last_applied_sequence,
            reflection: match controller.runtime.reflection {
                SphereReflectionState::Identity => "identity",
                SphereReflectionState::Sphere(_) => "sphere_reflection",
            },
            camera: CameraSnapshot {
                eye: controller.camera.eye,
                orientation: [
                    controller.camera.orientation.w,
                    controller.camera.orientation.x,
                    controller.camera.orientation.y,
                    controller.camera.orientation.z,
                ],
                right: basis.right,
                up: basis.up,
                forward: basis.forward,
                control_distance: controller.camera.control_distance,
                semantic_target: controller.camera.semantic_target,
                vertical_fov_radians: controller.camera.lens.vertical_fov_radians,
                near: controller.camera.lens.near,
                far: controller.camera.lens.far,
                camera_transition_remaining,
                surface_anchor_transition_remaining,
                surface_anchor_hop_height,
            },
            focus: FocusSnapshot {
                center: controller.focus.sphere.center,
                radius: controller.focus.sphere.radius,
                anchored: controller.focus.anchor.is_some(),
                focus_enabled: controller.focus.focus_enabled,
                inversion_enabled: controller.focus.inversion_enabled,
                focus_coordinate: controller.focus.focus_coordinate,
                angular_aperture: controller.focus.angular_aperture,
                focus_transition_remaining,
            },
            selected_focus: SelectedFocusSnapshot::from_navigation(
                &controller.focus,
                controller.runtime.reflection,
            )
            .map(SelectedFocusJsSnapshot::from),
            diagnostics: &controller.diagnostics.0,
        }
    }
}

#[derive(Serialize)]
struct CameraSnapshot {
    eye: [f64; 3],
    orientation: [f64; 4],
    right: [f64; 3],
    up: [f64; 3],
    forward: [f64; 3],
    control_distance: f64,
    semantic_target: Option<[f64; 3]>,
    vertical_fov_radians: f64,
    near: f64,
    far: f64,
    camera_transition_remaining: Option<f64>,
    surface_anchor_transition_remaining: Option<f64>,
    surface_anchor_hop_height: Option<f64>,
}

#[derive(Serialize)]
struct FocusSnapshot {
    center: [f64; 3],
    radius: f64,
    anchored: bool,
    focus_enabled: bool,
    inversion_enabled: bool,
    focus_coordinate: f64,
    angular_aperture: f64,
    focus_transition_remaining: Option<f64>,
}

#[derive(Serialize)]
pub(crate) struct SelectedFocusJsSnapshot {
    entity: String,
    source_bound_center: [f64; 3],
    source_bound_radius: f64,
    source_pivot: [f64; 3],
    margin: f64,
    output_pivot: Option<[f64; 3]>,
    output_radius: Option<f64>,
}

impl From<SelectedFocusSnapshot> for SelectedFocusJsSnapshot {
    fn from(selected: SelectedFocusSnapshot) -> Self {
        Self {
            entity: selected.entity.0.to_string(),
            source_bound_center: selected.source_bound.center,
            source_bound_radius: selected.source_bound.radius,
            source_pivot: selected.source_pivot,
            margin: selected.margin,
            output_pivot: selected.output_pivot,
            output_radius: selected.output_radius,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn synchronized_navigation_state(
    eye: &[f64],
    forward: &[f64],
    up: &[f64],
    control_distance: f64,
    semantic_target: &[f64],
    vertical_fov_radians: f64,
    near: f64,
    far: f64,
    focus_center: &[f64],
    focus_radius: f64,
    focus_enabled: bool,
    inversion_enabled: bool,
    focus_coordinate: f64,
    angular_aperture: f64,
) -> Result<(CameraRig, FocusNavigation), JsValue> {
    let eye = vector3(eye, "camera eye")?;
    let basis = CameraBasis::from_forward_up(
        vector3(forward, "camera forward")?,
        vector3(up, "camera up")?,
    )
    .map_err(js_error)?;
    let semantic_target = optional_vector3(semantic_target, "camera target")?;
    let camera = CameraRig::new(
        eye,
        basis,
        control_distance,
        semantic_target,
        perspective_lens(vertical_fov_radians, near, far)?,
    )
    .map_err(js_error)?;
    let sphere =
        FocusSphere::new(vector3(focus_center, "focus center")?, focus_radius).map_err(js_error)?;
    if !focus_coordinate.is_finite()
        || !(0.0..=1.0).contains(&focus_coordinate)
        || !angular_aperture.is_finite()
        || angular_aperture <= 0.0
    {
        return Err(JsValue::from_str(
            "focus coordinate must be in [0,1] and aperture must be positive",
        ));
    }
    Ok((
        camera,
        FocusNavigation {
            sphere,
            anchor: None,
            transition: None,
            focus_enabled,
            inversion_enabled,
            focus_coordinate,
            angular_aperture,
        },
    ))
}

pub(crate) fn vector3(values: &[f64], label: &str) -> Result<[f64; 3], JsValue> {
    let value: [f64; 3] = values
        .try_into()
        .map_err(|_| JsValue::from_str(&format!("{label} must contain exactly three values")))?;
    if value.into_iter().any(|component| !component.is_finite()) {
        return Err(JsValue::from_str(&format!("{label} must be finite")));
    }
    Ok(value)
}

pub(crate) fn stable_entity_id(value: &str) -> Result<StableEntityId, JsValue> {
    let entity = Uuid::parse_str(value)
        .map_err(|error| JsValue::from_str(&format!("focus entity must be a UUID: {error}")))?;
    if entity.is_nil() {
        return Err(JsValue::from_str("focus entity UUID must not be nil"));
    }
    Ok(StableEntityId(entity))
}

pub(crate) fn perspective_lens(
    vertical_fov_radians: f64,
    near: f64,
    far: f64,
) -> Result<PerspectiveLens, JsValue> {
    PerspectiveLens {
        vertical_fov_radians,
        near,
        far,
    }
    .validate()
    .map_err(js_error)
}

pub(crate) fn optional_vector3(values: &[f64], label: &str) -> Result<Option<[f64; 3]>, JsValue> {
    if values.is_empty() {
        Ok(None)
    } else {
        vector3(values, label).map(Some)
    }
}

pub(crate) fn parse_preset(value: &str) -> Result<NavigationPreset, JsValue> {
    match value {
        "hyperscope" => Ok(NavigationPreset::Hyperscope),
        "object" => Ok(NavigationPreset::Object),
        "fly" => Ok(NavigationPreset::Fly),
        "drone" => Ok(NavigationPreset::Drone),
        _ => Err(JsValue::from_str("unknown navigation preset")),
    }
}

pub(crate) fn preset_name(value: NavigationPreset) -> &'static str {
    match value {
        NavigationPreset::Hyperscope => "hyperscope",
        NavigationPreset::Object => "object",
        NavigationPreset::Fly => "fly",
        NavigationPreset::Drone => "drone",
    }
}

pub(crate) fn parse_easing(value: &str) -> Result<TransitionEasing, JsValue> {
    match value {
        "linear" => Ok(TransitionEasing::Linear),
        "smoothstep" => Ok(TransitionEasing::SmoothStep),
        "smootherstep" => Ok(TransitionEasing::SmootherStep),
        _ => Err(JsValue::from_str("unknown transition easing")),
    }
}

fn js_error(error: impl ToString) -> JsValue {
    JsValue::from_str(&error.to_string())
}
