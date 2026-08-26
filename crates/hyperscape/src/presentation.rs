//! Deterministic, renderer-independent presentation state.
//!
//! A presentation names assets, composes them as distinct layers, and maps
//! cues to complete semantic views. Browser code performs I/O and displays
//! text, but cue ordering, camera/focus transitions, and desired scene state
//! are owned here.

use crate::{
    CameraError, CameraRig, FocusNavigation, FocusSphere, NavigationAction, NavigationController,
    PerspectiveLens, SphereReflectionState, TransitionEasing,
};
use bevy_ecs::prelude::Resource;
use quilting_core::Quat;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use uuid::Uuid;

pub const PRESENTATION_VERSION: &str = "0.1";
/// Canonical interactive presentation used by the replay oracle and browser
/// release. Keeping it beside the parser makes packaged tests self-contained.
pub const HACKER_NIGHT_PRESENTATION_JSON: &str =
    include_str!("../fixtures/hacker-night.presentation.json");

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Presentation {
    pub version: String,
    pub id: Uuid,
    pub title: String,
    #[serde(default)]
    pub assets: Vec<PresentationAsset>,
    #[serde(default)]
    pub scenes: Vec<PresentationScene>,
    #[serde(default)]
    pub views: Vec<PresentationView>,
    #[serde(default)]
    pub cues: Vec<PresentationCue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresentationAsset {
    pub id: Uuid,
    pub name: String,
    pub uri: String,
    #[serde(default)]
    pub load: AssetLoadPolicy,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetLoadPolicy {
    Preload,
    #[default]
    OnCue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresentationScene {
    pub id: Uuid,
    pub name: String,
    #[serde(default)]
    pub layers: Vec<PresentationLayer>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresentationLayer {
    pub id: Uuid,
    pub name: String,
    pub asset: Uuid,
    #[serde(default)]
    pub transform: LayerTransform,
    #[serde(default = "default_true")]
    pub visible: bool,
    #[serde(default = "default_opacity")]
    pub opacity: f64,
}

/// Ordinary layer-local TRS, deliberately separate from conformal frames.
/// Quaternion order is `(w, x, y, z)` throughout the Hyperscape boundary.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LayerTransform {
    #[serde(default)]
    pub translation: [f64; 3],
    #[serde(default = "identity_quaternion")]
    pub rotation: [f64; 4],
    #[serde(default = "unit_scale")]
    pub scale: [f64; 3],
}

impl Default for LayerTransform {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: identity_quaternion(),
            scale: unit_scale(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresentationView {
    pub id: Uuid,
    pub name: String,
    pub camera: AuthoredCamera,
    #[serde(default)]
    pub focus: AuthoredFocus,
    #[serde(default)]
    pub layers: Vec<ViewLayerOverride>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AuthoredCamera {
    pub eye: [f64; 3],
    #[serde(default = "identity_quaternion")]
    pub orientation: [f64; 4],
    pub control_distance: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_target: Option<[f64; 3]>,
    #[serde(default = "default_vertical_fov")]
    pub vertical_fov_radians: f64,
    #[serde(default = "default_near")]
    pub near: f64,
    #[serde(default = "default_far")]
    pub far: f64,
}

impl Default for AuthoredCamera {
    fn default() -> Self {
        Self {
            eye: [0.0, 0.0, 3.0],
            orientation: identity_quaternion(),
            control_distance: 3.0,
            semantic_target: None,
            vertical_fov_radians: default_vertical_fov(),
            near: default_near(),
            far: default_far(),
        }
    }
}

impl AuthoredCamera {
    pub fn to_camera_rig(self) -> Result<CameraRig, PresentationError> {
        if self.orientation.into_iter().any(|value| !value.is_finite()) {
            return Err(invalid("camera orientation must be finite"));
        }
        let mut orientation = Quat::new(
            self.orientation[0],
            self.orientation[1],
            self.orientation[2],
            self.orientation[3],
        );
        let norm = orientation.norm();
        if !norm.is_finite() || norm <= 1.0e-12 {
            return Err(invalid("camera orientation must be nondegenerate"));
        }
        orientation = orientation / norm;
        if orientation.w < 0.0 {
            orientation = -orientation;
        }
        let camera = CameraRig {
            eye: self.eye,
            orientation,
            control_distance: self.control_distance,
            semantic_target: self.semantic_target,
            lens: PerspectiveLens {
                vertical_fov_radians: self.vertical_fov_radians,
                near: self.near,
                far: self.far,
            },
        };
        camera
            .validate()
            .map_err(|error| invalid(format!("invalid authored camera: {error}")))?;
        Ok(camera)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AuthoredFocus {
    pub center: [f64; 3],
    pub radius: f64,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub inversion_enabled: bool,
    #[serde(default = "default_focus_coordinate")]
    pub coordinate: f64,
    #[serde(default = "default_focus_aperture")]
    pub angular_aperture: f64,
}

impl Default for AuthoredFocus {
    fn default() -> Self {
        let focus = FocusNavigation::default();
        Self {
            center: focus.sphere.center,
            radius: focus.sphere.radius,
            enabled: focus.focus_enabled,
            inversion_enabled: focus.inversion_enabled,
            coordinate: focus.focus_coordinate,
            angular_aperture: focus.angular_aperture,
        }
    }
}

impl AuthoredFocus {
    pub fn to_focus_navigation(self) -> Result<FocusNavigation, PresentationError> {
        let sphere = FocusSphere::new(self.center, self.radius).map_err(invalid)?;
        if !self.coordinate.is_finite()
            || !(0.0..=1.0).contains(&self.coordinate)
            || !self.angular_aperture.is_finite()
            || self.angular_aperture <= 0.0
        {
            return Err(invalid(
                "focus coordinate must be in [0,1] and aperture must be positive",
            ));
        }
        Ok(FocusNavigation {
            sphere,
            anchor: None,
            transition: None,
            focus_enabled: self.enabled,
            inversion_enabled: self.inversion_enabled,
            focus_coordinate: self.coordinate,
            angular_aperture: self.angular_aperture,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ViewLayerOverride {
    pub layer: Uuid,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresentationCue {
    pub id: Uuid,
    pub scene: Uuid,
    pub view: Uuid,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<CueText>,
    #[serde(default)]
    pub animations: Vec<CueAnimation>,
    #[serde(default)]
    pub overlays: Vec<PresentationOverlay>,
    #[serde(default)]
    pub tessellation: PresentationTessellation,
    #[serde(default)]
    pub transition: PresentationTransition,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CueText {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eyebrow: Option<String>,
    pub heading: String,
    #[serde(default)]
    pub body: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CueAnimation {
    pub layer: Uuid,
    pub clip: String,
    #[serde(default)]
    pub time_seconds: f64,
    #[serde(default = "default_true")]
    pub playing: bool,
    #[serde(default = "default_animation_speed")]
    pub speed: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresentationOverlay {
    Wireframe,
    Lod,
    PatchBoundaries,
    ControlNet,
    Normals,
    Stretch,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PresentationTessellation {
    #[serde(default = "default_tessellation_density")]
    pub density: f64,
    #[serde(default = "default_true")]
    pub screen_attenuation: bool,
    #[serde(default = "default_min_pixels_per_subdivision")]
    pub min_pixels_per_subdivision: f64,
}

impl Default for PresentationTessellation {
    fn default() -> Self {
        Self {
            density: default_tessellation_density(),
            screen_attenuation: true,
            min_pixels_per_subdivision: default_min_pixels_per_subdivision(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PresentationTransition {
    #[serde(default = "default_transition_duration")]
    pub duration_seconds: f64,
    #[serde(default)]
    pub easing: TransitionEasing,
}

impl Default for PresentationTransition {
    fn default() -> Self {
        Self {
            duration_seconds: default_transition_duration(),
            easing: TransitionEasing::default(),
        }
    }
}

/// Fully resolved desired state returned to a browser/render adapter.
/// `required_assets` is a desired set: adapters may release prior on-cue
/// assets not present here, while retaining their own explicit caches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresentationSnapshot {
    pub cue_index: usize,
    pub cue_id: Uuid,
    pub scene_id: Uuid,
    pub view_id: Uuid,
    pub text: Option<CueText>,
    pub required_assets: Vec<PresentationAsset>,
    pub layers: Vec<PresentationLayerState>,
    pub animations: Vec<CueAnimation>,
    pub overlays: Vec<PresentationOverlay>,
    pub tessellation: PresentationTessellation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresentationLayerState {
    pub id: Uuid,
    pub name: String,
    pub asset: Uuid,
    pub transform: LayerTransform,
    pub visible: bool,
    pub opacity: f64,
}

#[derive(Resource, Debug, Clone)]
pub struct PresentationRuntime {
    presentation: Presentation,
    active_cue_index: Option<usize>,
    /// A finite target may be the pole of the chart crossed during a cue. In
    /// that case the transition uses its equivalent sight tangent and restores
    /// the authored target only after reaching the destination chart.
    pending_semantic_target: Option<[f64; 3]>,
    /// Successful manual aim-policy sequence visible when the pending target
    /// was created. A changed fence preempts restoration; future and rejected
    /// actions leave it unchanged.
    pending_semantic_target_policy_sequence: Option<u64>,
}

impl Presentation {
    pub fn from_json(json: &str) -> Result<Self, PresentationError> {
        let presentation: Self = serde_json::from_str(json)
            .map_err(|error| PresentationError::Json(error.to_string()))?;
        presentation.validate()?;
        Ok(presentation)
    }

    pub fn to_json_pretty(&self) -> Result<String, PresentationError> {
        self.validate()?;
        serde_json::to_string_pretty(self)
            .map_err(|error| PresentationError::Json(error.to_string()))
    }

    pub fn validate(&self) -> Result<(), PresentationError> {
        if self.version != PRESENTATION_VERSION {
            return Err(invalid(format!(
                "unsupported presentation version {:?}; expected {PRESENTATION_VERSION}",
                self.version
            )));
        }
        if self.title.trim().is_empty() {
            return Err(invalid("presentation title must not be empty"));
        }

        let mut identities = BTreeMap::<Uuid, String>::new();
        register_identity(&mut identities, self.id, "presentation")?;

        let mut asset_ids = BTreeSet::new();
        for (index, asset) in self.assets.iter().enumerate() {
            register_identity(&mut identities, asset.id, format!("asset {index}"))?;
            asset_ids.insert(asset.id);
            if asset.uri.trim().is_empty() {
                return Err(invalid(format!("asset {index} URI must not be empty")));
            }
        }

        let mut scene_ids = BTreeSet::new();
        let mut layer_scenes = BTreeMap::new();
        for (scene_index, scene) in self.scenes.iter().enumerate() {
            register_identity(&mut identities, scene.id, format!("scene {scene_index}"))?;
            scene_ids.insert(scene.id);
            for (layer_index, layer) in scene.layers.iter().enumerate() {
                register_identity(
                    &mut identities,
                    layer.id,
                    format!("scene {scene_index} layer {layer_index}"),
                )?;
                if !asset_ids.contains(&layer.asset) {
                    return Err(invalid(format!(
                        "scene {scene_index} layer {layer_index} references unknown asset {}",
                        layer.asset
                    )));
                }
                validate_layer(layer, scene_index, layer_index)?;
                layer_scenes.insert(layer.id, scene.id);
            }
        }

        let mut view_ids = BTreeSet::new();
        for (view_index, view) in self.views.iter().enumerate() {
            register_identity(&mut identities, view.id, format!("view {view_index}"))?;
            view_ids.insert(view.id);
            let camera = view.camera.to_camera_rig()?;
            let focus = view.focus.to_focus_navigation()?;
            if focus.inversion_enabled {
                let mut source_chart_camera = camera;
                source_chart_camera
                    .transport_between_reflections(
                        SphereReflectionState::Sphere(focus.sphere),
                        SphereReflectionState::Identity,
                    )
                    .map_err(|error| {
                        invalid(format!(
                            "view {view_index} camera is invalid in its inversion chart: {error}"
                        ))
                    })?;
            }
            let mut overridden = BTreeSet::new();
            for layer in &view.layers {
                if !layer_scenes.contains_key(&layer.layer) {
                    return Err(invalid(format!(
                        "view {view_index} references unknown layer {}",
                        layer.layer
                    )));
                }
                if !overridden.insert(layer.layer) {
                    return Err(invalid(format!(
                        "view {view_index} repeats layer override {}",
                        layer.layer
                    )));
                }
                if layer
                    .opacity
                    .is_some_and(|opacity| !opacity.is_finite() || !(0.0..=1.0).contains(&opacity))
                {
                    return Err(invalid(format!(
                        "view {view_index} layer opacity must be in [0,1]"
                    )));
                }
            }
        }

        if self.cues.is_empty() {
            return Err(invalid("presentation must contain at least one cue"));
        }
        for (cue_index, cue) in self.cues.iter().enumerate() {
            register_identity(&mut identities, cue.id, format!("cue {cue_index}"))?;
            if !scene_ids.contains(&cue.scene) {
                return Err(invalid(format!(
                    "cue {cue_index} references unknown scene {}",
                    cue.scene
                )));
            }
            if !view_ids.contains(&cue.view) {
                return Err(invalid(format!(
                    "cue {cue_index} references unknown view {}",
                    cue.view
                )));
            }
            if !cue.transition.duration_seconds.is_finite() || cue.transition.duration_seconds < 0.0
            {
                return Err(invalid(format!(
                    "cue {cue_index} transition duration must be finite and nonnegative"
                )));
            }
            if !cue.tessellation.density.is_finite()
                || !(1.0..=500.0).contains(&cue.tessellation.density)
                || !cue.tessellation.min_pixels_per_subdivision.is_finite()
                || !(1.0..=64.0).contains(&cue.tessellation.min_pixels_per_subdivision)
            {
                return Err(invalid(format!(
                    "cue {cue_index} tessellation density must be in [1,500] and pixel threshold in [1,64]"
                )));
            }
            for animation in &cue.animations {
                if layer_scenes.get(&animation.layer) != Some(&cue.scene) {
                    return Err(invalid(format!(
                        "cue {cue_index} animation layer {} is not in its scene",
                        animation.layer
                    )));
                }
                if animation.clip.trim().is_empty()
                    || !animation.time_seconds.is_finite()
                    || animation.time_seconds < 0.0
                    || !animation.speed.is_finite()
                {
                    return Err(invalid(format!(
                        "cue {cue_index} animation must have a clip, nonnegative time, and finite speed"
                    )));
                }
            }
            let mut overlays = BTreeSet::new();
            let mut surface_visualizations = 0;
            for overlay in &cue.overlays {
                if !overlays.insert(*overlay) {
                    return Err(invalid(format!(
                        "cue {cue_index} repeats presentation overlay {overlay:?}"
                    )));
                }
                if !matches!(overlay, PresentationOverlay::ControlNet) {
                    surface_visualizations += 1;
                }
            }
            if surface_visualizations > 1 {
                return Err(invalid(format!(
                    "cue {cue_index} selects more than one exclusive surface visualization"
                )));
            }
        }
        Ok(())
    }
}

impl PresentationRuntime {
    pub fn new(presentation: Presentation) -> Result<Self, PresentationError> {
        presentation.validate()?;
        Ok(Self {
            presentation,
            active_cue_index: None,
            pending_semantic_target: None,
            pending_semantic_target_policy_sequence: None,
        })
    }

    pub fn from_json(json: &str) -> Result<Self, PresentationError> {
        Self::new(Presentation::from_json(json)?)
    }

    pub fn presentation(&self) -> &Presentation {
        &self.presentation
    }

    pub fn active_cue_index(&self) -> Option<usize> {
        self.active_cue_index
    }

    pub fn activate_index(
        &mut self,
        index: usize,
        navigation: &mut NavigationController,
    ) -> Result<PresentationSnapshot, PresentationError> {
        let cue = self
            .presentation
            .cues
            .get(index)
            .ok_or_else(|| invalid(format!("cue index {index} is out of range")))?;
        let view = self
            .presentation
            .views
            .iter()
            .find(|view| view.id == cue.view)
            .expect("validated cue view");
        let target_focus = view.focus.to_focus_navigation()?;
        let target_reflection = if target_focus.inversion_enabled {
            SphereReflectionState::Sphere(target_focus.sphere)
        } else {
            SphereReflectionState::Identity
        };
        let authored_camera = view.camera.to_camera_rig()?;
        let mut camera_in_current_chart = authored_camera;
        let pending_semantic_target = match camera_in_current_chart
            .transport_between_reflections(target_reflection, navigation.runtime.reflection)
        {
            Ok(_) => None,
            Err(CameraError::ReflectionPole) if authored_camera.semantic_target.is_some() => {
                camera_in_current_chart = authored_camera;
                camera_in_current_chart.semantic_target = None;
                camera_in_current_chart
                    .transport_between_reflections(target_reflection, navigation.runtime.reflection)
                    .map_err(|error| {
                        invalid(format!(
                            "cue {index} camera intersects a reflection pole: {error}"
                        ))
                    })?;
                authored_camera.semantic_target
            }
            Err(error) => {
                return Err(invalid(format!(
                    "cue {index} camera intersects a reflection pole: {error}"
                )));
            }
        };

        let transition = cue.transition;
        push_navigation(
            navigation,
            NavigationAction::TransitionCamera {
                target: camera_in_current_chart,
                duration_seconds: transition.duration_seconds,
                easing: transition.easing,
            },
        )?;
        push_navigation(
            navigation,
            NavigationAction::TransitionFreeFocusSphere {
                target: target_focus.sphere,
                duration_seconds: transition.duration_seconds,
                easing: transition.easing,
            },
        )?;
        push_navigation(
            navigation,
            NavigationAction::SetFocusField {
                coordinate: target_focus.focus_coordinate,
                angular_aperture: target_focus.angular_aperture,
            },
        )?;
        push_navigation(
            navigation,
            NavigationAction::SetFocusEnabled(target_focus.focus_enabled),
        )?;
        push_navigation(
            navigation,
            NavigationAction::SetInversionEnabled(target_focus.inversion_enabled),
        )?;
        navigation
            .tick(0.0)
            .map_err(|error| PresentationError::Navigation(error.to_owned()))?;

        self.pending_semantic_target = pending_semantic_target;
        self.pending_semantic_target_policy_sequence = navigation
            .runtime
            .last_semantic_target_policy_sequence;
        self.reconcile_navigation(navigation);
        self.active_cue_index = Some(index);
        self.snapshot(index)
    }

    /// Advance the shared navigation clock and finish any semantic target that
    /// had to cross an intermediate chart as a free sight tangent.
    pub fn tick_navigation(
        &mut self,
        navigation: &mut NavigationController,
        delta_seconds: f64,
    ) -> Result<(), PresentationError> {
        navigation
            .tick(delta_seconds)
            .map_err(|error| PresentationError::Navigation(error.to_owned()))?;
        self.reconcile_navigation(navigation);
        Ok(())
    }

    pub fn reconcile_navigation(&mut self, navigation: &mut NavigationController) -> bool {
        let Some(target) = self.pending_semantic_target else {
            return false;
        };
        if navigation.runtime.last_semantic_target_policy_sequence
            != self.pending_semantic_target_policy_sequence
        {
            self.pending_semantic_target = None;
            self.pending_semantic_target_policy_sequence = None;
            return false;
        }
        let desired_reflection = if navigation.focus.inversion_enabled {
            SphereReflectionState::Sphere(navigation.focus.sphere)
        } else {
            SphereReflectionState::Identity
        };
        if navigation.runtime.camera_transition.is_some()
            || navigation.focus.transition.is_some()
            || navigation.runtime.reflection != desired_reflection
        {
            return false;
        }
        navigation.camera.semantic_target = Some(target);
        self.pending_semantic_target = None;
        self.pending_semantic_target_policy_sequence = None;
        true
    }

    pub fn jump_to_cue(
        &mut self,
        cue_id: Uuid,
        navigation: &mut NavigationController,
    ) -> Result<PresentationSnapshot, PresentationError> {
        let index = self
            .presentation
            .cues
            .iter()
            .position(|cue| cue.id == cue_id)
            .ok_or(PresentationError::UnknownCue(cue_id))?;
        self.activate_index(index, navigation)
    }

    pub fn advance(
        &mut self,
        navigation: &mut NavigationController,
    ) -> Result<PresentationSnapshot, PresentationError> {
        let last = self.presentation.cues.len() - 1;
        let index = self
            .active_cue_index
            .map_or(0, |index| index.saturating_add(1).min(last));
        self.activate_index(index, navigation)
    }

    pub fn reverse(
        &mut self,
        navigation: &mut NavigationController,
    ) -> Result<PresentationSnapshot, PresentationError> {
        let index = self
            .active_cue_index
            .map_or(0, |index| index.saturating_sub(1));
        self.activate_index(index, navigation)
    }

    pub fn snapshot(&self, index: usize) -> Result<PresentationSnapshot, PresentationError> {
        let cue = self
            .presentation
            .cues
            .get(index)
            .ok_or_else(|| invalid(format!("cue index {index} is out of range")))?;
        let scene = self
            .presentation
            .scenes
            .iter()
            .find(|scene| scene.id == cue.scene)
            .expect("validated cue scene");
        let view = self
            .presentation
            .views
            .iter()
            .find(|view| view.id == cue.view)
            .expect("validated cue view");
        let active_assets = scene
            .layers
            .iter()
            .map(|layer| layer.asset)
            .collect::<BTreeSet<_>>();
        let required_assets = self
            .presentation
            .assets
            .iter()
            .filter(|asset| {
                asset.load == AssetLoadPolicy::Preload || active_assets.contains(&asset.id)
            })
            .cloned()
            .collect();
        let overrides = view
            .layers
            .iter()
            .map(|overrides| (overrides.layer, overrides))
            .collect::<BTreeMap<_, _>>();
        let layers = scene
            .layers
            .iter()
            .map(|layer| {
                let overrides = overrides.get(&layer.id);
                PresentationLayerState {
                    id: layer.id,
                    name: layer.name.clone(),
                    asset: layer.asset,
                    transform: layer.transform,
                    visible: overrides
                        .and_then(|state| state.visible)
                        .unwrap_or(layer.visible),
                    opacity: overrides
                        .and_then(|state| state.opacity)
                        .unwrap_or(layer.opacity),
                }
            })
            .collect();
        Ok(PresentationSnapshot {
            cue_index: index,
            cue_id: cue.id,
            scene_id: cue.scene,
            view_id: cue.view,
            text: cue.text.clone(),
            required_assets,
            layers,
            animations: cue.animations.clone(),
            overlays: cue.overlays.clone(),
            tessellation: cue.tessellation,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresentationError {
    Json(String),
    Invalid(String),
    UnknownCue(Uuid),
    Navigation(String),
}

impl fmt::Display for PresentationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(message) => write!(formatter, "presentation JSON error: {message}"),
            Self::Invalid(message) => write!(formatter, "invalid presentation: {message}"),
            Self::UnknownCue(cue) => write!(formatter, "unknown presentation cue {cue}"),
            Self::Navigation(message) => {
                write!(formatter, "presentation navigation error: {message}")
            }
        }
    }
}

impl Error for PresentationError {}

fn validate_layer(
    layer: &PresentationLayer,
    scene_index: usize,
    layer_index: usize,
) -> Result<(), PresentationError> {
    let transform = layer.transform;
    if transform
        .translation
        .into_iter()
        .chain(transform.rotation)
        .chain(transform.scale)
        .any(|value| !value.is_finite())
        || transform.scale.into_iter().any(|value| value == 0.0)
    {
        return Err(invalid(format!(
            "scene {scene_index} layer {layer_index} transform must be finite with nonzero scale"
        )));
    }
    let rotation_norm_sq = transform
        .rotation
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>();
    if rotation_norm_sq <= 1.0e-24 {
        return Err(invalid(format!(
            "scene {scene_index} layer {layer_index} rotation must be nondegenerate"
        )));
    }
    if !layer.opacity.is_finite() || !(0.0..=1.0).contains(&layer.opacity) {
        return Err(invalid(format!(
            "scene {scene_index} layer {layer_index} opacity must be in [0,1]"
        )));
    }
    Ok(())
}

fn register_identity(
    identities: &mut BTreeMap<Uuid, String>,
    id: Uuid,
    label: impl Into<String>,
) -> Result<(), PresentationError> {
    let label = label.into();
    if id.is_nil() {
        return Err(invalid(format!("{label} has the nil UUID")));
    }
    if let Some(previous) = identities.insert(id, label.clone()) {
        return Err(invalid(format!(
            "{label} repeats UUID {id} already used by {previous}"
        )));
    }
    Ok(())
}

fn push_navigation(
    navigation: &mut NavigationController,
    action: NavigationAction,
) -> Result<(), PresentationError> {
    navigation
        .push(action)
        .map(|_| ())
        .map_err(|error| PresentationError::Navigation(error.to_owned()))
}

fn invalid(message: impl Into<String>) -> PresentationError {
    PresentationError::Invalid(message.into())
}

const fn identity_quaternion() -> [f64; 4] {
    [1.0, 0.0, 0.0, 0.0]
}

const fn unit_scale() -> [f64; 3] {
    [1.0; 3]
}

const fn default_true() -> bool {
    true
}

const fn default_opacity() -> f64 {
    1.0
}

const fn default_animation_speed() -> f64 {
    1.0
}

const fn default_tessellation_density() -> f64 {
    100.0
}

const fn default_min_pixels_per_subdivision() -> f64 {
    16.0
}

const fn default_transition_duration() -> f64 {
    0.7
}

const fn default_vertical_fov() -> f64 {
    std::f64::consts::FRAC_PI_3
}

const fn default_near() -> f64 {
    0.01
}

const fn default_far() -> f64 {
    10_000.0
}

const fn default_focus_coordinate() -> f64 {
    0.5
}

const fn default_focus_aperture() -> f64 {
    0.1
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = HACKER_NIGHT_PRESENTATION_JSON;

    fn assert_camera_near(actual: CameraRig, expected: CameraRig) {
        for (actual, expected) in actual.eye.into_iter().zip(expected.eye) {
            assert!((actual - expected).abs() < 1.0e-9);
        }
        assert!((actual.orientation.dot(expected.orientation).abs() - 1.0).abs() < 1.0e-9);
        assert!((actual.control_distance - expected.control_distance).abs() < 1.0e-9);
    }

    fn assert_focus_near(actual: &FocusNavigation, expected: &FocusNavigation) {
        for (actual, expected) in actual.sphere.center.into_iter().zip(expected.sphere.center) {
            assert!((actual - expected).abs() < 1.0e-9);
        }
        assert!((actual.sphere.radius - expected.sphere.radius).abs() < 1.0e-9);
        assert_eq!(actual.focus_enabled, expected.focus_enabled);
        assert_eq!(actual.inversion_enabled, expected.inversion_enabled);
        assert!((actual.focus_coordinate - expected.focus_coordinate).abs() < 1.0e-12);
        assert!((actual.angular_aperture - expected.angular_aperture).abs() < 1.0e-12);
    }

    #[test]
    fn fixture_resolves_distinct_assets_layers_and_cue_text() {
        let mut runtime = PresentationRuntime::from_json(FIXTURE).unwrap();
        let mut navigation = NavigationController::default();
        let composition_index = runtime
            .presentation()
            .cues
            .iter()
            .position(|cue| {
                cue.id == Uuid::parse_str("e0000000-0000-4000-8000-000000000002").unwrap()
            })
            .unwrap();
        let snapshot = runtime
            .activate_index(composition_index, &mut navigation)
            .unwrap();

        assert_eq!(snapshot.required_assets.len(), 2);
        assert_eq!(snapshot.layers.len(), 2);
        assert_ne!(snapshot.layers[0].asset, snapshot.layers[1].asset);
        assert_eq!(snapshot.text.unwrap().heading, "One scene, distinct assets");
        assert_eq!(runtime.active_cue_index(), Some(composition_index));
    }

    #[test]
    fn fixture_opens_with_projected_polytopes_before_the_animated_horse() {
        let mut runtime = PresentationRuntime::from_json(FIXTURE).unwrap();
        let mut navigation = NavigationController::default();
        let presentation = runtime.presentation();
        assert_eq!(presentation.assets.len(), 5);
        assert_eq!(presentation.cues.len(), 8);
        assert_eq!(
            presentation
                .assets
                .iter()
                .skip(2)
                .map(|asset| asset.uri.as_str())
                .collect::<Vec<_>>(),
            [
                "/polytopes/4-simplex.glb",
                "/polytopes/tesseract.glb",
                "/polytopes/16-cell.glb",
            ],
        );
        assert_eq!(
            presentation.cues[0].id,
            Uuid::parse_str("e0000000-0000-4000-8000-000000000007").unwrap(),
        );
        assert_eq!(
            presentation.cues[4].id,
            Uuid::parse_str("e0000000-0000-4000-8000-000000000001").unwrap(),
        );

        let opening = runtime.activate_index(0, &mut navigation).unwrap();
        assert_eq!(opening.required_assets.len(), 2);
        assert_eq!(opening.layers.len(), 2);
        assert!(!opening.layers[0].visible);
        assert!(opening.layers[1].visible);
        assert_eq!(opening.animations.len(), 1);
        assert!(!opening.animations[0].playing);
        assert_eq!(opening.overlays, [PresentationOverlay::PatchBoundaries]);

        let topology = runtime.activate_index(1, &mut navigation).unwrap();
        assert_eq!(topology.overlays, [PresentationOverlay::Wireframe]);
        assert!(!topology.tessellation.screen_attenuation);
        let lod = runtime.activate_index(2, &mut navigation).unwrap();
        assert_eq!(lod.overlays, [PresentationOverlay::Lod]);
        assert!(lod.tessellation.screen_attenuation);
        assert_eq!(lod.tessellation.min_pixels_per_subdivision, 8.0);
    }

    #[test]
    fn cue_transition_reaches_authored_composition_camera() {
        fn run(steps: &[f64]) -> NavigationController {
            let mut runtime = PresentationRuntime::from_json(FIXTURE).unwrap();
            let mut navigation = NavigationController::default();
            runtime.activate_index(0, &mut navigation).unwrap();
            runtime.tick_navigation(&mut navigation, 0.7).unwrap();
            let composition_index = runtime
                .presentation()
                .cues
                .iter()
                .position(|cue| {
                    cue.id == Uuid::parse_str("e0000000-0000-4000-8000-000000000002").unwrap()
                })
                .unwrap();
            runtime
                .activate_index(composition_index, &mut navigation)
                .unwrap();
            for step in steps {
                runtime.tick_navigation(&mut navigation, *step).unwrap();
            }
            navigation
        }

        let single = run(&[1.2]);
        let partitioned = run(&[0.1; 12]);
        let document = Presentation::from_json(FIXTURE).unwrap();
        let composition = document
            .cues
            .iter()
            .find(|cue| cue.id == Uuid::parse_str("e0000000-0000-4000-8000-000000000002").unwrap())
            .unwrap();
        let view = document
            .views
            .iter()
            .find(|view| view.id == composition.view)
            .unwrap();
        let expected_camera = view.camera.to_camera_rig().unwrap();
        let expected_focus = view.focus.to_focus_navigation().unwrap();
        assert_camera_near(single.camera, expected_camera);
        assert_camera_near(partitioned.camera, expected_camera);
        assert_focus_near(&single.focus, &expected_focus);
        assert_focus_near(&partitioned.focus, &expected_focus);
        assert_camera_near(single.camera, partitioned.camera);
        assert!(single.diagnostics.0.is_empty());
        assert!(partitioned.diagnostics.0.is_empty());
    }

    #[test]
    fn authored_lens_and_target_presence_interpolate_on_the_shared_camera_clock() {
        for semantic_target in [None, Some([0.0, 0.0, 0.0])] {
            let mut document = Presentation::from_json(FIXTURE).unwrap();
            let cue = document.cues[0].clone();
            let view = document
                .views
                .iter_mut()
                .find(|view| view.id == cue.view)
                .unwrap();
            view.camera.semantic_target = semantic_target;
            view.camera.vertical_fov_radians = 1.4;
            view.camera.near = 0.001;
            view.camera.far = 40_000.0;
            document.validate().unwrap();

            let mut runtime = PresentationRuntime::new(document).unwrap();
            let mut navigation = NavigationController::default();
            let start = navigation.camera;
            runtime.activate_index(0, &mut navigation).unwrap();
            assert_eq!(navigation.camera.lens, start.lens);
            assert_eq!(navigation.camera.semantic_target, None);

            runtime
                .tick_navigation(&mut navigation, cue.transition.duration_seconds * 0.5)
                .unwrap();
            let middle = navigation.camera;
            assert!((middle.lens.vertical_fov_radians
                - (start.lens.vertical_fov_radians + 1.4) * 0.5)
                .abs()
                < 1.0e-12);
            assert!((middle.lens.near - (start.lens.near * 0.001).sqrt()).abs() < 1.0e-12);
            assert!((middle.lens.far - (start.lens.far * 40_000.0).sqrt()).abs() < 1.0e-9);
            assert_eq!(middle.semantic_target.is_some(), semantic_target.is_some());

            runtime
                .tick_navigation(&mut navigation, cue.transition.duration_seconds * 0.5)
                .unwrap();
            assert_eq!(navigation.camera.lens.vertical_fov_radians, 1.4);
            assert_eq!(navigation.camera.lens.near, 0.001);
            assert_eq!(navigation.camera.lens.far, 40_000.0);
            assert_eq!(navigation.camera.semantic_target, semantic_target);
        }
    }

    #[test]
    fn educational_inversion_cue_crosses_a_safe_stable_sphere() {
        let mut runtime = PresentationRuntime::from_json(FIXTURE).unwrap();
        let mut navigation = NavigationController::default();
        let horse_index = runtime
            .presentation()
            .cues
            .iter()
            .position(|cue| {
                cue.id == Uuid::parse_str("e0000000-0000-4000-8000-000000000001").unwrap()
            })
            .unwrap();
        runtime
            .activate_index(horse_index, &mut navigation)
            .unwrap();
        runtime.tick_navigation(&mut navigation, 0.7).unwrap();

        let inversion_index = runtime
            .presentation()
            .cues
            .iter()
            .position(|cue| {
                cue.id == Uuid::parse_str("e0000000-0000-4000-8000-000000000006").unwrap()
            })
            .unwrap();
        runtime
            .activate_index(inversion_index, &mut navigation)
            .unwrap();
        for _ in 0..14 {
            runtime.tick_navigation(&mut navigation, 0.1).unwrap();
        }

        let view = runtime
            .presentation()
            .views
            .iter()
            .find(|view| view.id == runtime.presentation().cues[inversion_index].view)
            .unwrap();
        assert_camera_near(navigation.camera, view.camera.to_camera_rig().unwrap());
        assert_focus_near(
            &navigation.focus,
            &view.focus.to_focus_navigation().unwrap(),
        );
        assert_eq!(
            navigation.runtime.reflection,
            SphereReflectionState::Sphere(navigation.focus.sphere)
        );
        assert!(navigation.diagnostics.0.is_empty());
    }

    #[test]
    fn educational_lod_cue_resolves_its_tessellation_policy() {
        let mut runtime = PresentationRuntime::from_json(FIXTURE).unwrap();
        let mut navigation = NavigationController::default();
        let lod_index = runtime
            .presentation()
            .cues
            .iter()
            .position(|cue| {
                cue.id == Uuid::parse_str("e0000000-0000-4000-8000-000000000004").unwrap()
            })
            .unwrap();
        let snapshot = runtime.activate_index(lod_index, &mut navigation).unwrap();

        assert_eq!(snapshot.overlays, vec![PresentationOverlay::Lod]);
        assert_eq!(snapshot.tessellation.density, 100.0);
        assert!(snapshot.tessellation.screen_attenuation);
        assert_eq!(snapshot.tessellation.min_pixels_per_subdivision, 16.0);
    }

    #[test]
    fn finite_target_is_restored_after_leaving_its_inverted_pole_chart() {
        let mut runtime = PresentationRuntime::from_json(FIXTURE).unwrap();
        let mut navigation = NavigationController::default();
        let sphere = FocusSphere::new([0.0; 3], 3.0).unwrap();
        navigation.focus.sphere = sphere;
        navigation.focus.inversion_enabled = true;
        navigation.runtime.reflection = SphereReflectionState::Sphere(sphere);
        navigation.camera = CameraRig {
            eye: [0.0, 0.0, 6.0],
            semantic_target: None,
            control_distance: 6.0,
            ..CameraRig::default()
        };

        let horse_index = runtime
            .presentation()
            .cues
            .iter()
            .position(|cue| {
                cue.id == Uuid::parse_str("e0000000-0000-4000-8000-000000000001").unwrap()
            })
            .unwrap();
        runtime
            .activate_index(horse_index, &mut navigation)
            .unwrap();
        runtime.tick_navigation(&mut navigation, 0.7).unwrap();

        let horse_view = runtime.presentation().cues[horse_index].view;
        let expected = runtime
            .presentation()
            .views
            .iter()
            .find(|view| view.id == horse_view)
            .unwrap()
            .camera
            .to_camera_rig()
            .unwrap();
        assert_camera_near(navigation.camera, expected);
        assert_eq!(navigation.camera.semantic_target, Some([0.0; 3]));
        assert_eq!(
            navigation.runtime.reflection,
            SphereReflectionState::Identity
        );
        assert!(navigation.diagnostics.0.is_empty());
    }

    #[test]
    fn manual_aim_policy_preempts_a_pending_presentation_target() {
        let mut runtime = PresentationRuntime::from_json(FIXTURE).unwrap();
        let mut navigation = NavigationController::default();
        let sphere = FocusSphere::new([0.0; 3], 3.0).unwrap();
        navigation.focus.sphere = sphere;
        navigation.focus.inversion_enabled = true;
        navigation.runtime.reflection = SphereReflectionState::Sphere(sphere);
        navigation.camera = CameraRig {
            eye: [0.0, 0.0, 6.0],
            semantic_target: None,
            control_distance: 6.0,
            ..CameraRig::default()
        };

        runtime.activate_index(0, &mut navigation).unwrap();
        assert!(runtime.pending_semantic_target.is_some());
        navigation
            .push(NavigationAction::SetSemanticTargetEnabled(false))
            .unwrap();
        runtime.tick_navigation(&mut navigation, 0.0).unwrap();
        assert!(runtime.pending_semantic_target.is_none());
        runtime.tick_navigation(&mut navigation, 0.7).unwrap();

        assert_eq!(navigation.camera.semantic_target, None);
        assert!(navigation
            .runtime
            .last_semantic_target_policy_sequence
            .is_some());
    }

    #[test]
    fn rejected_aim_policy_does_not_preempt_a_pending_presentation_target() {
        let mut runtime = PresentationRuntime::from_json(FIXTURE).unwrap();
        let mut navigation = NavigationController::default();
        let sphere = FocusSphere::new([0.0; 3], 3.0).unwrap();
        navigation.focus.sphere = sphere;
        navigation.focus.inversion_enabled = true;
        navigation.runtime.reflection = SphereReflectionState::Sphere(sphere);
        navigation.camera = CameraRig {
            eye: [0.0, 0.0, 6.0],
            semantic_target: None,
            control_distance: 6.0,
            ..CameraRig::default()
        };

        runtime.activate_index(0, &mut navigation).unwrap();
        let surface_target = crate::SurfaceAnchorTarget::new(
            navigation.camera,
            [0.0, 1.0, 0.0],
        )
        .unwrap();
        navigation
            .push(NavigationAction::BeginSurfaceAnchorTransition {
                target: surface_target,
                scene_radius: 10.0,
                duration_seconds: 1.0,
                easing: TransitionEasing::Linear,
            })
            .unwrap();
        navigation.tick(0.0).unwrap();
        navigation
            .push(NavigationAction::SetSemanticTargetEnabled(true))
            .unwrap();
        runtime.tick_navigation(&mut navigation, 0.0).unwrap();

        assert!(runtime.pending_semantic_target.is_some());
        assert_eq!(
            navigation.runtime.last_semantic_target_policy_sequence,
            None
        );
        assert!(navigation.diagnostics.0.last().is_some_and(|diagnostic| {
            diagnostic.contains("point-target camera transport is unavailable")
        }));

        navigation
            .push(NavigationAction::CancelSurfaceAnchorTransition)
            .unwrap();
        runtime.tick_navigation(&mut navigation, 0.7).unwrap();
        assert!(runtime.pending_semantic_target.is_none());
        assert_eq!(navigation.camera.semantic_target, Some([0.0; 3]));
    }

    #[test]
    fn validation_rejects_cross_scene_animation_and_duplicate_identity() {
        let mut document = Presentation::from_json(FIXTURE).unwrap();
        document.cues[0].animations[0].layer = document.scenes[1].layers[1].id;
        assert!(
            document
                .validate()
                .unwrap_err()
                .to_string()
                .contains("is not in its scene")
        );

        let mut document = Presentation::from_json(FIXTURE).unwrap();
        document.views[0].id = document.assets[0].id;
        assert!(
            document
                .validate()
                .unwrap_err()
                .to_string()
                .contains("repeats UUID")
        );

        let mut document = Presentation::from_json(FIXTURE).unwrap();
        document.views[1].focus.inversion_enabled = true;
        document.views[1].camera.semantic_target = Some(document.views[1].focus.center);
        assert!(
            document
                .validate()
                .unwrap_err()
                .to_string()
                .contains("invalid in its inversion chart")
        );
    }

    #[test]
    fn validation_rejects_ambiguous_or_duplicate_surface_visualizations() {
        let mut document = Presentation::from_json(FIXTURE).unwrap();
        document.cues[0].overlays =
            vec![PresentationOverlay::Wireframe, PresentationOverlay::Normals];
        assert!(
            document
                .validate()
                .unwrap_err()
                .to_string()
                .contains("more than one exclusive surface visualization")
        );

        let mut document = Presentation::from_json(FIXTURE).unwrap();
        document.cues[0].overlays = vec![
            PresentationOverlay::PatchBoundaries,
            PresentationOverlay::PatchBoundaries,
        ];
        assert!(
            document
                .validate()
                .unwrap_err()
                .to_string()
                .contains("repeats presentation overlay")
        );

        let mut document = Presentation::from_json(FIXTURE).unwrap();
        document.cues[0].tessellation.min_pixels_per_subdivision = 65.0;
        assert!(
            document
                .validate()
                .unwrap_err()
                .to_string()
                .contains("pixel threshold in [1,64]")
        );
    }

    #[test]
    fn json_roundtrip_preserves_validated_document() {
        let presentation = Presentation::from_json(FIXTURE).unwrap();
        let encoded = presentation.to_json_pretty().unwrap();
        let recovered = Presentation::from_json(&encoded).unwrap();
        assert_eq!(recovered.id, presentation.id);
        assert_eq!(recovered.assets, presentation.assets);
        assert_eq!(recovered.scenes, presentation.scenes);
        assert_eq!(recovered.cues, presentation.cues);
        assert_eq!(recovered.views.len(), presentation.views.len());
        for (recovered, original) in recovered.views.iter().zip(&presentation.views) {
            assert_eq!(recovered.id, original.id);
            assert_camera_near(
                recovered.camera.to_camera_rig().unwrap(),
                original.camera.to_camera_rig().unwrap(),
            );
            assert_eq!(recovered.focus, original.focus);
            assert_eq!(recovered.layers, original.layers);
        }
    }
}
