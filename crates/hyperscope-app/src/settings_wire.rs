//! Feature-gated browser wire projection for Hyperscope routes.
//!
//! Route semantics and their serialized application packet belong together.
//! Platform bindings may percent-decode query pairs and serialize this value,
//! but must not maintain a parallel set of DTOs or conversions.

use serde::Serialize;

use crate::{
    FocusPostprocessSettings, HyperscopeRoute, PatchLabControlsWire, PatchLabSessionIntent,
    RenderSettings, RouteAnimationClock, RouteNavigationSettings, RoutePresentationSettings,
    RoutePrimaryAssetSettings, RouteRendererAssetSettings, HYPERSCOPE_CONTROL_SPECS,
};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RouteWireResult {
    pairs: Vec<[String; 2]>,
    resolved_pairs: Vec<[String; 2]>,
    diagnostics: Vec<RouteWireDiagnostic>,
    startup_settings: Option<RouteStartupWireSettings>,
    render_settings: Option<RouteRenderWireSettings>,
    navigation_settings: Option<RouteNavigationWireSettings>,
    patch_lab_session: Option<RoutePatchLabWireSession>,
    presentation_settings: Option<RoutePresentationWireSettings>,
    primary_asset_settings: Option<RoutePrimaryAssetWireSettings>,
    renderer_asset_settings: Option<RouteRendererAssetWireSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    selection: Option<RouteSelectionWire>,
    #[serde(skip_serializing_if = "Option::is_none")]
    animation_clock: Option<RouteAnimationClockWire>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteStartupWireSettings {
    render_settings: RouteRenderWireSettings,
    navigation_settings: RouteNavigationWireSettings,
    patch_lab_session: RoutePatchLabWireSession,
    presentation_settings: RoutePresentationWireSettings,
    primary_asset_settings: RoutePrimaryAssetWireSettings,
    renderer_asset_settings: RouteRendererAssetWireSettings,
    #[serde(skip_serializing_if = "Option::is_none")]
    selection: Option<RouteSelectionWire>,
    #[serde(skip_serializing_if = "Option::is_none")]
    animation_clock: Option<RouteAnimationClockWire>,
}

#[derive(Serialize)]
struct RouteWireDiagnostic {
    code: &'static str,
    key: String,
    value: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RouteControlSpecWire {
    key: &'static str,
    default_value: &'static str,
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    numeric_domain: Option<RouteNumericControlDomainWire>,
    #[serde(skip_serializing_if = "Option::is_none")]
    choices: Option<&'static [&'static str]>,
}

#[derive(Serialize)]
struct RouteNumericControlDomainWire {
    minimum: f64,
    maximum: f64,
    integral: bool,
    step: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteRenderWireSettings {
    style: &'static str,
    resolution_level: u8,
    density: f64,
    screen_attenuation: bool,
    min_pixels_per_subdivision: f64,
    atlas_exponent: u8,
    max_face_edge_ratio: u8,
    focus_postprocess: FocusPostprocessWireSettings,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FocusPostprocessWireSettings {
    enabled: bool,
    mode: u8,
    diagnostic_view: u8,
    blur_radius_pixels: u16,
    blur_strength: f64,
    focus_coordinate: f64,
    bandwidth: f64,
    normalize_range: bool,
    gaussian_passes: u8,
    kawase_passes: u8,
    kawase_offset: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteSelectionWire {
    asset_id: String,
    entity_id: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteAnimationClockWire {
    time_seconds: Option<f64>,
    speed: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RoutePatchLabWireSession {
    active: bool,
    controls: PatchLabControlsWire,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RoutePresentationWireSettings {
    enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    cue_id: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RoutePrimaryAssetWireSettings {
    uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    animation_clip: Option<u32>,
    playing: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteRendererAssetWireSettings {
    environment: String,
    matcap: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteNavigationWireSettings {
    transform: RouteTransformWireSettings,
    camera: RouteCameraWireSettings,
    space_mouse: RouteSpaceMouseWireSettings,
    surface_walk: RouteSurfaceWalkWireSettings,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteTransformWireSettings {
    kind: &'static str,
    center_controls: [f64; 3],
    radius_control: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteCameraWireSettings {
    zoom: f64,
    euler_radians: [f64; 3],
    position: [f64; 3],
    semantic_target_enabled: bool,
    vertical_fov_degrees: f64,
    focus_transition_seconds: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteSpaceMouseWireSettings {
    move_sensitivity: f64,
    rotate_sensitivity: f64,
    profile: &'static str,
    lock_horizon: bool,
    swap_yz: bool,
    accept_background_input: bool,
    hyperscope_pan_invert_mask: u8,
    hyperscope_rotate_invert_mask: u8,
    blender_pan_invert_mask: u8,
    blender_rotate_invert_mask: u8,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteSurfaceWalkWireSettings {
    base_radii_per_second: f64,
    base_eye_height: f64,
    speed_octave_steps: f64,
    body_scale_octave_steps: f64,
    eye_height_octave_steps: f64,
    smoothing_seconds: f64,
    tangent_pull_fraction: f64,
    fast_multiplier: f64,
    default_near: f64,
    minimum_near: f64,
    near_eye_fraction: f64,
}

impl HyperscopeRoute {
    /// Serialize every route read model from the same application-owned
    /// projection. The platform binding only forwards this value.
    pub fn wire_result(&self) -> RouteWireResult {
        let pairs = self
            .canonical_pairs()
            .into_iter()
            .map(|(key, value)| [key.to_owned(), value.to_owned()])
            .collect();
        let resolved_pairs = self
            .resolved_pairs()
            .into_iter()
            .map(|(key, value)| [key.to_owned(), value.to_owned()])
            .collect();
        let diagnostics = self
            .diagnostics()
            .iter()
            .map(|diagnostic| RouteWireDiagnostic {
                code: diagnostic.code.name(),
                key: diagnostic.key.clone(),
                value: diagnostic.value.clone(),
            })
            .collect();
        let render_settings = self.render_settings().ok().map(Into::into);
        let navigation_settings = self.navigation_settings().ok().map(Into::into);
        let patch_lab_session = self.patch_lab_session().ok().map(Into::into);
        let presentation_settings = self.presentation_settings().ok().map(Into::into);
        let primary_asset_settings = self.primary_asset_settings().ok().map(Into::into);
        let renderer_asset_settings = self.renderer_asset_settings().ok().map(Into::into);
        let selection =
            self.selected_identity()
                .ok()
                .flatten()
                .map(|identity| RouteSelectionWire {
                    asset_id: identity.asset.to_string(),
                    entity_id: identity.entity.to_string(),
                });
        let animation_clock = self.animation_clock().ok().flatten().map(Into::into);
        let startup_settings =
            self.startup_settings()
                .ok()
                .map(|settings| RouteStartupWireSettings {
                    render_settings: settings.render_settings.into(),
                    navigation_settings: settings.navigation_settings.into(),
                    patch_lab_session: settings.patch_lab_session.into(),
                    presentation_settings: settings.presentation_settings.into(),
                    primary_asset_settings: settings.primary_asset_settings.into(),
                    renderer_asset_settings: settings.renderer_asset_settings.into(),
                    selection: settings.selection.map(|identity| RouteSelectionWire {
                        asset_id: identity.asset.to_string(),
                        entity_id: identity.entity.to_string(),
                    }),
                    animation_clock: settings.animation_clock.map(Into::into),
                });

        RouteWireResult {
            pairs,
            resolved_pairs,
            diagnostics,
            startup_settings,
            render_settings,
            navigation_settings,
            patch_lab_session,
            presentation_settings,
            primary_asset_settings,
            renderer_asset_settings,
            selection,
            animation_clock,
        }
    }
}

pub fn hyperscope_control_specs_wire() -> Vec<RouteControlSpecWire> {
    HYPERSCOPE_CONTROL_SPECS
        .iter()
        .map(|spec| RouteControlSpecWire {
            key: spec.key,
            default_value: spec.default_value,
            kind: spec.kind.name(),
            numeric_domain: spec
                .numeric_domain
                .map(|domain| RouteNumericControlDomainWire {
                    minimum: domain.minimum,
                    maximum: domain.maximum,
                    integral: domain.integral,
                    step: domain.step,
                }),
            choices: (!spec.choices.is_empty()).then_some(spec.choices),
        })
        .collect()
}

impl From<RenderSettings> for RouteRenderWireSettings {
    fn from(settings: RenderSettings) -> Self {
        Self {
            style: settings.style.wire_name(),
            resolution_level: settings.resolution_level,
            density: settings.tessellation.density,
            screen_attenuation: settings.tessellation.screen_attenuation,
            min_pixels_per_subdivision: settings.tessellation.min_pixels_per_subdivision,
            atlas_exponent: settings.atlas_exponent,
            max_face_edge_ratio: settings.max_face_edge_ratio,
            focus_postprocess: settings.focus_postprocess.into(),
        }
    }
}

impl From<FocusPostprocessSettings> for FocusPostprocessWireSettings {
    fn from(settings: FocusPostprocessSettings) -> Self {
        Self {
            enabled: settings.enabled,
            mode: settings.mode.wire_index(),
            diagnostic_view: settings.diagnostic_view.wire_index(),
            blur_radius_pixels: settings.blur_radius_pixels,
            blur_strength: settings.blur_strength,
            focus_coordinate: settings.focus_coordinate,
            bandwidth: settings.bandwidth,
            normalize_range: settings.normalize_range,
            gaussian_passes: settings.gaussian_passes,
            kawase_passes: settings.kawase_passes,
            kawase_offset: settings.kawase_offset,
        }
    }
}

impl From<PatchLabSessionIntent> for RoutePatchLabWireSession {
    fn from(session: PatchLabSessionIntent) -> Self {
        Self {
            active: session.active,
            controls: session.controls.into(),
        }
    }
}

impl From<RoutePresentationSettings> for RoutePresentationWireSettings {
    fn from(settings: RoutePresentationSettings) -> Self {
        Self {
            enabled: settings.enabled,
            cue_id: settings.cue_id.map(|cue_id| cue_id.to_string()),
        }
    }
}

impl From<RoutePrimaryAssetSettings> for RoutePrimaryAssetWireSettings {
    fn from(settings: RoutePrimaryAssetSettings) -> Self {
        Self {
            uri: settings.uri,
            animation_clip: settings.animation_clip,
            playing: settings.playing,
        }
    }
}

impl From<RouteRendererAssetSettings> for RouteRendererAssetWireSettings {
    fn from(settings: RouteRendererAssetSettings) -> Self {
        Self {
            environment: settings.environment,
            matcap: settings.matcap,
        }
    }
}

impl From<RouteAnimationClock> for RouteAnimationClockWire {
    fn from(clock: RouteAnimationClock) -> Self {
        Self {
            time_seconds: clock.time_seconds,
            speed: clock.speed,
        }
    }
}

impl From<RouteNavigationSettings> for RouteNavigationWireSettings {
    fn from(settings: RouteNavigationSettings) -> Self {
        Self {
            transform: RouteTransformWireSettings {
                kind: settings.transform.kind.wire_name(),
                center_controls: settings.transform.center_controls,
                radius_control: settings.transform.radius_control,
            },
            camera: RouteCameraWireSettings {
                zoom: settings.camera.zoom,
                euler_radians: settings.camera.euler_radians,
                position: settings.camera.position,
                semantic_target_enabled: settings.camera.semantic_target_enabled,
                vertical_fov_degrees: settings.camera.vertical_fov_degrees,
                focus_transition_seconds: settings.camera.focus_transition_seconds,
            },
            space_mouse: RouteSpaceMouseWireSettings {
                move_sensitivity: settings.space_mouse.move_sensitivity,
                rotate_sensitivity: settings.space_mouse.rotate_sensitivity,
                profile: settings.space_mouse.profile.wire_name(),
                lock_horizon: settings.space_mouse.lock_horizon,
                swap_yz: settings.space_mouse.swap_yz,
                accept_background_input: settings.space_mouse.accept_background_input,
                hyperscope_pan_invert_mask: settings.space_mouse.hyperscope_pan_invert_mask,
                hyperscope_rotate_invert_mask: settings.space_mouse.hyperscope_rotate_invert_mask,
                blender_pan_invert_mask: settings.space_mouse.blender_pan_invert_mask,
                blender_rotate_invert_mask: settings.space_mouse.blender_rotate_invert_mask,
            },
            surface_walk: RouteSurfaceWalkWireSettings {
                base_radii_per_second: settings.surface_walk.base_radii_per_second,
                base_eye_height: settings.surface_walk.base_eye_height,
                speed_octave_steps: settings.surface_walk.speed_octave_steps,
                body_scale_octave_steps: settings.surface_walk.body_scale_octave_steps,
                eye_height_octave_steps: settings.surface_walk.eye_height_octave_steps,
                smoothing_seconds: settings.surface_walk.smoothing_seconds,
                tangent_pull_fraction: settings.surface_walk.tangent_pull_fraction,
                fast_multiplier: settings.surface_walk.fast_multiplier,
                default_near: settings.surface_walk.default_near,
                minimum_near: settings.surface_walk.minimum_near,
                near_eye_fraction: settings.surface_walk.near_eye_fraction,
            },
        }
    }
}

#[cfg(all(test, feature = "replay"))]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn route_wire_result_keeps_the_atomic_packet_and_focused_oracles_identical() {
        let route = HyperscopeRoute::from_pairs([
            ("mode", "wire"),
            ("glb", "scene.glb"),
            ("env", "studio_1k"),
            ("animtime", "1.25"),
        ]);
        let value = serde_json::to_value(route.wire_result()).unwrap();

        assert_eq!(
            value["startupSettings"]["renderSettings"],
            value["renderSettings"]
        );
        assert_eq!(
            value["startupSettings"]["navigationSettings"],
            value["navigationSettings"]
        );
        assert_eq!(
            value["startupSettings"]["patchLabSession"],
            value["patchLabSession"]
        );
        assert_eq!(
            value["startupSettings"]["presentationSettings"],
            value["presentationSettings"]
        );
        assert_eq!(
            value["startupSettings"]["primaryAssetSettings"],
            value["primaryAssetSettings"]
        );
        assert_eq!(
            value["startupSettings"]["rendererAssetSettings"],
            value["rendererAssetSettings"]
        );
        assert_eq!(
            value["startupSettings"]["animationClock"],
            value["animationClock"]
        );
        assert_eq!(value["renderSettings"]["style"], json!("wire"));
        assert_eq!(value["primaryAssetSettings"]["uri"], json!("scene.glb"));
        assert_eq!(
            value["rendererAssetSettings"]["environment"],
            json!("studio_1k")
        );

        let invalid = HyperscopeRoute::from_pairs([("unknown", "value")]);
        let invalid = serde_json::to_value(invalid.wire_result()).unwrap();
        assert_eq!(invalid["startupSettings"], serde_json::Value::Null);
        assert_eq!(invalid["diagnostics"][0]["code"], json!("unknown_key"));
    }
}
