use crate::app_shadow::FocusPostprocessShadow;
use hyperscope_app::{HyperscopeRoute, PatchLabControls, HYPERSCOPE_CONTROL_SPECS};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteShadowResult {
    pairs: Vec<[String; 2]>,
    resolved_pairs: Vec<[String; 2]>,
    diagnostics: Vec<RouteShadowDiagnostic>,
    render_settings: Option<RouteRenderSettings>,
    navigation_settings: Option<RouteNavigationSettings>,
    patch_lab_session: Option<RoutePatchLabSession>,
    #[serde(skip_serializing_if = "Option::is_none")]
    selection: Option<RouteSelection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    animation_clock: Option<RouteAnimationClock>,
}

#[derive(Serialize)]
struct RouteShadowDiagnostic {
    code: &'static str,
    key: String,
    value: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteControlSpec {
    key: &'static str,
    default_value: &'static str,
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    numeric_domain: Option<RouteNumericControlDomain>,
    #[serde(skip_serializing_if = "Option::is_none")]
    choices: Option<&'static [&'static str]>,
}

#[derive(Serialize)]
struct RouteNumericControlDomain {
    minimum: f64,
    maximum: f64,
    integral: bool,
    step: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteRenderSettings {
    style: &'static str,
    resolution_level: u8,
    density: f64,
    screen_attenuation: bool,
    min_pixels_per_subdivision: f64,
    atlas_exponent: u8,
    max_face_edge_ratio: u8,
    focus_postprocess: FocusPostprocessShadow,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteSelection {
    asset_id: String,
    entity_id: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteAnimationClock {
    time_seconds: Option<f64>,
    speed: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RoutePatchLabSession {
    active: bool,
    controls: RoutePatchLabControls,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RoutePatchLabControls {
    shape: &'static str,
    field: &'static str,
    manual_edge_exponents: [u8; 3],
    min_exponent: u8,
    max_exponent: u8,
    phase_microradians: u32,
    phase_radians: f64,
    bend_percent: u8,
    grid: u8,
    animate: bool,
}

impl From<PatchLabControls> for RoutePatchLabControls {
    fn from(controls: PatchLabControls) -> Self {
        Self {
            shape: controls.shape.wire_name(),
            field: controls.field.wire_name(),
            manual_edge_exponents: controls.manual_edge_exponents,
            min_exponent: controls.min_exponent,
            max_exponent: controls.max_exponent,
            phase_microradians: controls.phase_microradians,
            phase_radians: controls.phase_radians(),
            bend_percent: controls.bend_percent,
            grid: controls.grid,
            animate: controls.animate,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteNavigationSettings {
    transform: RouteTransformSettings,
    camera: RouteCameraSettings,
    space_mouse: RouteSpaceMouseSettings,
    surface_walk: RouteSurfaceWalkSettings,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteTransformSettings {
    kind: &'static str,
    center_controls: [f64; 3],
    radius_control: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteCameraSettings {
    zoom: f64,
    euler_radians: [f64; 3],
    position: [f64; 3],
    semantic_target_enabled: bool,
    vertical_fov_degrees: f64,
    focus_transition_seconds: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteSpaceMouseSettings {
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
struct RouteSurfaceWalkSettings {
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

/// Canonicalize already-decoded browser query pairs. Percent decoding and
/// encoding remain platform work; key identity, defaults, validation, and
/// ordering are Rust application policy.
#[wasm_bindgen(js_name = canonicalizeHyperscopeRoute)]
pub fn canonicalize_hyperscope_route(pairs: JsValue) -> Result<JsValue, JsValue> {
    let pairs: Vec<(String, String)> = serde_wasm_bindgen::from_value(pairs).map_err(js_error)?;
    let route = HyperscopeRoute::from_pairs(pairs);
    let pairs = route
        .canonical_pairs()
        .into_iter()
        .map(|(key, value)| [key.to_owned(), value.to_owned()])
        .collect();
    let resolved_pairs = route
        .resolved_pairs()
        .into_iter()
        .map(|(key, value)| [key.to_owned(), value.to_owned()])
        .collect();
    let diagnostics = route
        .diagnostics()
        .iter()
        .map(|diagnostic| RouteShadowDiagnostic {
            code: diagnostic.code.name(),
            key: diagnostic.key.clone(),
            value: diagnostic.value.clone(),
        })
        .collect();
    let render_settings = route
        .render_settings()
        .ok()
        .map(|settings| RouteRenderSettings {
            style: settings.style.wire_name(),
            resolution_level: settings.resolution_level,
            density: settings.tessellation.density,
            screen_attenuation: settings.tessellation.screen_attenuation,
            min_pixels_per_subdivision: settings.tessellation.min_pixels_per_subdivision,
            atlas_exponent: settings.atlas_exponent,
            max_face_edge_ratio: settings.max_face_edge_ratio,
            focus_postprocess: settings.focus_postprocess.into(),
        });
    let selection = route
        .selected_identity()
        .ok()
        .flatten()
        .map(|identity| RouteSelection {
            asset_id: identity.asset.to_string(),
            entity_id: identity.entity.to_string(),
        });
    let animation_clock = route
        .animation_clock()
        .ok()
        .flatten()
        .map(|clock| RouteAnimationClock {
            time_seconds: clock.time_seconds,
            speed: clock.speed,
        });
    let patch_lab_session = route
        .patch_lab_session()
        .ok()
        .map(|session| RoutePatchLabSession {
            active: session.active,
            controls: session.controls.into(),
        });
    let navigation_settings =
        route
            .navigation_settings()
            .ok()
            .map(|settings| RouteNavigationSettings {
                transform: RouteTransformSettings {
                    kind: settings.transform.kind.wire_name(),
                    center_controls: settings.transform.center_controls,
                    radius_control: settings.transform.radius_control,
                },
                camera: RouteCameraSettings {
                    zoom: settings.camera.zoom,
                    euler_radians: settings.camera.euler_radians,
                    position: settings.camera.position,
                    semantic_target_enabled: settings.camera.semantic_target_enabled,
                    vertical_fov_degrees: settings.camera.vertical_fov_degrees,
                    focus_transition_seconds: settings.camera.focus_transition_seconds,
                },
                space_mouse: RouteSpaceMouseSettings {
                    move_sensitivity: settings.space_mouse.move_sensitivity,
                    rotate_sensitivity: settings.space_mouse.rotate_sensitivity,
                    profile: settings.space_mouse.profile.wire_name(),
                    lock_horizon: settings.space_mouse.lock_horizon,
                    swap_yz: settings.space_mouse.swap_yz,
                    accept_background_input: settings.space_mouse.accept_background_input,
                    hyperscope_pan_invert_mask: settings.space_mouse.hyperscope_pan_invert_mask,
                    hyperscope_rotate_invert_mask: settings
                        .space_mouse
                        .hyperscope_rotate_invert_mask,
                    blender_pan_invert_mask: settings.space_mouse.blender_pan_invert_mask,
                    blender_rotate_invert_mask: settings.space_mouse.blender_rotate_invert_mask,
                },
                surface_walk: RouteSurfaceWalkSettings {
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
            });
    to_js(&RouteShadowResult {
        pairs,
        resolved_pairs,
        diagnostics,
        render_settings,
        navigation_settings,
        patch_lab_session,
        selection,
        animation_clock,
    })
}

#[wasm_bindgen(js_name = hyperscopeControlSpecs)]
pub fn hyperscope_control_specs() -> Result<JsValue, JsValue> {
    to_js(
        &HYPERSCOPE_CONTROL_SPECS
            .iter()
            .map(|spec| RouteControlSpec {
                key: spec.key,
                default_value: spec.default_value,
                kind: spec.kind.name(),
                numeric_domain: spec.numeric_domain.map(|domain| RouteNumericControlDomain {
                    minimum: domain.minimum,
                    maximum: domain.maximum,
                    integral: domain.integral,
                    step: domain.step,
                }),
                choices: (!spec.choices.is_empty()).then_some(spec.choices),
            })
            .collect::<Vec<_>>(),
    )
}

fn to_js(value: &impl Serialize) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value).map_err(js_error)
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
