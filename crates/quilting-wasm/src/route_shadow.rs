use hyperscope_app::{HyperscopeRoute, HYPERSCOPE_CONTROL_SPECS};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RouteShadowResult {
    pairs: Vec<[String; 2]>,
    resolved_pairs: Vec<[String; 2]>,
    diagnostics: Vec<RouteShadowDiagnostic>,
    render_settings: Option<RouteRenderSettings>,
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
        });
    to_js(&RouteShadowResult {
        pairs,
        resolved_pairs,
        diagnostics,
        render_settings,
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
