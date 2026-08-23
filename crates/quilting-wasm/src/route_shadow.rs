use hyperscope_app::{HyperscopeRoute, HYPERSCOPE_CONTROL_SPECS};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct RouteShadowResult {
    pairs: Vec<[String; 2]>,
    diagnostics: Vec<RouteShadowDiagnostic>,
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
    let diagnostics = route
        .diagnostics()
        .iter()
        .map(|diagnostic| RouteShadowDiagnostic {
            code: diagnostic.code.name(),
            key: diagnostic.key.clone(),
            value: diagnostic.value.clone(),
        })
        .collect();
    to_js(&RouteShadowResult { pairs, diagnostics })
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
