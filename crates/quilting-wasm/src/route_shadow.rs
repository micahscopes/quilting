use hyperscope_app::{hyperscope_control_specs_wire, HyperscopeRoute};
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// Canonicalize already-decoded browser query pairs. Percent decoding and
/// encoding remain platform work; key identity, defaults, validation, wire
/// shape, and ordering are Rust application policy.
#[wasm_bindgen(js_name = canonicalizeHyperscopeRoute)]
pub fn canonicalize_hyperscope_route(pairs: JsValue) -> Result<JsValue, JsValue> {
    let pairs: Vec<(String, String)> = serde_wasm_bindgen::from_value(pairs).map_err(js_error)?;
    let route = HyperscopeRoute::from_pairs(pairs);
    to_js(&route.wire_result())
}

#[wasm_bindgen(js_name = hyperscopeControlSpecs)]
pub fn hyperscope_control_specs() -> Result<JsValue, JsValue> {
    to_js(&hyperscope_control_specs_wire())
}

fn to_js(value: &impl Serialize) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value).map_err(js_error)
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
