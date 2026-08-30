//! JavaScript projection for backend-neutral application effects.

use hyperscope_app::PatchLabEffect;
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::JsValue;

pub(crate) fn patch_lab_effect_to_js(effect: &PatchLabEffect) -> JsValue {
    let object = Object::new();
    match effect {
        PatchLabEffect::BuildGeometry { job_id, geometry } => {
            set(&object, "type", "build_geometry");
            set(&object, "job_id", &job_id.to_string());
            set(&object, "shape", geometry.shape.wire_name());
            set_number(&object, "grid", f64::from(geometry.grid));
            set_number(&object, "bend_percent", f64::from(geometry.bend_percent));
        }
        PatchLabEffect::CancelGeometry { job_id } => {
            set(&object, "type", "cancel_geometry");
            set(&object, "job_id", &job_id.to_string());
        }
        PatchLabEffect::DiscardGeometry { geometry_job_id } => {
            set(&object, "type", "discard_geometry");
            set(&object, "geometry_job_id", &geometry_job_id.to_string());
        }
        PatchLabEffect::EvaluateLod {
            job_id,
            geometry_job_id,
            parameters,
        } => {
            set(&object, "type", "evaluate_lod");
            set(&object, "job_id", &job_id.to_string());
            set(&object, "geometry_job_id", &geometry_job_id.to_string());
            set(&object, "field", parameters.field.wire_name());
            set_number(
                &object,
                "phase_microradians",
                f64::from(parameters.phase_microradians),
            );
            set_number(&object, "min_exponent", f64::from(parameters.min_exponent));
            set_number(&object, "max_exponent", f64::from(parameters.max_exponent));
            let manual = Array::new();
            for exponent in parameters.manual_edge_exponents {
                manual.push(&JsValue::from_f64(f64::from(exponent)));
            }
            let _ = Reflect::set(
                &object,
                &JsValue::from_str("manual_edge_exponents"),
                &manual,
            );
            set_number(
                &object,
                "atlas_exponent",
                f64::from(parameters.atlas_exponent),
            );
            set_number(
                &object,
                "max_face_edge_ratio",
                f64::from(parameters.max_face_edge_ratio),
            );
        }
        PatchLabEffect::CancelLod {
            job_id,
            geometry_job_id,
        } => {
            set(&object, "type", "cancel_lod");
            set(&object, "job_id", &job_id.to_string());
            set(&object, "geometry_job_id", &geometry_job_id.to_string());
        }
    }
    object.into()
}

fn set(object: &Object, key: &str, value: &str) {
    let _ = Reflect::set(object, &JsValue::from_str(key), &JsValue::from_str(value));
}

fn set_number(object: &Object, key: &str, value: f64) {
    let _ = Reflect::set(object, &JsValue::from_str(key), &JsValue::from_f64(value));
}
