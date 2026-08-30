//! Shared control-registry projection for thin browser views.

use hyperscope_app::hyperscope_control_spec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NumericControlViewDomain {
    pub minimum: f64,
    pub maximum: f64,
    pub integral: bool,
    pub step: f64,
}

pub fn numeric_control_domain(key: &str) -> NumericControlViewDomain {
    let domain = hyperscope_control_spec(key)
        .and_then(|spec| spec.numeric_domain)
        .unwrap_or_else(|| panic!("control {key} must have a numeric domain"));
    NumericControlViewDomain {
        minimum: domain.minimum,
        maximum: domain.maximum,
        integral: domain.integral,
        step: domain.step,
    }
}
