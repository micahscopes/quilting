use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlValueKind {
    Text,
    Number,
    Toggle,
    OptionalUuid,
}

impl ControlValueKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Number => "number",
            Self::Toggle => "toggle",
            Self::OptionalUuid => "optional_uuid",
        }
    }

    fn accepts(self, value: &str) -> bool {
        match self {
            Self::Text => !value.is_empty(),
            Self::Number => value.parse::<f64>().is_ok_and(|number| number.is_finite()),
            Self::Toggle => matches!(value, "0" | "1"),
            Self::OptionalUuid => {
                value.is_empty()
                    || uuid::Uuid::parse_str(value).is_ok_and(|identifier| !identifier.is_nil())
            }
        }
    }

    fn equivalent(self, left: &str, right: &str) -> bool {
        match self {
            Self::Number => left
                .parse::<f64>()
                .ok()
                .zip(right.parse::<f64>().ok())
                .is_some_and(|(left, right)| {
                    left.is_finite() && right.is_finite() && left == right
                }),
            Self::OptionalUuid => match (uuid::Uuid::parse_str(left), uuid::Uuid::parse_str(right))
            {
                (Ok(left), Ok(right)) => left == right,
                _ => left == right,
            },
            Self::Text | Self::Toggle => left == right,
        }
    }
}

/// Stable application-level route metadata. DOM range/label/accessibility
/// metadata will extend this type; URL identity and defaults already live here
/// so future views cannot silently invent a second persistence contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlSpec {
    pub key: &'static str,
    pub default_value: &'static str,
    pub kind: ControlValueKind,
}

impl ControlSpec {
    pub fn is_default(self, value: &str) -> bool {
        self.kind.equivalent(value, self.default_value)
    }
}

macro_rules! spec {
    ($key:literal, $default:literal, $kind:ident) => {
        ControlSpec {
            key: $key,
            default_value: $default,
            kind: ControlValueKind::$kind,
        }
    };
}

/// Canonical order matches the current browser oracle. Once the route shadow
/// reaches parity, this ordering becomes the only serialization order.
pub const HYPERSCOPE_CONTROL_SPECS: &[ControlSpec] = &[
    spec!("glb", "horse.glb", Text),
    spec!("mode", "pbr", Text),
    spec!("xform", "identity", Text),
    spec!("mx", "5", Number),
    spec!("my", "0", Number),
    spec!("mz", "0", Number),
    spec!("mr", "20", Number),
    spec!("env", "rosendal_plains_1_1k", Text),
    spec!("matcap", "citric-acid", Text),
    spec!("res", "0", Number),
    spec!("density", "100", Number),
    spec!("atten", "1", Toggle),
    spec!("minpx", "16", Number),
    spec!("atlas", "7", Number),
    spec!("animate", "1", Toggle),
    spec!("anim", "-1", Number),
    spec!("fuzzy", "0", Toggle),
    spec!("fmode", "1", Number),
    spec!("fradius", "11", Number),
    spec!("fstr", "30", Number),
    spec!("ffocus", "62", Number),
    spec!("fbw", "10", Number),
    spec!("fnorm", "0", Toggle),
    spec!("fqual", "2", Number),
    spec!("fkaw", "1", Number),
    spec!("fkoff", "15", Number),
    spec!("smmove", "300", Number),
    spec!("smrotate", "300", Number),
    spec!("smnav", "hyperscope", Text),
    spec!("smlock", "1", Toggle),
    spec!("smswap", "0", Toggle),
    spec!("smpinv", "2", Number),
    spec!("smrinv", "1", Number),
    spec!("smbpinv", "0", Number),
    spec!("smbrinv", "0", Number),
    spec!("fov", "75", Number),
    spec!("interp", "70", Number),
    spec!("walksmooth", "18", Number),
    spec!("walkalign", "70", Number),
    spec!("walkspeed", "0", Number),
    spec!("walkscale", "0", Number),
    spec!("walkheight", "0", Number),
    spec!("walkimpl", "js", Text),
    spec!("lab", "0", Text),
    spec!("labfield", "edges", Text),
    spec!("laba", "3", Number),
    spec!("labb", "4", Number),
    spec!("labc", "4", Number),
    spec!("labmin", "1", Number),
    spec!("labmax", "6", Number),
    spec!("labphase", "0", Number),
    spec!("labbend", "55", Number),
    spec!("labgrid", "8", Number),
    spec!("labanimate", "0", Toggle),
    spec!("zoom", "3", Number),
    spec!("rx", "0.3", Number),
    spec!("ry", "0.5", Number),
    spec!("rz", "0", Number),
    spec!("px", "0", Number),
    spec!("py", "0", Number),
    spec!("pz", "0", Number),
    spec!("navshadow", "0", Toggle),
    spec!("presentation", "0", Toggle),
    spec!("cue", "", OptionalUuid),
    spec!("roundshadow", "0", Toggle),
    spec!("appshadow", "0", Toggle),
    spec!("routeshadow", "0", Toggle),
    spec!("rendershadow", "0", Toggle),
];

pub fn hyperscope_control_spec(key: &str) -> Option<&'static ControlSpec> {
    HYPERSCOPE_CONTROL_SPECS.iter().find(|spec| spec.key == key)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteDiagnosticCode {
    UnknownKey,
    DuplicateKey,
    InvalidValue,
}

impl RouteDiagnosticCode {
    pub fn name(self) -> &'static str {
        match self {
            Self::UnknownKey => "unknown_key",
            Self::DuplicateKey => "duplicate_key",
            Self::InvalidValue => "invalid_value",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteDiagnostic {
    pub code: RouteDiagnosticCode,
    pub key: String,
    pub value: String,
}

/// Decoded route values with deterministic first-value semantics. Invalid
/// known values are retained for shadow comparison but reported explicitly;
/// a later authority cutover can select fallback policy per ControlSpec.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HyperscopeRoute {
    values: BTreeMap<&'static str, String>,
    diagnostics: Vec<RouteDiagnostic>,
}

impl HyperscopeRoute {
    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Into<String>,
    {
        let mut route = Self::default();
        for (key, value) in pairs {
            let key = key.as_ref();
            let value = value.into();
            let Some(spec) = hyperscope_control_spec(key) else {
                route.diagnostics.push(RouteDiagnostic {
                    code: RouteDiagnosticCode::UnknownKey,
                    key: key.to_owned(),
                    value,
                });
                continue;
            };
            if route.values.contains_key(spec.key) {
                route.diagnostics.push(RouteDiagnostic {
                    code: RouteDiagnosticCode::DuplicateKey,
                    key: spec.key.to_owned(),
                    value,
                });
                continue;
            }
            if !spec.kind.accepts(&value) {
                route.diagnostics.push(RouteDiagnostic {
                    code: RouteDiagnosticCode::InvalidValue,
                    key: spec.key.to_owned(),
                    value: value.clone(),
                });
            }
            route.values.insert(spec.key, value);
        }
        route
    }

    pub fn value(&self, key: &str) -> Option<&str> {
        let spec = hyperscope_control_spec(key)?;
        Some(
            self.values
                .get(spec.key)
                .map_or(spec.default_value, String::as_str),
        )
    }

    pub fn canonical_pairs(&self) -> Vec<(&'static str, &str)> {
        HYPERSCOPE_CONTROL_SPECS
            .iter()
            .filter_map(|spec| {
                self.values
                    .get(spec.key)
                    .filter(|value| !spec.is_default(value))
                    .map(|value| (spec.key, value.as_str()))
            })
            .collect()
    }

    pub fn diagnostics(&self) -> &[RouteDiagnostic] {
        &self.diagnostics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn control_keys_are_unique_and_defaults_validate() {
        let mut keys = BTreeSet::new();
        for spec in HYPERSCOPE_CONTROL_SPECS {
            assert!(keys.insert(spec.key), "duplicate route key {}", spec.key);
            assert!(
                spec.kind.accepts(spec.default_value),
                "invalid default for {}",
                spec.key
            );
        }
    }

    #[test]
    fn canonical_route_omits_defaults_and_uses_spec_order() {
        let route = HyperscopeRoute::from_pairs([
            ("routeshadow", "1"),
            ("rx", "0.125"),
            ("mode", "lod"),
            ("glb", "horse.glb"),
            ("minpx", "16"),
            ("zoom", "3.00"),
            ("ry", "0.500"),
        ]);
        assert_eq!(route.value("zoom"), Some("3.00"));
        assert_eq!(
            route.canonical_pairs(),
            vec![("mode", "lod"), ("rx", "0.125"), ("routeshadow", "1")]
        );
        assert!(route.diagnostics().is_empty());
    }

    #[test]
    fn first_duplicate_wins_and_bad_or_unknown_values_are_diagnostic() {
        let route = HyperscopeRoute::from_pairs([
            ("mode", "wire"),
            ("mode", "pbr"),
            ("atten", "yes"),
            ("rx", "NaN"),
            ("mystery", "1"),
        ]);
        assert_eq!(route.value("mode"), Some("wire"));
        assert_eq!(route.value("atten"), Some("yes"));
        assert_eq!(route.diagnostics().len(), 4);
        assert_eq!(
            route
                .diagnostics()
                .iter()
                .map(|diagnostic| diagnostic.code)
                .collect::<Vec<_>>(),
            vec![
                RouteDiagnosticCode::DuplicateKey,
                RouteDiagnosticCode::InvalidValue,
                RouteDiagnosticCode::InvalidValue,
                RouteDiagnosticCode::UnknownKey,
            ]
        );
    }

    #[test]
    fn optional_presentation_cue_accepts_absence_or_a_non_nil_uuid() {
        let absent = HyperscopeRoute::from_pairs([("cue", "")]);
        assert!(absent.canonical_pairs().is_empty());
        assert!(absent.diagnostics().is_empty());

        let cue = "e0000000-0000-4000-8000-000000000004";
        let linked = HyperscopeRoute::from_pairs([("presentation", "1"), ("cue", cue)]);
        assert_eq!(
            linked.canonical_pairs(),
            vec![("presentation", "1"), ("cue", cue)]
        );
        assert!(linked.diagnostics().is_empty());

        for invalid in ["not-a-uuid", "00000000-0000-0000-0000-000000000000"] {
            let route = HyperscopeRoute::from_pairs([("cue", invalid)]);
            assert_eq!(route.diagnostics().len(), 1);
            assert_eq!(
                route.diagnostics()[0].code,
                RouteDiagnosticCode::InvalidValue
            );
        }
    }
}
