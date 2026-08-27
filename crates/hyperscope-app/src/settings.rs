use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlValueKind {
    Text,
    Number,
    Toggle,
    LodRatio,
    Implementation,
    OptionalUuid,
}

impl ControlValueKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Number => "number",
            Self::Toggle => "toggle",
            Self::LodRatio => "lod_ratio",
            Self::Implementation => "implementation",
            Self::OptionalUuid => "optional_uuid",
        }
    }

    fn accepts(self, value: &str) -> bool {
        match self {
            Self::Text => !value.is_empty(),
            Self::Number => value.parse::<f64>().is_ok_and(|number| number.is_finite()),
            Self::Toggle => matches!(value, "0" | "1"),
            Self::LodRatio => matches!(value, "2" | "4"),
            Self::Implementation => matches!(value, "js" | "shadow" | "rust"),
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
            Self::Text | Self::Toggle | Self::LodRatio | Self::Implementation => left == right,
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
    spec!("lodratio", "2", LodRatio),
    spec!("animate", "1", Toggle),
    spec!("anim", "-1", Number),
    spec!("fuzzy", "0", Toggle),
    spec!("fmode", "1", Number),
    spec!("fradius", "11", Number),
    spec!("fstr", "30", Number),
    spec!("ffocus", "62", Number),
    spec!("fbw", "10", Number),
    spec!("fnorm", "0", Toggle),
    spec!("fqual", "1", Number),
    spec!("fkaw", "3", Number),
    spec!("fkoff", "15", Number),
    spec!("smmove", "300", Number),
    spec!("smrotate", "300", Number),
    spec!("smnav", "hyperscope", Text),
    spec!("smlock", "1", Toggle),
    spec!("smswap", "0", Toggle),
    spec!("smbackground", "0", Toggle),
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
    spec!("walkimpl", "js", Implementation),
    spec!("navimpl", "js", Implementation),
    spec!("selectionimpl", "rust", Implementation),
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
    spec!("aim", "0", Toggle),
    spec!("selasset", "", OptionalUuid),
    spec!("selentity", "", OptionalUuid),
    spec!("presentation", "0", Toggle),
    spec!("cue", "", OptionalUuid),
    spec!("presentimpl", "rust", Implementation),
    spec!("roundshadow", "0", Toggle),
    spec!("appshadow", "0", Toggle),
    spec!("assetimpl", "rust", Implementation),
    spec!("sceneimpl", "rust", Implementation),
    spec!("routeimpl", "rust", Implementation),
    spec!("lodimpl", "js", Implementation),
    spec!("rendershadow", "0", Toggle),
    spec!("adaptiveshadow", "0", Toggle),
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
        route.validate_selection_pair();
        route
    }

    fn validate_selection_pair(&mut self) {
        let asset_present = self
            .values
            .get("selasset")
            .is_some_and(|value| !value.is_empty());
        let entity_present = self
            .values
            .get("selentity")
            .is_some_and(|value| !value.is_empty());
        if asset_present == entity_present {
            return;
        }
        let missing_key = if asset_present { "selentity" } else { "selasset" };
        self.diagnostics.push(RouteDiagnostic {
            code: RouteDiagnosticCode::InvalidValue,
            key: missing_key.to_owned(),
            value: String::new(),
        });
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
            ("routeimpl", "shadow"),
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
            vec![("mode", "lod"), ("rx", "0.125"), ("routeimpl", "shadow")]
        );
        assert!(route.diagnostics().is_empty());
    }

    #[test]
    fn finite_camera_target_policy_is_explicit_and_validated() {
        let free = HyperscopeRoute::from_pairs([("aim", "0")]);
        assert_eq!(free.value("aim"), Some("0"));
        assert!(free.canonical_pairs().is_empty());

        let aimed = HyperscopeRoute::from_pairs([("aim", "1")]);
        assert_eq!(aimed.value("aim"), Some("1"));
        assert_eq!(aimed.canonical_pairs(), vec![("aim", "1")]);
        assert!(aimed.diagnostics().is_empty());

        let invalid = HyperscopeRoute::from_pairs([("aim", "free")]);
        assert_eq!(invalid.value("aim"), Some("free"));
        assert_eq!(invalid.canonical_pairs(), vec![("aim", "free")]);
        assert_eq!(invalid.diagnostics().len(), 1);
        assert_eq!(
            invalid.diagnostics()[0].code,
            RouteDiagnosticCode::InvalidValue
        );
    }

    #[test]
    fn selected_identity_route_is_atomic_and_canonical() {
        let asset = "60000000-0000-4000-8000-000000000001";
        let entity = "70000000-0000-4000-8000-000000000001";
        let selected = HyperscopeRoute::from_pairs([
            ("selentity", entity),
            ("selasset", asset),
        ]);
        assert_eq!(selected.value("selasset"), Some(asset));
        assert_eq!(selected.value("selentity"), Some(entity));
        assert_eq!(
            selected.canonical_pairs(),
            vec![("selasset", asset), ("selentity", entity)]
        );
        assert!(selected.diagnostics().is_empty());

        for (key, value, missing_key) in [
            ("selasset", asset, "selentity"),
            ("selentity", entity, "selasset"),
        ] {
            let partial = HyperscopeRoute::from_pairs([(key, value)]);
            assert_eq!(partial.diagnostics().len(), 1);
            assert_eq!(partial.diagnostics()[0].code, RouteDiagnosticCode::InvalidValue);
            assert_eq!(partial.diagnostics()[0].key, missing_key);
        }
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
    fn lod_grading_route_admits_only_runtime_atlas_policies() {
        for accepted in ["2", "4"] {
            let route = HyperscopeRoute::from_pairs([("lodratio", accepted)]);
            assert!(route.diagnostics().is_empty());
            assert_eq!(route.value("lodratio"), Some(accepted));
        }
        for rejected in ["1", "3", "4.0", "8"] {
            let route = HyperscopeRoute::from_pairs([("lodratio", rejected)]);
            assert_eq!(route.diagnostics().len(), 1);
            assert_eq!(
                route.diagnostics()[0].code,
                RouteDiagnosticCode::InvalidValue,
            );
        }
    }

    #[test]
    fn implementation_routes_admit_only_measured_authority_modes() {
        for key in [
            "walkimpl",
            "navimpl",
            "selectionimpl",
            "presentimpl",
            "assetimpl",
            "sceneimpl",
            "routeimpl",
            "lodimpl",
        ] {
            for accepted in ["js", "shadow", "rust"] {
                let route = HyperscopeRoute::from_pairs([(key, accepted)]);
                assert!(route.diagnostics().is_empty());
                assert_eq!(route.value(key), Some(accepted));
            }
            for rejected in ["", "browser", "auto", "Rust"] {
                let route = HyperscopeRoute::from_pairs([(key, rejected)]);
                assert_eq!(route.diagnostics().len(), 1);
                assert_eq!(
                    route.diagnostics()[0].code,
                    RouteDiagnosticCode::InvalidValue,
                );
            }
        }
    }

    #[test]
    fn navigation_cutover_keeps_javascript_default_and_both_measured_rust_routes() {
        let default_route = HyperscopeRoute::from_pairs([("navimpl", "js")]);
        assert!(default_route.canonical_pairs().is_empty());

        for implementation in ["shadow", "rust"] {
            let route = HyperscopeRoute::from_pairs([("navimpl", implementation)]);
            assert_eq!(
                route.canonical_pairs(),
                vec![("navimpl", implementation)]
            );
            assert!(route.diagnostics().is_empty());
        }
    }

    #[test]
    fn rust_asset_authority_is_default_with_an_explicit_js_rollback() {
        let default_route = HyperscopeRoute::from_pairs([("assetimpl", "rust")]);
        assert_eq!(default_route.value("assetimpl"), Some("rust"));
        assert!(default_route.canonical_pairs().is_empty());

        let rollback_route = HyperscopeRoute::from_pairs([("assetimpl", "js")]);
        assert_eq!(rollback_route.canonical_pairs(), vec![("assetimpl", "js")]);
        assert!(rollback_route.diagnostics().is_empty());
    }

    #[test]
    fn rust_presentation_authority_is_default_with_an_explicit_js_rollback() {
        let default_route = HyperscopeRoute::from_pairs([("presentimpl", "rust")]);
        assert_eq!(default_route.value("presentimpl"), Some("rust"));
        assert!(default_route.canonical_pairs().is_empty());

        let rollback_route = HyperscopeRoute::from_pairs([("presentimpl", "js")]);
        assert_eq!(
            rollback_route.canonical_pairs(),
            vec![("presentimpl", "js")]
        );
        assert!(rollback_route.diagnostics().is_empty());
    }

    #[test]
    fn rust_selection_authority_is_default_with_an_explicit_js_rollback() {
        let default_route = HyperscopeRoute::from_pairs([("selectionimpl", "rust")]);
        assert_eq!(default_route.value("selectionimpl"), Some("rust"));
        assert!(default_route.canonical_pairs().is_empty());

        let rollback_route = HyperscopeRoute::from_pairs([("selectionimpl", "js")]);
        assert_eq!(
            rollback_route.canonical_pairs(),
            vec![("selectionimpl", "js")]
        );
        assert!(rollback_route.diagnostics().is_empty());
    }

    #[test]
    fn rust_scene_authority_is_default_with_an_explicit_js_rollback() {
        let default_route = HyperscopeRoute::from_pairs([("sceneimpl", "rust")]);
        assert_eq!(default_route.value("sceneimpl"), Some("rust"));
        assert!(default_route.canonical_pairs().is_empty());

        let rollback_route = HyperscopeRoute::from_pairs([("sceneimpl", "js")]);
        assert_eq!(rollback_route.canonical_pairs(), vec![("sceneimpl", "js")]);
        assert!(rollback_route.diagnostics().is_empty());
    }

    #[test]
    fn rust_route_authority_is_default_with_an_explicit_js_rollback() {
        let default_route = HyperscopeRoute::from_pairs([("routeimpl", "rust")]);
        assert_eq!(default_route.value("routeimpl"), Some("rust"));
        assert!(default_route.canonical_pairs().is_empty());

        let rollback_route = HyperscopeRoute::from_pairs([("routeimpl", "js")]);
        assert_eq!(rollback_route.canonical_pairs(), vec![("routeimpl", "js")]);
        assert!(rollback_route.diagnostics().is_empty());
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
