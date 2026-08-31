use std::collections::BTreeMap;

use crate::{
    FocusDiagnosticView, FocusPostprocessMode, FocusPostprocessSettings, RenderSettings,
};
use hyperscape::SurfaceWalkControls;
use hyperscape_protocol::{AssetEntityId, AssetId, EntityId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlValueKind {
    Text,
    Number,
    Toggle,
    Choice,
    RenderMode,
    ResolutionLevel,
    TessellationDensity,
    PixelFloor,
    AtlasExponent,
    LodRatio,
    Implementation,
    RenderBackend,
    OptionalUuid,
}

impl ControlValueKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Number => "number",
            Self::Toggle => "toggle",
            Self::Choice => "choice",
            Self::RenderMode => "render_mode",
            Self::ResolutionLevel => "resolution_level",
            Self::TessellationDensity => "tessellation_density",
            Self::PixelFloor => "pixel_floor",
            Self::AtlasExponent => "atlas_exponent",
            Self::LodRatio => "lod_ratio",
            Self::Implementation => "implementation",
            Self::RenderBackend => "render_backend",
            Self::OptionalUuid => "optional_uuid",
        }
    }

    fn accepts_shape(self, value: &str) -> bool {
        match self {
            Self::Text => !value.is_empty(),
            Self::Number => value.parse::<f64>().is_ok_and(|number| number.is_finite()),
            Self::Toggle => matches!(value, "0" | "1"),
            Self::Choice => !value.is_empty(),
            Self::RenderMode => matches!(
                value,
                "pbr" | "matcap" | "wire" | "normals" | "both" | "lod" | "stretch"
            ),
            Self::ResolutionLevel | Self::TessellationDensity | Self::AtlasExponent => {
                value.parse::<i64>().is_ok()
            }
            Self::PixelFloor => value.parse::<f64>().is_ok_and(|number| number.is_finite()),
            Self::LodRatio => matches!(value, "2" | "4"),
            Self::Implementation => matches!(value, "js" | "shadow" | "rust"),
            Self::RenderBackend => matches!(value, "webgl2" | "webgpu-shadow" | "webgpu"),
            Self::OptionalUuid => {
                value.is_empty()
                    || uuid::Uuid::parse_str(value).is_ok_and(|identifier| !identifier.is_nil())
            }
        }
    }

    fn equivalent(self, left: &str, right: &str) -> bool {
        match self {
            Self::Number
            | Self::ResolutionLevel
            | Self::TessellationDensity
            | Self::PixelFloor
            | Self::AtlasExponent => left
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
            Self::Text
            | Self::Toggle
            | Self::Choice
            | Self::RenderMode
            | Self::LodRatio
            | Self::Implementation
            | Self::RenderBackend => left == right,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NumericControlDomain {
    pub minimum: f64,
    pub maximum: f64,
    pub integral: bool,
    /// Preferred view increment. Admission validates the domain and numeric
    /// shape, not quantization, so copied high-precision values remain exact.
    pub step: f64,
}

impl NumericControlDomain {
    fn accepts(self, value: &str) -> bool {
        if self.integral {
            value.parse::<i64>().is_ok_and(|number| {
                let number = number as f64;
                (self.minimum..=self.maximum).contains(&number)
            })
        } else {
            value.parse::<f64>().is_ok_and(|number| {
                number.is_finite() && (self.minimum..=self.maximum).contains(&number)
            })
        }
    }
}

/// Stable application-level route metadata. DOM range/label/accessibility
/// metadata will extend this type; URL identity and defaults already live here
/// so future views cannot silently invent a second persistence contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlSpec {
    pub key: &'static str,
    pub default_value: &'static str,
    pub kind: ControlValueKind,
    pub numeric_domain: Option<NumericControlDomain>,
    pub choices: &'static [&'static str],
}

impl ControlSpec {
    pub fn accepts(self, value: &str) -> bool {
        self.kind.accepts_shape(value)
            && self
                .numeric_domain
                .is_none_or(|domain| domain.accepts(value))
            && (self.choices.is_empty() || self.choices.contains(&value))
    }

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
            numeric_domain: None,
            choices: &[],
        }
    };
}

macro_rules! numeric_spec {
    ($key:literal, $default:literal, $kind:ident, $minimum:expr, $maximum:expr, $integral:literal, $step:expr) => {
        ControlSpec {
            key: $key,
            default_value: $default,
            kind: ControlValueKind::$kind,
            numeric_domain: Some(NumericControlDomain {
                minimum: $minimum,
                maximum: $maximum,
                integral: $integral,
                step: $step,
            }),
            choices: &[],
        }
    };
}

macro_rules! choice_spec {
    ($key:literal, $default:literal, [$($choice:literal),+ $(,)?]) => {
        ControlSpec {
            key: $key,
            default_value: $default,
            kind: ControlValueKind::Choice,
            numeric_domain: None,
            choices: &[$($choice),+],
        }
    };
}

/// Canonical order matches the current browser oracle. Once the route shadow
/// reaches parity, this ordering becomes the only serialization order.
pub const HYPERSCOPE_CONTROL_SPECS: &[ControlSpec] = &[
    spec!("glb", "horse.glb", Text),
    spec!("gfx", "webgl2", RenderBackend),
    spec!("pickimpl", "js", Implementation),
    spec!("mode", "pbr", RenderMode),
    choice_spec!(
        "xform",
        "identity",
        ["identity", "sphere_reflection", "rotation", "translation"]
    ),
    numeric_spec!("mx", "5", Number, -30.0, 30.0, false, 0.1),
    numeric_spec!("my", "0", Number, -30.0, 30.0, false, 0.1),
    numeric_spec!("mz", "0", Number, -30.0, 30.0, false, 0.1),
    numeric_spec!("mr", "20", Number, 0.11, 50.0, false, 0.01),
    spec!("env", "rosendal_plains_1_1k", Text),
    spec!("matcap", "citric-acid", Text),
    numeric_spec!("res", "0", ResolutionLevel, 0.0, 6.0, true, 1.0),
    numeric_spec!("density", "100", TessellationDensity, 1.0, 500.0, true, 1.0),
    spec!("atten", "1", Toggle),
    numeric_spec!("minpx", "16", PixelFloor, 1.0, 64.0, false, 0.1),
    numeric_spec!("atlas", "7", AtlasExponent, 3.0, 9.0, true, 1.0),
    spec!("lodratio", "2", LodRatio),
    spec!("animate", "1", Toggle),
    numeric_spec!("anim", "-1", Number, -1.0, i32::MAX as f64, true, 1.0),
    numeric_spec!("animtime", "0", Number, -1e9, 1e9, false, 0.001),
    numeric_spec!("animspeed", "1", Number, -1e6, 1e6, false, 0.01),
    spec!("animclockimpl", "js", Implementation),
    spec!("animclipimpl", "js", Implementation),
    spec!("fuzzy", "0", Toggle),
    choice_spec!("fmode", "1", ["0", "1", "2", "3"]),
    choice_spec!("fdebug", "0", ["0", "1", "2", "3"]),
    numeric_spec!("fradius", "11", Number, 4.0, 128.0, true, 1.0),
    numeric_spec!("fstr", "30", Number, 1.0, 30.0, false, 0.1),
    numeric_spec!("ffocus", "62", Number, 0.0, 100.0, false, 0.1),
    numeric_spec!("fbw", "10", Number, 1.0, 50.0, false, 0.1),
    spec!("fnorm", "0", Toggle),
    numeric_spec!("fqual", "1", Number, 1.0, 4.0, true, 1.0),
    numeric_spec!("fkaw", "3", Number, 0.0, 4.0, true, 1.0),
    numeric_spec!("fkoff", "15", Number, 1.0, 30.0, false, 0.1),
    numeric_spec!("smmove", "300", Number, 10.0, 900.0, false, 1.0),
    numeric_spec!("smrotate", "300", Number, 10.0, 900.0, false, 1.0),
    choice_spec!(
        "smnav",
        "hyperscope",
        ["hyperscope", "object", "fly", "drone"]
    ),
    spec!("smlock", "1", Toggle),
    spec!("smswap", "0", Toggle),
    spec!("smbackground", "0", Toggle),
    numeric_spec!("smpinv", "2", Number, 0.0, 7.0, true, 1.0),
    numeric_spec!("smrinv", "1", Number, 0.0, 7.0, true, 1.0),
    numeric_spec!("smbpinv", "0", Number, 0.0, 7.0, true, 1.0),
    numeric_spec!("smbrinv", "0", Number, 0.0, 7.0, true, 1.0),
    numeric_spec!("fov", "75", Number, 35.0, 110.0, true, 1.0),
    numeric_spec!("interp", "70", Number, 5.0, 500.0, true, 1.0),
    numeric_spec!("walksmooth", "18", Number, 0.0, 150.0, true, 1.0),
    numeric_spec!("walkalign", "70", Number, 0.0, 100.0, true, 1.0),
    numeric_spec!("walkspeed", "0", Number, -400.0, 400.0, false, 1.0),
    numeric_spec!("walkscale", "0", Number, -800.0, 800.0, false, 1.0),
    numeric_spec!("walkheight", "0", Number, -400.0, 400.0, false, 1.0),
    spec!("navstateimpl", "js", Implementation),
    spec!("walkimpl", "js", Implementation),
    spec!("navimpl", "js", Implementation),
    spec!("selectionimpl", "rust", Implementation),
    spec!("patchlabimpl", "js", Implementation),
    choice_spec!("lab", "0", ["0", "triangle", "plane", "cube"]),
    choice_spec!(
        "labfield",
        "edges",
        ["edges", "wave", "radial", "sweep", "uniform"]
    ),
    numeric_spec!("laba", "3", Number, 0.0, 9.0, true, 1.0),
    numeric_spec!("labb", "4", Number, 0.0, 9.0, true, 1.0),
    numeric_spec!("labc", "4", Number, 0.0, 9.0, true, 1.0),
    numeric_spec!("labmin", "1", Number, 0.0, 9.0, true, 1.0),
    numeric_spec!("labmax", "6", Number, 0.0, 9.0, true, 1.0),
    numeric_spec!(
        "labphase",
        "0",
        Number,
        0.0,
        std::f64::consts::TAU,
        false,
        0.001
    ),
    numeric_spec!("labbend", "55", Number, 0.0, 100.0, true, 1.0),
    numeric_spec!("labgrid", "8", Number, 2.0, 16.0, true, 1.0),
    spec!("labanimate", "0", Toggle),
    numeric_spec!("zoom", "3", Number, 0.1, 1000.0, false, 0.01),
    numeric_spec!(
        "rx",
        "0.3",
        Number,
        -std::f64::consts::FRAC_PI_2 + 0.01,
        std::f64::consts::FRAC_PI_2 - 0.01,
        false,
        0.001
    ),
    numeric_spec!("ry", "0.5", Number, -1e6, 1e6, false, 0.001),
    numeric_spec!(
        "rz",
        "0",
        Number,
        -std::f64::consts::PI,
        std::f64::consts::PI,
        false,
        0.001
    ),
    numeric_spec!("px", "0", Number, -1e6, 1e6, false, 0.001),
    numeric_spec!("py", "0", Number, -1e6, 1e6, false, 0.001),
    numeric_spec!("pz", "0", Number, -1e6, 1e6, false, 0.001),
    spec!("aim", "0", Toggle),
    spec!("selasset", "", OptionalUuid),
    spec!("selentity", "", OptionalUuid),
    spec!("presentation", "0", Toggle),
    spec!("cue", "", OptionalUuid),
    spec!("presentimpl", "rust", Implementation),
    spec!("gfxpresentimpl", "rust", Implementation),
    spec!("roundshadow", "0", Toggle),
    spec!("appshadow", "0", Toggle),
    spec!("assetimpl", "rust", Implementation),
    spec!("sceneimpl", "rust", Implementation),
    spec!("routeimpl", "rust", Implementation),
    spec!("renderstateimpl", "rust", Implementation),
    spec!("lodimpl", "js", Implementation),
    spec!("rendershadow", "0", Toggle),
    spec!("adaptiveshadow", "0", Toggle),
    spec!("rootgroupshadow", "0", Toggle),
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

/// Partial clip-relative animation-clock intent authored by a URL. Absence of
/// a field is semantically different from explicitly linking its default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteAnimationClock {
    pub time_seconds: Option<f64>,
    pub speed: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteTransformKind {
    Identity,
    SphereReflection,
    Rotation,
    Translation,
}

impl RouteTransformKind {
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::SphereReflection => "sphere_reflection",
            Self::Rotation => "rotation",
            Self::Translation => "translation",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteTransformSettings {
    pub kind: RouteTransformKind,
    pub center_controls: [f64; 3],
    pub radius_control: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteCameraSettings {
    pub zoom: f64,
    pub euler_radians: [f64; 3],
    pub position: [f64; 3],
    pub semantic_target_enabled: bool,
    pub vertical_fov_degrees: f64,
    pub focus_transition_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteSpaceMouseProfile {
    Hyperscope,
    Object,
    Fly,
    Drone,
}

impl RouteSpaceMouseProfile {
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::Hyperscope => "hyperscope",
            Self::Object => "object",
            Self::Fly => "fly",
            Self::Drone => "drone",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteSpaceMouseSettings {
    pub move_sensitivity: f64,
    pub rotate_sensitivity: f64,
    pub profile: RouteSpaceMouseProfile,
    pub lock_horizon: bool,
    pub swap_yz: bool,
    pub accept_background_input: bool,
    pub hyperscope_pan_invert_mask: u8,
    pub hyperscope_rotate_invert_mask: u8,
    pub blender_pan_invert_mask: u8,
    pub blender_rotate_invert_mask: u8,
}

/// Device-independent navigation preferences owned by the application.
///
/// Raw HID policy, axis maps, and browser focus permissions remain adapter
/// concerns. These values instead change semantic transitions and surface
/// locomotion, so mouse, SpaceMouse, replay, Blender, and future game inputs
/// must all observe one committed packet.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(default, rename_all = "camelCase"))]
pub struct NavigationSettings {
    pub transition_seconds: f64,
    pub surface_walk: SurfaceWalkControls,
}

impl Default for NavigationSettings {
    fn default() -> Self {
        Self {
            transition_seconds: 0.7,
            surface_walk: SurfaceWalkControls::default(),
        }
    }
}

impl NavigationSettings {
    pub fn validate(self) -> Result<Self, &'static str> {
        if !self.transition_seconds.is_finite() || !(0.0..=5.0).contains(&self.transition_seconds) {
            return Err("navigation transition duration must be finite and in [0,5]");
        }
        self.surface_walk
            .metrics(1.0, false)
            .map_err(|_| "surface-walk controls are invalid")?;
        Ok(self)
    }
}

/// One typed startup value for browser, replay, and future Blender navigation
/// adapters. URL/UI units are converted here; adapters consume semantic
/// seconds and fractions and do not maintain a parallel route parser.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteNavigationSettings {
    pub transform: RouteTransformSettings,
    pub camera: RouteCameraSettings,
    pub space_mouse: RouteSpaceMouseSettings,
    pub surface_walk: SurfaceWalkControls,
}

impl RouteNavigationSettings {
    /// Project startup-only camera, transform, and HID values away, retaining
    /// the semantic packet that belongs in `hyperscope-app` state.
    pub fn application_settings(self) -> Result<NavigationSettings, &'static str> {
        NavigationSettings {
            transition_seconds: self.camera.focus_transition_seconds,
            surface_walk: self.surface_walk,
        }
        .validate()
    }
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
            if !spec.accepts(&value) {
                route.diagnostics.push(RouteDiagnostic {
                    code: RouteDiagnosticCode::InvalidValue,
                    key: spec.key.to_owned(),
                    value: value.clone(),
                });
            }
            route.values.insert(spec.key, value);
        }
        route.validate_selection_pair();
        route.validate_patch_lab_atlas();
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
        let missing_key = if asset_present {
            "selentity"
        } else {
            "selasset"
        };
        self.diagnostics.push(RouteDiagnostic {
            code: RouteDiagnosticCode::InvalidValue,
            key: missing_key.to_owned(),
            value: String::new(),
        });
    }

    fn validate_patch_lab_atlas(&mut self) {
        let Some(atlas_exponent) = self
            .value("atlas")
            .and_then(|value| value.parse::<u8>().ok())
        else {
            return;
        };
        for key in ["laba", "labb", "labc", "labmin", "labmax"] {
            let Some(value) = self.values.get(key) else {
                continue;
            };
            if value
                .parse::<u8>()
                .is_ok_and(|exponent| exponent > atlas_exponent)
            {
                self.diagnostics.push(RouteDiagnostic {
                    code: RouteDiagnosticCode::InvalidValue,
                    key: key.to_owned(),
                    value: value.clone(),
                });
            }
        }
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

    /// Return every canonical control value in registry order, including
    /// defaults for omitted controls. This is the authoritative startup read
    /// model; unlike `canonical_pairs`, it is not intended for compact URLs.
    pub fn resolved_pairs(&self) -> Vec<(&'static str, &str)> {
        HYPERSCOPE_CONTROL_SPECS
            .iter()
            .map(|spec| {
                (
                    spec.key,
                    self.values
                        .get(spec.key)
                        .map_or(spec.default_value, String::as_str),
                )
            })
            .collect()
    }

    /// Resolve the route's complete renderer-independent render policy.
    ///
    /// Omitted values come from the canonical Rust control registry. Browser
    /// adapters may convert the resulting numbers into control values, but
    /// must not apply another default or range policy after route admission.
    pub fn render_settings(&self) -> Result<RenderSettings, &'static str> {
        let style = match self.value("mode") {
            Some("both") => "matcap_wire",
            Some(style @ ("pbr" | "matcap" | "wire" | "normals" | "lod" | "stretch")) => style,
            _ => return Err("route render mode is invalid"),
        };
        let resolution_level = self
            .value("res")
            .and_then(|value| value.parse::<u8>().ok())
            .ok_or("route resolution level is invalid")?;
        let density = self
            .value("density")
            .and_then(|value| value.parse::<f64>().ok())
            .ok_or("route tessellation density is invalid")?;
        let screen_attenuation = match self.value("atten") {
            Some("0") => false,
            Some("1") => true,
            _ => return Err("route screen attenuation value is invalid"),
        };
        let min_pixels_per_subdivision = self
            .value("minpx")
            .and_then(|value| value.parse::<f64>().ok())
            .ok_or("route pixel floor is invalid")?;
        let atlas_exponent = self
            .value("atlas")
            .and_then(|value| value.parse::<u8>().ok())
            .ok_or("route atlas exponent is invalid")?;
        let max_face_edge_ratio = self
            .value("lodratio")
            .and_then(|value| value.parse::<u8>().ok())
            .ok_or("route face-edge ratio is invalid")?;
        let focus_enabled = match self.value("fuzzy") {
            Some("0") => false,
            Some("1") => true,
            _ => return Err("route focus postprocess enable value is invalid"),
        };
        let focus_mode = self
            .value("fmode")
            .and_then(|value| value.parse::<u8>().ok())
            .and_then(FocusPostprocessMode::from_wire_index)
            .ok_or("route focus postprocess mode is invalid")?;
        let focus_diagnostic_view = self
            .value("fdebug")
            .and_then(|value| value.parse::<u8>().ok())
            .and_then(FocusDiagnosticView::from_wire_index)
            .ok_or("route focus diagnostic view is invalid")?;
        let blur_radius_pixels = self
            .value("fradius")
            .and_then(|value| value.parse::<u16>().ok())
            .ok_or("route focus blur radius is invalid")?;
        let blur_strength = self
            .value("fstr")
            .and_then(|value| value.parse::<f64>().ok())
            .map(|value| value / 10.0)
            .ok_or("route focus blur strength is invalid")?;
        let focus_coordinate = self
            .value("ffocus")
            .and_then(|value| value.parse::<f64>().ok())
            .map(|value| value / 100.0)
            .ok_or("route focus coordinate is invalid")?;
        let bandwidth = self
            .value("fbw")
            .and_then(|value| value.parse::<f64>().ok())
            .map(|value| value / 100.0)
            .ok_or("route focus bandwidth is invalid")?;
        let normalize_range = match self.value("fnorm") {
            Some("0") => false,
            Some("1") => true,
            _ => return Err("route focus normalization value is invalid"),
        };
        let gaussian_passes = self
            .value("fqual")
            .and_then(|value| value.parse::<u8>().ok())
            .ok_or("route Gaussian pass count is invalid")?;
        let kawase_passes = self
            .value("fkaw")
            .and_then(|value| value.parse::<u8>().ok())
            .ok_or("route Kawase pass count is invalid")?;
        let kawase_offset = self
            .value("fkoff")
            .and_then(|value| value.parse::<f64>().ok())
            .map(|value| value / 10.0)
            .ok_or("route Kawase offset is invalid")?;

        RenderSettings::from_wire_values(
            style,
            resolution_level,
            density,
            screen_attenuation,
            min_pixels_per_subdivision,
            atlas_exponent,
            max_face_edge_ratio,
        )
        .and_then(|settings| {
            settings.with_focus_postprocess(FocusPostprocessSettings {
                enabled: focus_enabled,
                mode: focus_mode,
                diagnostic_view: focus_diagnostic_view,
                blur_radius_pixels,
                blur_strength,
                focus_coordinate,
                bandwidth,
                normalize_range,
                gaussian_passes,
                kawase_passes,
                kawase_offset,
            })
        })
    }

    /// Resolve the optional stable asset-scoped selection carried by a route.
    /// Runtime node indices never enter this boundary.
    pub fn selected_identity(&self) -> Result<Option<AssetEntityId>, &'static str> {
        let asset = self.value("selasset").unwrap_or_default();
        let entity = self.value("selentity").unwrap_or_default();
        if asset.is_empty() && entity.is_empty() {
            return Ok(None);
        }
        if asset.is_empty() || entity.is_empty() {
            return Err("route selection identity must contain both IDs");
        }
        let asset = uuid::Uuid::parse_str(asset)
            .ok()
            .and_then(|value| AssetId::new(value).ok())
            .ok_or("route selection asset ID is invalid")?;
        let entity = uuid::Uuid::parse_str(entity)
            .ok()
            .and_then(|value| EntityId::new(value).ok())
            .ok_or("route selection entity ID is invalid")?;
        AssetEntityId::new(asset, entity)
            .map(Some)
            .map_err(|_| "route selection identity is invalid")
    }

    pub fn animation_clock(&self) -> Result<Option<RouteAnimationClock>, &'static str> {
        let time_seconds = self
            .values
            .get("animtime")
            .map(|value| {
                if !hyperscope_control_spec("animtime").is_some_and(|spec| spec.accepts(value)) {
                    return Err("route animation time is invalid");
                }
                value
                    .parse::<f64>()
                    .ok()
                    .filter(|value| value.is_finite())
                    .ok_or("route animation time is invalid")
            })
            .transpose()?;
        let speed = self
            .values
            .get("animspeed")
            .map(|value| {
                if !hyperscope_control_spec("animspeed").is_some_and(|spec| spec.accepts(value)) {
                    return Err("route animation speed is invalid");
                }
                value
                    .parse::<f64>()
                    .ok()
                    .filter(|value| value.is_finite())
                    .ok_or("route animation speed is invalid")
            })
            .transpose()?;
        if time_seconds.is_none() && speed.is_none() {
            Ok(None)
        } else {
            Ok(Some(RouteAnimationClock {
                time_seconds,
                speed,
            }))
        }
    }

    pub fn navigation_settings(&self) -> Result<RouteNavigationSettings, &'static str> {
        let number = |key, error| {
            self.value(key)
                .filter(|value| {
                    hyperscope_control_spec(key).is_some_and(|spec| spec.accepts(value))
                })
                .and_then(|value| value.parse::<f64>().ok())
                .filter(|value| value.is_finite())
                .ok_or(error)
        };
        let toggle = |key, error| match self.value(key) {
            Some("0") => Ok(false),
            Some("1") => Ok(true),
            _ => Err(error),
        };
        let mask = |key, error| {
            self.value(key)
                .filter(|value| {
                    hyperscope_control_spec(key).is_some_and(|spec| spec.accepts(value))
                })
                .and_then(|value| value.parse::<u8>().ok())
                .ok_or(error)
        };

        let transform_kind = match self.value("xform") {
            Some("identity") => RouteTransformKind::Identity,
            Some("sphere_reflection") => RouteTransformKind::SphereReflection,
            Some("rotation") => RouteTransformKind::Rotation,
            Some("translation") => RouteTransformKind::Translation,
            _ => return Err("route transform kind is invalid"),
        };
        let space_mouse_profile = match self.value("smnav") {
            Some("hyperscope") => RouteSpaceMouseProfile::Hyperscope,
            Some("object") => RouteSpaceMouseProfile::Object,
            Some("fly") => RouteSpaceMouseProfile::Fly,
            Some("drone") => RouteSpaceMouseProfile::Drone,
            _ => return Err("route SpaceMouse profile is invalid"),
        };
        let surface_walk = SurfaceWalkControls {
            speed_octave_steps: number("walkspeed", "route walk speed is invalid")?,
            body_scale_octave_steps: number("walkscale", "route walk scale is invalid")?,
            eye_height_octave_steps: number("walkheight", "route walk height is invalid")?,
            smoothing_seconds: number("walksmooth", "route walk smoothing is invalid")? / 100.0,
            tangent_pull_fraction: number("walkalign", "route walk alignment is invalid")? / 100.0,
            ..SurfaceWalkControls::default()
        };
        surface_walk
            .metrics(1.0, false)
            .map_err(|_| "route surface-walk controls are invalid")?;

        Ok(RouteNavigationSettings {
            transform: RouteTransformSettings {
                kind: transform_kind,
                center_controls: [
                    number("mx", "route transform center X is invalid")?,
                    number("my", "route transform center Y is invalid")?,
                    number("mz", "route transform center Z is invalid")?,
                ],
                radius_control: number("mr", "route transform radius is invalid")?,
            },
            camera: RouteCameraSettings {
                zoom: number("zoom", "route camera zoom is invalid")?,
                euler_radians: [
                    number("rx", "route camera pitch is invalid")?,
                    number("ry", "route camera yaw is invalid")?,
                    number("rz", "route camera roll is invalid")?,
                ],
                position: [
                    number("px", "route camera X is invalid")?,
                    number("py", "route camera Y is invalid")?,
                    number("pz", "route camera Z is invalid")?,
                ],
                semantic_target_enabled: toggle("aim", "route camera target mode is invalid")?,
                vertical_fov_degrees: number("fov", "route camera FOV is invalid")?,
                focus_transition_seconds: number(
                    "interp",
                    "route focus transition duration is invalid",
                )? / 100.0,
            },
            space_mouse: RouteSpaceMouseSettings {
                move_sensitivity: number("smmove", "route SpaceMouse move sensitivity is invalid")?,
                rotate_sensitivity: number(
                    "smrotate",
                    "route SpaceMouse rotation sensitivity is invalid",
                )?,
                profile: space_mouse_profile,
                lock_horizon: toggle("smlock", "route SpaceMouse horizon lock is invalid")?,
                swap_yz: toggle("smswap", "route SpaceMouse axis swap is invalid")?,
                accept_background_input: toggle(
                    "smbackground",
                    "route SpaceMouse background policy is invalid",
                )?,
                hyperscope_pan_invert_mask: mask(
                    "smpinv",
                    "route Hyperscope pan inversion mask is invalid",
                )?,
                hyperscope_rotate_invert_mask: mask(
                    "smrinv",
                    "route Hyperscope rotation inversion mask is invalid",
                )?,
                blender_pan_invert_mask: mask(
                    "smbpinv",
                    "route Blender pan inversion mask is invalid",
                )?,
                blender_rotate_invert_mask: mask(
                    "smbrinv",
                    "route Blender rotation inversion mask is invalid",
                )?,
            },
            surface_walk,
        })
    }

    pub fn diagnostics(&self) -> &[RouteDiagnostic] {
        &self.diagnostics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscape::{PresentationTessellation, RenderStyle};
    use std::collections::BTreeSet;

    #[test]
    fn control_keys_are_unique_and_defaults_validate() {
        let mut keys = BTreeSet::new();
        for spec in HYPERSCOPE_CONTROL_SPECS {
            assert!(keys.insert(spec.key), "duplicate route key {}", spec.key);
            assert!(
                spec.accepts(spec.default_value),
                "invalid default for {}",
                spec.key
            );
            if let Some(domain) = spec.numeric_domain {
                assert!(domain.minimum.is_finite());
                assert!(domain.maximum.is_finite());
                assert!(domain.minimum <= domain.maximum);
                assert!(domain.step.is_finite() && domain.step > 0.0);
            }
            assert!(spec.choices.is_empty() || spec.choices.contains(&spec.default_value));
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
        let resolved = route.resolved_pairs();
        assert_eq!(resolved.len(), HYPERSCOPE_CONTROL_SPECS.len());
        assert_eq!(resolved[0], ("glb", "horse.glb"));
        assert_eq!(
            resolved.iter().copied().find(|(key, _)| *key == "mode"),
            Some(("mode", "lod")),
        );
        assert_eq!(resolved.last(), Some(&("rootgroupshadow", "0")));
        assert!(route.diagnostics().is_empty());
    }

    #[test]
    fn render_backend_route_is_explicit_and_closed() {
        let default = HyperscopeRoute::from_pairs([("gfx", "webgl2")]);
        assert!(default.canonical_pairs().is_empty());
        assert!(default.diagnostics().is_empty());

        for backend in ["webgpu-shadow", "webgpu"] {
            let route = HyperscopeRoute::from_pairs([("gfx", backend)]);
            assert_eq!(route.canonical_pairs(), vec![("gfx", backend)]);
            assert!(route.diagnostics().is_empty());
        }

        let invalid = HyperscopeRoute::from_pairs([("gfx", "auto")]);
        assert_eq!(invalid.diagnostics().len(), 1);
        assert_eq!(
            invalid.diagnostics()[0].code,
            RouteDiagnosticCode::InvalidValue
        );
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
    fn animation_clock_route_is_numeric_ordered_and_optional() {
        let route = HyperscopeRoute::from_pairs([
            ("animspeed", "-0.5"),
            ("animclockimpl", "shadow"),
            ("animtime", "1.25"),
        ]);
        assert_eq!(
            route.canonical_pairs(),
            vec![
                ("animtime", "1.25"),
                ("animspeed", "-0.5"),
                ("animclockimpl", "shadow"),
            ]
        );
        assert!(route.diagnostics().is_empty());
        assert_eq!(
            route.animation_clock().unwrap(),
            Some(RouteAnimationClock {
                time_seconds: Some(1.25),
                speed: Some(-0.5),
            })
        );

        let defaults = HyperscopeRoute::from_pairs([("animtime", "0.0"), ("animspeed", "1.00")]);
        assert!(defaults.canonical_pairs().is_empty());
        assert_eq!(
            defaults.animation_clock().unwrap(),
            Some(RouteAnimationClock {
                time_seconds: Some(0.0),
                speed: Some(1.0),
            }),
            "explicit defaults remain authored clock intent even when omitted from a compact URL",
        );
        assert_eq!(HyperscopeRoute::default().animation_clock(), Ok(None));

        let invalid = HyperscopeRoute::from_pairs([("animtime", "NaN")]);
        assert_eq!(invalid.diagnostics().len(), 1);
        assert!(invalid.animation_clock().is_err());
        assert!(HyperscopeRoute::from_pairs([("animspeed", "1000001")])
            .animation_clock()
            .is_err());
        assert_eq!(
            invalid.diagnostics()[0].code,
            RouteDiagnosticCode::InvalidValue
        );
    }

    #[test]
    fn selected_identity_route_is_atomic_and_canonical() {
        let asset = "60000000-0000-4000-8000-000000000001";
        let entity = "70000000-0000-4000-8000-000000000001";
        let selected = HyperscopeRoute::from_pairs([("selentity", entity), ("selasset", asset)]);
        assert_eq!(selected.value("selasset"), Some(asset));
        assert_eq!(selected.value("selentity"), Some(entity));
        assert_eq!(
            selected.selected_identity().unwrap(),
            Some(
                AssetEntityId::new(
                    AssetId::new(uuid::Uuid::parse_str(asset).unwrap()).unwrap(),
                    EntityId::new(uuid::Uuid::parse_str(entity).unwrap()).unwrap(),
                )
                .unwrap()
            )
        );
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
            assert_eq!(
                partial.diagnostics()[0].code,
                RouteDiagnosticCode::InvalidValue
            );
            assert_eq!(partial.diagnostics()[0].key, missing_key);
            assert!(partial.selected_identity().is_err());
        }
        assert_eq!(HyperscopeRoute::default().selected_identity(), Ok(None));
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
    fn render_control_routes_validate_the_actual_browser_contract() {
        for accepted in ["pbr", "matcap", "wire", "normals", "both", "lod", "stretch"] {
            let route = HyperscopeRoute::from_pairs([("mode", accepted)]);
            assert!(route.diagnostics().is_empty());
        }
        for rejected in ["", "matcap_wire", "normal", "PBR", "browser_magic"] {
            let route = HyperscopeRoute::from_pairs([("mode", rejected)]);
            assert_eq!(route.diagnostics().len(), 1);
        }

        for (key, accepted, rejected) in [
            ("res", ["0", "3", "6"], ["-1", "7", "3.5"]),
            ("density", ["1", "237", "500"], ["0", "501", "12.5"]),
            ("atlas", ["3", "7", "9"], ["2", "10", "7.5"]),
        ] {
            for value in accepted {
                assert!(HyperscopeRoute::from_pairs([(key, value)])
                    .diagnostics()
                    .is_empty());
            }
            for value in rejected {
                assert_eq!(
                    HyperscopeRoute::from_pairs([(key, value)])
                        .diagnostics()
                        .len(),
                    1,
                );
            }
        }

        for accepted in ["1", "16.0", "48.25", "64"] {
            assert!(HyperscopeRoute::from_pairs([("minpx", accepted)])
                .diagnostics()
                .is_empty());
        }
        for rejected in ["0", "0.999", "64.001", "65", "NaN"] {
            assert_eq!(
                HyperscopeRoute::from_pairs([("minpx", rejected)])
                    .diagnostics()
                    .len(),
                1,
            );
        }
        assert!(HyperscopeRoute::from_pairs([("minpx", "16.0")])
            .canonical_pairs()
            .is_empty());

        let route = HyperscopeRoute::from_pairs([
            ("mode", "both"),
            ("res", "4"),
            ("density", "237"),
            ("atten", "0"),
            ("minpx", "48.25"),
            ("atlas", "9"),
            ("lodratio", "4"),
            ("fuzzy", "1"),
            ("fmode", "2"),
            ("fdebug", "2"),
            ("fradius", "24"),
            ("fstr", "17.5"),
            ("ffocus", "31.25"),
            ("fbw", "8.5"),
            ("fnorm", "1"),
            ("fqual", "2"),
            ("fkaw", "4"),
            ("fkoff", "22.5"),
        ]);
        assert_eq!(
            route.render_settings().unwrap(),
            RenderSettings {
                style: RenderStyle::MatcapWire,
                resolution_level: 4,
                tessellation: PresentationTessellation {
                    density: 237.0,
                    screen_attenuation: false,
                    min_pixels_per_subdivision: 48.25,
                },
                atlas_exponent: 9,
                max_face_edge_ratio: 4,
                focus_postprocess: FocusPostprocessSettings {
                    enabled: true,
                    mode: FocusPostprocessMode::Hybrid,
                    diagnostic_view: FocusDiagnosticView::DistanceField,
                    blur_radius_pixels: 24,
                    blur_strength: 1.75,
                    focus_coordinate: 0.3125,
                    bandwidth: 0.085,
                    normalize_range: true,
                    gaussian_passes: 2,
                    kawase_passes: 4,
                    kawase_offset: 2.25,
                },
            }
        );
        assert_eq!(
            HyperscopeRoute::default().render_settings().unwrap(),
            RenderSettings::default(),
        );
        assert!(HyperscopeRoute::from_pairs([("mode", "matcap_wire")])
            .render_settings()
            .is_err());
    }

    #[test]
    fn startup_control_contracts_match_browser_ranges_and_closed_choices() {
        for (key, accepted, rejected) in [
            ("xform", "sphere_reflection", "reflection"),
            ("fmode", "3", "4"),
            ("smnav", "drone", "turntable"),
            ("lab", "cube", "sphere"),
            ("labfield", "radial", "noise"),
        ] {
            assert!(HyperscopeRoute::from_pairs([(key, accepted)])
                .diagnostics()
                .is_empty());
            assert_eq!(
                HyperscopeRoute::from_pairs([(key, rejected)])
                    .diagnostics()
                    .len(),
                1,
            );
        }

        for (key, accepted, rejected) in [
            ("mx", "-30", "-30.01"),
            ("mr", "0.11", "0.10"),
            ("animtime", "1000000000", "1000000001"),
            ("anim", "-1", "1.5"),
            ("fradius", "128", "128.5"),
            ("smmove", "10.5", "9.99"),
            ("smpinv", "7", "7.1"),
            ("fov", "110", "111"),
            ("walkscale", "-800", "-801"),
            ("labphase", "6.283", "6.284"),
            ("labgrid", "16", "17"),
            ("zoom", "1000", "1001"),
            ("rx", "1.5", "1.57"),
            ("rz", "-3.14", "-3.15"),
            ("px", "1000000", "1000001"),
        ] {
            assert!(
                HyperscopeRoute::from_pairs([(key, accepted)])
                    .diagnostics()
                    .is_empty(),
                "{key}={accepted}"
            );
            assert_eq!(
                HyperscopeRoute::from_pairs([(key, rejected)])
                    .diagnostics()
                    .len(),
                1,
                "{key}={rejected}",
            );
        }

        assert!(HyperscopeRoute::from_pairs([("atlas", "9"), ("laba", "9")])
            .diagnostics()
            .is_empty());
        let over_resident_atlas = HyperscopeRoute::from_pairs([("atlas", "7"), ("laba", "8")]);
        assert_eq!(over_resident_atlas.diagnostics().len(), 1);
        assert_eq!(over_resident_atlas.diagnostics()[0].key, "laba");
    }

    #[test]
    fn implementation_routes_admit_only_measured_authority_modes() {
        for key in [
            "pickimpl",
            "navstateimpl",
            "walkimpl",
            "navimpl",
            "selectionimpl",
            "patchlabimpl",
            "presentimpl",
            "gfxpresentimpl",
            "assetimpl",
            "sceneimpl",
            "routeimpl",
            "renderstateimpl",
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
            assert_eq!(route.canonical_pairs(), vec![("navimpl", implementation)]);
            assert!(route.diagnostics().is_empty());
        }
    }

    #[test]
    fn render_settings_cutover_uses_rust_by_default_with_measured_rollbacks() {
        let default_route = HyperscopeRoute::from_pairs([("renderstateimpl", "rust")]);
        assert!(default_route.canonical_pairs().is_empty());

        for implementation in ["js", "shadow"] {
            let route = HyperscopeRoute::from_pairs([("renderstateimpl", implementation)]);
            assert_eq!(
                route.canonical_pairs(),
                vec![("renderstateimpl", implementation)],
            );
            assert!(route.diagnostics().is_empty());
        }
    }

    #[test]
    fn navigation_route_resolves_semantic_units_without_browser_conversion() {
        let route = HyperscopeRoute::from_pairs([
            ("xform", "sphere_reflection"),
            ("mx", "-2.5"),
            ("mr", "7.25"),
            ("zoom", "12.5"),
            ("rx", "-0.25"),
            ("ry", "1.5"),
            ("rz", "0.75"),
            ("px", "3.0"),
            ("aim", "1"),
            ("fov", "108"),
            ("interp", "250"),
            ("smnav", "drone"),
            ("smbackground", "1"),
            ("smpinv", "7"),
            ("walksmooth", "45"),
            ("walkalign", "25"),
            ("walkspeed", "100"),
            ("walkscale", "-100"),
            ("walkheight", "50"),
        ]);
        assert!(route.diagnostics().is_empty());
        let settings = route.navigation_settings().unwrap();
        assert_eq!(
            settings.transform.kind,
            RouteTransformKind::SphereReflection
        );
        assert_eq!(settings.transform.kind.wire_name(), "sphere_reflection");
        assert_eq!(settings.transform.center_controls, [-2.5, 0.0, 0.0]);
        assert_eq!(settings.transform.radius_control, 7.25);
        assert_eq!(settings.camera.zoom, 12.5);
        assert_eq!(settings.camera.euler_radians, [-0.25, 1.5, 0.75]);
        assert_eq!(settings.camera.position, [3.0, 0.0, 0.0]);
        assert!(settings.camera.semantic_target_enabled);
        assert_eq!(settings.camera.vertical_fov_degrees, 108.0);
        assert_eq!(settings.camera.focus_transition_seconds, 2.5);
        assert_eq!(settings.space_mouse.profile, RouteSpaceMouseProfile::Drone);
        assert_eq!(settings.space_mouse.profile.wire_name(), "drone");
        assert!(settings.space_mouse.accept_background_input);
        assert_eq!(settings.space_mouse.hyperscope_pan_invert_mask, 7);
        assert_eq!(settings.surface_walk.smoothing_seconds, 0.45);
        assert_eq!(settings.surface_walk.tangent_pull_fraction, 0.25);
        assert_eq!(settings.surface_walk.speed_octave_steps, 100.0);
        assert_eq!(settings.surface_walk.body_scale_octave_steps, -100.0);
        assert_eq!(settings.surface_walk.eye_height_octave_steps, 50.0);
        let application = settings.application_settings().unwrap();
        assert_eq!(application.transition_seconds, 2.5);
        assert_eq!(application.surface_walk.smoothing_seconds, 0.45);
        assert_eq!(application.surface_walk.tangent_pull_fraction, 0.25);
    }

    #[test]
    fn navigation_route_rejects_invalid_semantic_controls() {
        let invalid = HyperscopeRoute::from_pairs([("walksmooth", "151")]);
        assert_eq!(invalid.diagnostics().len(), 1);
        assert!(invalid.navigation_settings().is_err());

        let mut settings = NavigationSettings::default();
        settings.transition_seconds = f64::NAN;
        assert!(settings.validate().is_err());
        let mut settings = NavigationSettings::default();
        settings.surface_walk.minimum_near = 1.0;
        assert!(settings.validate().is_err());
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
    fn rust_graphics_presentation_policy_is_default_with_explicit_rollbacks() {
        let default_route = HyperscopeRoute::from_pairs([("gfxpresentimpl", "rust")]);
        assert_eq!(default_route.value("gfxpresentimpl"), Some("rust"));
        assert!(default_route.canonical_pairs().is_empty());

        for implementation in ["js", "shadow"] {
            let route = HyperscopeRoute::from_pairs([("gfxpresentimpl", implementation)]);
            assert_eq!(
                route.canonical_pairs(),
                vec![("gfxpresentimpl", implementation)]
            );
            assert!(route.diagnostics().is_empty());
        }
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
    fn retained_gpu_pick_comparison_is_opt_in() {
        let default_route = HyperscopeRoute::from_pairs([("pickimpl", "js")]);
        assert_eq!(default_route.value("pickimpl"), Some("js"));
        assert!(default_route.canonical_pairs().is_empty());

        for implementation in ["shadow", "rust"] {
            let route = HyperscopeRoute::from_pairs([("pickimpl", implementation)]);
            assert_eq!(route.value("pickimpl"), Some(implementation));
            assert_eq!(
                route.canonical_pairs(),
                vec![("pickimpl", implementation)]
            );
            assert!(route.diagnostics().is_empty());
        }
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
