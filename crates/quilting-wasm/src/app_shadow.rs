use crate::navigation::synchronized_navigation_state;
use hyperscape::{Presentation, PresentationSnapshot};
use hyperscape_protocol::{AssetDescriptor, AssetId, RequestId};
use hyperscope_app::{
    AppCommit, AppEffect, AppEvent, AppFrameSnapshot, AppStore, AssetLoadCompletion,
    AssetLoadOutcome, AssetStatus, CommitDisposition, EffectCompletion, FrameTick,
    NavigationSynchronization, PresentationAction, SemanticAction, Timed,
};
use serde::Serialize;
use uuid::Uuid;
use wasm_bindgen::prelude::*;

/// Opt-in WASM adapter for comparing browser asset jobs with the Rust app
/// reducer. It observes fetch/file acquisition only and never loads a model,
/// mutates the renderer, or chooses a browser transport.
#[wasm_bindgen]
pub struct HyperscopeAppShadow {
    store: AppStore,
}

#[wasm_bindgen]
impl HyperscopeAppShadow {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            store: AppStore::default(),
        }
    }

    #[wasm_bindgen(js_name = requestAsset)]
    #[allow(clippy::too_many_arguments)]
    pub fn request_asset(
        &self,
        sequence: u32,
        at_seconds: f64,
        request_id: &str,
        asset_id: &str,
        uri: &str,
        media_type: &str,
    ) -> Result<JsValue, JsValue> {
        let request_id = request_id_from_str(request_id)?;
        let asset = AssetDescriptor {
            id: asset_id_from_str(asset_id)?,
            uri: uri.to_owned(),
            media_type: (!media_type.is_empty()).then(|| media_type.to_owned()),
            content_digest: None,
        };
        let commit = self
            .store
            .dispatch(AppEvent::Input(Timed {
                sequence: u64::from(sequence),
                at_seconds,
                value: SemanticAction::RequestAsset { request_id, asset },
            }))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }

    #[wasm_bindgen(js_name = cancelAsset)]
    pub fn cancel_asset(
        &self,
        sequence: u32,
        at_seconds: f64,
        asset_id: &str,
    ) -> Result<JsValue, JsValue> {
        let commit = self
            .store
            .dispatch(AppEvent::Input(Timed {
                sequence: u64::from(sequence),
                at_seconds,
                value: SemanticAction::CancelAsset(asset_id_from_str(asset_id)?),
            }))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }

    #[wasm_bindgen(js_name = completeAssetLoaded)]
    pub fn complete_asset_loaded(
        &self,
        request_id: &str,
        asset_id: &str,
        byte_length: u32,
    ) -> Result<JsValue, JsValue> {
        self.complete_asset(
            request_id,
            asset_id,
            AssetLoadOutcome::Loaded {
                byte_length: byte_length as usize,
                content_digest: None,
            },
        )
    }

    #[wasm_bindgen(js_name = completeAssetFailed)]
    pub fn complete_asset_failed(
        &self,
        request_id: &str,
        asset_id: &str,
        code: &str,
        message: &str,
        retryable: bool,
    ) -> Result<JsValue, JsValue> {
        self.complete_asset(
            request_id,
            asset_id,
            AssetLoadOutcome::Failed {
                code: code.to_owned(),
                message: message.to_owned(),
                retryable,
            },
        )
    }

    #[wasm_bindgen(js_name = advanceFrame)]
    pub fn advance_frame(
        &self,
        elapsed_seconds: f64,
        delta_seconds: f64,
    ) -> Result<JsValue, JsValue> {
        let commit = self
            .store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds,
                delta_seconds,
            }))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }

    /// Replace settled camera/focus state before a low-rate authored
    /// transition. The app reducer validates and commits the replacement.
    #[wasm_bindgen(js_name = synchronizeNavigation)]
    #[allow(clippy::too_many_arguments)]
    pub fn synchronize_navigation(
        &self,
        eye: &[f64],
        forward: &[f64],
        up: &[f64],
        control_distance: f64,
        semantic_target: &[f64],
        focus_center: &[f64],
        focus_radius: f64,
        focus_enabled: bool,
        inversion_enabled: bool,
        focus_coordinate: f64,
        angular_aperture: f64,
    ) -> Result<JsValue, JsValue> {
        let (camera, focus) = synchronized_navigation_state(
            eye,
            forward,
            up,
            control_distance,
            semantic_target,
            focus_center,
            focus_radius,
            focus_enabled,
            inversion_enabled,
            focus_coordinate,
            angular_aperture,
        )?;
        let commit = self
            .store
            .dispatch(AppEvent::NavigationSynchronized(
                NavigationSynchronization { camera, focus },
            ))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }

    /// Advance the app-owned presentation clock by the same delta as the
    /// incumbent controller and return a compact pose/focus parity snapshot.
    #[wasm_bindgen(js_name = tickPresentation)]
    pub fn tick_presentation(&self, delta_seconds: f64) -> Result<JsValue, JsValue> {
        let current = self.store.frame_snapshot();
        self.store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: current.elapsed_seconds + delta_seconds,
                delta_seconds,
            }))
            .map_err(js_error)?;
        navigation_to_js(self.store.frame_snapshot())
    }

    /// Admit a validated presentation document without activating a cue or
    /// changing navigation state.
    #[wasm_bindgen(js_name = loadPresentation)]
    pub fn load_presentation(&self, json: &str) -> Result<JsValue, JsValue> {
        let presentation = Presentation::from_json(json).map_err(js_error)?;
        let commit = self
            .store
            .dispatch(AppEvent::PresentationLoaded(presentation))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }

    /// Mirror low-rate cue intent. This shadow compares resolved desired state;
    /// the existing navigation controller remains frame/camera authority until
    /// a separate pose-parity gate is enabled.
    #[wasm_bindgen(js_name = present)]
    pub fn present(&self, sequence: u32, action: &str, cue_id: &str) -> Result<JsValue, JsValue> {
        let action = match action {
            "start" => PresentationAction::Start,
            "advance" => PresentationAction::Advance,
            "reverse" => PresentationAction::Reverse,
            "jump" => PresentationAction::JumpToCue(parse_uuid(cue_id, "cue ID")?),
            "clear" => PresentationAction::Clear,
            _ => return Err(JsValue::from_str("unknown presentation action")),
        };
        let commit = self
            .store
            .dispatch(AppEvent::Input(Timed {
                sequence: u64::from(sequence),
                at_seconds: self.store.frame_snapshot().elapsed_seconds,
                value: SemanticAction::Present(action),
            }))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }

    /// A bounded, UI-shaped projection. The app store publishes asset and
    /// diagnostic vectors before its summary revision commit fence.
    pub fn snapshot(&self) -> Result<JsValue, JsValue> {
        let summary = self.store.summary_snapshot();
        let assets = self
            .store
            .asset_snapshot()
            .into_iter()
            .map(|asset| ShadowAsset {
                id: asset.descriptor.id.to_string(),
                uri: asset.descriptor.uri,
                status: ShadowAssetStatus::from(asset.status),
            })
            .collect();
        let diagnostics = self
            .store
            .diagnostic_snapshot()
            .into_iter()
            .map(|diagnostic| ShadowDiagnostic {
                revision: diagnostic.revision.to_string(),
                code: diagnostic.code,
                message: diagnostic.message,
            })
            .collect();
        let presentation =
            self.store
                .presentation_snapshot()
                .map(|presentation| ShadowPresentation {
                    id: presentation.presentation_id.to_string(),
                    title: presentation.title,
                    cue_count: presentation.cue_count,
                    active: presentation.active,
                });
        to_js(&ShadowSnapshot {
            revision: summary.revision.to_string(),
            assets,
            loading_assets: summary.loading_assets,
            diagnostics,
            presentation,
        })
    }
}

impl Default for HyperscopeAppShadow {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperscopeAppShadow {
    fn complete_asset(
        &self,
        request_id: &str,
        asset_id: &str,
        outcome: AssetLoadOutcome,
    ) -> Result<JsValue, JsValue> {
        let commit = self
            .store
            .dispatch(AppEvent::EffectCompleted(EffectCompletion::AssetLoad(
                AssetLoadCompletion {
                    request_id: request_id_from_str(request_id)?,
                    asset_id: asset_id_from_str(asset_id)?,
                    outcome,
                },
            )))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowCommit {
    revision: String,
    disposition: &'static str,
    published_ui: bool,
    effects: Vec<ShadowEffect>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ShadowEffect {
    FetchAsset {
        request_id: String,
        asset_id: String,
        uri: String,
    },
    CancelAssetLoad {
        request_id: String,
        asset_id: String,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowSnapshot {
    revision: String,
    assets: Vec<ShadowAsset>,
    loading_assets: usize,
    diagnostics: Vec<ShadowDiagnostic>,
    presentation: Option<ShadowPresentation>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPresentation {
    id: String,
    title: String,
    cue_count: usize,
    active: Option<PresentationSnapshot>,
}

#[derive(Serialize)]
struct ShadowNavigationSnapshot {
    elapsed_seconds: f64,
    reflection: &'static str,
    camera: ShadowCameraSnapshot,
    focus: ShadowFocusSnapshot,
}

#[derive(Serialize)]
struct ShadowCameraSnapshot {
    eye: [f64; 3],
    right: [f64; 3],
    up: [f64; 3],
    forward: [f64; 3],
    control_distance: f64,
    semantic_target: Option<[f64; 3]>,
    camera_transition_remaining: Option<f64>,
}

#[derive(Serialize)]
struct ShadowFocusSnapshot {
    center: [f64; 3],
    radius: f64,
    anchored: bool,
    focus_enabled: bool,
    inversion_enabled: bool,
    focus_coordinate: f64,
    angular_aperture: f64,
    focus_transition_remaining: Option<f64>,
}

#[derive(Serialize)]
struct ShadowAsset {
    id: String,
    uri: String,
    status: ShadowAssetStatus,
}

#[derive(Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum ShadowAssetStatus {
    Loading {
        request_id: String,
    },
    Ready {
        byte_length: usize,
    },
    Failed {
        code: String,
        message: String,
        retryable: bool,
    },
    Cancelled,
}

impl From<AssetStatus> for ShadowAssetStatus {
    fn from(status: AssetStatus) -> Self {
        match status {
            AssetStatus::Loading { request_id } => Self::Loading {
                request_id: request_id.to_string(),
            },
            AssetStatus::Ready { byte_length, .. } => Self::Ready { byte_length },
            AssetStatus::Failed {
                code,
                message,
                retryable,
            } => Self::Failed {
                code,
                message,
                retryable,
            },
            AssetStatus::Cancelled => Self::Cancelled,
        }
    }
}

#[derive(Serialize)]
struct ShadowDiagnostic {
    revision: String,
    code: &'static str,
    message: String,
}

fn navigation_to_js(frame: AppFrameSnapshot) -> Result<JsValue, JsValue> {
    let basis = frame.camera.basis();
    let focus_transition_remaining = frame
        .focus
        .transition
        .map(|transition| (transition.duration_seconds - transition.elapsed_seconds).max(0.0));
    to_js(&ShadowNavigationSnapshot {
        elapsed_seconds: frame.elapsed_seconds,
        reflection: match frame.reflection {
            hyperscape::SphereReflectionState::Identity => "identity",
            hyperscape::SphereReflectionState::Sphere(_) => "sphere_reflection",
        },
        camera: ShadowCameraSnapshot {
            eye: frame.camera.eye,
            right: basis.right,
            up: basis.up,
            forward: basis.forward,
            control_distance: frame.camera.control_distance,
            semantic_target: frame.camera.semantic_target,
            camera_transition_remaining: frame.camera_transition_remaining,
        },
        focus: ShadowFocusSnapshot {
            center: frame.focus.sphere.center,
            radius: frame.focus.sphere.radius,
            anchored: frame.focus.anchor.is_some(),
            focus_enabled: frame.focus.focus_enabled,
            inversion_enabled: frame.focus.inversion_enabled,
            focus_coordinate: frame.focus.focus_coordinate,
            angular_aperture: frame.focus.angular_aperture,
            focus_transition_remaining,
        },
    })
}

fn commit_to_js(commit: &AppCommit) -> Result<JsValue, JsValue> {
    let effects = commit
        .effects
        .iter()
        .map(|effect| match effect {
            AppEffect::FetchAsset { request_id, asset } => ShadowEffect::FetchAsset {
                request_id: request_id.to_string(),
                asset_id: asset.id.to_string(),
                uri: asset.uri.clone(),
            },
            AppEffect::CancelAssetLoad {
                request_id,
                asset_id,
            } => ShadowEffect::CancelAssetLoad {
                request_id: request_id.to_string(),
                asset_id: asset_id.to_string(),
            },
        })
        .collect();
    to_js(&ShadowCommit {
        revision: commit.revision.to_string(),
        disposition: match commit.disposition {
            CommitDisposition::Applied => "applied",
            CommitDisposition::IgnoredStale => "ignored_stale",
        },
        published_ui: commit.published_ui,
        effects,
    })
}

fn request_id_from_str(value: &str) -> Result<RequestId, JsValue> {
    RequestId::new(parse_uuid(value, "request ID")?).map_err(js_error)
}

fn asset_id_from_str(value: &str) -> Result<AssetId, JsValue> {
    AssetId::new(parse_uuid(value, "asset ID")?).map_err(js_error)
}

fn parse_uuid(value: &str, context: &str) -> Result<Uuid, JsValue> {
    Uuid::parse_str(value)
        .map_err(|error| JsValue::from_str(&format!("{context} must be a UUID: {error}")))
}

fn to_js(value: &impl Serialize) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value).map_err(js_error)
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
