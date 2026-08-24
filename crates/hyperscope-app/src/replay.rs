//! Versioned, adapter-independent application replay traces.
//!
//! A trace records semantic inputs, reducer outcomes, and compact committed
//! state. It deliberately excludes DOM events, device reports, renderer
//! resources, and wall clocks so native tools, browsers, Blender adapters, and
//! future render backends can consume the same oracle.

use crate::{
    AppCommit, AppEffect, AppEvent, AppStore, AssetLoadCompletion, AssetLoadOutcome, AssetStatus,
    AuthoredRevision, CommitDisposition, EffectCompletion, FrameTick, NavigationSynchronization,
    PresentationAction, ReceivedPresence, SemanticAction, Timed,
};
use hyperscape::{
    AuthoredCamera, AuthoredFocus, FocusSphere, NavigationAction, NavigationFrame,
    NavigationPreset, PerspectiveLens, Presentation, SphereReflectionState, SurfaceAnchorTarget,
    TransitionEasing,
};
use hyperscape_protocol::{
    AssetDescriptor, AssetEntityId, AssetId, AuthoredEnvelope, EntityId, EphemeralPresence, PeerId,
    PresenceEnvelope, RequestId,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use uuid::Uuid;

pub const APP_REPLAY_VERSION: &str = "hyperscope-app-replay/0.7";
pub const LEGACY_APP_REPLAY_VERSION_0_6: &str = "hyperscope-app-replay/0.6";
pub const LEGACY_APP_REPLAY_VERSION_0_5: &str = "hyperscope-app-replay/0.5";
pub const LEGACY_APP_REPLAY_VERSION_0_4: &str = "hyperscope-app-replay/0.4";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplaySchema {
    V0_4,
    V0_5,
    V0_6,
    V0_7,
}
pub const APP_REPLAY_FINGERPRINT_ALGORITHM: &str = "fnv1a-128-json";
const FNV1A_128_OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
const FNV1A_128_PRIME: u128 = 0x0000000001000000000000000000013b;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayScript {
    pub version: String,
    pub events: Vec<AppReplayEvent>,
}

impl AppReplayScript {
    pub fn new(events: Vec<AppReplayEvent>) -> Self {
        Self {
            version: APP_REPLAY_VERSION.to_owned(),
            events,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AppReplayEvent {
    LoadPresentation {
        presentation: Presentation,
    },
    SynchronizeNavigation {
        camera: AuthoredCamera,
        focus: AuthoredFocus,
    },
    Navigate {
        sequence: u64,
        at_seconds: f64,
        action: ReplayNavigationAction,
    },
    Present {
        sequence: u64,
        at_seconds: f64,
        action: ReplayPresentationAction,
    },
    RequestAsset {
        sequence: u64,
        at_seconds: f64,
        request_id: RequestId,
        asset: AssetDescriptor,
    },
    CancelAsset {
        sequence: u64,
        at_seconds: f64,
        asset_id: AssetId,
    },
    CompleteAssetLoad {
        request_id: RequestId,
        asset_id: AssetId,
        outcome: ReplayAssetLoadOutcome,
    },
    ReceivePresence {
        envelope: PresenceEnvelope,
        received_at_seconds: f64,
    },
    ApplyAuthoredRevision {
        projection_revision: u64,
        commands: Vec<AuthoredEnvelope>,
    },
    Frame {
        elapsed_seconds: f64,
        delta_seconds: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ReplayAssetLoadOutcome {
    Loaded {
        byte_length: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content_digest: Option<[u8; 32]>,
    },
    Failed {
        code: String,
        message: String,
        retryable: bool,
    },
}

impl TryFrom<ReplayAssetLoadOutcome> for AssetLoadOutcome {
    type Error = String;

    fn try_from(outcome: ReplayAssetLoadOutcome) -> Result<Self, Self::Error> {
        match outcome {
            ReplayAssetLoadOutcome::Loaded {
                byte_length,
                content_digest,
            } => Ok(Self::Loaded {
                byte_length: usize::try_from(byte_length)
                    .map_err(|_| "asset byte length exceeds this target's address space")?,
                content_digest,
            }),
            ReplayAssetLoadOutcome::Failed {
                code,
                message,
                retryable,
            } => Ok(Self::Failed {
                code,
                message,
                retryable,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ReplayNavigationAction {
    SetPreset {
        preset: ReplayNavigationPreset,
    },
    ApplyFrame {
        frame: ReplayNavigationFrame,
    },
    SetCamera {
        camera: AuthoredCamera,
    },
    SetPerspectiveLens {
        vertical_fov_radians: f64,
        near: f64,
        far: f64,
    },
    SetSemanticTargetEnabled {
        enabled: bool,
    },
    TransitionCamera {
        target: AuthoredCamera,
        duration_seconds: f64,
        easing: TransitionEasing,
    },
    BeginSurfaceAnchorTransition {
        target: ReplaySurfaceAnchorTarget,
        scene_radius: f64,
        duration_seconds: f64,
        easing: TransitionEasing,
    },
    UpdateSurfaceAnchorTarget {
        target: ReplaySurfaceAnchorTarget,
    },
    CancelSurfaceAnchorTransition,
    AnchorFocus {
        /// Optional only for reading traces before asset-scoped selection in
        /// replay 0.7.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        asset_id: Option<Uuid>,
        #[serde(rename = "entity_id", alias = "entity")]
        entity: Uuid,
        source_bound: ReplayFocusSphere,
        /// Optional only for reading 0.4 traces. Newer traces retain the
        /// actual clicked/object pivot instead of substituting the bound center.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_pivot: Option<[f64; 3]>,
        margin: f64,
        duration_seconds: f64,
        easing: TransitionEasing,
    },
    DetachFocus,
    SetFreeFocusSphere {
        sphere: ReplayFocusSphere,
    },
    TransitionFreeFocusSphere {
        target: ReplayFocusSphere,
        duration_seconds: f64,
        easing: TransitionEasing,
    },
    TranslateFocus {
        delta: [f64; 3],
    },
    ScaleFocusLog {
        log_delta: f64,
    },
    SetFocusEnabled {
        enabled: bool,
    },
    SetFocusField {
        coordinate: f64,
        angular_aperture: f64,
    },
    SetInversionEnabled {
        enabled: bool,
    },
    ToggleInversion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayNavigationPreset {
    Hyperscope,
    Object,
    Fly,
    Drone,
}

impl From<ReplayNavigationPreset> for NavigationPreset {
    fn from(preset: ReplayNavigationPreset) -> Self {
        match preset {
            ReplayNavigationPreset::Hyperscope => Self::Hyperscope,
            ReplayNavigationPreset::Object => Self::Object,
            ReplayNavigationPreset::Fly => Self::Fly,
            ReplayNavigationPreset::Drone => Self::Drone,
        }
    }
}

impl From<NavigationPreset> for ReplayNavigationPreset {
    fn from(preset: NavigationPreset) -> Self {
        match preset {
            NavigationPreset::Hyperscope => Self::Hyperscope,
            NavigationPreset::Object => Self::Object,
            NavigationPreset::Fly => Self::Fly,
            NavigationPreset::Drone => Self::Drone,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayNavigationFrame {
    pub translation: [f64; 3],
    pub rotation: [f64; 3],
    pub dolly_log: f64,
    pub horizon_locked: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayFocusSphere {
    pub center: [f64; 3],
    pub radius: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplaySurfaceAnchorTarget {
    pub camera: AuthoredCamera,
    pub normal: [f64; 3],
}

impl TryFrom<ReplayNavigationAction> for NavigationAction {
    type Error = String;

    fn try_from(action: ReplayNavigationAction) -> Result<Self, Self::Error> {
        match action {
            ReplayNavigationAction::SetPreset { preset } => Ok(Self::SetPreset(preset.into())),
            ReplayNavigationAction::ApplyFrame { frame } => {
                finite3(frame.translation, "navigation translation")?;
                finite3(frame.rotation, "navigation rotation")?;
                finite(frame.dolly_log, "navigation dolly")?;
                Ok(Self::ApplyFrame(NavigationFrame {
                    translation: frame.translation,
                    rotation: frame.rotation,
                    dolly_log: frame.dolly_log,
                    horizon_locked: frame.horizon_locked,
                }))
            }
            ReplayNavigationAction::SetCamera { camera } => Ok(Self::SetCamera(
                camera.to_camera_rig().map_err(|error| error.to_string())?,
            )),
            ReplayNavigationAction::SetPerspectiveLens {
                vertical_fov_radians,
                near,
                far,
            } => PerspectiveLens {
                vertical_fov_radians,
                near,
                far,
            }
            .validate()
            .map(Self::SetPerspectiveLens)
            .map_err(|error| error.to_string()),
            ReplayNavigationAction::SetSemanticTargetEnabled { enabled } => {
                Ok(Self::SetSemanticTargetEnabled(enabled))
            }
            ReplayNavigationAction::TransitionCamera {
                target,
                duration_seconds,
                easing,
            } => {
                nonnegative(duration_seconds, "camera transition duration")?;
                Ok(Self::TransitionCamera {
                    target: target.to_camera_rig().map_err(|error| error.to_string())?,
                    duration_seconds,
                    easing,
                })
            }
            ReplayNavigationAction::BeginSurfaceAnchorTransition {
                target,
                scene_radius,
                duration_seconds,
                easing,
            } => {
                positive(scene_radius, "surface anchor scene radius")?;
                nonnegative(duration_seconds, "surface anchor transition duration")?;
                Ok(Self::BeginSurfaceAnchorTransition {
                    target: target.try_into()?,
                    scene_radius,
                    duration_seconds,
                    easing,
                })
            }
            ReplayNavigationAction::UpdateSurfaceAnchorTarget { target } => {
                Ok(Self::UpdateSurfaceAnchorTarget(target.try_into()?))
            }
            ReplayNavigationAction::CancelSurfaceAnchorTransition => {
                Ok(Self::CancelSurfaceAnchorTransition)
            }
            ReplayNavigationAction::AnchorFocus {
                asset_id,
                entity,
                source_bound,
                source_pivot,
                margin,
                duration_seconds,
                easing,
            } => {
                let asset_id = asset_id.ok_or_else(|| {
                    "replay 0.7+ focus anchor requires an explicit asset identity".to_owned()
                })?;
                finite(margin, "focus anchor margin")?;
                nonnegative(duration_seconds, "focus transition duration")?;
                let source_pivot = source_pivot.ok_or_else(|| {
                    "replay 0.5+ focus anchor requires an explicit source pivot".to_owned()
                })?;
                finite3(source_pivot, "focus anchor source pivot")?;
                Ok(Self::AnchorFocus {
                    identity: AssetEntityId::new(
                        AssetId::new(asset_id).map_err(|error| error.to_string())?,
                        EntityId::new(entity).map_err(|error| error.to_string())?,
                    )
                    .map_err(|error| error.to_string())?,
                    source_bound: source_bound.try_into()?,
                    source_pivot,
                    margin,
                    duration_seconds,
                    easing,
                })
            }
            ReplayNavigationAction::DetachFocus => Ok(Self::DetachFocus),
            ReplayNavigationAction::SetFreeFocusSphere { sphere } => {
                Ok(Self::SetFreeFocusSphere(sphere.try_into()?))
            }
            ReplayNavigationAction::TransitionFreeFocusSphere {
                target,
                duration_seconds,
                easing,
            } => {
                nonnegative(duration_seconds, "focus transition duration")?;
                Ok(Self::TransitionFreeFocusSphere {
                    target: target.try_into()?,
                    duration_seconds,
                    easing,
                })
            }
            ReplayNavigationAction::TranslateFocus { delta } => {
                finite3(delta, "focus translation")?;
                Ok(Self::TranslateFocus(delta))
            }
            ReplayNavigationAction::ScaleFocusLog { log_delta } => {
                finite(log_delta, "focus logarithmic scale")?;
                Ok(Self::ScaleFocusLog(log_delta))
            }
            ReplayNavigationAction::SetFocusEnabled { enabled } => {
                Ok(Self::SetFocusEnabled(enabled))
            }
            ReplayNavigationAction::SetFocusField {
                coordinate,
                angular_aperture,
            } => {
                if !coordinate.is_finite()
                    || !(0.0..=1.0).contains(&coordinate)
                    || !angular_aperture.is_finite()
                    || angular_aperture <= 0.0
                {
                    return Err(
                        "focus coordinate must be in [0,1] and aperture must be positive".into(),
                    );
                }
                Ok(Self::SetFocusField {
                    coordinate,
                    angular_aperture,
                })
            }
            ReplayNavigationAction::SetInversionEnabled { enabled } => {
                Ok(Self::SetInversionEnabled(enabled))
            }
            ReplayNavigationAction::ToggleInversion => Ok(Self::ToggleInversion),
        }
    }
}

impl TryFrom<ReplayFocusSphere> for FocusSphere {
    type Error = String;

    fn try_from(sphere: ReplayFocusSphere) -> Result<Self, Self::Error> {
        Self::new(sphere.center, sphere.radius).map_err(str::to_owned)
    }
}

impl From<FocusSphere> for ReplayFocusSphere {
    fn from(sphere: FocusSphere) -> Self {
        Self {
            center: sphere.center,
            radius: sphere.radius,
        }
    }
}

impl TryFrom<ReplaySurfaceAnchorTarget> for SurfaceAnchorTarget {
    type Error = String;

    fn try_from(target: ReplaySurfaceAnchorTarget) -> Result<Self, Self::Error> {
        Self::new(
            target
                .camera
                .to_camera_rig()
                .map_err(|error| error.to_string())?,
            target.normal,
        )
        .map_err(|error| error.to_string())
    }
}

fn finite(value: f64, label: &str) -> Result<(), String> {
    value
        .is_finite()
        .then_some(())
        .ok_or_else(|| format!("{label} must be finite"))
}

fn finite3(value: [f64; 3], label: &str) -> Result<(), String> {
    value
        .into_iter()
        .all(f64::is_finite)
        .then_some(())
        .ok_or_else(|| format!("{label} must be finite"))
}

fn nonnegative(value: f64, label: &str) -> Result<(), String> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(format!("{label} must be finite and nonnegative"))
    }
}

fn positive(value: f64, label: &str) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(format!("{label} must be finite and positive"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ReplayPresentationAction {
    Start,
    Advance,
    Reverse,
    JumpToCue { cue: Uuid },
    Clear,
}

impl From<ReplayPresentationAction> for PresentationAction {
    fn from(action: ReplayPresentationAction) -> Self {
        match action {
            ReplayPresentationAction::Start => Self::Start,
            ReplayPresentationAction::Advance => Self::Advance,
            ReplayPresentationAction::Reverse => Self::Reverse,
            ReplayPresentationAction::JumpToCue { cue } => Self::JumpToCue(cue),
            ReplayPresentationAction::Clear => Self::Clear,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayTrace {
    pub version: String,
    pub records: Vec<AppReplayRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayRecord {
    pub ordinal: usize,
    pub event: AppReplayEvent,
    pub outcome: AppReplayOutcome,
    pub state: AppReplayState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum AppReplayOutcome {
    Committed {
        revision: u64,
        disposition: ReplayCommitDisposition,
        published_ui: bool,
        effects: Vec<AppReplayEffect>,
    },
    Rejected {
        error: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayCommitDisposition {
    Applied,
    IgnoredStale,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AppReplayEffect {
    FetchAsset {
        request_id: RequestId,
        asset: AssetDescriptor,
    },
    CancelAssetLoad {
        request_id: RequestId,
        asset_id: AssetId,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayState {
    pub revision: u64,
    pub elapsed_seconds: f64,
    pub authored_projection_revision: Option<u64>,
    pub active_cue: Option<Uuid>,
    pub active_scene: Option<Uuid>,
    pub active_view: Option<Uuid>,
    pub assets: Vec<AppReplayAssetState>,
    pub presence: Vec<AppReplayPresenceState>,
    pub diagnostics: Vec<AppReplayDiagnosticState>,
    pub reflection: ReplayReflection,
    pub navigation: AppReplayNavigationState,
    pub camera: AppReplayCameraState,
    pub focus: AppReplayFocusState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayAssetState {
    pub descriptor: AssetDescriptor,
    pub status: ReplayAssetStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ReplayAssetStatus {
    Loading {
        request_id: RequestId,
    },
    Ready {
        byte_length: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content_digest: Option<[u8; 32]>,
    },
    Failed {
        code: String,
        message: String,
        retryable: bool,
    },
    Cancelled,
}

impl From<AssetStatus> for ReplayAssetStatus {
    fn from(status: AssetStatus) -> Self {
        match status {
            AssetStatus::Loading { request_id } => Self::Loading { request_id },
            AssetStatus::Ready {
                byte_length,
                content_digest,
            } => Self::Ready {
                byte_length: u64::try_from(byte_length)
                    .expect("supported Rust targets have at most 64-bit usize"),
                content_digest,
            },
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayPresenceState {
    pub peer: PeerId,
    pub sequence: u64,
    pub expires_at_seconds: f64,
    pub presence: EphemeralPresence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayDiagnosticState {
    pub revision: u64,
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayNavigationState {
    pub preset: ReplayNavigationPreset,
    pub pending_actions: u64,
    pub last_applied_sequence: Option<u64>,
    pub surface_anchor_transition_remaining_seconds: Option<f64>,
    pub surface_anchor_hop_height: Option<f64>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayReflection {
    Identity,
    SphereReflection,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayCameraState {
    pub eye: [f64; 3],
    pub orientation_wxyz: [f64; 4],
    pub control_distance: f64,
    pub semantic_target: Option<[f64; 3]>,
    pub vertical_fov_radians: f64,
    pub near: f64,
    pub far: f64,
    pub transition_remaining_seconds: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayFocusState {
    pub center: [f64; 3],
    pub radius: f64,
    pub selected: Option<AppReplaySelectedFocusState>,
    pub focus_enabled: bool,
    pub inversion_enabled: bool,
    pub coordinate: f64,
    pub angular_aperture: f64,
    pub transition_remaining_seconds: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplaySelectedFocusState {
    pub asset_id: Uuid,
    pub entity_id: Uuid,
    pub source_bound: ReplayFocusSphere,
    pub source_pivot: [f64; 3],
    pub margin: f64,
    pub output_pivot: Option<[f64; 3]>,
    pub output_radius: Option<f64>,
}

pub fn run_app_replay(script: &AppReplayScript) -> Result<AppReplayTrace, AppReplayError> {
    let schema = match script.version.as_str() {
        LEGACY_APP_REPLAY_VERSION_0_4 => ReplaySchema::V0_4,
        LEGACY_APP_REPLAY_VERSION_0_5 => ReplaySchema::V0_5,
        LEGACY_APP_REPLAY_VERSION_0_6 => ReplaySchema::V0_6,
        APP_REPLAY_VERSION => ReplaySchema::V0_7,
        _ => return Err(AppReplayError::UnsupportedVersion(script.version.clone())),
    };
    let store = AppStore::default();
    let mut records = Vec::with_capacity(script.events.len());
    for (ordinal, event) in script.events.iter().cloned().enumerate() {
        let outcome = replay_event(&store, &event, schema);
        records.push(AppReplayRecord {
            ordinal,
            event,
            outcome,
            state: replay_state(&store),
        });
    }
    Ok(AppReplayTrace {
        version: APP_REPLAY_VERSION.to_owned(),
        records,
    })
}

/// Builds a deterministic successful walkthrough of every cue in a
/// presentation. The initial navigation synchronization makes the starting
/// state explicit instead of depending on a platform adapter's defaults.
pub fn presentation_walkthrough_replay(presentation: Presentation) -> AppReplayScript {
    let cues = presentation
        .cues
        .iter()
        .map(|cue| (cue.id, cue.transition.duration_seconds))
        .collect::<Vec<_>>();
    let mut events = vec![
        AppReplayEvent::LoadPresentation { presentation },
        AppReplayEvent::SynchronizeNavigation {
            camera: AuthoredCamera::default(),
            focus: AuthoredFocus::default(),
        },
    ];
    let Some((_, first_duration)) = cues.first().copied() else {
        return AppReplayScript::new(events);
    };

    let mut elapsed_seconds = 0.0;
    events.push(AppReplayEvent::Present {
        sequence: 1,
        at_seconds: elapsed_seconds,
        action: ReplayPresentationAction::Start,
    });
    elapsed_seconds += first_duration;
    events.push(AppReplayEvent::Frame {
        elapsed_seconds,
        delta_seconds: first_duration,
    });

    for (sequence, (cue, duration)) in (2_u64..).zip(cues.into_iter().skip(1)) {
        events.push(AppReplayEvent::Present {
            sequence,
            at_seconds: elapsed_seconds,
            action: ReplayPresentationAction::JumpToCue { cue },
        });
        elapsed_seconds += duration;
        events.push(AppReplayEvent::Frame {
            elapsed_seconds,
            delta_seconds: duration,
        });
    }
    AppReplayScript::new(events)
}

pub fn app_replay_fingerprint(trace: &AppReplayTrace) -> Result<String, serde_json::Error> {
    let encoded = serde_json::to_vec(trace)?;
    let mut fingerprint = FNV1A_128_OFFSET;
    for byte in encoded {
        fingerprint ^= u128::from(byte);
        fingerprint = fingerprint.wrapping_mul(FNV1A_128_PRIME);
    }
    Ok(format!("{fingerprint:032x}"))
}

fn replay_event(store: &AppStore, event: &AppReplayEvent, schema: ReplaySchema) -> AppReplayOutcome {
    let result: Result<AppCommit, String> = replay_app_event(event, schema)
        .and_then(|event| store.dispatch(event).map_err(|error| error.to_string()));
    match result {
        Ok(commit) => AppReplayOutcome::Committed {
            revision: commit.revision,
            disposition: match commit.disposition {
                CommitDisposition::Applied => ReplayCommitDisposition::Applied,
                CommitDisposition::IgnoredStale => ReplayCommitDisposition::IgnoredStale,
            },
            published_ui: commit.published_ui,
            effects: commit.effects.iter().map(replay_effect).collect(),
        },
        Err(error) => AppReplayOutcome::Rejected { error },
    }
}

fn replay_app_event(event: &AppReplayEvent, schema: ReplaySchema) -> Result<AppEvent, String> {
    match event {
        AppReplayEvent::LoadPresentation { presentation } => {
            Ok(AppEvent::PresentationLoaded(presentation.clone()))
        }
        AppReplayEvent::SynchronizeNavigation { camera, focus } => camera
            .to_camera_rig()
            .and_then(|camera| {
                focus
                    .to_focus_navigation()
                    .map(|focus| NavigationSynchronization { camera, focus })
            })
            .map_err(|error| error.to_string())
            .map(AppEvent::NavigationSynchronized),
        AppReplayEvent::Navigate {
            sequence,
            at_seconds,
            action,
        } => navigation_action_for_replay_version(*action, schema).map(|action| {
            AppEvent::Input(Timed {
                sequence: *sequence,
                at_seconds: *at_seconds,
                value: SemanticAction::Navigate(action),
            })
        }),
        AppReplayEvent::Present {
            sequence,
            at_seconds,
            action,
        } => Ok(AppEvent::Input(Timed {
            sequence: *sequence,
            at_seconds: *at_seconds,
            value: SemanticAction::Present((*action).into()),
        })),
        AppReplayEvent::RequestAsset {
            sequence,
            at_seconds,
            request_id,
            asset,
        } => Ok(AppEvent::Input(Timed {
            sequence: *sequence,
            at_seconds: *at_seconds,
            value: SemanticAction::RequestAsset {
                request_id: *request_id,
                asset: asset.clone(),
            },
        })),
        AppReplayEvent::CancelAsset {
            sequence,
            at_seconds,
            asset_id,
        } => Ok(AppEvent::Input(Timed {
            sequence: *sequence,
            at_seconds: *at_seconds,
            value: SemanticAction::CancelAsset(*asset_id),
        })),
        AppReplayEvent::CompleteAssetLoad {
            request_id,
            asset_id,
            outcome,
        } => Ok(AppEvent::EffectCompleted(EffectCompletion::AssetLoad(
            AssetLoadCompletion {
                request_id: *request_id,
                asset_id: *asset_id,
                outcome: outcome.clone().try_into()?,
            },
        ))),
        AppReplayEvent::ReceivePresence {
            envelope,
            received_at_seconds,
        } => Ok(AppEvent::RemotePresence(ReceivedPresence {
            envelope: envelope.clone(),
            received_at_seconds: *received_at_seconds,
        })),
        AppReplayEvent::ApplyAuthoredRevision {
            projection_revision,
            commands,
        } => Ok(AppEvent::AuthoredRevision(AuthoredRevision {
            projection_revision: *projection_revision,
            commands: commands.clone(),
        })),
        AppReplayEvent::Frame {
            elapsed_seconds,
            delta_seconds,
        } => Ok(AppEvent::Frame(FrameTick {
            elapsed_seconds: *elapsed_seconds,
            delta_seconds: *delta_seconds,
        })),
    }
}

fn navigation_action_for_replay_version(
    mut action: ReplayNavigationAction,
    schema: ReplaySchema,
) -> Result<NavigationAction, String> {
    if matches!(schema, ReplaySchema::V0_4 | ReplaySchema::V0_5)
        && matches!(
            action,
            ReplayNavigationAction::SetPerspectiveLens { .. }
                | ReplayNavigationAction::SetSemanticTargetEnabled { .. }
        )
    {
        return Err("camera lens/target-mode actions require replay 0.6".to_owned());
    }
    if schema == ReplaySchema::V0_4 {
        if let ReplayNavigationAction::AnchorFocus {
            source_bound,
            source_pivot,
            ..
        } = &mut action
        {
            if source_pivot.is_none() {
                *source_pivot = Some(source_bound.center);
            }
        }
    }
    if schema != ReplaySchema::V0_7
        && matches!(
            action,
            ReplayNavigationAction::AnchorFocus { asset_id: None, .. }
        )
    {
        return Err(
            "legacy focus anchor has an unscoped entity identity; add asset_id before replaying"
                .to_owned(),
        );
    }
    NavigationAction::try_from(action)
}

fn replay_effect(effect: &AppEffect) -> AppReplayEffect {
    match effect {
        AppEffect::FetchAsset { request_id, asset } => AppReplayEffect::FetchAsset {
            request_id: *request_id,
            asset: asset.clone(),
        },
        AppEffect::CancelAssetLoad {
            request_id,
            asset_id,
        } => AppReplayEffect::CancelAssetLoad {
            request_id: *request_id,
            asset_id: *asset_id,
        },
    }
}

fn replay_state(store: &AppStore) -> AppReplayState {
    let state = store.lock_state();
    let frame = state.frame_snapshot();
    let active = state.active_presentation.as_ref();
    let navigation = &state.navigation;
    let focus_transition_remaining = frame
        .focus
        .transition
        .map(|transition| (transition.duration_seconds - transition.elapsed_seconds).max(0.0));
    AppReplayState {
        revision: frame.revision,
        elapsed_seconds: frame.elapsed_seconds,
        authored_projection_revision: state.authored_projection_revision,
        active_cue: active.map(|snapshot| snapshot.cue_id),
        active_scene: active.map(|snapshot| snapshot.scene_id),
        active_view: active.map(|snapshot| snapshot.view_id),
        assets: state
            .asset_read_models()
            .into_iter()
            .map(|asset| AppReplayAssetState {
                descriptor: asset.descriptor,
                status: asset.status.into(),
            })
            .collect(),
        presence: state
            .presence_read_models()
            .into_iter()
            .map(|presence| AppReplayPresenceState {
                peer: presence.peer,
                sequence: presence.sequence,
                expires_at_seconds: presence.expires_at_seconds,
                presence: presence.presence,
            })
            .collect(),
        diagnostics: state
            .diagnostic_read_models()
            .into_iter()
            .map(|diagnostic| AppReplayDiagnosticState {
                revision: diagnostic.revision,
                code: diagnostic.code.to_owned(),
                message: diagnostic.message,
            })
            .collect(),
        reflection: match frame.reflection {
            SphereReflectionState::Identity => ReplayReflection::Identity,
            SphereReflectionState::Sphere(_) => ReplayReflection::SphereReflection,
        },
        navigation: AppReplayNavigationState {
            preset: navigation.runtime.preset.into(),
            pending_actions: u64::try_from(navigation.queue.len())
                .expect("supported Rust targets have at most 64-bit usize"),
            last_applied_sequence: navigation.runtime.last_applied_sequence,
            surface_anchor_transition_remaining_seconds: navigation
                .surface_walk
                .anchor_transition()
                .map(|transition| {
                    (transition.duration_seconds - transition.elapsed_seconds).max(0.0)
                }),
            surface_anchor_hop_height: navigation
                .surface_walk
                .anchor_transition()
                .map(|transition| transition.hop_height),
            diagnostics: navigation.diagnostics.0.clone(),
        },
        camera: AppReplayCameraState {
            eye: frame.camera.eye,
            orientation_wxyz: [
                frame.camera.orientation.w,
                frame.camera.orientation.x,
                frame.camera.orientation.y,
                frame.camera.orientation.z,
            ],
            control_distance: frame.camera.control_distance,
            semantic_target: frame.camera.semantic_target,
            vertical_fov_radians: frame.camera.lens.vertical_fov_radians,
            near: frame.camera.lens.near,
            far: frame.camera.lens.far,
            transition_remaining_seconds: frame.camera_transition_remaining,
        },
        focus: AppReplayFocusState {
            center: frame.focus.sphere.center,
            radius: frame.focus.sphere.radius,
            selected: frame
                .selected_focus
                .map(|selected| AppReplaySelectedFocusState {
                    asset_id: selected.identity.asset.as_uuid(),
                    entity_id: selected.identity.entity.as_uuid(),
                    source_bound: selected.source_bound.into(),
                    source_pivot: selected.source_pivot,
                    margin: selected.margin,
                    output_pivot: selected.output_pivot,
                    output_radius: selected.output_radius,
                }),
            focus_enabled: frame.focus.focus_enabled,
            inversion_enabled: frame.focus.inversion_enabled,
            coordinate: frame.focus.focus_coordinate,
            angular_aperture: frame.focus.angular_aperture,
            transition_remaining_seconds: focus_transition_remaining,
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppReplayError {
    UnsupportedVersion(String),
}

impl fmt::Display for AppReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported app replay version {version:?}")
            }
        }
    }
}

impl Error for AppReplayError {}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscape_protocol::AuthoredCommand;

    const FIXTURE: &str = include_str!("../../../examples/hacker-night.presentation.json");
    const GOLDEN: &str = include_str!("../../../examples/hacker-night.replay.fingerprint");
    const NAVIGATION_FIXTURE: &str = include_str!("../../../examples/navigation.app-replay.json");
    const NAVIGATION_GOLDEN: &str = include_str!("../../../examples/navigation.replay.fingerprint");
    const ORCHESTRATION_FIXTURE: &str =
        include_str!("../../../examples/orchestration.app-replay.json");
    const ORCHESTRATION_GOLDEN: &str =
        include_str!("../../../examples/orchestration.replay.fingerprint");
    const UNKNOWN_CUE: &str = "f0000000-0000-4000-8000-000000000099";

    fn fixture() -> Presentation {
        Presentation::from_json(FIXTURE).unwrap()
    }

    fn assert_semantic_state_eq(left: &AppReplayState, right: &AppReplayState) {
        assert_eq!(left.active_cue, right.active_cue);
        assert_eq!(left.active_scene, right.active_scene);
        assert_eq!(left.active_view, right.active_view);
        assert_eq!(
            left.authored_projection_revision,
            right.authored_projection_revision
        );
        assert_eq!(left.assets, right.assets);
        assert_eq!(left.presence, right.presence);
        assert_eq!(left.diagnostics, right.diagnostics);
        assert_eq!(left.reflection, right.reflection);
        assert_eq!(left.navigation, right.navigation);
        assert_eq!(left.camera, right.camera);
        assert_eq!(left.focus, right.focus);
    }

    fn committed_effects(record: &AppReplayRecord) -> &[AppReplayEffect] {
        match &record.outcome {
            AppReplayOutcome::Committed { effects, .. } => effects,
            AppReplayOutcome::Rejected { error } => {
                panic!("expected committed replay record, rejected with {error}")
            }
        }
    }

    fn navigation_action_name(action: &ReplayNavigationAction) -> &'static str {
        match action {
            ReplayNavigationAction::SetPreset { .. } => "set_preset",
            ReplayNavigationAction::ApplyFrame { .. } => "apply_frame",
            ReplayNavigationAction::SetCamera { .. } => "set_camera",
            ReplayNavigationAction::SetPerspectiveLens { .. } => "set_perspective_lens",
            ReplayNavigationAction::SetSemanticTargetEnabled { .. } => {
                "set_semantic_target_enabled"
            }
            ReplayNavigationAction::TransitionCamera { .. } => "transition_camera",
            ReplayNavigationAction::BeginSurfaceAnchorTransition { .. } => {
                "begin_surface_anchor_transition"
            }
            ReplayNavigationAction::UpdateSurfaceAnchorTarget { .. } => {
                "update_surface_anchor_target"
            }
            ReplayNavigationAction::CancelSurfaceAnchorTransition => {
                "cancel_surface_anchor_transition"
            }
            ReplayNavigationAction::AnchorFocus { .. } => "anchor_focus",
            ReplayNavigationAction::DetachFocus => "detach_focus",
            ReplayNavigationAction::SetFreeFocusSphere { .. } => "set_free_focus_sphere",
            ReplayNavigationAction::TransitionFreeFocusSphere { .. } => {
                "transition_free_focus_sphere"
            }
            ReplayNavigationAction::TranslateFocus { .. } => "translate_focus",
            ReplayNavigationAction::ScaleFocusLog { .. } => "scale_focus_log",
            ReplayNavigationAction::SetFocusEnabled { .. } => "set_focus_enabled",
            ReplayNavigationAction::SetFocusField { .. } => "set_focus_field",
            ReplayNavigationAction::SetInversionEnabled { .. } => "set_inversion_enabled",
            ReplayNavigationAction::ToggleInversion => "toggle_inversion",
        }
    }

    fn authoritative_navigation_action_name(action: &NavigationAction) -> &'static str {
        match action {
            NavigationAction::SetPreset(_) => "set_preset",
            NavigationAction::ApplyFrame(_) => "apply_frame",
            NavigationAction::SetCamera(_) => "set_camera",
            NavigationAction::SetPerspectiveLens(_) => "set_perspective_lens",
            NavigationAction::SetSemanticTargetEnabled(_) => "set_semantic_target_enabled",
            NavigationAction::TransitionCamera { .. } => "transition_camera",
            NavigationAction::BeginSurfaceAnchorTransition { .. } => {
                "begin_surface_anchor_transition"
            }
            NavigationAction::UpdateSurfaceAnchorTarget(_) => "update_surface_anchor_target",
            NavigationAction::CancelSurfaceAnchorTransition => "cancel_surface_anchor_transition",
            NavigationAction::AnchorFocus { .. } => "anchor_focus",
            NavigationAction::DetachFocus => "detach_focus",
            NavigationAction::SetFreeFocusSphere(_) => "set_free_focus_sphere",
            NavigationAction::TransitionFreeFocusSphere { .. } => "transition_free_focus_sphere",
            NavigationAction::TranslateFocus(_) => "translate_focus",
            NavigationAction::ScaleFocusLog(_) => "scale_focus_log",
            NavigationAction::SetFocusEnabled(_) => "set_focus_enabled",
            NavigationAction::SetFocusField { .. } => "set_focus_field",
            NavigationAction::SetInversionEnabled(_) => "set_inversion_enabled",
            NavigationAction::ToggleInversion => "toggle_inversion",
        }
    }

    fn authoritative_app_event_name(event: &AppEvent) -> &'static str {
        match event {
            AppEvent::Input(timed) => match &timed.value {
                SemanticAction::Navigate(_) => "navigate",
                SemanticAction::Present(_) => "present",
                SemanticAction::RequestAsset { .. } => "request_asset",
                SemanticAction::CancelAsset(_) => "cancel_asset",
            },
            AppEvent::PresentationLoaded(_) => "presentation_loaded",
            AppEvent::NavigationSynchronized(_) => "navigation_synchronized",
            AppEvent::Frame(_) => "frame",
            AppEvent::EffectCompleted(completion) => match completion {
                EffectCompletion::AssetLoad(_) => "effect_completed_asset_load",
            },
            AppEvent::RemotePresence(_) => "remote_presence",
            AppEvent::AuthoredRevision(_) => "authored_revision",
        }
    }

    fn authoritative_authored_command_name(command: &AuthoredCommand) -> &'static str {
        match command {
            AuthoredCommand::UpsertAsset { .. } => "upsert_asset",
            AuthoredCommand::SetEntityTransform { .. } => "set_entity_transform",
            AuthoredCommand::RemoveEntity { .. } => "remove_entity",
        }
    }

    #[test]
    fn fixtures_cover_the_authoritative_application_event_surface() {
        let navigation: AppReplayScript = serde_json::from_str(NAVIGATION_FIXTURE).unwrap();
        let orchestration: AppReplayScript = serde_json::from_str(ORCHESTRATION_FIXTURE).unwrap();
        let presentation = presentation_walkthrough_replay(fixture());
        let covered = presentation
            .events
            .iter()
            .chain(navigation.events.iter())
            .chain(orchestration.events.iter())
            .filter_map(|event| replay_app_event(event, ReplaySchema::V0_6).ok())
            .map(|event| authoritative_app_event_name(&event))
            .collect::<std::collections::BTreeSet<_>>();
        let authored_covered = orchestration
            .events
            .iter()
            .filter_map(|event| match event {
                AppReplayEvent::ApplyAuthoredRevision { commands, .. } => Some(commands),
                _ => None,
            })
            .flatten()
            .map(|envelope| authoritative_authored_command_name(&envelope.command))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            covered,
            std::collections::BTreeSet::from([
                "authored_revision",
                "cancel_asset",
                "effect_completed_asset_load",
                "frame",
                "navigate",
                "navigation_synchronized",
                "presentation_loaded",
                "present",
                "remote_presence",
                "request_asset",
            ])
        );
        assert_eq!(
            authored_covered,
            std::collections::BTreeSet::from([
                "remove_entity",
                "set_entity_transform",
                "upsert_asset",
            ])
        );
    }

    #[test]
    fn walkthrough_trace_is_deterministic_and_json_roundtrips() {
        let script = presentation_walkthrough_replay(fixture());
        let encoded_script = serde_json::to_string(&script).unwrap();
        let decoded_script: AppReplayScript = serde_json::from_str(&encoded_script).unwrap();
        assert_eq!(decoded_script, script);

        let first = run_app_replay(&script).unwrap();
        let second = run_app_replay(&decoded_script).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.records.len(), 14);
        assert!(first.records.iter().all(|record| matches!(
            record.outcome,
            AppReplayOutcome::Committed {
                disposition: ReplayCommitDisposition::Applied,
                ..
            }
        )));
        assert_eq!(
            first.records.last().unwrap().state.active_cue,
            fixture().cues.last().map(|cue| cue.id)
        );

        let encoded_trace = serde_json::to_string(&first).unwrap();
        let decoded_trace: AppReplayTrace = serde_json::from_str(&encoded_trace).unwrap();
        assert_eq!(decoded_trace, first);
        assert_eq!(
            app_replay_fingerprint(&first).unwrap(),
            app_replay_fingerprint(&second).unwrap()
        );
        assert_eq!(
            format!(
                "{APP_REPLAY_FINGERPRINT_ALGORITHM}:{}",
                app_replay_fingerprint(&first).unwrap()
            ),
            GOLDEN.trim()
        );
    }

    #[test]
    fn rejected_event_is_recorded_without_mutating_committed_state() {
        let presentation = fixture();
        let mut script = AppReplayScript::new(vec![
            AppReplayEvent::LoadPresentation { presentation },
            AppReplayEvent::Present {
                sequence: 1,
                at_seconds: 0.0,
                action: ReplayPresentationAction::Start,
            },
        ]);
        script.events.push(AppReplayEvent::Present {
            sequence: 2,
            at_seconds: 0.0,
            action: ReplayPresentationAction::JumpToCue {
                cue: Uuid::parse_str(UNKNOWN_CUE).unwrap(),
            },
        });

        let trace = run_app_replay(&script).unwrap();
        assert!(matches!(
            trace.records.last().unwrap().outcome,
            AppReplayOutcome::Rejected { .. }
        ));
        let before = &trace.records[trace.records.len() - 2].state;
        let after = &trace.records.last().unwrap().state;
        assert_eq!(after.revision, before.revision);
        assert_semantic_state_eq(after, before);
    }

    #[test]
    fn navigation_fixture_covers_the_semantic_vocabulary_and_matches_its_golden() {
        let script: AppReplayScript = serde_json::from_str(NAVIGATION_FIXTURE).unwrap();
        let encoded = serde_json::to_string(&script).unwrap();
        assert_eq!(
            serde_json::from_str::<AppReplayScript>(&encoded).unwrap(),
            script
        );

        let covered = script
            .events
            .iter()
            .filter_map(|event| match event {
                AppReplayEvent::Navigate { action, .. } => Some(navigation_action_name(action)),
                _ => None,
            })
            .collect::<std::collections::BTreeSet<_>>();
        let authoritative_covered = script
            .events
            .iter()
            .filter_map(|event| match event {
                AppReplayEvent::Navigate { action, .. } => NavigationAction::try_from(*action)
                    .ok()
                    .map(|action| authoritative_navigation_action_name(&action)),
                _ => None,
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            covered,
            std::collections::BTreeSet::from([
                "anchor_focus",
                "apply_frame",
                "begin_surface_anchor_transition",
                "cancel_surface_anchor_transition",
                "detach_focus",
                "scale_focus_log",
                "set_camera",
                "set_perspective_lens",
                "set_semantic_target_enabled",
                "set_focus_enabled",
                "set_focus_field",
                "set_free_focus_sphere",
                "set_inversion_enabled",
                "set_preset",
                "toggle_inversion",
                "transition_camera",
                "transition_free_focus_sphere",
                "translate_focus",
                "update_surface_anchor_target",
            ])
        );
        assert_eq!(authoritative_covered, covered);

        let trace = run_app_replay(&script).unwrap();
        assert_eq!(trace.records.len(), 31);
        assert_eq!(trace.records[1].state.camera.eye, [0.0, 0.0, 4.0]);
        assert_eq!(trace.records[1].state.navigation.pending_actions, 1);
        assert_eq!(
            trace.records[1].state.navigation.last_applied_sequence,
            None
        );
        assert_eq!(
            trace.records[8].state.navigation.last_applied_sequence,
            Some(6)
        );
        assert_eq!(
            trace.records[15]
                .state
                .navigation
                .surface_anchor_transition_remaining_seconds,
            Some(1.0)
        );
        assert_eq!(
            trace.records[18]
                .state
                .navigation
                .surface_anchor_transition_remaining_seconds,
            None
        );
        let selected = trace.records[18]
            .state
            .focus
            .selected
            .as_ref()
            .expect("anchor-focus frame exposes selected source and output state");
        assert_eq!(
            selected.asset_id,
            Uuid::parse_str("60000000-0000-4000-8000-000000000001").unwrap()
        );
        assert_eq!(
            selected.entity_id,
            Uuid::parse_str("70000000-0000-4000-8000-000000000001").unwrap()
        );
        assert_eq!(selected.source_bound.center, [1.5, 0.0, 0.0]);
        assert_eq!(selected.source_pivot, [1.7, 0.1, 0.0]);
        assert!(selected.output_pivot.is_some());
        assert!(selected.output_radius.is_some_and(|radius| radius > 0.0));
        assert!(matches!(
            trace.records.last().unwrap().outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("entity ID must not be nil")
        ));
        let before = &trace.records[trace.records.len() - 2].state;
        let after = &trace.records.last().unwrap().state;
        assert_eq!(after.revision, before.revision);
        assert_semantic_state_eq(after, before);
        assert_eq!(after.navigation.preset, ReplayNavigationPreset::Fly);
        assert_eq!(after.navigation.pending_actions, 0);
        assert_eq!(after.navigation.last_applied_sequence, Some(18));
        assert!(after.navigation.diagnostics.is_empty());
        assert_eq!(after.focus.selected, None);
        assert_eq!(after.reflection, ReplayReflection::Identity);
        assert_eq!(after.camera.vertical_fov_radians, 1.25);
        assert_eq!(after.camera.near, 0.002);
        assert_eq!(after.camera.far, 25_000.0);
        assert!(after.camera.semantic_target.is_some());
        assert_eq!(
            format!(
                "{APP_REPLAY_FINGERPRINT_ALGORITHM}:{}",
                app_replay_fingerprint(&trace).unwrap()
            ),
            NAVIGATION_GOLDEN.trim()
        );
        let trace_json = serde_json::to_string(&trace).unwrap();
        assert_eq!(
            serde_json::from_str::<AppReplayTrace>(&trace_json).unwrap(),
            trace
        );
    }

    #[test]
    fn legacy_unscoped_anchor_focus_is_not_given_a_false_durable_scope() {
        let action: ReplayNavigationAction = serde_json::from_str(
            r#"{
                "action":"anchor_focus",
                "entity":"70000000-0000-4000-8000-000000000001",
                "source_bound":{"center":[1.0,2.0,3.0],"radius":0.5},
                "margin":1.1,
                "duration_seconds":0.7,
                "easing":"smoother_step"
            }"#,
        )
        .unwrap();
        let legacy = AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_4.to_owned(),
            events: vec![
                AppReplayEvent::Navigate {
                    sequence: 0,
                    at_seconds: 0.0,
                    action,
                },
                AppReplayEvent::Frame {
                    elapsed_seconds: 0.0,
                    delta_seconds: 0.0,
                },
            ],
        };
        let trace = run_app_replay(&legacy).unwrap();
        assert_eq!(trace.version, APP_REPLAY_VERSION);
        assert!(matches!(
            trace.records[0].outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("unscoped entity identity")
        ));
        assert_eq!(trace.records[1].state.focus.selected, None);
    }

    #[test]
    fn current_anchor_focus_rejects_an_omitted_pivot() {
        let action: ReplayNavigationAction = serde_json::from_str(
            r#"{
                "action":"anchor_focus",
                "asset_id":"60000000-0000-4000-8000-000000000001",
                "entity":"70000000-0000-4000-8000-000000000001",
                "source_bound":{"center":[1.0,2.0,3.0],"radius":0.5},
                "margin":1.1,
                "duration_seconds":0.7,
                "easing":"smoother_step"
            }"#,
        )
        .unwrap();
        let current = AppReplayScript::new(vec![AppReplayEvent::Navigate {
            sequence: 0,
            at_seconds: 0.0,
            action,
        }]);
        let trace = run_app_replay(&current).unwrap();
        assert!(matches!(
            trace.records[0].outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("requires an explicit source pivot")
        ));
        assert_eq!(trace.records[0].state.focus.selected, None);
    }

    #[test]
    fn replay_0_5_does_not_migrate_unscoped_focus_identity() {
        let action: ReplayNavigationAction = serde_json::from_str(
            r#"{
                "action":"anchor_focus",
                "entity":"70000000-0000-4000-8000-000000000001",
                "source_bound":{"center":[1.0,2.0,3.0],"radius":0.5},
                "margin":1.1,
                "duration_seconds":0.7,
                "easing":"smoother_step"
            }"#,
        )
        .unwrap();
        let script = AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_5.to_owned(),
            events: vec![AppReplayEvent::Navigate {
                sequence: 0,
                at_seconds: 0.0,
                action,
            }],
        };
        let trace = run_app_replay(&script).unwrap();
        assert_eq!(trace.version, APP_REPLAY_VERSION);
        assert!(matches!(
            trace.records[0].outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("unscoped entity identity")
        ));
    }

    #[test]
    fn current_anchor_focus_rejects_an_omitted_asset_scope() {
        let action: ReplayNavigationAction = serde_json::from_str(
            r#"{
                "action":"anchor_focus",
                "entity":"70000000-0000-4000-8000-000000000001",
                "source_bound":{"center":[1.0,2.0,3.0],"radius":0.5},
                "source_pivot":[1.0,2.0,3.0],
                "margin":1.1,
                "duration_seconds":0.7,
                "easing":"smoother_step"
            }"#,
        )
        .unwrap();
        let trace = run_app_replay(&AppReplayScript::new(vec![AppReplayEvent::Navigate {
            sequence: 0,
            at_seconds: 0.0,
            action,
        }]))
        .unwrap();
        assert!(matches!(
            trace.records[0].outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("requires an explicit asset identity")
        ));
        assert_eq!(trace.records[0].state.focus.selected, None);
    }

    #[test]
    fn replay_distinguishes_the_same_entity_uuid_in_two_assets() {
        let anchor = |asset_id| ReplayNavigationAction::AnchorFocus {
            asset_id: Some(asset_id),
            entity: Uuid::from_u128(7),
            source_bound: ReplayFocusSphere {
                center: [0.0; 3],
                radius: 1.0,
            },
            source_pivot: Some([0.0; 3]),
            margin: 1.1,
            duration_seconds: 0.0,
            easing: TransitionEasing::SmootherStep,
        };
        let first_asset = Uuid::from_u128(5);
        let second_asset = Uuid::from_u128(6);
        let trace = run_app_replay(&AppReplayScript::new(vec![
            AppReplayEvent::Navigate {
                sequence: 0,
                at_seconds: 0.0,
                action: anchor(first_asset),
            },
            AppReplayEvent::Frame {
                elapsed_seconds: 0.0,
                delta_seconds: 0.0,
            },
            AppReplayEvent::Navigate {
                sequence: 1,
                at_seconds: 0.0,
                action: anchor(second_asset),
            },
            AppReplayEvent::Frame {
                elapsed_seconds: 0.0,
                delta_seconds: 0.0,
            },
        ]))
        .unwrap();

        assert_eq!(
            trace.records[1]
                .state
                .focus
                .selected
                .as_ref()
                .unwrap()
                .asset_id,
            first_asset,
        );
        let selected = trace.records[3].state.focus.selected.as_ref().unwrap();
        assert_eq!(selected.asset_id, second_asset);
        assert_eq!(selected.entity_id, Uuid::from_u128(7));
    }

    #[test]
    fn legacy_replays_reject_0_6_camera_policy_actions() {
        for version in [
            LEGACY_APP_REPLAY_VERSION_0_4,
            LEGACY_APP_REPLAY_VERSION_0_5,
        ] {
            for action in [
                ReplayNavigationAction::SetPerspectiveLens {
                    vertical_fov_radians: 1.2,
                    near: 0.01,
                    far: 10_000.0,
                },
                ReplayNavigationAction::SetSemanticTargetEnabled { enabled: true },
            ] {
                let script = AppReplayScript {
                    version: version.to_owned(),
                    events: vec![AppReplayEvent::Navigate {
                        sequence: 0,
                        at_seconds: 0.0,
                        action,
                    }],
                };
                let trace = run_app_replay(&script).unwrap();
                assert!(matches!(
                    trace.records[0].outcome,
                    AppReplayOutcome::Rejected { ref error }
                        if error.contains("require replay 0.6")
                ));
            }
        }
    }

    #[test]
    fn replay_0_6_retains_camera_policy_actions() {
        let script = AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_6.to_owned(),
            events: vec![
                AppReplayEvent::Navigate {
                    sequence: 0,
                    at_seconds: 0.0,
                    action: ReplayNavigationAction::SetPerspectiveLens {
                        vertical_fov_radians: 1.2,
                        near: 0.01,
                        far: 10_000.0,
                    },
                },
                AppReplayEvent::Frame {
                    elapsed_seconds: 0.0,
                    delta_seconds: 0.0,
                },
            ],
        };
        let trace = run_app_replay(&script).unwrap();
        assert!(trace
            .records
            .iter()
            .all(|record| matches!(record.outcome, AppReplayOutcome::Committed { .. })));
        let camera = &trace.records[1].state.camera;
        assert_eq!(camera.vertical_fov_radians, 1.2);
        assert_eq!(camera.near, 0.01);
        assert_eq!(camera.far, 10_000.0);
    }

    #[test]
    fn explicit_anchor_pivot_roundtrips_in_the_current_schema() {
        let action = ReplayNavigationAction::AnchorFocus {
            asset_id: Some(Uuid::from_u128(6)),
            entity: Uuid::from_u128(7),
            source_bound: ReplayFocusSphere {
                center: [1.0, 2.0, 3.0],
                radius: 0.5,
            },
            source_pivot: Some([1.25, 2.1, 2.75]),
            margin: 1.1,
            duration_seconds: 0.7,
            easing: TransitionEasing::SmoothStep,
        };
        let json = serde_json::to_string(&action).unwrap();
        assert!(json.contains("asset_id"));
        assert!(json.contains("entity_id"));
        assert!(json.contains("source_pivot"));
        assert_eq!(
            serde_json::from_str::<ReplayNavigationAction>(&json).unwrap(),
            action
        );
    }

    #[test]
    fn orchestration_fixture_covers_effects_presence_and_authored_admission() {
        let script: AppReplayScript = serde_json::from_str(ORCHESTRATION_FIXTURE).unwrap();
        let encoded = serde_json::to_string(&script).unwrap();
        assert_eq!(
            serde_json::from_str::<AppReplayScript>(&encoded).unwrap(),
            script
        );

        let trace = run_app_replay(&script).unwrap();
        assert_eq!(trace.records.len(), 19);
        let (first_request, first_asset) = match &script.events[0] {
            AppReplayEvent::RequestAsset {
                request_id, asset, ..
            } => (*request_id, asset.clone()),
            event => panic!("expected first asset request, observed {event:?}"),
        };
        let (second_request, second_asset) = match &script.events[1] {
            AppReplayEvent::RequestAsset {
                request_id, asset, ..
            } => (*request_id, asset.clone()),
            event => panic!("expected replacement asset request, observed {event:?}"),
        };
        assert_eq!(
            committed_effects(&trace.records[0]),
            &[AppReplayEffect::FetchAsset {
                request_id: first_request,
                asset: first_asset.clone(),
            }]
        );
        assert_eq!(
            committed_effects(&trace.records[1]),
            &[
                AppReplayEffect::CancelAssetLoad {
                    request_id: first_request,
                    asset_id: first_asset.id,
                },
                AppReplayEffect::FetchAsset {
                    request_id: second_request,
                    asset: second_asset,
                },
            ]
        );
        assert!(matches!(
            trace.records[2].outcome,
            AppReplayOutcome::Committed {
                disposition: ReplayCommitDisposition::IgnoredStale,
                ..
            }
        ));
        assert!(matches!(
            trace.records[3].state.assets[0].status,
            ReplayAssetStatus::Failed {
                ref code,
                retryable: false,
                ..
            } if code == "decode_failed"
        ));
        assert!(matches!(
            trace.records[5].state.assets[0].status,
            ReplayAssetStatus::Cancelled
        ));
        let third_request = match &script.events[4] {
            AppReplayEvent::RequestAsset { request_id, .. } => *request_id,
            event => panic!("expected retry asset request, observed {event:?}"),
        };
        assert_eq!(
            committed_effects(&trace.records[5]),
            &[AppReplayEffect::CancelAssetLoad {
                request_id: third_request,
                asset_id: first_asset.id,
            }]
        );
        assert!(matches!(
            trace.records[6].outcome,
            AppReplayOutcome::Committed {
                disposition: ReplayCommitDisposition::IgnoredStale,
                ..
            }
        ));
        assert_eq!(
            trace.records[8].state.assets[1].status,
            ReplayAssetStatus::Ready {
                byte_length: 2048,
                content_digest: Some([7; 32]),
            }
        );

        assert_eq!(trace.records[9].state.presence[0].sequence, 1);
        assert_eq!(trace.records[9].state.presence[0].expires_at_seconds, 2.5);
        assert!(matches!(
            trace.records[10].outcome,
            AppReplayOutcome::Committed {
                disposition: ReplayCommitDisposition::IgnoredStale,
                ..
            }
        ));
        assert_eq!(trace.records[11].state.presence[0].sequence, 2);
        assert_eq!(trace.records[11].state.presence[0].expires_at_seconds, 4.2);
        assert!(matches!(
            trace.records[12].outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("unsupported protocol version")
        ));
        assert_eq!(
            trace.records[12].state.revision,
            trace.records[11].state.revision
        );

        assert_eq!(
            trace.records[13].state.authored_projection_revision,
            Some(1)
        );
        assert!(matches!(
            trace.records[14].outcome,
            AppReplayOutcome::Committed {
                disposition: ReplayCommitDisposition::IgnoredStale,
                ..
            }
        ));
        assert!(matches!(
            trace.records[15].outcome,
            AppReplayOutcome::Rejected { ref error } if error.contains("entity ID must not be nil")
        ));
        assert_eq!(
            trace.records[15].state.revision,
            trace.records[14].state.revision
        );
        assert_eq!(trace.records[16].state.presence.len(), 1);
        assert!(trace.records[17].state.presence.is_empty());
        assert!(matches!(
            trace.records[18].outcome,
            AppReplayOutcome::Rejected { ref error } if error.contains("unknown asset")
        ));
        assert_semantic_state_eq(&trace.records[18].state, &trace.records[17].state);
        assert_eq!(
            trace.records[18]
                .state
                .diagnostics
                .iter()
                .map(|diagnostic| diagnostic.code.as_str())
                .collect::<Vec<_>>(),
            [
                "stale_effect_completion",
                "stale_effect_completion",
                "stale_authored_revision",
            ]
        );
        assert_eq!(
            format!(
                "{APP_REPLAY_FINGERPRINT_ALGORITHM}:{}",
                app_replay_fingerprint(&trace).unwrap()
            ),
            ORCHESTRATION_GOLDEN.trim()
        );
        let trace_json = serde_json::to_string(&trace).unwrap();
        assert_eq!(
            serde_json::from_str::<AppReplayTrace>(&trace_json).unwrap(),
            trace
        );
    }

    #[test]
    fn completed_transition_is_cadence_independent_in_replay() {
        let presentation = fixture();
        let destination = presentation.cues[5].id;
        let prefix = vec![
            AppReplayEvent::LoadPresentation { presentation },
            AppReplayEvent::Present {
                sequence: 1,
                at_seconds: 0.0,
                action: ReplayPresentationAction::Start,
            },
            AppReplayEvent::Frame {
                elapsed_seconds: 0.7,
                delta_seconds: 0.7,
            },
            AppReplayEvent::Present {
                sequence: 2,
                at_seconds: 0.7,
                action: ReplayPresentationAction::JumpToCue { cue: destination },
            },
        ];
        let mut single = prefix.clone();
        single.push(AppReplayEvent::Frame {
            elapsed_seconds: 1.9,
            delta_seconds: 1.2,
        });
        let mut partitioned = prefix;
        for step in 1..=12 {
            partitioned.push(AppReplayEvent::Frame {
                elapsed_seconds: 0.7 + f64::from(step) * 0.1,
                delta_seconds: 0.1,
            });
        }

        let single = run_app_replay(&AppReplayScript::new(single)).unwrap();
        let partitioned = run_app_replay(&AppReplayScript::new(partitioned)).unwrap();
        let single = &single.records.last().unwrap().state;
        let partitioned = &partitioned.records.last().unwrap().state;
        assert!((single.elapsed_seconds - partitioned.elapsed_seconds).abs() < 1.0e-12);
        assert_semantic_state_eq(single, partitioned);
    }

    #[test]
    fn semantic_navigation_transition_is_cadence_independent_in_replay() {
        let target = AuthoredCamera {
            eye: [3.0, 1.0, 7.0],
            control_distance: 7.0,
            semantic_target: Some([0.0; 3]),
            ..AuthoredCamera::default()
        };
        let prefix = vec![AppReplayEvent::Navigate {
            sequence: 0,
            at_seconds: 0.0,
            action: ReplayNavigationAction::TransitionCamera {
                target,
                duration_seconds: 1.2,
                easing: TransitionEasing::SmootherStep,
            },
        }];
        let mut single = prefix.clone();
        single.push(AppReplayEvent::Frame {
            elapsed_seconds: 1.2,
            delta_seconds: 1.2,
        });
        let mut partitioned = prefix;
        for step in 1..=12 {
            partitioned.push(AppReplayEvent::Frame {
                elapsed_seconds: f64::from(step) * 0.1,
                delta_seconds: 0.1,
            });
        }

        let single = run_app_replay(&AppReplayScript::new(single)).unwrap();
        let partitioned = run_app_replay(&AppReplayScript::new(partitioned)).unwrap();
        let single = &single.records.last().unwrap().state;
        let partitioned = &partitioned.records.last().unwrap().state;
        assert!((single.elapsed_seconds - partitioned.elapsed_seconds).abs() < 1.0e-12);
        assert_semantic_state_eq(single, partitioned);
    }

    #[test]
    fn unsupported_trace_version_fails_before_dispatch() {
        let script = AppReplayScript {
            version: "hyperscope-app-replay/9.9".to_owned(),
            events: vec![AppReplayEvent::LoadPresentation {
                presentation: fixture(),
            }],
        };
        assert_eq!(
            run_app_replay(&script),
            Err(AppReplayError::UnsupportedVersion(script.version))
        );
    }
}
