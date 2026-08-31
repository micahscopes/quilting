//! Versioned, adapter-independent application replay traces.
//!
//! A trace records semantic inputs, reducer outcomes, and compact committed
//! state. It deliberately excludes DOM events, device reports, renderer
//! resources, and wall clocks so native tools, browsers, Blender adapters, and
//! future render backends can consume the same oracle.

use crate::{
    AnimationAction, AnimationClipDescriptor, AnimationClipSelectionCompletion,
    AnimationClipSelectionOutcome, AppCommit, AppEffect, AppEvent, AppStore, AssetLoadCompletion,
    AssetLoadOutcome, AssetLoadScope, AssetMetadata, AssetStatus, AuthoredRevision,
    CommitDisposition, EffectCompletion, FrameTick, NavigationSettings, NavigationSynchronization,
    PatchLabEffect, PresentationAction, PresentationAnimationResidencyBinding,
    PrimarySceneInstallCompletion, PrimarySceneInstallMetadata, PrimarySceneInstallOutcome,
    ReceivedPresence, RenderSettings, SemanticAction, Timed,
};
use hyperscape::{
    AuthoredCamera, AuthoredFocus, FocusSphere, InteractionAction, InteractionHit,
    NavigationAction, NavigationFrame, NavigationPreset, PerspectiveLens, Presentation,
    SphereReflectionState, SurfaceAnchorTarget, TransitionEasing, TurntableFrame,
};
use hyperscape_protocol::{
    AssetDescriptor, AssetEntityId, AssetId, AuthoredEnvelope, EntityId, EphemeralPresence, PeerId,
    PresenceEnvelope, RequestId, WireTransform,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use uuid::Uuid;

pub const APP_REPLAY_VERSION: &str = "hyperscope-app-replay/0.26";
pub const LEGACY_APP_REPLAY_VERSION_0_25: &str = "hyperscope-app-replay/0.25";
pub const LEGACY_APP_REPLAY_VERSION_0_24: &str = "hyperscope-app-replay/0.24";
pub const LEGACY_APP_REPLAY_VERSION_0_23: &str = "hyperscope-app-replay/0.23";
pub const LEGACY_APP_REPLAY_VERSION_0_22: &str = "hyperscope-app-replay/0.22";
pub const LEGACY_APP_REPLAY_VERSION_0_21: &str = "hyperscope-app-replay/0.21";
pub const LEGACY_APP_REPLAY_VERSION_0_20: &str = "hyperscope-app-replay/0.20";
pub const LEGACY_APP_REPLAY_VERSION_0_19: &str = "hyperscope-app-replay/0.19";
pub const LEGACY_APP_REPLAY_VERSION_0_18: &str = "hyperscope-app-replay/0.18";
pub const LEGACY_APP_REPLAY_VERSION_0_17: &str = "hyperscope-app-replay/0.17";
pub const LEGACY_APP_REPLAY_VERSION_0_16: &str = "hyperscope-app-replay/0.16";
pub const LEGACY_APP_REPLAY_VERSION_0_15: &str = "hyperscope-app-replay/0.15";
pub const LEGACY_APP_REPLAY_VERSION_0_14: &str = "hyperscope-app-replay/0.14";
pub const LEGACY_APP_REPLAY_VERSION_0_13: &str = "hyperscope-app-replay/0.13";
pub const LEGACY_APP_REPLAY_VERSION_0_12: &str = "hyperscope-app-replay/0.12";
pub const LEGACY_APP_REPLAY_VERSION_0_11: &str = "hyperscope-app-replay/0.11";
pub const LEGACY_APP_REPLAY_VERSION_0_10: &str = "hyperscope-app-replay/0.10";
pub const LEGACY_APP_REPLAY_VERSION_0_9: &str = "hyperscope-app-replay/0.9";
pub const LEGACY_APP_REPLAY_VERSION_0_8: &str = "hyperscope-app-replay/0.8";
pub const LEGACY_APP_REPLAY_VERSION_0_7: &str = "hyperscope-app-replay/0.7";
pub const LEGACY_APP_REPLAY_VERSION_0_6: &str = "hyperscope-app-replay/0.6";
pub const LEGACY_APP_REPLAY_VERSION_0_5: &str = "hyperscope-app-replay/0.5";
pub const LEGACY_APP_REPLAY_VERSION_0_4: &str = "hyperscope-app-replay/0.4";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReplaySchema {
    V0_4,
    V0_5,
    V0_6,
    V0_7,
    V0_8,
    V0_9,
    V0_10,
    V0_11,
    V0_12,
    V0_13,
    V0_14,
    V0_15,
    V0_16,
    V0_17,
    V0_18,
    V0_19,
    V0_20,
    V0_21,
    V0_22,
    V0_23,
    V0_24,
    V0_25,
    V0_26,
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
    UpdatePresentationAnimationResidency {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        binding: Option<ReplayPresentationAnimationResidencyBinding>,
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
    Interact {
        sequence: u64,
        at_seconds: f64,
        action: ReplayInteractionAction,
    },
    Present {
        sequence: u64,
        at_seconds: f64,
        action: ReplayPresentationAction,
    },
    Animate {
        sequence: u64,
        at_seconds: f64,
        action: ReplayAnimationAction,
    },
    SetRenderSettings {
        sequence: u64,
        at_seconds: f64,
        settings: RenderSettings,
    },
    SetNavigationSettings {
        sequence: u64,
        at_seconds: f64,
        settings: NavigationSettings,
    },
    RequestAsset {
        sequence: u64,
        at_seconds: f64,
        request_id: RequestId,
        asset: AssetDescriptor,
        #[serde(default, skip_serializing_if = "ReplayAssetLoadScope::is_asset")]
        scope: ReplayAssetLoadScope,
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
    CompletePrimarySceneInstall {
        request_id: RequestId,
        asset_id: AssetId,
        outcome: ReplayPrimarySceneInstallOutcome,
    },
    CompleteAnimationClipSelection {
        job_id: u64,
        scene_request_id: RequestId,
        asset_id: AssetId,
        clip_index: u32,
        outcome: ReplayAnimationClipSelectionOutcome,
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ReplayAnimationAction {
    SetPlaying { playing: bool },
    TogglePlaying,
    Seek { time_seconds: f64 },
    SetSpeed { speed: f64 },
    SetClock {
        playing: bool,
        time_seconds: f64,
        speed: f64,
    },
    SelectClip { index: u32 },
}

impl From<ReplayAnimationAction> for AnimationAction {
    fn from(action: ReplayAnimationAction) -> Self {
        match action {
            ReplayAnimationAction::SetPlaying { playing } => Self::SetPlaying(playing),
            ReplayAnimationAction::TogglePlaying => Self::TogglePlaying,
            ReplayAnimationAction::Seek { time_seconds } => Self::Seek(time_seconds),
            ReplayAnimationAction::SetSpeed { speed } => Self::SetSpeed(speed),
            ReplayAnimationAction::SetClock {
                playing,
                time_seconds,
                speed,
            } => Self::SetClock(crate::AnimationClock {
                playing,
                time_seconds,
                speed,
            }),
            ReplayAnimationAction::SelectClip { index } => Self::SelectClip(index),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayAssetLoadScope {
    #[default]
    Asset,
    PrimaryScene,
}

impl ReplayAssetLoadScope {
    fn is_asset(scope: &Self) -> bool {
        *scope == Self::Asset
    }
}

impl From<ReplayAssetLoadScope> for AssetLoadScope {
    fn from(scope: ReplayAssetLoadScope) -> Self {
        match scope {
            ReplayAssetLoadScope::Asset => Self::Asset,
            ReplayAssetLoadScope::PrimaryScene => Self::PrimaryScene,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ReplayAssetLoadOutcome {
    Loaded {
        byte_length: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content_digest: Option<[u8; 32]>,
        #[serde(default, skip_serializing_if = "AssetMetadata::is_empty")]
        metadata: AssetMetadata,
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
                metadata,
            } => Ok(Self::Loaded {
                byte_length: usize::try_from(byte_length)
                    .map_err(|_| "asset byte length exceeds this target's address space")?,
                content_digest,
                metadata,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayAnimationClipDescriptor {
    pub index: u32,
    pub name: String,
    pub time_min_seconds: f64,
    pub time_max_seconds: f64,
}

impl From<AnimationClipDescriptor> for ReplayAnimationClipDescriptor {
    fn from(clip: AnimationClipDescriptor) -> Self {
        Self {
            index: clip.index,
            name: clip.name,
            time_min_seconds: clip.time_min_seconds,
            time_max_seconds: clip.time_max_seconds,
        }
    }
}

impl From<ReplayAnimationClipDescriptor> for AnimationClipDescriptor {
    fn from(clip: ReplayAnimationClipDescriptor) -> Self {
        Self {
            index: clip.index,
            name: clip.name,
            time_min_seconds: clip.time_min_seconds,
            time_max_seconds: clip.time_max_seconds,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayPrimarySceneInstallMetadata {
    pub num_vertices: u32,
    pub num_faces: u32,
    pub animation_clips: Vec<ReplayAnimationClipDescriptor>,
}

impl From<PrimarySceneInstallMetadata> for ReplayPrimarySceneInstallMetadata {
    fn from(metadata: PrimarySceneInstallMetadata) -> Self {
        Self {
            num_vertices: metadata.num_vertices,
            num_faces: metadata.num_faces,
            animation_clips: metadata
                .animation_clips
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<ReplayPrimarySceneInstallMetadata> for PrimarySceneInstallMetadata {
    fn from(metadata: ReplayPrimarySceneInstallMetadata) -> Self {
        Self {
            num_vertices: metadata.num_vertices,
            num_faces: metadata.num_faces,
            animation_clips: metadata
                .animation_clips
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ReplayPrimarySceneInstallOutcome {
    Installed {
        metadata: ReplayPrimarySceneInstallMetadata,
    },
    Failed {
        code: String,
        message: String,
        retryable: bool,
    },
}

impl From<ReplayPrimarySceneInstallOutcome> for PrimarySceneInstallOutcome {
    fn from(outcome: ReplayPrimarySceneInstallOutcome) -> Self {
        match outcome {
            ReplayPrimarySceneInstallOutcome::Installed { metadata } => {
                Self::Installed(metadata.into())
            }
            ReplayPrimarySceneInstallOutcome::Failed {
                code,
                message,
                retryable,
            } => Self::Failed {
                code,
                message,
                retryable,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ReplayAnimationClipSelectionOutcome {
    Selected,
    Failed {
        code: String,
        message: String,
        retryable: bool,
    },
}

impl From<ReplayAnimationClipSelectionOutcome> for AnimationClipSelectionOutcome {
    fn from(outcome: ReplayAnimationClipSelectionOutcome) -> Self {
        match outcome {
            ReplayAnimationClipSelectionOutcome::Selected => Self::Selected,
            ReplayAnimationClipSelectionOutcome::Failed {
                code,
                message,
                retryable,
            } => Self::Failed {
                code,
                message,
                retryable,
            },
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
    ApplyCameraIntent {
        preset: ReplayNavigationPreset,
        semantic_target_enabled: bool,
        frame: ReplayNavigationFrame,
    },
    ApplyTurntableIntent {
        semantic_target_enabled: bool,
        frame: ReplayTurntableFrame,
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
    AimAtSelection {
        duration_seconds: f64,
        easing: TransitionEasing,
    },
    ReframeSelection {
        viewport_aspect: f64,
        margin: f64,
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
    RefitFocusAndToggleInversion {
        duration_seconds: f64,
        easing: TransitionEasing,
    },
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
pub struct ReplayTurntableFrame {
    pub pan: [f64; 2],
    pub pitch: f64,
    pub yaw: f64,
    pub dolly_log: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayFocusSphere {
    pub center: [f64; 3],
    pub radius: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayInteractionSurfacePoint {
    pub face: u32,
    pub barycentric: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayInteractionHit {
    pub asset_id: Uuid,
    pub entity_id: Uuid,
    pub source_bound: ReplayFocusSphere,
    pub source_pivot: [f64; 3],
    pub output_distance: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<ReplayInteractionSurfacePoint>,
}

impl TryFrom<ReplayInteractionHit> for InteractionHit {
    type Error = String;

    fn try_from(hit: ReplayInteractionHit) -> Result<Self, Self::Error> {
        let mut interaction = InteractionHit::new(
            AssetEntityId::new(
                AssetId::new(hit.asset_id).map_err(|error| error.to_string())?,
                EntityId::new(hit.entity_id).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?,
            hit.source_bound.try_into()?,
            hit.source_pivot,
            hit.output_distance,
        )
        .map_err(str::to_owned)?;
        if let Some(surface) = hit.surface {
            interaction = interaction
                .with_surface(surface.face, surface.barycentric)
                .map_err(str::to_owned)?;
        }
        Ok(interaction)
    }
}

impl From<InteractionHit> for ReplayInteractionHit {
    fn from(hit: InteractionHit) -> Self {
        Self {
            asset_id: hit.identity.asset.as_uuid(),
            entity_id: hit.identity.entity.as_uuid(),
            source_bound: hit.source_bound.into(),
            source_pivot: hit.source_pivot,
            output_distance: hit.output_distance,
            surface: hit.surface.map(|surface| ReplayInteractionSurfacePoint {
                face: surface.face,
                barycentric: surface.barycentric,
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ReplayInteractionAction {
    SetHover {
        hit: Option<ReplayInteractionHit>,
    },
    SetProximityHover {
        hit: Option<ReplayInteractionHit>,
    },
    PressPrimary,
    ReleasePrimary,
    CancelPrimary,
}

impl TryFrom<ReplayInteractionAction> for InteractionAction {
    type Error = String;

    fn try_from(action: ReplayInteractionAction) -> Result<Self, Self::Error> {
        match action {
            ReplayInteractionAction::SetHover { hit } => Ok(Self::SetHover(
                hit.map(InteractionHit::try_from).transpose()?,
            )),
            ReplayInteractionAction::SetProximityHover { hit } => Ok(
                Self::SetProximityHover(hit.map(InteractionHit::try_from).transpose()?),
            ),
            ReplayInteractionAction::PressPrimary => Ok(Self::PressPrimary),
            ReplayInteractionAction::ReleasePrimary => Ok(Self::ReleasePrimary),
            ReplayInteractionAction::CancelPrimary => Ok(Self::CancelPrimary),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplaySurfaceAnchorTarget {
    pub camera: AuthoredCamera,
    pub normal: [f64; 3],
}

fn replay_navigation_frame(frame: ReplayNavigationFrame) -> Result<NavigationFrame, String> {
    finite3(frame.translation, "navigation translation")?;
    finite3(frame.rotation, "navigation rotation")?;
    finite(frame.dolly_log, "navigation dolly")?;
    Ok(NavigationFrame {
        translation: frame.translation,
        rotation: frame.rotation,
        dolly_log: frame.dolly_log,
        horizon_locked: frame.horizon_locked,
    })
}

fn replay_turntable_frame(frame: ReplayTurntableFrame) -> Result<TurntableFrame, String> {
    if frame.pan.into_iter().any(|value| !value.is_finite()) {
        return Err("turntable pan must remain finite".to_owned());
    }
    finite(frame.pitch, "turntable pitch")?;
    finite(frame.yaw, "turntable yaw")?;
    finite(frame.dolly_log, "turntable dolly")?;
    Ok(TurntableFrame {
        pan: frame.pan,
        pitch: frame.pitch,
        yaw: frame.yaw,
        dolly_log: frame.dolly_log,
    })
}

impl TryFrom<ReplayNavigationAction> for NavigationAction {
    type Error = String;

    fn try_from(action: ReplayNavigationAction) -> Result<Self, Self::Error> {
        match action {
            ReplayNavigationAction::SetPreset { preset } => Ok(Self::SetPreset(preset.into())),
            ReplayNavigationAction::ApplyFrame { frame } => {
                replay_navigation_frame(frame).map(Self::ApplyFrame)
            }
            ReplayNavigationAction::ApplyCameraIntent {
                preset,
                semantic_target_enabled,
                frame,
            } => Ok(Self::ApplyCameraIntent {
                preset: preset.into(),
                semantic_target_enabled,
                frame: replay_navigation_frame(frame)?,
            }),
            ReplayNavigationAction::ApplyTurntableIntent {
                semantic_target_enabled,
                frame,
            } => Ok(Self::ApplyTurntableIntent {
                semantic_target_enabled,
                frame: replay_turntable_frame(frame)?,
            }),
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
            ReplayNavigationAction::AimAtSelection {
                duration_seconds,
                easing,
            } => {
                nonnegative(duration_seconds, "selected aim transition duration")?;
                Ok(Self::AimAtSelection {
                    duration_seconds,
                    easing,
                })
            }
            ReplayNavigationAction::ReframeSelection {
                viewport_aspect,
                margin,
                duration_seconds,
                easing,
            } => {
                positive(viewport_aspect, "selection reframe viewport aspect")?;
                if !margin.is_finite() || margin < 1.0 {
                    return Err("selection reframe margin must be finite and at least one".into());
                }
                nonnegative(duration_seconds, "selection reframe transition duration")?;
                Ok(Self::ReframeSelection {
                    viewport_aspect,
                    margin,
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
                    enabled: None,
                    coordinate,
                    angular_aperture,
                })
            }
            ReplayNavigationAction::SetInversionEnabled { enabled } => {
                Ok(Self::SetInversionEnabled(enabled))
            }
            ReplayNavigationAction::ToggleInversion => Ok(Self::ToggleInversion),
            ReplayNavigationAction::RefitFocusAndToggleInversion {
                duration_seconds,
                easing,
            } => {
                nonnegative(duration_seconds, "focus transition duration")?;
                Ok(Self::RefitFocusAndToggleInversion {
                    duration_seconds,
                    easing,
                })
            }
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
    InstallPrimaryScene {
        request_id: RequestId,
        asset_id: AssetId,
    },
    CancelPrimarySceneInstall {
        request_id: RequestId,
        asset_id: AssetId,
    },
    SelectAnimationClip {
        job_id: u64,
        scene_request_id: RequestId,
        asset_id: AssetId,
        clip_index: u32,
    },
    CancelAnimationClipSelection {
        job_id: u64,
        scene_request_id: RequestId,
        asset_id: AssetId,
        clip_index: u32,
    },
    PatchLab {
        effect: PatchLabEffect,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presentation_animation_residency: Option<ReplayPresentationAnimationResidencyBinding>,
    pub animation_playing: bool,
    pub animation_time_seconds: f64,
    pub animation_speed: f64,
    #[serde(default)]
    pub navigation_settings: NavigationSettings,
    pub render_settings: RenderSettings,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loading_primary_scene_asset: Option<AssetId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loading_primary_scene_request: Option<RequestId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ready_primary_asset: Option<AppReplayPrimaryAssetState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub installing_primary_scene_asset: Option<AssetId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub installing_primary_scene_request: Option<RequestId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub installed_primary_scene: Option<AppReplayInstalledPrimarySceneState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_animation_clip: Option<AppReplayActiveAnimationClipState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_animation_clip: Option<AppReplayPendingAnimationClipState>,
    pub assets: Vec<AppReplayAssetState>,
    pub authored_assets: Vec<AssetDescriptor>,
    pub authored_entities: Vec<AppReplayAuthoredEntityState>,
    pub presence: Vec<AppReplayPresenceState>,
    pub diagnostics: Vec<AppReplayDiagnosticState>,
    pub reflection: ReplayReflection,
    pub navigation: AppReplayNavigationState,
    #[serde(default)]
    pub interaction: AppReplayInteractionState,
    pub camera: AppReplayCameraState,
    pub focus: AppReplayFocusState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayPresentationAnimationResidencyBinding {
    pub presentation_asset_id: AssetId,
    pub scene_request_id: RequestId,
    pub resident_asset_id: AssetId,
}

impl From<ReplayPresentationAnimationResidencyBinding>
    for PresentationAnimationResidencyBinding
{
    fn from(binding: ReplayPresentationAnimationResidencyBinding) -> Self {
        Self {
            presentation_asset_id: binding.presentation_asset_id,
            scene_request_id: binding.scene_request_id,
            resident_asset_id: binding.resident_asset_id,
        }
    }
}

impl From<PresentationAnimationResidencyBinding>
    for ReplayPresentationAnimationResidencyBinding
{
    fn from(binding: PresentationAnimationResidencyBinding) -> Self {
        Self {
            presentation_asset_id: binding.presentation_asset_id,
            scene_request_id: binding.scene_request_id,
            resident_asset_id: binding.resident_asset_id,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayAuthoredEntityState {
    pub entity: EntityId,
    pub transform: WireTransform,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayAssetState {
    pub descriptor: AssetDescriptor,
    pub status: ReplayAssetStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayPrimaryAssetState {
    pub request_id: RequestId,
    pub descriptor: AssetDescriptor,
    pub byte_length: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_digest: Option<[u8; 32]>,
    #[serde(default, skip_serializing_if = "AssetMetadata::is_empty")]
    pub metadata: AssetMetadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayInstalledPrimarySceneState {
    pub asset: AppReplayPrimaryAssetState,
    pub install: ReplayPrimarySceneInstallMetadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayActiveAnimationClipState {
    pub scene_request_id: RequestId,
    pub asset_id: AssetId,
    pub clip: ReplayAnimationClipDescriptor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayPendingAnimationClipState {
    pub job_id: u64,
    pub scene_request_id: RequestId,
    pub asset_id: AssetId,
    pub clip: ReplayAnimationClipDescriptor,
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
        #[serde(default, skip_serializing_if = "AssetMetadata::is_empty")]
        metadata: AssetMetadata,
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
                metadata,
            } => Self::Ready {
                byte_length: u64::try_from(byte_length)
                    .expect("supported Rust targets have at most 64-bit usize"),
                content_digest,
                metadata,
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

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppReplayInteractionState {
    pub revision: u64,
    pub integrated_until_seconds: f64,
    pub last_applied_sequence: Option<u64>,
    pub hovered: Option<ReplayInteractionHit>,
    pub active: Option<ReplayInteractionHit>,
    pub selected: Option<ReplayInteractionIdentity>,
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayInteractionIdentity {
    pub asset_id: Uuid,
    pub entity_id: Uuid,
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
        LEGACY_APP_REPLAY_VERSION_0_7 => ReplaySchema::V0_7,
        LEGACY_APP_REPLAY_VERSION_0_8 => ReplaySchema::V0_8,
        LEGACY_APP_REPLAY_VERSION_0_9 => ReplaySchema::V0_9,
        LEGACY_APP_REPLAY_VERSION_0_10 => ReplaySchema::V0_10,
        LEGACY_APP_REPLAY_VERSION_0_11 => ReplaySchema::V0_11,
        LEGACY_APP_REPLAY_VERSION_0_12 => ReplaySchema::V0_12,
        LEGACY_APP_REPLAY_VERSION_0_13 => ReplaySchema::V0_13,
        LEGACY_APP_REPLAY_VERSION_0_14 => ReplaySchema::V0_14,
        LEGACY_APP_REPLAY_VERSION_0_15 => ReplaySchema::V0_15,
        LEGACY_APP_REPLAY_VERSION_0_16 => ReplaySchema::V0_16,
        LEGACY_APP_REPLAY_VERSION_0_17 => ReplaySchema::V0_17,
        LEGACY_APP_REPLAY_VERSION_0_18 => ReplaySchema::V0_18,
        LEGACY_APP_REPLAY_VERSION_0_19 => ReplaySchema::V0_19,
        LEGACY_APP_REPLAY_VERSION_0_20 => ReplaySchema::V0_20,
        LEGACY_APP_REPLAY_VERSION_0_21 => ReplaySchema::V0_21,
        LEGACY_APP_REPLAY_VERSION_0_22 => ReplaySchema::V0_22,
        LEGACY_APP_REPLAY_VERSION_0_23 => ReplaySchema::V0_23,
        LEGACY_APP_REPLAY_VERSION_0_24 => ReplaySchema::V0_24,
        LEGACY_APP_REPLAY_VERSION_0_25 => ReplaySchema::V0_25,
        APP_REPLAY_VERSION => ReplaySchema::V0_26,
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
    let initial_render_settings = RenderSettings {
        style: hyperscape::RenderStyle::Stretch,
        resolution_level: 6,
        tessellation: hyperscape::PresentationTessellation {
            density: 12.0,
            screen_attenuation: false,
            min_pixels_per_subdivision: 64.0,
        },
        atlas_exponent: 9,
        max_face_edge_ratio: 4,
        ..RenderSettings::default()
    };
    let mut events = vec![
        AppReplayEvent::LoadPresentation { presentation },
        AppReplayEvent::SynchronizeNavigation {
            camera: AuthoredCamera::default(),
            focus: AuthoredFocus::default(),
        },
        AppReplayEvent::SetRenderSettings {
            sequence: 1,
            at_seconds: 0.0,
            settings: initial_render_settings,
        },
    ];
    let Some((_, first_duration)) = cues.first().copied() else {
        return AppReplayScript::new(events);
    };

    let mut elapsed_seconds = 0.0;
    events.push(AppReplayEvent::Present {
        sequence: 2,
        at_seconds: elapsed_seconds,
        action: ReplayPresentationAction::Start,
    });
    elapsed_seconds += first_duration;
    events.push(AppReplayEvent::Frame {
        elapsed_seconds,
        delta_seconds: first_duration,
    });

    for (sequence, (cue, duration)) in (3_u64..).zip(cues.into_iter().skip(1)) {
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

fn replay_event(
    store: &AppStore,
    event: &AppReplayEvent,
    schema: ReplaySchema,
) -> AppReplayOutcome {
    let result: Result<AppCommit, String> = replay_app_event(event, schema)
        .and_then(|event| store.dispatch(event).map_err(|error| error.to_string()))
        .map(|mut commit| {
            // Schemas through 0.19 ended primary loading at decoded bytes. A
            // current reducer emits the new renderer-install effect after the
            // same completion, so erase that newly introduced job/effect when
            // interpreting an old script. This is an adapter migration fence,
            // not a second application mutation path: only replay can request
            // historical reducer semantics.
            if !matches!(
                schema,
                ReplaySchema::V0_20
                    | ReplaySchema::V0_21
                    | ReplaySchema::V0_22
                    | ReplaySchema::V0_23
                    | ReplaySchema::V0_24
                    | ReplaySchema::V0_25
                    | ReplaySchema::V0_26
            ) {
                let pending_install = commit.effects.iter().find_map(|effect| match effect {
                    AppEffect::InstallPrimaryScene {
                        request_id,
                        asset_id,
                    } => Some((*request_id, *asset_id)),
                    _ => None,
                });
                if let Some(pending_install) = pending_install {
                    {
                        let mut state = store.lock_state();
                        if state.primary_scene_install == Some(pending_install) {
                            state.primary_scene_install = None;
                        }
                    }
                    store.flush_read_models();
                    commit.effects.retain(|effect| {
                        !matches!(effect, AppEffect::InstallPrimaryScene { .. })
                    });
                }
            }
            // Version 0.20 recorded the installed clip catalog but did not
            // project an application-owned active clip. Preserve that exact
            // state when the current reducer derives clip zero from a
            // successful installation.
            if schema == ReplaySchema::V0_20
                && matches!(event, AppReplayEvent::CompletePrimarySceneInstall { .. })
            {
                {
                    let mut state = store.lock_state();
                    state.active_animation_clip = None;
                    state.pending_animation_clip = None;
                }
                store.flush_read_models();
            }
            commit
        });
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
        AppReplayEvent::UpdatePresentationAnimationResidency { binding } => {
            if !matches!(
                schema,
                ReplaySchema::V0_22
                    | ReplaySchema::V0_23
                    | ReplaySchema::V0_24
                    | ReplaySchema::V0_25
                    | ReplaySchema::V0_26
            ) {
                return Err(
                    "presentation animation residency requires app replay 0.22".to_owned(),
                );
            }
            Ok(AppEvent::PresentationAnimationResidencyChanged(
                binding.map(Into::into),
            ))
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
        AppReplayEvent::Interact {
            sequence,
            at_seconds,
            action,
        } => {
            if !matches!(schema, ReplaySchema::V0_25 | ReplaySchema::V0_26) {
                return Err("interaction actions require app replay 0.25".to_owned());
            }
            Ok(AppEvent::Input(Timed {
                sequence: *sequence,
                at_seconds: *at_seconds,
                value: SemanticAction::Interact((*action).try_into()?),
            }))
        }
        AppReplayEvent::Present {
            sequence,
            at_seconds,
            action,
        } => Ok(AppEvent::Input(Timed {
            sequence: *sequence,
            at_seconds: *at_seconds,
            value: SemanticAction::Present((*action).into()),
        })),
        AppReplayEvent::Animate {
            sequence,
            at_seconds,
            action,
        } => {
            if !matches!(
                schema,
                ReplaySchema::V0_11
                    | ReplaySchema::V0_12
                    | ReplaySchema::V0_13
                    | ReplaySchema::V0_14
                    | ReplaySchema::V0_15
                    | ReplaySchema::V0_16
                    | ReplaySchema::V0_17
                    | ReplaySchema::V0_18
                    | ReplaySchema::V0_19
                    | ReplaySchema::V0_20
                    | ReplaySchema::V0_21
                    | ReplaySchema::V0_22
                    | ReplaySchema::V0_23
                    | ReplaySchema::V0_24
                    | ReplaySchema::V0_25
                    | ReplaySchema::V0_26
            ) {
                return Err("animation playback actions require app replay 0.11".to_owned());
            }
            if !matches!(
                schema,
                ReplaySchema::V0_17
                    | ReplaySchema::V0_18
                    | ReplaySchema::V0_19
                    | ReplaySchema::V0_20
                    | ReplaySchema::V0_21
                    | ReplaySchema::V0_22
                    | ReplaySchema::V0_23
                    | ReplaySchema::V0_24
                    | ReplaySchema::V0_25
                    | ReplaySchema::V0_26
            )
                && matches!(
                    action,
                    ReplayAnimationAction::Seek { .. }
                        | ReplayAnimationAction::SetSpeed { .. }
                        | ReplayAnimationAction::SetClock { .. }
                )
            {
                return Err("animation clock actions require app replay 0.17".to_owned());
            }
            if !matches!(
                schema,
                ReplaySchema::V0_21
                    | ReplaySchema::V0_22
                    | ReplaySchema::V0_23
                    | ReplaySchema::V0_24
                    | ReplaySchema::V0_25
                    | ReplaySchema::V0_26
            )
                && matches!(action, ReplayAnimationAction::SelectClip { .. })
            {
                return Err("animation clip selection requires app replay 0.21".to_owned());
            }
            Ok(AppEvent::Input(Timed {
                sequence: *sequence,
                at_seconds: *at_seconds,
                value: SemanticAction::Animate((*action).into()),
            }))
        }
        AppReplayEvent::SetRenderSettings {
            sequence,
            at_seconds,
            settings,
        } => {
            if !matches!(
                schema,
                ReplaySchema::V0_18
                    | ReplaySchema::V0_19
                    | ReplaySchema::V0_20
                    | ReplaySchema::V0_21
                    | ReplaySchema::V0_22
                    | ReplaySchema::V0_23
                    | ReplaySchema::V0_24
                    | ReplaySchema::V0_25
                    | ReplaySchema::V0_26
            ) {
                return Err("render settings actions require app replay 0.18".to_owned());
            }
            if !matches!(
                schema,
                ReplaySchema::V0_24 | ReplaySchema::V0_25 | ReplaySchema::V0_26
            )
                && settings.focus_postprocess != crate::FocusPostprocessSettings::default()
            {
                return Err("focus postprocess settings require app replay 0.24".to_owned());
            }
            if schema != ReplaySchema::V0_26
                && settings.focus_postprocess.diagnostic_view
                    != crate::FocusDiagnosticView::Composite
            {
                return Err("focus diagnostic views require app replay 0.26".to_owned());
            }
            Ok(AppEvent::Input(Timed {
                sequence: *sequence,
                at_seconds: *at_seconds,
                value: SemanticAction::SetRenderSettings(*settings),
            }))
        }
        AppReplayEvent::SetNavigationSettings {
            sequence,
            at_seconds,
            settings,
        } => {
            if !matches!(
                schema,
                ReplaySchema::V0_23
                    | ReplaySchema::V0_24
                    | ReplaySchema::V0_25
                    | ReplaySchema::V0_26
            ) {
                return Err("navigation settings actions require app replay 0.23".to_owned());
            }
            Ok(AppEvent::Input(Timed {
                sequence: *sequence,
                at_seconds: *at_seconds,
                value: SemanticAction::SetNavigationSettings(*settings),
            }))
        }
        AppReplayEvent::RequestAsset {
            sequence,
            at_seconds,
            request_id,
            asset,
            scope,
        } => Ok(AppEvent::Input(Timed {
            sequence: *sequence,
            at_seconds: *at_seconds,
            value: SemanticAction::RequestAsset {
                request_id: *request_id,
                asset: asset.clone(),
                scope: match schema {
                    ReplaySchema::V0_9
                    | ReplaySchema::V0_10
                    | ReplaySchema::V0_11
                    | ReplaySchema::V0_12
                    | ReplaySchema::V0_13
                    | ReplaySchema::V0_14
                    | ReplaySchema::V0_15
                    | ReplaySchema::V0_16
                    | ReplaySchema::V0_17
                    | ReplaySchema::V0_18
                    | ReplaySchema::V0_19
                    | ReplaySchema::V0_20
                    | ReplaySchema::V0_21
                    | ReplaySchema::V0_22
                    | ReplaySchema::V0_23
                    | ReplaySchema::V0_24
                    | ReplaySchema::V0_25
                    | ReplaySchema::V0_26 => (*scope).into(),
                    _ if *scope == ReplayAssetLoadScope::Asset => AssetLoadScope::Asset,
                    _ => {
                        return Err(
                            "primary_scene asset scope requires app replay schema 0.9".to_owned()
                        )
                    }
                },
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
        } => {
            if matches!(
                outcome,
                ReplayAssetLoadOutcome::Loaded { metadata, .. } if !metadata.is_empty()
            ) && !matches!(
                schema,
                ReplaySchema::V0_12
                    | ReplaySchema::V0_13
                    | ReplaySchema::V0_14
                    | ReplaySchema::V0_15
                    | ReplaySchema::V0_16
                    | ReplaySchema::V0_17
                    | ReplaySchema::V0_18
                    | ReplaySchema::V0_19
                    | ReplaySchema::V0_20
                    | ReplaySchema::V0_21
                    | ReplaySchema::V0_22
                    | ReplaySchema::V0_23
                    | ReplaySchema::V0_24
                    | ReplaySchema::V0_25
                    | ReplaySchema::V0_26
            ) {
                return Err("asset provenance requires app replay 0.12".to_owned());
            }
            Ok(AppEvent::EffectCompleted(EffectCompletion::AssetLoad(
                AssetLoadCompletion {
                    request_id: *request_id,
                    asset_id: *asset_id,
                    outcome: outcome.clone().try_into()?,
                },
            )))
        }
        AppReplayEvent::CompletePrimarySceneInstall {
            request_id,
            asset_id,
            outcome,
        } => {
            if !matches!(
                schema,
                ReplaySchema::V0_20
                    | ReplaySchema::V0_21
                    | ReplaySchema::V0_22
                    | ReplaySchema::V0_23
                    | ReplaySchema::V0_24
                    | ReplaySchema::V0_25
                    | ReplaySchema::V0_26
            ) {
                return Err("primary scene installation requires app replay 0.20".to_owned());
            }
            Ok(AppEvent::EffectCompleted(
                EffectCompletion::PrimarySceneInstall(PrimarySceneInstallCompletion {
                    request_id: *request_id,
                    asset_id: *asset_id,
                    outcome: outcome.clone().into(),
                }),
            ))
        }
        AppReplayEvent::CompleteAnimationClipSelection {
            job_id,
            scene_request_id,
            asset_id,
            clip_index,
            outcome,
        } => {
            if !matches!(
                schema,
                ReplaySchema::V0_21
                    | ReplaySchema::V0_22
                    | ReplaySchema::V0_23
                    | ReplaySchema::V0_24
                    | ReplaySchema::V0_25
                    | ReplaySchema::V0_26
            ) {
                return Err("animation clip selection requires app replay 0.21".to_owned());
            }
            Ok(AppEvent::EffectCompleted(
                EffectCompletion::AnimationClipSelection(AnimationClipSelectionCompletion {
                    job_id: *job_id,
                    scene_request_id: *scene_request_id,
                    asset_id: *asset_id,
                    clip_index: *clip_index,
                    outcome: outcome.clone().into(),
                }),
            ))
        }
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
    if !matches!(
        schema,
        ReplaySchema::V0_13
            | ReplaySchema::V0_14
            | ReplaySchema::V0_15
            | ReplaySchema::V0_16
            | ReplaySchema::V0_17
            | ReplaySchema::V0_18
            | ReplaySchema::V0_19
            | ReplaySchema::V0_20
            | ReplaySchema::V0_21
            | ReplaySchema::V0_22
            | ReplaySchema::V0_23
            | ReplaySchema::V0_24
            | ReplaySchema::V0_25
            | ReplaySchema::V0_26
    )
        && matches!(action, ReplayNavigationAction::ApplyCameraIntent { .. })
    {
        return Err("atomic camera intent requires replay 0.13".to_owned());
    }
    if !matches!(
        schema,
        ReplaySchema::V0_14
            | ReplaySchema::V0_15
            | ReplaySchema::V0_16
            | ReplaySchema::V0_17
            | ReplaySchema::V0_18
            | ReplaySchema::V0_19
            | ReplaySchema::V0_20
            | ReplaySchema::V0_21
            | ReplaySchema::V0_22
            | ReplaySchema::V0_23
            | ReplaySchema::V0_24
            | ReplaySchema::V0_25
            | ReplaySchema::V0_26
    )
        && matches!(action, ReplayNavigationAction::ApplyTurntableIntent { .. })
    {
        return Err("atomic turntable intent requires replay 0.14".to_owned());
    }
    if !matches!(
        schema,
        ReplaySchema::V0_15
            | ReplaySchema::V0_16
            | ReplaySchema::V0_17
            | ReplaySchema::V0_18
            | ReplaySchema::V0_19
            | ReplaySchema::V0_20
            | ReplaySchema::V0_21
            | ReplaySchema::V0_22
            | ReplaySchema::V0_23
            | ReplaySchema::V0_24
            | ReplaySchema::V0_25
            | ReplaySchema::V0_26
    )
        && matches!(action, ReplayNavigationAction::ReframeSelection { .. })
    {
        return Err("selected camera reframe requires replay 0.15".to_owned());
    }
    if !matches!(
        schema,
        ReplaySchema::V0_16
            | ReplaySchema::V0_17
            | ReplaySchema::V0_18
            | ReplaySchema::V0_19
            | ReplaySchema::V0_20
            | ReplaySchema::V0_21
            | ReplaySchema::V0_22
            | ReplaySchema::V0_23
            | ReplaySchema::V0_24
            | ReplaySchema::V0_25
            | ReplaySchema::V0_26
    )
        && matches!(action, ReplayNavigationAction::AimAtSelection { .. })
    {
        return Err("selected camera aim requires replay 0.16".to_owned());
    }
    if matches!(schema, ReplaySchema::V0_4 | ReplaySchema::V0_5)
        && matches!(
            action,
            ReplayNavigationAction::SetPerspectiveLens { .. }
                | ReplayNavigationAction::SetSemanticTargetEnabled { .. }
        )
    {
        return Err("camera lens/target-mode actions require replay 0.6".to_owned());
    }
    if !matches!(
        schema,
        ReplaySchema::V0_10
            | ReplaySchema::V0_11
            | ReplaySchema::V0_12
            | ReplaySchema::V0_13
            | ReplaySchema::V0_14
            | ReplaySchema::V0_15
            | ReplaySchema::V0_16
            | ReplaySchema::V0_17
            | ReplaySchema::V0_18
            | ReplaySchema::V0_19
            | ReplaySchema::V0_20
            | ReplaySchema::V0_21
            | ReplaySchema::V0_22
            | ReplaySchema::V0_23
            | ReplaySchema::V0_24
            | ReplaySchema::V0_25
            | ReplaySchema::V0_26
    ) && matches!(
        action,
        ReplayNavigationAction::RefitFocusAndToggleInversion { .. }
    ) {
        return Err("selected inversion gesture requires replay 0.10".to_owned());
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
    if !matches!(
        schema,
        ReplaySchema::V0_7
            | ReplaySchema::V0_8
            | ReplaySchema::V0_9
            | ReplaySchema::V0_10
            | ReplaySchema::V0_11
            | ReplaySchema::V0_12
            | ReplaySchema::V0_13
            | ReplaySchema::V0_14
            | ReplaySchema::V0_15
            | ReplaySchema::V0_16
            | ReplaySchema::V0_17
            | ReplaySchema::V0_18
            | ReplaySchema::V0_19
            | ReplaySchema::V0_20
            | ReplaySchema::V0_21
            | ReplaySchema::V0_22
            | ReplaySchema::V0_23
            | ReplaySchema::V0_24
            | ReplaySchema::V0_25
            | ReplaySchema::V0_26
    ) && matches!(
        action,
        ReplayNavigationAction::AnchorFocus { asset_id: None, .. }
    ) {
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
        AppEffect::InstallPrimaryScene {
            request_id,
            asset_id,
        } => AppReplayEffect::InstallPrimaryScene {
            request_id: *request_id,
            asset_id: *asset_id,
        },
        AppEffect::CancelPrimarySceneInstall {
            request_id,
            asset_id,
        } => AppReplayEffect::CancelPrimarySceneInstall {
            request_id: *request_id,
            asset_id: *asset_id,
        },
        AppEffect::SelectAnimationClip {
            job_id,
            scene_request_id,
            asset_id,
            clip_index,
        } => AppReplayEffect::SelectAnimationClip {
            job_id: *job_id,
            scene_request_id: *scene_request_id,
            asset_id: *asset_id,
            clip_index: *clip_index,
        },
        AppEffect::CancelAnimationClipSelection {
            job_id,
            scene_request_id,
            asset_id,
            clip_index,
        } => AppReplayEffect::CancelAnimationClipSelection {
            job_id: *job_id,
            scene_request_id: *scene_request_id,
            asset_id: *asset_id,
            clip_index: *clip_index,
        },
        AppEffect::PatchLab(effect) => AppReplayEffect::PatchLab {
            effect: effect.clone(),
        },
    }
}

fn replay_state(store: &AppStore) -> AppReplayState {
    let state = store.lock_state();
    let frame = state.frame_snapshot();
    let authored = state.authored_scene_read_model();
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
        presentation_animation_residency: state
            .presentation_animation_residency
            .map(Into::into),
        animation_playing: state.animation.playing,
        animation_time_seconds: state.animation.time_seconds,
        animation_speed: state.animation.speed,
        navigation_settings: state.navigation_settings,
        render_settings: state.render_settings,
        loading_primary_scene_asset: state.primary_scene_load.map(|(_, asset)| asset),
        loading_primary_scene_request: state.primary_scene_load.map(|(request, _)| request),
        ready_primary_asset: state.ready_primary_asset.as_ref().map(|asset| {
            AppReplayPrimaryAssetState {
                request_id: asset.request_id,
                descriptor: asset.descriptor.clone(),
                byte_length: u64::try_from(asset.byte_length)
                    .expect("supported Rust targets have at most 64-bit usize"),
                content_digest: asset.content_digest,
                metadata: asset.metadata.clone(),
            }
        }),
        installing_primary_scene_asset: state
            .primary_scene_install
            .map(|(_, asset)| asset),
        installing_primary_scene_request: state
            .primary_scene_install
            .map(|(request, _)| request),
        installed_primary_scene: state.installed_primary_scene.as_ref().map(|scene| {
            AppReplayInstalledPrimarySceneState {
                asset: AppReplayPrimaryAssetState {
                    request_id: scene.asset.request_id,
                    descriptor: scene.asset.descriptor.clone(),
                    byte_length: u64::try_from(scene.asset.byte_length)
                        .expect("supported Rust targets have at most 64-bit usize"),
                    content_digest: scene.asset.content_digest,
                    metadata: scene.asset.metadata.clone(),
                },
                install: scene.install.clone().into(),
            }
        }),
        active_animation_clip: state.active_animation_clip.as_ref().map(|active| {
            AppReplayActiveAnimationClipState {
                scene_request_id: active.scene_request_id,
                asset_id: active.asset_id,
                clip: active.clip.clone().into(),
            }
        }),
        pending_animation_clip: state.pending_animation_clip.as_ref().map(|pending| {
            AppReplayPendingAnimationClipState {
                job_id: pending.job_id,
                scene_request_id: pending.scene_request_id,
                asset_id: pending.asset_id,
                clip: pending.clip.clone().into(),
            }
        }),
        assets: state
            .asset_read_models()
            .into_iter()
            .map(|asset| AppReplayAssetState {
                descriptor: asset.descriptor,
                status: asset.status.into(),
            })
            .collect(),
        authored_assets: authored.assets,
        authored_entities: authored
            .entities
            .into_iter()
            .map(|entity| AppReplayAuthoredEntityState {
                entity: entity.entity,
                transform: entity.transform,
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
        interaction: AppReplayInteractionState {
            revision: frame.interaction.revision,
            integrated_until_seconds: frame.interaction.integrated_until_seconds,
            last_applied_sequence: frame.interaction.last_applied_sequence,
            hovered: frame.interaction.hovered.map(Into::into),
            active: frame.interaction.active.map(Into::into),
            selected: frame
                .interaction
                .selected
                .map(|identity| ReplayInteractionIdentity {
                    asset_id: identity.asset.as_uuid(),
                    entity_id: identity.entity.as_uuid(),
                }),
            diagnostics: state.interaction.diagnostics.0.clone(),
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

    const FIXTURE: &str = hyperscape::HACKER_NIGHT_PRESENTATION_JSON;
    const GOLDEN: &str = include_str!("../fixtures/hacker-night.replay.fingerprint");
    const NAVIGATION_FIXTURE: &str = include_str!("../fixtures/navigation.app-replay.json");
    const NAVIGATION_GOLDEN: &str = include_str!("../fixtures/navigation.replay.fingerprint");
    const ORCHESTRATION_FIXTURE: &str = include_str!("../fixtures/orchestration.app-replay.json");
    const ORCHESTRATION_GOLDEN: &str = include_str!("../fixtures/orchestration.replay.fingerprint");
    const INTERACTION_FIXTURE: &str = include_str!("../fixtures/interaction.app-replay.json");
    const INTERACTION_GOLDEN: &str = include_str!("../fixtures/interaction.replay.fingerprint");
    const UNKNOWN_CUE: &str = "f0000000-0000-4000-8000-000000000099";

    fn fixture() -> Presentation {
        Presentation::from_json(FIXTURE).unwrap()
    }

    fn assert_semantic_state_eq(left: &AppReplayState, right: &AppReplayState) {
        assert_eq!(left.active_cue, right.active_cue);
        assert_eq!(left.active_scene, right.active_scene);
        assert_eq!(left.active_view, right.active_view);
        assert_eq!(left.animation_playing, right.animation_playing);
        assert_eq!(left.animation_time_seconds, right.animation_time_seconds);
        assert_eq!(left.animation_speed, right.animation_speed);
        assert_eq!(left.navigation_settings, right.navigation_settings);
        assert_eq!(left.render_settings, right.render_settings);
        assert_eq!(
            left.loading_primary_scene_asset,
            right.loading_primary_scene_asset
        );
        assert_eq!(
            left.loading_primary_scene_request,
            right.loading_primary_scene_request
        );
        assert_eq!(left.ready_primary_asset, right.ready_primary_asset);
        assert_eq!(
            left.installing_primary_scene_asset,
            right.installing_primary_scene_asset
        );
        assert_eq!(
            left.installing_primary_scene_request,
            right.installing_primary_scene_request
        );
        assert_eq!(
            left.installed_primary_scene,
            right.installed_primary_scene
        );
        assert_eq!(left.active_animation_clip, right.active_animation_clip);
        assert_eq!(left.pending_animation_clip, right.pending_animation_clip);
        assert_eq!(
            left.authored_projection_revision,
            right.authored_projection_revision
        );
        assert_eq!(left.assets, right.assets);
        assert_eq!(left.authored_assets, right.authored_assets);
        assert_eq!(left.authored_entities, right.authored_entities);
        assert_eq!(left.presence, right.presence);
        assert_eq!(left.diagnostics, right.diagnostics);
        assert_eq!(left.reflection, right.reflection);
        assert_eq!(left.navigation, right.navigation);
        assert_eq!(left.interaction.revision, right.interaction.revision);
        assert!(
            (left.interaction.integrated_until_seconds
                - right.interaction.integrated_until_seconds)
                .abs()
                <= 1.0e-12
        );
        assert_eq!(
            left.interaction.last_applied_sequence,
            right.interaction.last_applied_sequence
        );
        assert_eq!(left.interaction.hovered, right.interaction.hovered);
        assert_eq!(left.interaction.active, right.interaction.active);
        assert_eq!(left.interaction.selected, right.interaction.selected);
        assert_eq!(left.interaction.diagnostics, right.interaction.diagnostics);
        assert_eq!(left.camera, right.camera);
        assert_eq!(left.focus, right.focus);
    }

    #[test]
    fn interaction_replay_preserves_surface_identity_and_is_cadence_independent() {
        let script: AppReplayScript = serde_json::from_str(INTERACTION_FIXTURE).unwrap();
        let trace = run_app_replay(&script).unwrap();
        let final_state = &trace.records.last().unwrap().state;
        let hovered = final_state.interaction.hovered.unwrap();
        assert_eq!(
            hovered.asset_id,
            Uuid::from_u128(0xd0000000000040008000000000000001)
        );
        assert_eq!(
            hovered.entity_id,
            Uuid::from_u128(0xe0000000000040008000000000000001)
        );
        assert_eq!(hovered.source_pivot, [1.25, 2.5, 3.75]);
        assert_eq!(hovered.output_distance, 4.5);
        assert_eq!(
            hovered.surface,
            Some(ReplayInteractionSurfacePoint {
                face: 17,
                barycentric: [0.5, 0.25, 0.25],
            })
        );
        assert_eq!(final_state.interaction.active, None);
        assert_eq!(final_state.interaction.last_applied_sequence, Some(2));
        assert_eq!(
            final_state.interaction.selected,
            Some(ReplayInteractionIdentity {
                asset_id: hovered.asset_id,
                entity_id: hovered.entity_id,
            })
        );
        let selected = final_state
            .focus
            .selected
            .as_ref()
            .expect("interaction activation selects through navigation authority");
        assert_eq!(selected.asset_id, hovered.asset_id);
        assert_eq!(selected.entity_id, hovered.entity_id);
        assert_eq!(selected.source_pivot, hovered.source_pivot);

        let encoded = serde_json::to_string(&trace).unwrap();
        assert_eq!(serde_json::from_str::<AppReplayTrace>(&encoded).unwrap(), trace);
        assert_eq!(
            format!(
                "{APP_REPLAY_FINGERPRINT_ALGORITHM}:{}",
                app_replay_fingerprint(&trace).unwrap()
            ),
            INTERACTION_GOLDEN.trim()
        );

        let mut split = script.clone();
        split.events.pop();
        split.events.insert(
            1,
            AppReplayEvent::Frame {
                elapsed_seconds: 0.1,
                delta_seconds: 0.1,
            },
        );
        split.events.insert(
            3,
            AppReplayEvent::Frame {
                elapsed_seconds: 0.2,
                delta_seconds: 0.1,
            },
        );
        split.events.push(AppReplayEvent::Frame {
            elapsed_seconds: 0.3,
            delta_seconds: 0.09999999999999998,
        });
        let split_trace = run_app_replay(&split).unwrap();
        assert_semantic_state_eq(
            &trace.records.last().unwrap().state,
            &split_trace.records.last().unwrap().state,
        );

        let mut legacy = script;
        legacy.version = LEGACY_APP_REPLAY_VERSION_0_24.to_owned();
        let legacy_trace = run_app_replay(&legacy).unwrap();
        assert_eq!(
            legacy_trace
                .records
                .iter()
                .filter(|record| matches!(record.outcome, AppReplayOutcome::Rejected { .. }))
                .count(),
            3
        );
        assert_eq!(legacy_trace.records.last().unwrap().state.focus.selected, None);
        assert_eq!(
            legacy_trace
                .records
                .first()
                .and_then(|record| match &record.outcome {
                    AppReplayOutcome::Rejected { error } => Some(error.as_str()),
                    _ => None,
                }),
            Some("interaction actions require app replay 0.25")
        );

        let mut invalid: AppReplayScript = serde_json::from_str(INTERACTION_FIXTURE).unwrap();
        if let AppReplayEvent::Interact {
            action: ReplayInteractionAction::SetHover { hit: Some(hit) },
            ..
        } = &mut invalid.events[0]
        {
            hit.asset_id = Uuid::nil();
        } else {
            panic!("interaction fixture starts with a concrete hover hit");
        }
        let invalid_trace = run_app_replay(&invalid).unwrap();
        assert!(matches!(
            invalid_trace.records[0].outcome,
            AppReplayOutcome::Rejected { ref error } if error.contains("must not be nil")
        ));
        assert_eq!(
            invalid_trace.records.last().unwrap().state.focus.selected,
            None
        );
    }

    #[test]
    fn replay_records_animation_playback_actions_and_rejects_legacy_use() {
        let events = vec![
            AppReplayEvent::Animate {
                sequence: 1,
                at_seconds: 0.0,
                action: ReplayAnimationAction::SetPlaying { playing: false },
            },
            AppReplayEvent::Animate {
                sequence: 2,
                at_seconds: 0.0,
                action: ReplayAnimationAction::TogglePlaying,
            },
        ];
        let trace = run_app_replay(&AppReplayScript::new(events.clone())).unwrap();
        assert!(!trace.records[0].state.animation_playing);
        assert!(trace.records[1].state.animation_playing);
        assert!(trace
            .records
            .iter()
            .all(|record| matches!(record.outcome, AppReplayOutcome::Committed { .. })));

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_10.to_owned(),
            events,
        })
        .unwrap();
        assert!(legacy.records.iter().all(|record| matches!(
            &record.outcome,
            AppReplayOutcome::Rejected { error }
                if error.contains("require app replay 0.11")
        )));
    }

    #[test]
    fn replay_0_17_records_animation_clock_without_reinterpreting_0_16() {
        let events = vec![AppReplayEvent::Animate {
            sequence: 1,
            at_seconds: 0.0,
            action: ReplayAnimationAction::SetClock {
                playing: true,
                time_seconds: 2.0,
                speed: -0.5,
            },
        }];
        let current = run_app_replay(&AppReplayScript::new(events.clone())).unwrap();
        assert!(matches!(
            current.records[0].outcome,
            AppReplayOutcome::Committed { .. }
        ));
        assert_eq!(current.records[0].state.animation_time_seconds, 2.0);
        assert_eq!(current.records[0].state.animation_speed, -0.5);

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_16.to_owned(),
            events,
        })
        .unwrap();
        assert!(matches!(
            &legacy.records[0].outcome,
            AppReplayOutcome::Rejected { error }
                if error.contains("require app replay 0.17")
        ));
        assert_eq!(legacy.records[0].state.animation_time_seconds, 0.0);
        assert_eq!(legacy.records[0].state.animation_speed, 1.0);
    }

    #[test]
    fn replay_0_18_records_atomic_render_settings_without_reinterpreting_0_17() {
        let settings = RenderSettings {
            style: hyperscape::RenderStyle::Lod,
            resolution_level: 5,
            tessellation: hyperscape::PresentationTessellation {
                density: 321.5,
                screen_attenuation: true,
                min_pixels_per_subdivision: 48.25,
            },
            atlas_exponent: 9,
            max_face_edge_ratio: 4,
            ..RenderSettings::default()
        };
        let events = vec![AppReplayEvent::SetRenderSettings {
            sequence: 1,
            at_seconds: 0.0,
            settings,
        }];
        let current = run_app_replay(&AppReplayScript::new(events.clone())).unwrap();
        assert!(matches!(
            current.records[0].outcome,
            AppReplayOutcome::Committed { .. }
        ));
        assert_eq!(current.records[0].state.render_settings, settings);

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_17.to_owned(),
            events,
        })
        .unwrap();
        assert!(matches!(
            &legacy.records[0].outcome,
            AppReplayOutcome::Rejected { error }
                if error.contains("require app replay 0.18")
        ));
        assert_eq!(
            legacy.records[0].state.render_settings,
            RenderSettings::default(),
        );
    }

    #[test]
    fn replay_0_23_records_atomic_navigation_settings_without_reinterpreting_0_22() {
        let mut settings = NavigationSettings::default();
        settings.transition_seconds = 2.5;
        settings.surface_walk.smoothing_seconds = 0.45;
        settings.surface_walk.tangent_pull_fraction = 0.25;
        settings.surface_walk.speed_octave_steps = 100.0;
        settings.surface_walk.body_scale_octave_steps = -100.0;
        settings.surface_walk.eye_height_octave_steps = 50.0;
        let events = vec![AppReplayEvent::SetNavigationSettings {
            sequence: 1,
            at_seconds: 0.0,
            settings,
        }];
        let script = AppReplayScript::new(events.clone());
        let encoded = serde_json::to_string(&script).unwrap();
        let decoded: AppReplayScript = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, script);

        let current = run_app_replay(&decoded).unwrap();
        assert!(matches!(
            current.records[0].outcome,
            AppReplayOutcome::Committed { .. }
        ));
        assert_eq!(current.records[0].state.navigation_settings, settings);
        let mut legacy_state_json = serde_json::to_value(&current.records[0].state).unwrap();
        legacy_state_json
            .as_object_mut()
            .unwrap()
            .remove("navigationSettings");
        let legacy_state: AppReplayState = serde_json::from_value(legacy_state_json).unwrap();
        assert_eq!(
            legacy_state.navigation_settings,
            NavigationSettings::default(),
        );

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_22.to_owned(),
            events,
        })
        .unwrap();
        assert!(matches!(
            &legacy.records[0].outcome,
            AppReplayOutcome::Rejected { error }
                if error.contains("require app replay 0.23")
        ));
        assert_eq!(
            legacy.records[0].state.navigation_settings,
            NavigationSettings::default(),
        );
    }

    #[test]
    fn replay_0_24_records_focus_postprocess_without_reinterpreting_0_23() {
        let focus_postprocess = crate::FocusPostprocessSettings {
            enabled: true,
            mode: crate::FocusPostprocessMode::Spheroidal,
            diagnostic_view: crate::FocusDiagnosticView::Composite,
            blur_radius_pixels: 48,
            blur_strength: 1.75,
            focus_coordinate: 0.25,
            bandwidth: 0.08,
            normalize_range: true,
            gaussian_passes: 2,
            kawase_passes: 4,
            kawase_offset: 2.25,
        };
        let settings = RenderSettings::default()
            .with_focus_postprocess(focus_postprocess)
            .unwrap();
        let events = vec![AppReplayEvent::SetRenderSettings {
            sequence: 1,
            at_seconds: 0.0,
            settings,
        }];
        let current = run_app_replay(&AppReplayScript::new(events.clone())).unwrap();
        assert!(matches!(
            current.records[0].outcome,
            AppReplayOutcome::Committed { .. }
        ));
        assert_eq!(
            current.records[0].state.render_settings.focus_postprocess,
            focus_postprocess,
        );

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_23.to_owned(),
            events,
        })
        .unwrap();
        assert!(matches!(
            &legacy.records[0].outcome,
            AppReplayOutcome::Rejected { error }
                if error.contains("require app replay 0.24")
        ));
        assert_eq!(
            legacy.records[0].state.render_settings,
            RenderSettings::default(),
        );

        let mut legacy_json = serde_json::to_value(RenderSettings::default()).unwrap();
        legacy_json
            .as_object_mut()
            .unwrap()
            .remove("focusPostprocess");
        let decoded: RenderSettings = serde_json::from_value(legacy_json).unwrap();
        assert_eq!(decoded.focus_postprocess, Default::default());
    }

    #[test]
    fn replay_0_26_records_focus_diagnostics_without_reinterpreting_0_25() {
        let focus_postprocess = crate::FocusPostprocessSettings {
            diagnostic_view: crate::FocusDiagnosticView::Firmness,
            ..crate::FocusPostprocessSettings::default()
        };
        let settings = RenderSettings::default()
            .with_focus_postprocess(focus_postprocess)
            .unwrap();
        let events = vec![AppReplayEvent::SetRenderSettings {
            sequence: 1,
            at_seconds: 0.0,
            settings,
        }];
        let current = run_app_replay(&AppReplayScript::new(events.clone())).unwrap();
        assert!(matches!(
            current.records[0].outcome,
            AppReplayOutcome::Committed { .. }
        ));
        assert_eq!(
            current.records[0]
                .state
                .render_settings
                .focus_postprocess
                .diagnostic_view,
            crate::FocusDiagnosticView::Firmness,
        );

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_25.to_owned(),
            events,
        })
        .unwrap();
        assert!(matches!(
            &legacy.records[0].outcome,
            AppReplayOutcome::Rejected { error }
                if error.contains("require app replay 0.26")
        ));
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
            ReplayNavigationAction::ApplyCameraIntent { .. } => "apply_camera_intent",
            ReplayNavigationAction::ApplyTurntableIntent { .. } => "apply_turntable_intent",
            ReplayNavigationAction::SetCamera { .. } => "set_camera",
            ReplayNavigationAction::SetPerspectiveLens { .. } => "set_perspective_lens",
            ReplayNavigationAction::SetSemanticTargetEnabled { .. } => {
                "set_semantic_target_enabled"
            }
            ReplayNavigationAction::TransitionCamera { .. } => "transition_camera",
            ReplayNavigationAction::AimAtSelection { .. } => "aim_at_selection",
            ReplayNavigationAction::ReframeSelection { .. } => "reframe_selection",
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
            ReplayNavigationAction::RefitFocusAndToggleInversion { .. } => {
                "refit_focus_and_toggle_inversion"
            }
        }
    }

    fn authoritative_navigation_action_name(action: &NavigationAction) -> &'static str {
        match action {
            NavigationAction::SetPreset(_) => "set_preset",
            NavigationAction::ApplyFrame(_) => "apply_frame",
            NavigationAction::ApplyCameraIntent { .. } => "apply_camera_intent",
            NavigationAction::ApplyTurntableIntent { .. } => "apply_turntable_intent",
            NavigationAction::SetCamera(_) => "set_camera",
            NavigationAction::SetPerspectiveLens(_) => "set_perspective_lens",
            NavigationAction::SetSemanticTargetEnabled(_) => "set_semantic_target_enabled",
            NavigationAction::TransitionCamera { .. } => "transition_camera",
            NavigationAction::AimAtSelection { .. } => "aim_at_selection",
            NavigationAction::ReframeSelection { .. } => "reframe_selection",
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
            NavigationAction::RefitFocusAndToggleInversion { .. } => {
                "refit_focus_and_toggle_inversion"
            }
        }
    }

    fn authoritative_app_event_name(event: &AppEvent) -> &'static str {
        match event {
            AppEvent::Input(timed) => match &timed.value {
                SemanticAction::Navigate(_) => "navigate",
                SemanticAction::Interact(_) => "interact",
                SemanticAction::Present(_) => "present",
                SemanticAction::Animate(_) => "animate",
                SemanticAction::SetNavigationSettings(_) => "set_navigation_settings",
                SemanticAction::SetRenderSettings(_) => "set_render_settings",
                SemanticAction::SetPatchLab(_) => "set_patch_lab",
                SemanticAction::RequestAsset { .. } => "request_asset",
                SemanticAction::CancelAsset(_) => "cancel_asset",
            },
            AppEvent::PresentationLoaded(_) => "presentation_loaded",
            AppEvent::PresentationAnimationResidencyChanged(_) => {
                "presentation_animation_residency_changed"
            }
            AppEvent::NavigationSynchronized(_) => "navigation_synchronized",
            AppEvent::Frame(_) => "frame",
            AppEvent::EffectCompleted(completion) => match completion {
                EffectCompletion::AssetLoad(_) => "effect_completed_asset_load",
                EffectCompletion::PrimarySceneInstall(_) => {
                    "effect_completed_primary_scene_install"
                }
                EffectCompletion::AnimationClipSelection(_) => {
                    "effect_completed_animation_clip_selection"
                }
                EffectCompletion::PatchLab(_) => "effect_completed_patch_lab",
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
        let current_animation_events = [
            AppReplayEvent::SetNavigationSettings {
                sequence: 0,
                at_seconds: 0.0,
                settings: NavigationSettings::default(),
            },
            AppReplayEvent::Interact {
                sequence: 0,
                at_seconds: 0.0,
                action: ReplayInteractionAction::SetHover {
                    hit: Some(ReplayInteractionHit {
                        asset_id: Uuid::from_u128(0xd0),
                        entity_id: Uuid::from_u128(0xe0),
                        source_bound: ReplayFocusSphere {
                            center: [0.0, 0.0, 0.0],
                            radius: 1.0,
                        },
                        source_pivot: [0.0, 0.0, 0.0],
                        output_distance: 1.0,
                        surface: None,
                    }),
                },
            },
            AppReplayEvent::CompletePrimarySceneInstall {
                request_id: RequestId::from_u128(0xb0).unwrap(),
                asset_id: AssetId::from_u128(0xa0).unwrap(),
                outcome: ReplayPrimarySceneInstallOutcome::Installed {
                    metadata: ReplayPrimarySceneInstallMetadata {
                        num_vertices: 3,
                        num_faces: 1,
                        animation_clips: Vec::new(),
                    },
                },
            },
            AppReplayEvent::Animate {
                sequence: 0,
                at_seconds: 0.0,
                action: ReplayAnimationAction::SelectClip { index: 0 },
            },
            AppReplayEvent::CompleteAnimationClipSelection {
                job_id: 0,
                scene_request_id: RequestId::from_u128(0xb0).unwrap(),
                asset_id: AssetId::from_u128(0xa0).unwrap(),
                clip_index: 0,
                outcome: ReplayAnimationClipSelectionOutcome::Selected,
            },
            AppReplayEvent::UpdatePresentationAnimationResidency {
                binding: Some(ReplayPresentationAnimationResidencyBinding {
                    presentation_asset_id: AssetId::from_u128(0xc0).unwrap(),
                    scene_request_id: RequestId::from_u128(0xb0).unwrap(),
                    resident_asset_id: AssetId::from_u128(0xa0).unwrap(),
                }),
            },
        ];
        let covered = presentation
            .events
            .iter()
            .chain(navigation.events.iter())
            .chain(orchestration.events.iter())
            .chain(current_animation_events.iter())
            .filter_map(|event| replay_app_event(event, ReplaySchema::V0_26).ok())
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
                "animate",
                "cancel_asset",
                "effect_completed_animation_clip_selection",
                "effect_completed_asset_load",
                "effect_completed_primary_scene_install",
                "frame",
                "interact",
                "navigate",
                "navigation_synchronized",
                "presentation_animation_residency_changed",
                "presentation_loaded",
                "present",
                "remote_presence",
                "request_asset",
                "set_navigation_settings",
                "set_render_settings",
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
        let presentation = fixture();
        let expected_records = 3 + presentation.cues.len() * 2;
        let script = presentation_walkthrough_replay(presentation);
        let encoded_script = serde_json::to_string(&script).unwrap();
        let decoded_script: AppReplayScript = serde_json::from_str(&encoded_script).unwrap();
        assert_eq!(decoded_script, script);

        let first = run_app_replay(&script).unwrap();
        let second = run_app_replay(&decoded_script).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.records.len(), expected_records);
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
                "aim_at_selection",
                "apply_camera_intent",
                "apply_turntable_intent",
                "apply_frame",
                "begin_surface_anchor_transition",
                "cancel_surface_anchor_transition",
                "detach_focus",
                "refit_focus_and_toggle_inversion",
                "reframe_selection",
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
        assert_eq!(trace.records.len(), 40);
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
        assert_eq!(after.navigation.last_applied_sequence, Some(24));
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
    fn replay_0_9_records_primary_scene_replacement_without_reinterpreting_0_8() {
        let descriptor = |id: u128, uri: &str| AssetDescriptor {
            id: AssetId::from_u128(id).unwrap(),
            uri: uri.to_owned(),
            media_type: Some("model/gltf-binary".to_owned()),
            content_digest: None,
        };
        let first_asset = descriptor(1, "horse.glb");
        let second_asset = descriptor(2, "chess.glb");
        let first_request = RequestId::from_u128(10).unwrap();
        let second_request = RequestId::from_u128(11).unwrap();
        let events = vec![
            AppReplayEvent::RequestAsset {
                sequence: 1,
                at_seconds: 0.0,
                request_id: first_request,
                asset: first_asset.clone(),
                scope: ReplayAssetLoadScope::PrimaryScene,
            },
            AppReplayEvent::RequestAsset {
                sequence: 2,
                at_seconds: 0.0,
                request_id: second_request,
                asset: second_asset.clone(),
                scope: ReplayAssetLoadScope::PrimaryScene,
            },
        ];

        let trace = run_app_replay(&AppReplayScript::new(events.clone())).unwrap();
        assert_eq!(
            committed_effects(&trace.records[1]),
            &[
                AppReplayEffect::CancelAssetLoad {
                    request_id: first_request,
                    asset_id: first_asset.id,
                },
                AppReplayEffect::FetchAsset {
                    request_id: second_request,
                    asset: second_asset.clone(),
                },
            ]
        );
        assert_eq!(
            trace.records[1].state.loading_primary_scene_asset,
            Some(second_asset.id)
        );
        assert_eq!(
            trace.records[1].state.loading_primary_scene_request,
            Some(second_request)
        );

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_8.to_owned(),
            events: vec![events[0].clone()],
        })
        .unwrap();
        assert!(matches!(
            legacy.records[0].outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("requires app replay schema 0.9")
        ));
        assert_eq!(legacy.records[0].state.revision, 0);
        assert!(legacy.records[0].state.assets.is_empty());
    }

    #[test]
    fn replay_0_20_records_renderer_install_without_reinterpreting_0_19() {
        let asset = AssetDescriptor {
            id: AssetId::from_u128(1).unwrap(),
            uri: "horse.glb".to_owned(),
            media_type: Some("model/gltf-binary".to_owned()),
            content_digest: None,
        };
        let request_id = RequestId::from_u128(10).unwrap();
        let decoded_events = vec![
            AppReplayEvent::RequestAsset {
                sequence: 1,
                at_seconds: 0.0,
                request_id,
                asset: asset.clone(),
                scope: ReplayAssetLoadScope::PrimaryScene,
            },
            AppReplayEvent::CompleteAssetLoad {
                request_id,
                asset_id: asset.id,
                outcome: ReplayAssetLoadOutcome::Loaded {
                    byte_length: 123,
                    content_digest: Some([9; 32]),
                    metadata: AssetMetadata::default(),
                },
            },
        ];

        let mut current_events = decoded_events.clone();
        current_events.push(AppReplayEvent::CompletePrimarySceneInstall {
            request_id,
            asset_id: asset.id,
            outcome: ReplayPrimarySceneInstallOutcome::Installed {
                metadata: ReplayPrimarySceneInstallMetadata {
                    num_vertices: 796,
                    num_faces: 984,
                    animation_clips: vec![ReplayAnimationClipDescriptor {
                        index: 0,
                        name: "gallop".to_owned(),
                        time_min_seconds: 0.0,
                        time_max_seconds: 1.5,
                    }],
                },
            },
        });
        let current = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_20.to_owned(),
            events: current_events,
        })
        .unwrap();
        assert_eq!(
            current.records[1].state.ready_primary_asset,
            Some(AppReplayPrimaryAssetState {
                request_id,
                descriptor: asset.clone(),
                byte_length: 123,
                content_digest: Some([9; 32]),
                metadata: AssetMetadata::default(),
            })
        );
        assert_eq!(
            committed_effects(&current.records[1]),
            &[AppReplayEffect::InstallPrimaryScene {
                request_id,
                asset_id: asset.id,
            }]
        );
        assert_eq!(
            current.records[1].state.installing_primary_scene_request,
            Some(request_id)
        );
        assert_eq!(
            current.records[2]
                .state
                .installed_primary_scene
                .as_ref()
                .unwrap()
                .asset
                .descriptor,
            asset
        );

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_19.to_owned(),
            events: decoded_events,
        })
        .unwrap();
        assert_eq!(legacy.records[1].state.ready_primary_asset, current.records[1].state.ready_primary_asset);
        assert_eq!(legacy.records[1].state.installing_primary_scene_asset, None);
        assert_eq!(legacy.records[1].state.installed_primary_scene, None);
        assert!(committed_effects(&legacy.records[1]).is_empty());
    }

    #[test]
    fn replay_0_21_records_clip_jobs_without_reinterpreting_0_20() {
        let asset = AssetDescriptor {
            id: AssetId::from_u128(1).unwrap(),
            uri: "horse.glb".to_owned(),
            media_type: Some("model/gltf-binary".to_owned()),
            content_digest: None,
        };
        let request_id = RequestId::from_u128(10).unwrap();
        let events = vec![
            AppReplayEvent::RequestAsset {
                sequence: 1,
                at_seconds: 0.0,
                request_id,
                asset: asset.clone(),
                scope: ReplayAssetLoadScope::PrimaryScene,
            },
            AppReplayEvent::CompleteAssetLoad {
                request_id,
                asset_id: asset.id,
                outcome: ReplayAssetLoadOutcome::Loaded {
                    byte_length: 123,
                    content_digest: None,
                    metadata: AssetMetadata::default(),
                },
            },
            AppReplayEvent::CompletePrimarySceneInstall {
                request_id,
                asset_id: asset.id,
                outcome: ReplayPrimarySceneInstallOutcome::Installed {
                    metadata: ReplayPrimarySceneInstallMetadata {
                        num_vertices: 796,
                        num_faces: 984,
                        animation_clips: vec![
                            ReplayAnimationClipDescriptor {
                                index: 0,
                                name: "gallop".to_owned(),
                                time_min_seconds: 0.0,
                                time_max_seconds: 1.5,
                            },
                            ReplayAnimationClipDescriptor {
                                index: 1,
                                name: "turn".to_owned(),
                                time_min_seconds: 2.0,
                                time_max_seconds: 3.0,
                            },
                        ],
                    },
                },
            },
            AppReplayEvent::Animate {
                sequence: 2,
                at_seconds: 0.0,
                action: ReplayAnimationAction::SelectClip { index: 1 },
            },
            AppReplayEvent::CompleteAnimationClipSelection {
                job_id: 0,
                scene_request_id: request_id,
                asset_id: asset.id,
                clip_index: 1,
                outcome: ReplayAnimationClipSelectionOutcome::Selected,
            },
        ];

        let current = run_app_replay(&AppReplayScript::new(events.clone())).unwrap();
        assert_eq!(
            current.records[2]
                .state
                .active_animation_clip
                .as_ref()
                .unwrap()
                .clip
                .index,
            0
        );
        assert_eq!(
            committed_effects(&current.records[3]),
            &[AppReplayEffect::SelectAnimationClip {
                job_id: 0,
                scene_request_id: request_id,
                asset_id: asset.id,
                clip_index: 1,
            }]
        );
        assert_eq!(
            current.records[3]
                .state
                .pending_animation_clip
                .as_ref()
                .unwrap()
                .clip
                .index,
            1
        );
        assert_eq!(
            current.records[4]
                .state
                .active_animation_clip
                .as_ref()
                .unwrap()
                .clip
                .index,
            1
        );

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_20.to_owned(),
            events,
        })
        .unwrap();
        assert_eq!(legacy.records[2].state.active_animation_clip, None);
        assert!(matches!(
            legacy.records[3].outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("requires app replay 0.21")
        ));
        assert_eq!(legacy.records[3].state.pending_animation_clip, None);
        assert!(matches!(
            legacy.records[4].outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("requires app replay 0.21")
        ));
    }

    #[test]
    fn replay_0_22_records_presentation_animation_residency_without_reinterpreting_0_21() {
        let resident_asset = AssetDescriptor {
            id: AssetId::from_u128(1).unwrap(),
            uri: "cached-horse.glb".to_owned(),
            media_type: Some("model/gltf-binary".to_owned()),
            content_digest: None,
        };
        let presentation_asset_id = AssetId::new(
            Uuid::parse_str("a0000000-0000-4000-8000-000000000001").unwrap(),
        )
        .unwrap();
        let request_id = RequestId::from_u128(10).unwrap();
        let binding = ReplayPresentationAnimationResidencyBinding {
            presentation_asset_id,
            scene_request_id: request_id,
            resident_asset_id: resident_asset.id,
        };
        let events = vec![
            AppReplayEvent::LoadPresentation {
                presentation: fixture(),
            },
            AppReplayEvent::Present {
                sequence: 1,
                at_seconds: 0.0,
                action: ReplayPresentationAction::Start,
            },
            AppReplayEvent::RequestAsset {
                sequence: 2,
                at_seconds: 0.0,
                request_id,
                asset: resident_asset.clone(),
                scope: ReplayAssetLoadScope::PrimaryScene,
            },
            AppReplayEvent::CompleteAssetLoad {
                request_id,
                asset_id: resident_asset.id,
                outcome: ReplayAssetLoadOutcome::Loaded {
                    byte_length: 123,
                    content_digest: None,
                    metadata: AssetMetadata::default(),
                },
            },
            AppReplayEvent::CompletePrimarySceneInstall {
                request_id,
                asset_id: resident_asset.id,
                outcome: ReplayPrimarySceneInstallOutcome::Installed {
                    metadata: ReplayPrimarySceneInstallMetadata {
                        num_vertices: 796,
                        num_faces: 984,
                        animation_clips: vec![
                            ReplayAnimationClipDescriptor {
                                index: 0,
                                name: "idle".to_owned(),
                                time_min_seconds: 0.0,
                                time_max_seconds: 1.0,
                            },
                            ReplayAnimationClipDescriptor {
                                index: 1,
                                name: "horse_A_".to_owned(),
                                time_min_seconds: 0.0,
                                time_max_seconds: 1.5,
                            },
                        ],
                    },
                },
            },
            AppReplayEvent::UpdatePresentationAnimationResidency {
                binding: Some(binding),
            },
        ];

        let current = run_app_replay(&AppReplayScript::new(events.clone())).unwrap();
        assert_eq!(
            current.records[5].state.presentation_animation_residency,
            Some(binding),
        );
        assert_eq!(
            committed_effects(&current.records[5]),
            &[AppReplayEffect::SelectAnimationClip {
                job_id: 0,
                scene_request_id: request_id,
                asset_id: resident_asset.id,
                clip_index: 1,
            }],
        );

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_21.to_owned(),
            events,
        })
        .unwrap();
        assert!(matches!(
            legacy.records[5].outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("requires app replay 0.22")
        ));
        assert_eq!(
            legacy.records[5].state.presentation_animation_residency,
            None,
        );
    }

    #[test]
    fn replay_0_12_records_asset_provenance_without_reinterpreting_0_11() {
        let asset = AssetDescriptor {
            id: AssetId::from_u128(1).unwrap(),
            uri: "credited.glb".to_owned(),
            media_type: Some("model/gltf-binary".to_owned()),
            content_digest: None,
        };
        let request_id = RequestId::from_u128(2).unwrap();
        let metadata = AssetMetadata {
            title: Some("Credited model".to_owned()),
            author: Some("Example Artist".to_owned()),
            license: Some("CC-BY-4.0".to_owned()),
            source: Some("https://example.test/credited-model".to_owned()),
            ..AssetMetadata::default()
        };
        let events = vec![
            AppReplayEvent::RequestAsset {
                sequence: 0,
                at_seconds: 0.0,
                request_id,
                asset: asset.clone(),
                scope: ReplayAssetLoadScope::Asset,
            },
            AppReplayEvent::CompleteAssetLoad {
                request_id,
                asset_id: asset.id,
                outcome: ReplayAssetLoadOutcome::Loaded {
                    byte_length: 42,
                    content_digest: None,
                    metadata: metadata.clone(),
                },
            },
        ];

        let current = run_app_replay(&AppReplayScript::new(events.clone())).unwrap();
        assert_eq!(
            current.records[1].state.assets[0].status,
            ReplayAssetStatus::Ready {
                byte_length: 42,
                content_digest: None,
                metadata,
            }
        );

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_11.to_owned(),
            events,
        })
        .unwrap();
        assert!(matches!(
            legacy.records[1].outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("asset provenance requires app replay 0.12")
        ));
        assert!(matches!(
            legacy.records[1].state.assets[0].status,
            ReplayAssetStatus::Loading { .. }
        ));
    }

    #[test]
    fn replay_0_13_adds_atomic_camera_intent_without_reinterpreting_0_12() {
        let intent = AppReplayEvent::Navigate {
            sequence: 0,
            at_seconds: 0.0,
            action: ReplayNavigationAction::ApplyCameraIntent {
                preset: ReplayNavigationPreset::Object,
                semantic_target_enabled: true,
                frame: ReplayNavigationFrame {
                    translation: [0.1, 0.0, 0.0],
                    rotation: [0.0; 3],
                    dolly_log: 0.0,
                    horizon_locked: false,
                },
            },
        };
        let current = run_app_replay(&AppReplayScript::new(vec![
            intent.clone(),
            AppReplayEvent::Frame {
                elapsed_seconds: 0.0,
                delta_seconds: 0.0,
            },
        ]))
        .unwrap();
        assert!(matches!(
            current.records[0].outcome,
            AppReplayOutcome::Committed { .. }
        ));
        assert_eq!(
            current.records[1].state.navigation.preset,
            ReplayNavigationPreset::Object
        );
        assert!(current.records[1].state.camera.semantic_target.is_some());

        let replay_0_13 = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_13.to_owned(),
            events: vec![intent.clone()],
        })
        .unwrap();
        assert!(matches!(
            replay_0_13.records[0].outcome,
            AppReplayOutcome::Committed { .. }
        ));

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_12.to_owned(),
            events: vec![intent],
        })
        .unwrap();
        assert!(matches!(
            legacy.records[0].outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("requires replay 0.13")
        ));
        assert_eq!(legacy.records[0].state.revision, 0);
    }

    #[test]
    fn replay_0_14_adds_atomic_turntable_intent_without_reinterpreting_0_13() {
        let intent = AppReplayEvent::Navigate {
            sequence: 0,
            at_seconds: 0.0,
            action: ReplayNavigationAction::ApplyTurntableIntent {
                semantic_target_enabled: false,
                frame: ReplayTurntableFrame {
                    pan: [0.1, -0.2],
                    pitch: 0.03,
                    yaw: -0.04,
                    dolly_log: 0.1,
                },
            },
        };
        let current = run_app_replay(&AppReplayScript::new(vec![
            intent.clone(),
            AppReplayEvent::Frame {
                elapsed_seconds: 0.0,
                delta_seconds: 0.0,
            },
        ]))
        .unwrap();
        assert!(matches!(
            current.records[0].outcome,
            AppReplayOutcome::Committed { .. }
        ));
        assert_ne!(current.records[1].state.camera.eye, [0.0, 0.0, 3.0]);
        assert!(current.records[1].state.camera.semantic_target.is_none());

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_13.to_owned(),
            events: vec![intent],
        })
        .unwrap();
        assert!(matches!(
            legacy.records[0].outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("requires replay 0.14")
        ));
        assert_eq!(legacy.records[0].state.revision, 0);
    }

    #[test]
    fn replay_0_15_adds_selected_reframe_without_reinterpreting_0_14() {
        let reframe = AppReplayEvent::Navigate {
            sequence: 1,
            at_seconds: 0.0,
            action: ReplayNavigationAction::ReframeSelection {
                viewport_aspect: 16.0 / 9.0,
                margin: 1.15,
                duration_seconds: 0.0,
                easing: TransitionEasing::SmootherStep,
            },
        };
        let events = vec![
            AppReplayEvent::Navigate {
                sequence: 0,
                at_seconds: 0.0,
                action: ReplayNavigationAction::AnchorFocus {
                    asset_id: Some(
                        Uuid::parse_str("60000000-0000-4000-8000-000000000001").unwrap(),
                    ),
                    entity: Uuid::parse_str("70000000-0000-4000-8000-000000000001")
                        .unwrap(),
                    source_bound: ReplayFocusSphere {
                        center: [2.0, 0.0, 0.0],
                        radius: 0.5,
                    },
                    source_pivot: Some([2.0, 0.0, 0.0]),
                    margin: 1.1,
                    duration_seconds: 0.0,
                    easing: TransitionEasing::Linear,
                },
            },
            AppReplayEvent::Frame {
                elapsed_seconds: 0.0,
                delta_seconds: 0.0,
            },
            reframe.clone(),
            AppReplayEvent::Frame {
                elapsed_seconds: 0.0,
                delta_seconds: 0.0,
            },
        ];
        let current = run_app_replay(&AppReplayScript::new(events)).unwrap();
        assert!(current.records.iter().all(|record| matches!(
            record.outcome,
            AppReplayOutcome::Committed { .. }
        )));
        assert_eq!(
            current.records[3].state.camera.semantic_target,
            Some([2.0, 0.0, 0.0])
        );

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_14.to_owned(),
            events: vec![reframe],
        })
        .unwrap();
        assert!(matches!(
            legacy.records[0].outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("requires replay 0.15")
        ));
        assert_eq!(legacy.records[0].state.revision, 0);
    }

    #[test]
    fn replay_0_16_adds_selected_aim_without_reinterpreting_0_15() {
        let aim = AppReplayEvent::Navigate {
            sequence: 1,
            at_seconds: 0.0,
            action: ReplayNavigationAction::AimAtSelection {
                duration_seconds: 0.0,
                easing: TransitionEasing::SmootherStep,
            },
        };
        let events = vec![
            AppReplayEvent::Navigate {
                sequence: 0,
                at_seconds: 0.0,
                action: ReplayNavigationAction::AnchorFocus {
                    asset_id: Some(
                        Uuid::parse_str("60000000-0000-4000-8000-000000000001").unwrap(),
                    ),
                    entity: Uuid::parse_str("70000000-0000-4000-8000-000000000001")
                        .unwrap(),
                    source_bound: ReplayFocusSphere {
                        center: [2.0, 0.0, 0.0],
                        radius: 0.5,
                    },
                    source_pivot: Some([2.0, 0.0, 0.0]),
                    margin: 1.1,
                    duration_seconds: 0.0,
                    easing: TransitionEasing::Linear,
                },
            },
            AppReplayEvent::Frame {
                elapsed_seconds: 0.0,
                delta_seconds: 0.0,
            },
            aim.clone(),
            AppReplayEvent::Frame {
                elapsed_seconds: 0.0,
                delta_seconds: 0.0,
            },
        ];
        let current = run_app_replay(&AppReplayScript::new(events)).unwrap();
        assert!(current.records.iter().all(|record| matches!(
            record.outcome,
            AppReplayOutcome::Committed { .. }
        )));
        assert_eq!(
            current.records[3].state.camera.semantic_target,
            Some([2.0, 0.0, 0.0])
        );
        assert_eq!(current.records[3].state.camera.control_distance, 3.0);

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_15.to_owned(),
            events: vec![aim],
        })
        .unwrap();
        assert!(matches!(
            legacy.records[0].outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("requires replay 0.16")
        ));
        assert_eq!(legacy.records[0].state.revision, 0);
    }

    #[test]
    fn replay_0_10_adds_the_atomic_selected_inversion_gesture() {
        let event = AppReplayEvent::Navigate {
            sequence: 0,
            at_seconds: 0.0,
            action: ReplayNavigationAction::RefitFocusAndToggleInversion {
                duration_seconds: 0.7,
                easing: TransitionEasing::SmootherStep,
            },
        };
        let current = run_app_replay(&AppReplayScript::new(vec![event.clone()])).unwrap();
        assert!(matches!(
            current.records[0].outcome,
            AppReplayOutcome::Committed { .. }
        ));

        let legacy = run_app_replay(&AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_9.to_owned(),
            events: vec![event],
        })
        .unwrap();
        assert!(matches!(
            legacy.records[0].outcome,
            AppReplayOutcome::Rejected { ref error }
                if error.contains("requires replay 0.10")
        ));
        assert_eq!(legacy.records[0].state.revision, 0);
        assert_eq!(
            legacy.records[0].state.reflection,
            ReplayReflection::Identity
        );
    }

    #[test]
    fn replay_0_7_retains_asset_scoped_focus_actions() {
        let script = AppReplayScript {
            version: LEGACY_APP_REPLAY_VERSION_0_7.to_owned(),
            events: vec![
                AppReplayEvent::Navigate {
                    sequence: 0,
                    at_seconds: 0.0,
                    action: ReplayNavigationAction::AnchorFocus {
                        asset_id: Some(Uuid::from_u128(5)),
                        entity: Uuid::from_u128(6),
                        source_bound: ReplayFocusSphere {
                            center: [1.0, 2.0, 3.0],
                            radius: 0.5,
                        },
                        source_pivot: Some([1.1, 2.0, 3.0]),
                        margin: 1.1,
                        duration_seconds: 0.0,
                        easing: TransitionEasing::SmootherStep,
                    },
                },
                AppReplayEvent::Frame {
                    elapsed_seconds: 0.0,
                    delta_seconds: 0.0,
                },
            ],
        };
        let trace = run_app_replay(&script).unwrap();
        assert_eq!(trace.version, APP_REPLAY_VERSION);
        assert_eq!(
            trace.records[1]
                .state
                .focus
                .selected
                .as_ref()
                .unwrap()
                .asset_id,
            Uuid::from_u128(5)
        );
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
        for version in [LEGACY_APP_REPLAY_VERSION_0_4, LEGACY_APP_REPLAY_VERSION_0_5] {
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
                metadata: AssetMetadata::default(),
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
        assert_eq!(trace.records[13].state.authored_assets.len(), 1);
        assert_eq!(
            trace.records[13].state.authored_assets[0].uri,
            "authored-scene.glb"
        );
        assert_eq!(trace.records[13].state.authored_entities.len(), 1);
        assert_eq!(
            trace.records[13].state.authored_entities[0].entity,
            EntityId::new(Uuid::parse_str("e0000000-0000-4000-8000-000000000010").unwrap())
                .unwrap()
        );
        assert_eq!(
            trace.records[13].state.authored_entities[0]
                .transform
                .translation,
            [1.0, 2.0, 3.0]
        );
        assert!(matches!(
            trace.records[14].outcome,
            AppReplayOutcome::Committed {
                disposition: ReplayCommitDisposition::IgnoredStale,
                ..
            }
        ));
        assert_eq!(
            trace.records[14].state.authored_entities,
            trace.records[13].state.authored_entities
        );
        assert!(matches!(
            trace.records[15].outcome,
            AppReplayOutcome::Rejected { ref error } if error.contains("entity ID must not be nil")
        ));
        assert_eq!(
            trace.records[15].state.revision,
            trace.records[14].state.revision
        );
        assert_eq!(
            trace.records[15].state.authored_entities,
            trace.records[14].state.authored_entities
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
