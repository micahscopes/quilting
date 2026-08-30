//! Application authority above Hyperscape's semantic runtime and below any
//! browser, Blender, or persistence adapter.
//!
//! The boundary is `AppEvent -> reducer -> AppCommit + AppEffect`. Effects are
//! values; an adapter performs them and returns an [`EffectCompletion`] event.
//! High-rate frame/navigation/presence events update authoritative state but do
//! not publish UI signals automatically. A view adapter calls
//! [`AppStore::flush_read_models`] at its chosen cadence.

#[cfg(feature = "replay")]
mod replay;
mod animation_pose;
mod peer;
mod patch_lab;
mod presentation_animation;
mod settings;

#[cfg(feature = "replay")]
pub use replay::*;
pub use animation_pose::*;
pub use peer::*;
pub use patch_lab::*;
pub use presentation_animation::*;
pub use settings::*;

use futures_signals::signal::{Mutable, MutableSignalCloned};
use futures_signals::signal_vec::{MutableSignalVec, MutableVec};
use hyperscape::{
    extract_packed_presentation_scene, CameraRig, FocusNavigation, FocusSphere, NavigationAction,
    NavigationController, NavigationPreset, PackedPresentationLayerBinding,
    PackedPresentationNode, PackedPresentationSceneError, Presentation, PresentationAsset,
    PresentationRuntime, PresentationSnapshot, PresentationTessellation, RenderStyle,
    ScheduledNavigationAction, SphereReflectionState,
};
use hyperscape_protocol::{
    AssetDescriptor, AssetEntityId, AssetId, AuthoredCommand, AuthoredEnvelope, EntityId,
    EphemeralPresence, PeerId, PresenceEnvelope, RequestId, WireTransform,
};
pub use quilting_gltf::GltfAssetMetadata as AssetMetadata;
use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};
use uuid::Uuid;

const MAX_DIAGNOSTICS: usize = 256;
const SESSION_NODE_ENTITY_PREFIX: u128 = 0xeeeeeeee_0000_4000_8000_000000000000;

/// Build an application-local identity for one glTF node in a session-scoped
/// asset load.
///
/// The asset ID is allocated by the application load lane, so the pair cannot
/// be confused with durable Blender/authored identity even though it uses the
/// same validated runtime key. The recognizable UUID prefix and exact node
/// payload make the mapping deterministic, allocation-free, and injective for
/// every `u32` glTF node index. Adapters must never promote this pair into an
/// authored command or HHHS history.
pub fn session_node_identity(asset: AssetId, source_node: u32) -> AssetEntityId {
    let entity = EntityId::from_u128(SESSION_NODE_ENTITY_PREFIX | (u128::from(source_node) + 1))
        .expect("the session-node namespace and nonzero node payload form a non-nil UUID");
    AssetEntityId::new(asset, entity)
        .expect("a validated application asset and session entity form a valid identity")
}

#[derive(Debug, Clone, PartialEq)]
pub struct Timed<T> {
    pub sequence: u64,
    pub at_seconds: f64,
    pub value: T,
}

impl<T> Timed<T> {
    fn validate_time(&self) -> Result<(), ReduceError> {
        if !self.at_seconds.is_finite() || self.at_seconds < 0.0 {
            Err(ReduceError::InvalidTime)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SemanticAction {
    Navigate(NavigationAction),
    Present(PresentationAction),
    Animate(AnimationAction),
    SetNavigationSettings(NavigationSettings),
    SetRenderSettings(RenderSettings),
    SetPatchLab(PatchLabSessionIntent),
    RequestAsset {
        request_id: RequestId,
        asset: AssetDescriptor,
        scope: AssetLoadScope,
    },
    CancelAsset(AssetId),
}

/// One atomic, renderer-independent application rendering policy.
///
/// Environment assets and backend resource handles remain adapter concerns;
/// these values determine semantic presentation and tessellation behavior and
/// therefore must not be split across independently ordered browser signals.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "camelCase"))]
pub struct RenderSettings {
    pub style: RenderStyle,
    /// Zero selects adaptive resolution; positive values select `2^level`.
    pub resolution_level: u8,
    pub tessellation: PresentationTessellation,
    pub atlas_exponent: u8,
    pub max_face_edge_ratio: u8,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            style: RenderStyle::Pbr,
            resolution_level: 0,
            tessellation: PresentationTessellation::default(),
            atlas_exponent: 7,
            max_face_edge_ratio: 2,
        }
    }
}

impl RenderSettings {
    /// Decode the backend-neutral control vocabulary into one validated
    /// semantic value. Browser, Leptos, route, and future Blender adapters use
    /// this constructor instead of maintaining parallel style parsing.
    pub fn from_wire_values(
        style: &str,
        resolution_level: u8,
        density: f64,
        screen_attenuation: bool,
        min_pixels_per_subdivision: f64,
        atlas_exponent: u8,
        max_face_edge_ratio: u8,
    ) -> Result<Self, &'static str> {
        Self {
            style: RenderStyle::from_wire_name(style)
                .ok_or("unknown backend-neutral render style")?,
            resolution_level,
            tessellation: PresentationTessellation {
                density,
                screen_attenuation,
                min_pixels_per_subdivision,
            },
            atlas_exponent,
            max_face_edge_ratio,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self, &'static str> {
        if self.resolution_level > 6 {
            return Err("render resolution level must be in [0,6]");
        }
        if !self.tessellation.density.is_finite()
            || !(1.0..=500.0).contains(&self.tessellation.density)
        {
            return Err("tessellation density must be finite and in [1,500]");
        }
        if !self.tessellation.min_pixels_per_subdivision.is_finite()
            || !(1.0..=64.0).contains(&self.tessellation.min_pixels_per_subdivision)
        {
            return Err("pixel floor must be finite and in [1,64]");
        }
        if !(3..=9).contains(&self.atlas_exponent) {
            return Err("resident atlas exponent must be in [3,9]");
        }
        if !matches!(self.max_face_edge_ratio, 2 | 4) {
            return Err("maximum face-edge ratio must be 2 or 4");
        }
        Ok(self)
    }

    fn with_presentation(self, snapshot: &PresentationSnapshot) -> Result<Self, &'static str> {
        Self {
            style: snapshot.render_style,
            tessellation: snapshot.tessellation,
            ..self
        }
        .validate()
    }
}

/// Renderer-independent primary animation clock. The application owns scene
/// time and transport intent; a renderer maps this unwrapped clock into the
/// active clip's authored time range and evaluates the resulting pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimationClock {
    pub playing: bool,
    pub time_seconds: f64,
    pub speed: f64,
}

impl Default for AnimationClock {
    fn default() -> Self {
        Self {
            playing: true,
            time_seconds: 0.0,
            speed: 1.0,
        }
    }
}

impl AnimationClock {
    fn validate(self) -> Result<Self, ReduceError> {
        if !self.time_seconds.is_finite() || !self.speed.is_finite() {
            Err(ReduceError::InvalidAnimationClock)
        } else {
            Ok(self)
        }
    }

    fn advanced(self, delta_seconds: f64) -> Result<Self, ReduceError> {
        if !self.playing {
            return Ok(self);
        }
        Self {
            time_seconds: self.time_seconds + delta_seconds * self.speed,
            ..self
        }
        .validate()
    }

    /// Map the unwrapped scene clock into one resident clip. This keeps loop
    /// semantics backend-independent, including reverse playback across zero.
    pub fn clip_time(self, time_min: f64, duration: f64) -> Option<f64> {
        if !self.time_seconds.is_finite()
            || !time_min.is_finite()
            || !duration.is_finite()
            || duration <= 0.0
        {
            return None;
        }
        Some(time_min + self.time_seconds.rem_euclid(duration))
    }
}

/// Semantic animation transport edits. `SetClock` is the atomic restoration
/// boundary used by presentation cues, routes, and future authored controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnimationAction {
    SetPlaying(bool),
    TogglePlaying,
    Seek(f64),
    SetSpeed(f64),
    SetClock(AnimationClock),
    SelectClip(u32),
}

/// Concurrency policy for an asset acquisition request.
///
/// Presentation layers and other independently composited resources use
/// [`AssetLoadScope::Asset`]. Loads intended to replace the renderer's primary
/// scene use [`AssetLoadScope::PrimaryScene`], which makes a later request
/// cancel the preceding primary-scene request even when their asset IDs differ.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AssetLoadScope {
    #[default]
    Asset,
    PrimaryScene,
}

/// Low-rate presentation intent. Cue activation and the navigation transition
/// it starts commit transactionally inside [`AppState`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentationAction {
    Start,
    Advance,
    Reverse,
    JumpToCue(Uuid),
    Clear,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTick {
    pub elapsed_seconds: f64,
    pub delta_seconds: f64,
}

/// Settled semantic navigation state supplied by a platform adapter before a
/// low-rate authored transition. Process-local focus anchors and in-flight
/// transitions are intentionally not admissible through this boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct NavigationSynchronization {
    pub camera: CameraRig,
    pub focus: FocusNavigation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReceivedPresence {
    pub envelope: PresenceEnvelope,
    /// Local monotonic receipt time. Sender wall clocks never determine TTL.
    pub received_at_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AuthoredRevision {
    pub projection_revision: u64,
    pub commands: Vec<AuthoredEnvelope>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetLoadOutcome {
    Loaded {
        byte_length: usize,
        content_digest: Option<[u8; 32]>,
        metadata: AssetMetadata,
    },
    Failed {
        code: String,
        message: String,
        retryable: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetLoadCompletion {
    pub request_id: RequestId,
    pub asset_id: AssetId,
    pub outcome: AssetLoadOutcome,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrimarySceneInstallOutcome {
    Installed(PrimarySceneInstallMetadata),
    Failed {
        code: String,
        message: String,
        retryable: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrimarySceneInstallCompletion {
    pub request_id: RequestId,
    pub asset_id: AssetId,
    pub outcome: PrimarySceneInstallOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnimationClipSelectionOutcome {
    Selected,
    Failed {
        code: String,
        message: String,
        retryable: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimationClipSelectionCompletion {
    pub job_id: u64,
    pub scene_request_id: RequestId,
    pub asset_id: AssetId,
    pub clip_index: u32,
    pub outcome: AnimationClipSelectionOutcome,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EffectCompletion {
    AssetLoad(AssetLoadCompletion),
    PrimarySceneInstall(PrimarySceneInstallCompletion),
    AnimationClipSelection(AnimationClipSelectionCompletion),
    PatchLab(PatchLabCompletion),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AppEvent {
    Input(Timed<SemanticAction>),
    PresentationLoaded(Presentation),
    /// Process-local evidence connecting an authored presentation asset to an
    /// exact renderer residency. It is never durable or replicated.
    PresentationAnimationResidencyChanged(Option<PresentationAnimationResidencyBinding>),
    NavigationSynchronized(NavigationSynchronization),
    Frame(FrameTick),
    EffectCompleted(EffectCompletion),
    RemotePresence(ReceivedPresence),
    AuthoredRevision(AuthoredRevision),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppEffect {
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
    PatchLab(PatchLabEffect),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitDisposition {
    Applied,
    IgnoredStale,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppCommit {
    pub revision: u64,
    pub effects: Vec<AppEffect>,
    pub disposition: CommitDisposition,
    /// Whether this dispatch published futures-signals read models. `false`
    /// means authoritative state advanced and the adapter may publish later.
    pub published_ui: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetStatus {
    Loading {
        request_id: RequestId,
    },
    Ready {
        byte_length: usize,
        content_digest: Option<[u8; 32]>,
        metadata: AssetMetadata,
    },
    Failed {
        code: String,
        message: String,
        retryable: bool,
    },
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetReadModel {
    pub descriptor: AssetDescriptor,
    pub status: AssetStatus,
}

/// The last primary-scene asset whose load/decode completed successfully.
///
/// A replacement request does not erase this ready candidate while it is in
/// flight or if it fails. Renderer installation is a later asynchronous
/// boundary and must not be inferred from this read model. Keeping the
/// successful request and metadata separate from the mutable asset-load record
/// also represents same-asset reloads without pretending pending bytes are
/// already decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimaryAssetReadModel {
    pub request_id: RequestId,
    pub descriptor: AssetDescriptor,
    pub byte_length: usize,
    pub content_digest: Option<[u8; 32]>,
    pub metadata: AssetMetadata,
}

/// One animation clip made available by an installed primary scene.
///
/// The index is the renderer-facing glTF animation index. Clip times retain
/// their authored range; the application clock remains unwrapped and maps
/// into this range only when extracting a renderer frame.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimationClipDescriptor {
    pub index: u32,
    pub name: String,
    pub time_min_seconds: f64,
    pub time_max_seconds: f64,
}

impl AnimationClipDescriptor {
    pub fn duration_seconds(&self) -> f64 {
        self.time_max_seconds - self.time_min_seconds
    }
}

/// Renderer-independent facts proven only after a decoded primary asset has
/// been uploaded and activated successfully.
///
/// These values deliberately exclude GPU handles and backend-specific batch
/// layouts. They are stable application inputs shared by WebGL2, WebGPU,
/// replay, presentation, and future Blender adapters.
#[derive(Debug, Clone, PartialEq)]
pub struct PrimarySceneInstallMetadata {
    pub num_vertices: u32,
    pub num_faces: u32,
    pub animation_clips: Vec<AnimationClipDescriptor>,
}

impl PrimarySceneInstallMetadata {
    pub fn validate(self) -> Result<Self, PrimarySceneInstallMetadataError> {
        if self.num_vertices == 0 || self.num_faces == 0 {
            return Err(PrimarySceneInstallMetadataError::EmptyGeometry);
        }
        for (expected, clip) in self.animation_clips.iter().enumerate() {
            let expected = u32::try_from(expected)
                .map_err(|_| PrimarySceneInstallMetadataError::TooManyAnimationClips)?;
            if clip.index != expected {
                return Err(PrimarySceneInstallMetadataError::NonCanonicalClipIndex {
                    expected,
                    actual: clip.index,
                });
            }
            if !clip.time_min_seconds.is_finite()
                || !clip.time_max_seconds.is_finite()
                || clip.time_max_seconds < clip.time_min_seconds
            {
                return Err(PrimarySceneInstallMetadataError::InvalidClipRange {
                    index: clip.index,
                });
            }
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimarySceneInstallMetadataError {
    EmptyGeometry,
    TooManyAnimationClips,
    NonCanonicalClipIndex { expected: u32, actual: u32 },
    InvalidClipRange { index: u32 },
}

impl fmt::Display for PrimarySceneInstallMetadataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGeometry => {
                formatter.write_str("installed primary scene must contain vertices and faces")
            }
            Self::TooManyAnimationClips => {
                formatter.write_str("installed primary scene has more than u32::MAX clips")
            }
            Self::NonCanonicalClipIndex { expected, actual } => write!(
                formatter,
                "animation clip index must be canonical: expected {expected}, got {actual}",
            ),
            Self::InvalidClipRange { index } => write!(
                formatter,
                "animation clip {index} must have a finite, nondecreasing authored time range",
            ),
        }
    }
}

impl Error for PrimarySceneInstallMetadataError {}

/// The renderer-resident primary scene. A newer decoded candidate does not
/// replace this read model until its own installation completion is accepted.
#[derive(Debug, Clone, PartialEq)]
pub struct InstalledPrimarySceneReadModel {
    pub asset: PrimaryAssetReadModel,
    pub install: PrimarySceneInstallMetadata,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActiveAnimationClipReadModel {
    pub scene_request_id: RequestId,
    pub asset_id: AssetId,
    pub clip: AnimationClipDescriptor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PendingAnimationClipReadModel {
    pub job_id: u64,
    pub scene_request_id: RequestId,
    pub asset_id: AssetId,
    pub clip: AnimationClipDescriptor,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AnimationClipSelectionReadModel {
    pub active: Option<ActiveAnimationClipReadModel>,
    pub pending: Option<PendingAnimationClipReadModel>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticReadModel {
    pub revision: u64,
    pub code: &'static str,
    pub message: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AppSummary {
    /// This is a commit fence: runtime asset, authored scene, and diagnostic
    /// SignalVec replacements are published before this revision changes.
    pub revision: u64,
    pub authored_projection_revision: Option<u64>,
    pub assets: usize,
    pub loading_assets: usize,
    /// The one in-flight load permitted to replace the renderer's primary
    /// scene. This is ephemeral job state, not rendered-scene identity.
    pub loading_primary_scene_asset: Option<AssetId>,
    pub loading_primary_scene_request: Option<RequestId>,
    pub ready_primary_asset: Option<AssetId>,
    pub ready_primary_request: Option<RequestId>,
    pub installing_primary_scene_asset: Option<AssetId>,
    pub installing_primary_scene_request: Option<RequestId>,
    pub installed_primary_scene_asset: Option<AssetId>,
    pub installed_primary_scene_request: Option<RequestId>,
    pub active_animation_clip: Option<u32>,
    pub pending_animation_clip_job: Option<u64>,
    pub pending_animation_clip: Option<u32>,
    pub active_peers: usize,
    pub authored_assets: usize,
    pub authored_entities: usize,
    pub diagnostics: usize,
    pub presentation_loaded: bool,
    pub active_cue: Option<Uuid>,
    pub animation_playing: bool,
    pub animation_time_seconds: f64,
    pub animation_speed: f64,
    pub patch_lab_active: bool,
    pub patch_lab_geometry_job: Option<u64>,
    pub patch_lab_lod_job: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AppFrameSnapshot {
    pub revision: u64,
    pub elapsed_seconds: f64,
    pub animation: AnimationClock,
    pub active_animation_clip: Option<AnimationClipFrameSnapshot>,
    pub navigation_preset: NavigationPreset,
    pub pending_navigation_actions: usize,
    pub last_applied_navigation_sequence: Option<u64>,
    pub camera: CameraRig,
    pub focus: FocusNavigation,
    /// Selected object and clicked pivot projected into the active output
    /// chart. Source selection survives a projection pole; only the derived
    /// output values become absent.
    pub selected_focus: Option<SelectedFocusSnapshot>,
    pub reflection: SphereReflectionState,
    pub camera_transition_remaining: Option<f64>,
    pub surface_anchor_transition_remaining: Option<f64>,
    pub surface_anchor_hop_height: Option<f64>,
}

/// Allocation-free renderer sample for the application-authoritative active
/// clip. Names remain in the low-rate installed-scene projection; a frame
/// carries only stable identity, authored range, and mapped sample time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimationClipFrameSnapshot {
    pub scene_request_id: RequestId,
    pub asset_id: AssetId,
    pub clip_index: u32,
    pub time_min_seconds: f64,
    pub time_max_seconds: f64,
    pub sample_time_seconds: f64,
}

impl AnimationClipFrameSnapshot {
    fn from_active(active: &ActiveAnimationClipReadModel, clock: AnimationClock) -> Self {
        let clip = &active.clip;
        let duration = clip.duration_seconds();
        let sample_time_seconds = if duration > 0.0 {
            clock
                .clip_time(clip.time_min_seconds, duration)
                .unwrap_or(clip.time_min_seconds)
        } else {
            clip.time_min_seconds
        };
        Self {
            scene_request_id: active.scene_request_id,
            asset_id: active.asset_id,
            clip_index: clip.index,
            time_min_seconds: clip.time_min_seconds,
            time_max_seconds: clip.time_max_seconds,
            sample_time_seconds,
        }
    }
}

/// Low-rate renderer/control projection published before the summary revision
/// fence. Render extraction can sample the same value directly from AppState.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AppRenderSnapshot {
    pub revision: u64,
    pub settings: RenderSettings,
}

/// Low-rate device-independent navigation preference projection. Camera and
/// focus poses remain in [`AppFrameSnapshot`]; raw device policy remains in
/// the platform adapter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AppNavigationSettingsSnapshot {
    pub revision: u64,
    pub settings: NavigationSettings,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectedFocusSnapshot {
    pub identity: AssetEntityId,
    pub source_bound: FocusSphere,
    pub source_pivot: [f64; 3],
    pub margin: f64,
    pub output_pivot: Option<[f64; 3]>,
    pub output_radius: Option<f64>,
}

impl SelectedFocusSnapshot {
    /// Build the selected-focus read model from one authoritative navigation
    /// state. Mapping always starts in the ordinary source chart; changing the
    /// output chart never mutates durable/source selection identity.
    pub fn from_navigation(
        focus: &FocusNavigation,
        reflection: SphereReflectionState,
    ) -> Option<Self> {
        let anchor = focus.anchor?;
        let projected = SphereReflectionState::Identity
            .transport_point_and_directions(reflection, anchor.source_pivot, [])
            .ok()
            .and_then(|transport| {
                let output_radius = anchor.source_bound.radius * transport.local_scale;
                (output_radius.is_finite() && output_radius > 0.0)
                    .then_some((transport.point, output_radius))
            });
        Some(Self {
            identity: anchor.identity,
            source_bound: anchor.source_bound,
            source_pivot: anchor.source_pivot,
            margin: anchor.margin,
            output_pivot: projected.map(|(point, _)| point),
            output_radius: projected.map(|(_, radius)| radius),
        })
    }
}

/// Low-rate FRP projection. Camera/focus motion stays in
/// [`AppFrameSnapshot`], so ticking a transition does not clone cue assets and
/// layers into every render frame.
#[derive(Debug, Clone, PartialEq)]
pub struct PresentationReadModel {
    pub presentation_id: Uuid,
    pub title: String,
    pub cue_count: usize,
    pub assets: Vec<PresentationAsset>,
    pub active: Option<PresentationSnapshot>,
    pub animation_residency: Option<PresentationAnimationResidencyBinding>,
}

/// One coherent application revision of the active presentation composition.
/// Renderer handles enter through [`PackedPresentationLayerBinding`]; all
/// semantic layer and authored transform state is sampled under one lock.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivePresentationSceneReadModel {
    pub revision: u64,
    pub authored_projection_revision: Option<u64>,
    pub cue_id: Uuid,
    pub scene_id: Uuid,
    pub nodes: Vec<PackedPresentationNode>,
    pub unmatched_authored_entities: Vec<EntityId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresentationSceneReadError {
    NoActivePresentation,
    Extraction(PackedPresentationSceneError),
}

impl fmt::Display for PresentationSceneReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoActivePresentation => {
                formatter.write_str("no active presentation cue is available for extraction")
            }
            Self::Extraction(error) => error.fmt(formatter),
        }
    }
}

impl Error for PresentationSceneReadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::NoActivePresentation => None,
            Self::Extraction(error) => Some(error),
        }
    }
}

impl From<PackedPresentationSceneError> for PresentationSceneReadError {
    fn from(error: PackedPresentationSceneError) -> Self {
        Self::Extraction(error)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PeerPresenceReadModel {
    pub peer: PeerId,
    pub sequence: u64,
    pub expires_at_seconds: f64,
    pub presence: EphemeralPresence,
}

/// One durable authored entity projection. The protocol currently exposes
/// transform and removal commands only, so presence in this collection means
/// the latest accepted authored revision retains a transform for the entity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuthoredEntityReadModel {
    pub entity: EntityId,
    pub transform: WireTransform,
}

/// Deterministic low-rate projection of the materialized authored command
/// lane. Assets and entities are key-sorted; the projection revision is the
/// upstream atomic checkpoint fence, distinct from the local app revision.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthoredSceneReadModel {
    pub projection_revision: Option<u64>,
    pub assets: Vec<AssetDescriptor>,
    pub entities: Vec<AuthoredEntityReadModel>,
}

#[derive(Debug, Clone)]
struct AssetRecord {
    descriptor: AssetDescriptor,
    status: AssetStatus,
}

#[derive(Debug, Clone)]
struct PresenceRecord {
    sequence: u64,
    expires_at_seconds: f64,
    presence: EphemeralPresence,
}

/// The reducer's complete mutable state. Fields remain private so adapters
/// cannot bypass event ordering or effect-generation policy.
#[derive(Debug, Clone)]
pub struct AppState {
    revision: u64,
    frame_elapsed_seconds: f64,
    /// Next locally allocated sequence for semantic inputs that commit at the
    /// current application time. Navigation retains its own queue sequence
    /// because presentation activation and navigation synchronization also
    /// participate in that ordering domain.
    next_direct_input_sequence: Option<u64>,
    navigation: NavigationController,
    presentation: Option<PresentationRuntime>,
    active_presentation: Option<PresentationSnapshot>,
    presentation_animation_residency: Option<PresentationAnimationResidencyBinding>,
    animation: AnimationClock,
    navigation_settings: NavigationSettings,
    render_settings: RenderSettings,
    patch_lab: PatchLabRuntime,
    assets: BTreeMap<AssetId, AssetRecord>,
    primary_scene_load: Option<(RequestId, AssetId)>,
    ready_primary_asset: Option<PrimaryAssetReadModel>,
    primary_scene_install: Option<(RequestId, AssetId)>,
    installed_primary_scene: Option<InstalledPrimarySceneReadModel>,
    next_animation_clip_job: Option<u64>,
    active_animation_clip: Option<ActiveAnimationClipReadModel>,
    pending_animation_clip: Option<PendingAnimationClipReadModel>,
    presence: BTreeMap<PeerId, PresenceRecord>,
    authored_projection_revision: Option<u64>,
    authored_assets: BTreeMap<AssetId, AssetDescriptor>,
    authored_entities: BTreeMap<EntityId, WireTransform>,
    diagnostics: VecDeque<DiagnosticReadModel>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            revision: 0,
            frame_elapsed_seconds: 0.0,
            next_direct_input_sequence: Some(0),
            navigation: NavigationController::default(),
            presentation: None,
            active_presentation: None,
            presentation_animation_residency: None,
            animation: AnimationClock::default(),
            navigation_settings: NavigationSettings::default(),
            render_settings: RenderSettings::default(),
            patch_lab: PatchLabRuntime::default(),
            assets: BTreeMap::new(),
            primary_scene_load: None,
            ready_primary_asset: None,
            primary_scene_install: None,
            installed_primary_scene: None,
            next_animation_clip_job: Some(0),
            active_animation_clip: None,
            pending_animation_clip: None,
            presence: BTreeMap::new(),
            authored_projection_revision: None,
            authored_assets: BTreeMap::new(),
            authored_entities: BTreeMap::new(),
            diagnostics: VecDeque::new(),
        }
    }
}

impl AppState {
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn frame_snapshot(&self) -> AppFrameSnapshot {
        let camera_transition_remaining =
            self.navigation.runtime.camera_transition.map(|transition| {
                (transition.duration_seconds - transition.elapsed_seconds).max(0.0)
            });
        let surface_anchor_transition_remaining = self
            .navigation
            .surface_walk
            .anchor_transition()
            .map(|transition| (transition.duration_seconds - transition.elapsed_seconds).max(0.0));
        let selected_focus = SelectedFocusSnapshot::from_navigation(
            &self.navigation.focus,
            self.navigation.runtime.reflection,
        );
        AppFrameSnapshot {
            revision: self.revision,
            elapsed_seconds: self.frame_elapsed_seconds,
            animation: self.animation,
            active_animation_clip: self
                .active_animation_clip
                .as_ref()
                .map(|active| AnimationClipFrameSnapshot::from_active(active, self.animation)),
            navigation_preset: self.navigation.runtime.preset,
            pending_navigation_actions: self.navigation.queue.len(),
            last_applied_navigation_sequence: self.navigation.runtime.last_applied_sequence,
            camera: self.navigation.camera,
            focus: self.navigation.focus.clone(),
            selected_focus,
            reflection: self.navigation.runtime.reflection,
            camera_transition_remaining,
            surface_anchor_transition_remaining,
            surface_anchor_hop_height: self
                .navigation
                .surface_walk
                .anchor_transition()
                .map(|transition| transition.hop_height),
        }
    }

    pub fn render_snapshot(&self) -> AppRenderSnapshot {
        AppRenderSnapshot {
            revision: self.revision,
            settings: self.render_settings,
        }
    }

    pub fn navigation_settings_snapshot(&self) -> AppNavigationSettingsSnapshot {
        AppNavigationSettingsSnapshot {
            revision: self.revision,
            settings: self.navigation_settings,
        }
    }

    pub fn reduce(&mut self, event: AppEvent) -> Result<AppCommit, ReduceError> {
        let direct_input_sequence = match &event {
            AppEvent::Input(Timed {
                value: SemanticAction::Navigate(_),
                ..
            }) => None,
            AppEvent::Input(Timed { sequence, .. }) => Some(*sequence),
            _ => None,
        };
        let mut effects = Vec::new();
        let mut disposition = CommitDisposition::Applied;
        let mut publish_ui = true;

        match event {
            AppEvent::Input(timed) => {
                timed.validate_time()?;
                let input_may_run_now = timed.at_seconds <= self.frame_elapsed_seconds;
                match timed.value {
                    SemanticAction::Navigate(action) => {
                        self.navigation
                            .queue
                            .insert(ScheduledNavigationAction {
                                sequence: timed.sequence,
                                at_seconds: timed.at_seconds,
                                action,
                            })
                            .map_err(ReduceError::Navigation)?;
                        publish_ui = false;
                    }
                    SemanticAction::Present(action) => {
                        if !input_may_run_now {
                            return Err(ReduceError::FuturePresentationInput);
                        }
                        let mut staged = self.clone();
                        staged.apply_presentation_action(action, &mut effects)?;
                        *self = staged;
                    }
                    SemanticAction::Animate(action) => {
                        if !input_may_run_now {
                            return Err(ReduceError::FutureAnimationInput);
                        }
                        match action {
                            AnimationAction::SelectClip(index) => {
                                self.request_animation_clip(index, &mut effects)?;
                            }
                            action => {
                                let animation = match action {
                                    AnimationAction::SetPlaying(playing) => AnimationClock {
                                        playing,
                                        ..self.animation
                                    },
                                    AnimationAction::TogglePlaying => AnimationClock {
                                        playing: !self.animation.playing,
                                        ..self.animation
                                    },
                                    AnimationAction::Seek(time_seconds) => AnimationClock {
                                        time_seconds,
                                        ..self.animation
                                    },
                                    AnimationAction::SetSpeed(speed) => AnimationClock {
                                        speed,
                                        ..self.animation
                                    },
                                    AnimationAction::SetClock(clock) => clock,
                                    AnimationAction::SelectClip(_) => unreachable!(),
                                }
                                .validate()?;
                                self.animation = animation;
                            }
                        }
                    }
                    SemanticAction::SetRenderSettings(settings) => {
                        if !input_may_run_now {
                            return Err(ReduceError::FutureRenderSettingsInput);
                        }
                        let settings = settings
                            .validate()
                            .map_err(ReduceError::InvalidRenderSettings)?;
                        let mut staged = self.clone();
                        let patch_lab_policy_changed = staged.render_settings.atlas_exponent
                            != settings.atlas_exponent
                            || staged.render_settings.max_face_edge_ratio
                                != settings.max_face_edge_ratio;
                        staged.render_settings = settings;
                        if patch_lab_policy_changed {
                            staged
                                .patch_lab
                                .render_settings_changed(settings, &mut effects)?;
                        }
                        *self = staged;
                    }
                    SemanticAction::SetNavigationSettings(settings) => {
                        if !input_may_run_now {
                            return Err(ReduceError::FutureNavigationSettingsInput);
                        }
                        self.navigation_settings = settings
                            .validate()
                            .map_err(ReduceError::InvalidNavigationSettings)?;
                    }
                    SemanticAction::SetPatchLab(intent) => {
                        if !input_may_run_now {
                            return Err(ReduceError::FutureEffectInput);
                        }
                        let mut staged = self.clone();
                        staged.patch_lab.apply_session(
                            intent,
                            staged.render_settings,
                            staged.frame_elapsed_seconds,
                            &mut effects,
                        )?;
                        *self = staged;
                    }
                    SemanticAction::RequestAsset {
                        request_id,
                        asset,
                        scope,
                    } => {
                        if !input_may_run_now {
                            return Err(ReduceError::FutureEffectInput);
                        }
                        request_id
                            .validate()
                            .map_err(|error| ReduceError::Wire(error.to_string()))?;
                        asset
                            .validate()
                            .map_err(|error| ReduceError::Wire(error.to_string()))?;
                        let mut cancelled = Vec::new();
                        if scope == AssetLoadScope::PrimaryScene {
                            if let Some((request_id, asset_id)) =
                                self.primary_scene_install.take()
                            {
                                effects.push(AppEffect::CancelPrimarySceneInstall {
                                    request_id,
                                    asset_id,
                                });
                            }
                            if let Some((previous_request, previous_asset)) =
                                self.primary_scene_load.take()
                            {
                                if let Some(record) = self.assets.get_mut(&previous_asset) {
                                    if record.status
                                        == (AssetStatus::Loading {
                                            request_id: previous_request,
                                        })
                                    {
                                        record.status = AssetStatus::Cancelled;
                                    }
                                }
                                cancelled.push((previous_request, previous_asset));
                            }
                        }
                        if let Some(AssetRecord {
                            status:
                                AssetStatus::Loading {
                                    request_id: previous_request,
                                },
                            ..
                        }) = self.assets.get(&asset.id)
                        {
                            let previous_request = *previous_request;
                            if !cancelled.contains(&(previous_request, asset.id)) {
                                cancelled.push((previous_request, asset.id));
                            }
                            if self.primary_scene_load == Some((previous_request, asset.id)) {
                                self.primary_scene_load = None;
                            }
                        }
                        for (request_id, asset_id) in cancelled {
                            effects.push(AppEffect::CancelAssetLoad {
                                request_id,
                                asset_id,
                            });
                        }
                        self.assets.insert(
                            asset.id,
                            AssetRecord {
                                descriptor: asset.clone(),
                                status: AssetStatus::Loading { request_id },
                            },
                        );
                        if scope == AssetLoadScope::PrimaryScene {
                            self.primary_scene_load = Some((request_id, asset.id));
                        }
                        effects.push(AppEffect::FetchAsset { request_id, asset });
                    }
                    SemanticAction::CancelAsset(asset_id) => {
                        if !input_may_run_now {
                            return Err(ReduceError::FutureEffectInput);
                        }
                        let record = self
                            .assets
                            .get_mut(&asset_id)
                            .ok_or(ReduceError::UnknownAsset(asset_id))?;
                        if let AssetStatus::Loading { request_id } = record.status {
                            effects.push(AppEffect::CancelAssetLoad {
                                request_id,
                                asset_id,
                            });
                            if self.primary_scene_load == Some((request_id, asset_id)) {
                                self.primary_scene_load = None;
                            }
                        }
                        if self
                            .primary_scene_install
                            .is_some_and(|(_, pending_asset)| pending_asset == asset_id)
                        {
                            let (request_id, asset_id) = self
                                .primary_scene_install
                                .take()
                                .expect("matching primary install remains pending");
                            effects.push(AppEffect::CancelPrimarySceneInstall {
                                request_id,
                                asset_id,
                            });
                        }
                        record.status = AssetStatus::Cancelled;
                    }
                }
            }
            AppEvent::PresentationLoaded(presentation) => {
                let presentation = PresentationRuntime::new(presentation)
                    .map_err(|error| ReduceError::Presentation(error.to_string()))?;
                self.presentation = Some(presentation);
                self.active_presentation = None;
                self.presentation_animation_residency = None;
            }
            AppEvent::PresentationAnimationResidencyChanged(binding) => {
                let mut staged = self.clone();
                staged.set_presentation_animation_residency(binding, &mut effects)?;
                *self = staged;
            }
            AppEvent::NavigationSynchronized(synchronization) => {
                self.synchronize_navigation(synchronization)?;
                publish_ui = false;
            }
            AppEvent::Frame(frame) => {
                if !frame.elapsed_seconds.is_finite()
                    || !frame.delta_seconds.is_finite()
                    || frame.elapsed_seconds < self.frame_elapsed_seconds
                    || frame.delta_seconds < 0.0
                {
                    return Err(ReduceError::InvalidTime);
                }
                let next_animation = self.animation.advanced(frame.delta_seconds)?;
                self.navigation
                    .advance_to(frame.elapsed_seconds)
                    .map_err(ReduceError::Navigation)?;
                if let Some(presentation) = self.presentation.as_mut() {
                    presentation.reconcile_navigation(&mut self.navigation);
                }
                self.animation = next_animation;
                self.frame_elapsed_seconds = frame.elapsed_seconds;
                self.patch_lab.advance(
                    frame.elapsed_seconds,
                    self.render_settings,
                    &mut effects,
                )?;
                self.presence
                    .retain(|_, record| record.expires_at_seconds > frame.elapsed_seconds);
                publish_ui = false;
            }
            AppEvent::EffectCompleted(EffectCompletion::AssetLoad(completion)) => {
                let active = self.assets.get(&completion.asset_id).is_some_and(|record| {
                    record.status
                        == AssetStatus::Loading {
                            request_id: completion.request_id,
                        }
                });
                if !active {
                    disposition = CommitDisposition::IgnoredStale;
                    self.push_diagnostic(
                        "stale_effect_completion",
                        format!(
                            "ignored asset completion {} for inactive request {}",
                            completion.asset_id, completion.request_id
                        ),
                    );
                } else {
                    let completed_primary = self.primary_scene_load
                        == Some((completion.request_id, completion.asset_id));
                    if completed_primary {
                        self.primary_scene_load = None;
                    }
                    let descriptor = self
                        .assets
                        .get(&completion.asset_id)
                        .expect("active request implies asset record")
                        .descriptor
                        .clone();
                    let status = match completion.outcome {
                        AssetLoadOutcome::Loaded {
                            byte_length,
                            content_digest,
                            metadata,
                        } => {
                            if completed_primary {
                                self.ready_primary_asset = Some(PrimaryAssetReadModel {
                                    request_id: completion.request_id,
                                    descriptor,
                                    byte_length,
                                    content_digest,
                                    metadata: metadata.clone(),
                                });
                                self.primary_scene_install =
                                    Some((completion.request_id, completion.asset_id));
                                effects.push(AppEffect::InstallPrimaryScene {
                                    request_id: completion.request_id,
                                    asset_id: completion.asset_id,
                                });
                            }
                            AssetStatus::Ready {
                                byte_length,
                                content_digest,
                                metadata,
                            }
                        }
                        AssetLoadOutcome::Failed {
                            code,
                            message,
                            retryable,
                        } => AssetStatus::Failed {
                            code,
                            message,
                            retryable,
                        },
                    };
                    self.assets
                        .get_mut(&completion.asset_id)
                        .expect("active request implies asset record")
                        .status = status;
                }
            }
            AppEvent::EffectCompleted(EffectCompletion::PrimarySceneInstall(completion)) => {
                let active = self.primary_scene_install
                    == Some((completion.request_id, completion.asset_id));
                if !active {
                    disposition = CommitDisposition::IgnoredStale;
                    self.push_diagnostic(
                        "stale_primary_scene_install_completion",
                        format!(
                            "ignored primary scene install completion {} for inactive request {}",
                            completion.asset_id, completion.request_id
                        ),
                    );
                } else {
                    let validated_outcome = match completion.outcome {
                        PrimarySceneInstallOutcome::Installed(metadata) => {
                            PrimarySceneInstallOutcome::Installed(metadata.validate().map_err(
                                ReduceError::InvalidPrimarySceneInstallMetadata,
                            )?)
                        }
                        failed @ PrimarySceneInstallOutcome::Failed { .. } => failed,
                    };
                    self.primary_scene_install = None;
                    match validated_outcome {
                        PrimarySceneInstallOutcome::Installed(install) => {
                            let asset = self
                                .ready_primary_asset
                                .as_ref()
                                .filter(|asset| {
                                    asset.request_id == completion.request_id
                                        && asset.descriptor.id == completion.asset_id
                                })
                                .cloned()
                                .expect("an active primary install has a decoded asset candidate");
                            if let Some(pending) = self.pending_animation_clip.take() {
                                effects.push(AppEffect::CancelAnimationClipSelection {
                                    job_id: pending.job_id,
                                    scene_request_id: pending.scene_request_id,
                                    asset_id: pending.asset_id,
                                    clip_index: pending.clip.index,
                                });
                            }
                            self.active_animation_clip =
                                install.animation_clips.first().cloned().map(|clip| {
                                    ActiveAnimationClipReadModel {
                                        scene_request_id: asset.request_id,
                                        asset_id: asset.descriptor.id,
                                        clip,
                                    }
                                });
                            if self.presentation_animation_residency.is_some_and(|binding| {
                                binding.scene_request_id != asset.request_id
                                    || binding.resident_asset_id != asset.descriptor.id
                            }) {
                                self.presentation_animation_residency = None;
                            }
                            self.installed_primary_scene =
                                Some(InstalledPrimarySceneReadModel { asset, install });
                        }
                        PrimarySceneInstallOutcome::Failed {
                            code,
                            message,
                            retryable,
                        } => self.push_diagnostic(
                            "primary_scene_install_failed",
                            format!(
                                "primary scene {} install failed ({code}, retryable={retryable}): {message}",
                                completion.asset_id
                            ),
                        ),
                    }
                }
            }
            AppEvent::EffectCompleted(EffectCompletion::AnimationClipSelection(completion)) => {
                let active = self.pending_animation_clip.as_ref().is_some_and(|pending| {
                    pending.job_id == completion.job_id
                        && pending.scene_request_id == completion.scene_request_id
                        && pending.asset_id == completion.asset_id
                        && pending.clip.index == completion.clip_index
                });
                if !active {
                    disposition = CommitDisposition::IgnoredStale;
                    self.push_diagnostic(
                        "stale_animation_clip_completion",
                        format!(
                            "ignored animation clip {} completion for inactive job {}",
                            completion.clip_index, completion.job_id
                        ),
                    );
                } else {
                    let pending = self
                        .pending_animation_clip
                        .take()
                        .expect("matching animation clip selection remains pending");
                    match completion.outcome {
                        AnimationClipSelectionOutcome::Selected => {
                            self.active_animation_clip = Some(ActiveAnimationClipReadModel {
                                scene_request_id: pending.scene_request_id,
                                asset_id: pending.asset_id,
                                clip: pending.clip,
                            });
                        }
                        AnimationClipSelectionOutcome::Failed {
                            code,
                            message,
                            retryable,
                        } => self.push_diagnostic(
                            "animation_clip_selection_failed",
                            format!(
                                "animation clip {} selection failed ({code}, retryable={retryable}): {message}",
                                pending.clip.index
                            ),
                        ),
                    }
                }
            }
            AppEvent::EffectCompleted(EffectCompletion::PatchLab(completion)) => {
                match self
                    .patch_lab
                    .complete(completion, self.render_settings, &mut effects)?
                {
                    PatchLabCompletionDisposition::Applied => {}
                    PatchLabCompletionDisposition::Stale(message) => {
                        disposition = CommitDisposition::IgnoredStale;
                        self.push_diagnostic("stale_patch_lab_completion", message);
                    }
                }
            }
            AppEvent::RemotePresence(received) => {
                received
                    .envelope
                    .validate()
                    .map_err(|error| ReduceError::Wire(error.to_string()))?;
                let expires_at_seconds = received
                    .envelope
                    .presence
                    .expires_at_seconds(received.received_at_seconds)
                    .map_err(|error| ReduceError::Wire(error.to_string()))?;
                let sender = received.envelope.header.sender;
                let sequence = received.envelope.header.sequence;
                if self
                    .presence
                    .get(&sender)
                    .is_some_and(|current| current.sequence >= sequence)
                {
                    disposition = CommitDisposition::IgnoredStale;
                } else {
                    self.presence.insert(
                        sender,
                        PresenceRecord {
                            sequence,
                            expires_at_seconds,
                            presence: received.envelope.presence,
                        },
                    );
                }
                publish_ui = false;
            }
            AppEvent::AuthoredRevision(revision) => {
                for command in &revision.commands {
                    command
                        .validate()
                        .map_err(|error| ReduceError::Wire(error.to_string()))?;
                }
                if self
                    .authored_projection_revision
                    .is_some_and(|current| revision.projection_revision <= current)
                {
                    disposition = CommitDisposition::IgnoredStale;
                    self.push_diagnostic(
                        "stale_authored_revision",
                        format!(
                            "ignored authored projection revision {}",
                            revision.projection_revision
                        ),
                    );
                } else {
                    let mut authored_assets = self.authored_assets.clone();
                    let mut authored_entities = self.authored_entities.clone();
                    for envelope in &revision.commands {
                        match &envelope.command {
                            AuthoredCommand::UpsertAsset { asset } => {
                                authored_assets.insert(asset.id, asset.clone());
                            }
                            AuthoredCommand::SetEntityTransform { entity, transform } => {
                                authored_entities.insert(*entity, *transform);
                            }
                            AuthoredCommand::RemoveEntity { entity } => {
                                authored_entities.remove(entity);
                            }
                        }
                    }
                    self.authored_assets = authored_assets;
                    self.authored_entities = authored_entities;
                    self.authored_projection_revision = Some(revision.projection_revision);
                }
            }
        }

        if let Some(sequence) = direct_input_sequence {
            self.next_direct_input_sequence = match (
                self.next_direct_input_sequence,
                sequence.checked_add(1),
            ) {
                (_, None) | (None, _) => None,
                (Some(current), Some(next)) => Some(current.max(next)),
            };
        }
        self.revision = self.revision.saturating_add(1);
        for diagnostic in &mut self.diagnostics {
            if diagnostic.revision == u64::MAX {
                diagnostic.revision = self.revision;
            }
        }
        Ok(AppCommit {
            revision: self.revision,
            effects,
            disposition,
            published_ui: publish_ui,
        })
    }

    fn apply_presentation_action(
        &mut self,
        action: PresentationAction,
        effects: &mut Vec<AppEffect>,
    ) -> Result<(), ReduceError> {
        if action == PresentationAction::Clear {
            self.presentation = None;
            self.active_presentation = None;
            self.presentation_animation_residency = None;
            return Ok(());
        }

        // Presentation activation starts several navigation transitions. Work
        // on clones so any reference, pole, or navigation failure leaves both
        // authorities at the preceding committed revision.
        let mut presentation = self
            .presentation
            .clone()
            .ok_or(ReduceError::NoPresentation)?;
        let mut navigation = self.navigation.clone();
        let snapshot = match action {
            PresentationAction::Start => presentation.activate_index(0, &mut navigation),
            PresentationAction::Advance => presentation.advance(&mut navigation),
            PresentationAction::Reverse => presentation.reverse(&mut navigation),
            PresentationAction::JumpToCue(cue) => presentation.jump_to_cue(cue, &mut navigation),
            PresentationAction::Clear => unreachable!("clear handled before transactional clone"),
        }
        .map_err(|error| ReduceError::Presentation(error.to_string()))?;
        let animation = if let [animation] = snapshot.animations.as_slice() {
            AnimationClock {
                playing: animation.playing,
                time_seconds: animation.time_seconds,
                speed: animation.speed,
            }
            .validate()?
        } else {
            self.animation
        };
        let render_settings = self
            .render_settings
            .with_presentation(&snapshot)
            .map_err(ReduceError::InvalidRenderSettings)?;
        let resolved_animation = self.resolve_presentation_animation(&snapshot)?;
        self.presentation = Some(presentation);
        self.navigation = navigation;
        self.animation = animation;
        self.render_settings = render_settings;
        self.active_presentation = Some(snapshot);
        if let Some(resolved) = resolved_animation {
            self.request_animation_clip(resolved.clip.index, effects)?;
            self.animation = resolved.clock;
        }
        Ok(())
    }

    fn set_presentation_animation_residency(
        &mut self,
        binding: Option<PresentationAnimationResidencyBinding>,
        effects: &mut Vec<AppEffect>,
    ) -> Result<(), ReduceError> {
        let Some(binding) = binding else {
            self.presentation_animation_residency = None;
            return Ok(());
        };
        let presentation = self
            .presentation
            .as_ref()
            .ok_or(ReduceError::NoPresentation)?;
        let asset_exists = presentation.presentation().assets.iter().any(|asset| {
            AssetId::new(asset.id).is_ok_and(|asset_id| asset_id == binding.presentation_asset_id)
        });
        if !asset_exists {
            return Err(ReduceError::UnknownPresentationAnimationAsset(
                binding.presentation_asset_id,
            ));
        }
        let installed = self
            .installed_primary_scene
            .as_ref()
            .ok_or(ReduceError::NoInstalledPrimaryScene)?;
        if binding.scene_request_id != installed.asset.request_id
            || binding.resident_asset_id != installed.asset.descriptor.id
        {
            return Err(ReduceError::PresentationAnimation(
                PresentationAnimationResolutionError::ResidentSceneMismatch {
                    expected_request: binding.scene_request_id,
                    actual_request: installed.asset.request_id,
                    expected_asset: binding.resident_asset_id,
                    actual_asset: installed.asset.descriptor.id,
                }
                .to_string(),
            ));
        }
        let resolved = self
            .active_presentation
            .as_ref()
            .map(|snapshot| resolve_presentation_animation(snapshot, binding, installed))
            .transpose()
            .map_err(|error| ReduceError::PresentationAnimation(error.to_string()))?
            .flatten();
        self.presentation_animation_residency = Some(binding);
        if let Some(resolved) = resolved {
            self.request_animation_clip(resolved.clip.index, effects)?;
            self.animation = resolved.clock;
        }
        Ok(())
    }

    fn resolve_presentation_animation(
        &self,
        snapshot: &PresentationSnapshot,
    ) -> Result<Option<ResolvedPresentationAnimation>, ReduceError> {
        let (Some(binding), Some(installed)) = (
            self.presentation_animation_residency,
            self.installed_primary_scene.as_ref(),
        ) else {
            return Ok(None);
        };
        resolve_presentation_animation(snapshot, binding, installed)
            .map_err(|error| ReduceError::PresentationAnimation(error.to_string()))
    }

    fn synchronize_navigation(
        &mut self,
        synchronization: NavigationSynchronization,
    ) -> Result<(), ReduceError> {
        synchronization
            .camera
            .validate()
            .map_err(|error| ReduceError::NavigationState(error.to_string()))?;
        FocusSphere::new(
            synchronization.focus.sphere.center,
            synchronization.focus.sphere.radius,
        )
        .map_err(|error| ReduceError::NavigationState(error.to_owned()))?;
        if synchronization.focus.anchor.is_some() || synchronization.focus.transition.is_some() {
            return Err(ReduceError::NavigationState(
                "synchronized focus state must be free and settled".to_owned(),
            ));
        }
        if !synchronization.focus.focus_coordinate.is_finite()
            || !(0.0..=1.0).contains(&synchronization.focus.focus_coordinate)
            || !synchronization.focus.angular_aperture.is_finite()
            || synchronization.focus.angular_aperture <= 0.0
        {
            return Err(ReduceError::NavigationState(
                "focus coordinate must be in [0,1] and aperture must be positive".to_owned(),
            ));
        }

        let mut navigation = NavigationController::default();
        navigation
            .advance_to(self.frame_elapsed_seconds)
            .map_err(ReduceError::Navigation)?;
        navigation.camera = synchronization.camera;
        navigation.focus = synchronization.focus;
        navigation.runtime.reflection = if navigation.focus.inversion_enabled {
            SphereReflectionState::Sphere(navigation.focus.sphere)
        } else {
            SphereReflectionState::Identity
        };
        self.navigation = navigation;
        Ok(())
    }

    fn request_animation_clip(
        &mut self,
        index: u32,
        effects: &mut Vec<AppEffect>,
    ) -> Result<(), ReduceError> {
        let scene = self
            .installed_primary_scene
            .as_ref()
            .ok_or(ReduceError::NoInstalledPrimaryScene)?;
        let clip = scene
            .install
            .animation_clips
            .iter()
            .find(|clip| clip.index == index)
            .cloned()
            .ok_or(ReduceError::UnknownAnimationClip(index))?;
        let scene_request_id = scene.asset.request_id;
        let asset_id = scene.asset.descriptor.id;

        if self.pending_animation_clip.as_ref().is_some_and(|pending| {
            pending.scene_request_id == scene_request_id
                && pending.asset_id == asset_id
                && pending.clip.index == index
        }) {
            return Ok(());
        }

        if self.active_animation_clip.as_ref().is_some_and(|active| {
            active.scene_request_id == scene_request_id
                && active.asset_id == asset_id
                && active.clip.index == index
        }) {
            if let Some(pending) = self.pending_animation_clip.take() {
                effects.push(AppEffect::CancelAnimationClipSelection {
                    job_id: pending.job_id,
                    scene_request_id: pending.scene_request_id,
                    asset_id: pending.asset_id,
                    clip_index: pending.clip.index,
                });
            }
            return Ok(());
        }

        let job_id = self
            .next_animation_clip_job
            .ok_or(ReduceError::AnimationClipJobExhausted)?;
        if let Some(pending) = self.pending_animation_clip.take() {
            effects.push(AppEffect::CancelAnimationClipSelection {
                job_id: pending.job_id,
                scene_request_id: pending.scene_request_id,
                asset_id: pending.asset_id,
                clip_index: pending.clip.index,
            });
        }
        self.next_animation_clip_job = job_id.checked_add(1);
        self.pending_animation_clip = Some(PendingAnimationClipReadModel {
            job_id,
            scene_request_id,
            asset_id,
            clip,
        });
        effects.push(AppEffect::SelectAnimationClip {
            job_id,
            scene_request_id,
            asset_id,
            clip_index: index,
        });
        Ok(())
    }

    fn push_diagnostic(&mut self, code: &'static str, message: String) {
        if self.diagnostics.len() == MAX_DIAGNOSTICS {
            self.diagnostics.pop_front();
        }
        self.diagnostics.push_back(DiagnosticReadModel {
            revision: u64::MAX,
            code,
            message,
        });
    }

    fn summary(&self) -> AppSummary {
        let (patch_lab_active, patch_lab_geometry_job, patch_lab_lod_job) =
            self.patch_lab.summary();
        AppSummary {
            revision: self.revision,
            authored_projection_revision: self.authored_projection_revision,
            assets: self.assets.len(),
            loading_assets: self
                .assets
                .values()
                .filter(|record| matches!(record.status, AssetStatus::Loading { .. }))
                .count(),
            loading_primary_scene_asset: self.primary_scene_load.map(|(_, asset)| asset),
            loading_primary_scene_request: self.primary_scene_load.map(|(request, _)| request),
            ready_primary_asset: self
                .ready_primary_asset
                .as_ref()
                .map(|scene| scene.descriptor.id),
            ready_primary_request: self
                .ready_primary_asset
                .as_ref()
                .map(|scene| scene.request_id),
            installing_primary_scene_asset: self
                .primary_scene_install
                .map(|(_, asset)| asset),
            installing_primary_scene_request: self
                .primary_scene_install
                .map(|(request, _)| request),
            installed_primary_scene_asset: self
                .installed_primary_scene
                .as_ref()
                .map(|scene| scene.asset.descriptor.id),
            installed_primary_scene_request: self
                .installed_primary_scene
                .as_ref()
                .map(|scene| scene.asset.request_id),
            active_animation_clip: self
                .active_animation_clip
                .as_ref()
                .map(|active| active.clip.index),
            pending_animation_clip_job: self
                .pending_animation_clip
                .as_ref()
                .map(|pending| pending.job_id),
            pending_animation_clip: self
                .pending_animation_clip
                .as_ref()
                .map(|pending| pending.clip.index),
            active_peers: self.presence.len(),
            authored_assets: self.authored_assets.len(),
            authored_entities: self.authored_entities.len(),
            diagnostics: self.diagnostics.len(),
            presentation_loaded: self.presentation.is_some(),
            active_cue: self
                .active_presentation
                .as_ref()
                .map(|snapshot| snapshot.cue_id),
            animation_playing: self.animation.playing,
            animation_time_seconds: self.animation.time_seconds,
            animation_speed: self.animation.speed,
            patch_lab_active,
            patch_lab_geometry_job,
            patch_lab_lod_job,
        }
    }

    fn presentation_read_model(&self) -> Option<PresentationReadModel> {
        self.presentation.as_ref().map(|runtime| {
            let presentation = runtime.presentation();
            PresentationReadModel {
                presentation_id: presentation.id,
                title: presentation.title.clone(),
                cue_count: presentation.cues.len(),
                assets: presentation.assets.clone(),
                active: self.active_presentation.clone(),
                animation_residency: self.presentation_animation_residency,
            }
        })
    }

    fn asset_read_models(&self) -> Vec<AssetReadModel> {
        self.assets
            .values()
            .map(|record| AssetReadModel {
                descriptor: record.descriptor.clone(),
                status: record.status.clone(),
            })
            .collect()
    }

    fn primary_asset_read_model(&self) -> Option<PrimaryAssetReadModel> {
        self.ready_primary_asset.clone()
    }

    fn installed_primary_scene_read_model(&self) -> Option<InstalledPrimarySceneReadModel> {
        self.installed_primary_scene.clone()
    }

    fn animation_clip_selection_read_model(&self) -> AnimationClipSelectionReadModel {
        AnimationClipSelectionReadModel {
            active: self.active_animation_clip.clone(),
            pending: self.pending_animation_clip.clone(),
        }
    }

    fn diagnostic_read_models(&self) -> Vec<DiagnosticReadModel> {
        self.diagnostics.iter().cloned().collect()
    }

    fn presence_read_models(&self) -> Vec<PeerPresenceReadModel> {
        self.presence
            .iter()
            .map(|(&peer, record)| PeerPresenceReadModel {
                peer,
                sequence: record.sequence,
                expires_at_seconds: record.expires_at_seconds,
                presence: record.presence.clone(),
            })
            .collect()
    }

    fn authored_scene_read_model(&self) -> AuthoredSceneReadModel {
        AuthoredSceneReadModel {
            projection_revision: self.authored_projection_revision,
            assets: self.authored_assets.values().cloned().collect(),
            entities: self
                .authored_entities
                .iter()
                .map(|(&entity, &transform)| AuthoredEntityReadModel { entity, transform })
                .collect(),
        }
    }
}

/// Thread-safe reducer host with futures-signals projections. Signal vectors
/// are projections only; callers cannot mutate the reducer through them.
#[derive(Clone)]
pub struct AppStore {
    state: Arc<Mutex<AppState>>,
    summary: Mutable<AppSummary>,
    navigation: Mutable<AppFrameSnapshot>,
    navigation_settings: Mutable<AppNavigationSettingsSnapshot>,
    render: Mutable<AppRenderSnapshot>,
    patch_lab: Mutable<PatchLabReadModel>,
    assets: MutableVec<AssetReadModel>,
    primary_asset: Mutable<Option<PrimaryAssetReadModel>>,
    installed_primary_scene: Mutable<Option<InstalledPrimarySceneReadModel>>,
    animation_clip_selection: Mutable<AnimationClipSelectionReadModel>,
    authored_assets: MutableVec<AssetDescriptor>,
    authored_entities: MutableVec<AuthoredEntityReadModel>,
    diagnostics: MutableVec<DiagnosticReadModel>,
    presentation: Mutable<Option<PresentationReadModel>>,
}

impl Default for AppStore {
    fn default() -> Self {
        Self::new(AppState::default())
    }
}

impl AppStore {
    pub fn new(state: AppState) -> Self {
        let summary = state.summary();
        let navigation = state.frame_snapshot();
        let navigation_settings = state.navigation_settings_snapshot();
        let render = state.render_snapshot();
        let patch_lab = state.patch_lab.read_model();
        let assets = state.asset_read_models();
        let primary_asset = state.primary_asset_read_model();
        let installed_primary_scene = state.installed_primary_scene_read_model();
        let animation_clip_selection = state.animation_clip_selection_read_model();
        let authored = state.authored_scene_read_model();
        let diagnostics = state.diagnostic_read_models();
        let presentation = state.presentation_read_model();
        Self {
            state: Arc::new(Mutex::new(state)),
            summary: Mutable::new(summary),
            navigation: Mutable::new(navigation),
            navigation_settings: Mutable::new(navigation_settings),
            render: Mutable::new(render),
            patch_lab: Mutable::new(patch_lab),
            assets: MutableVec::new_with_values(assets),
            primary_asset: Mutable::new(primary_asset),
            installed_primary_scene: Mutable::new(installed_primary_scene),
            animation_clip_selection: Mutable::new(animation_clip_selection),
            authored_assets: MutableVec::new_with_values(authored.assets),
            authored_entities: MutableVec::new_with_values(authored.entities),
            diagnostics: MutableVec::new_with_values(diagnostics),
            presentation: Mutable::new(presentation),
        }
    }

    pub fn dispatch(&self, event: AppEvent) -> Result<AppCommit, ReduceError> {
        let commit = self.lock_state().reduce(event)?;
        if commit.published_ui {
            self.flush_read_models();
        }
        Ok(commit)
    }

    /// Dispatch one locally authored semantic action at the current virtual
    /// application time and allocate its sequence under the reducer lock.
    ///
    /// Immediately applied actions share one monotonically increasing direct
    /// input sequence. Explicitly sequenced replay or adapter inputs advance
    /// that allocator when admitted. Navigation delegates to its queue-owned
    /// allocator instead, preserving presentation ordering and the deliberate
    /// reset performed by [`AppEvent::NavigationSynchronized`].
    pub fn dispatch_semantic(
        &self,
        action: SemanticAction,
    ) -> Result<(u64, AppCommit), ReduceError> {
        if let SemanticAction::Navigate(action) = action {
            return self.dispatch_navigation(action);
        }

        let (sequence, commit) = {
            let mut state = self.lock_state();
            let sequence = state
                .next_direct_input_sequence
                .ok_or(ReduceError::InputSequenceExhausted)?;
            let at_seconds = state.frame_elapsed_seconds;
            let commit = state.reduce(AppEvent::Input(Timed {
                sequence,
                at_seconds,
                value: action,
            }))?;
            (sequence, commit)
        };
        if commit.published_ui {
            self.flush_read_models();
        }
        Ok((sequence, commit))
    }

    /// Queue a local semantic navigation action at the current virtual frame
    /// time, allocating its sequence from the same authority used by
    /// presentation and explicitly sequenced replay input.
    ///
    /// The action remains pending until the next navigation integration
    /// boundary: normally a frame dispatch, or a presentation cue activation
    /// whose transactional zero-time tick must apply its own queued actions.
    /// This matches [`NavigationController::push`]. Allocation and insertion
    /// occur under one lock so concurrent adapters cannot race a
    /// snapshot-derived counter.
    pub fn dispatch_navigation(
        &self,
        action: NavigationAction,
    ) -> Result<(u64, AppCommit), ReduceError> {
        let (sequence, commit) = {
            let mut state = self.lock_state();
            let sequence = state.navigation.next_sequence();
            let at_seconds = state.frame_elapsed_seconds;
            let commit = state.reduce(AppEvent::Input(Timed {
                sequence,
                at_seconds,
                value: SemanticAction::Navigate(action),
            }))?;
            (sequence, commit)
        };
        if commit.published_ui {
            self.flush_read_models();
        }
        Ok((sequence, commit))
    }

    /// Publish a coherent read-model batch after a high-rate event burst. The
    /// summary is set last and its revision acts as the consumer commit fence.
    pub fn flush_read_models(&self) -> u64 {
        let state = self.lock_state();
        let assets = state.asset_read_models();
        let primary_asset = state.primary_asset_read_model();
        let installed_primary_scene = state.installed_primary_scene_read_model();
        let animation_clip_selection = state.animation_clip_selection_read_model();
        let authored = state.authored_scene_read_model();
        let diagnostics = state.diagnostic_read_models();
        let presentation = state.presentation_read_model();
        let navigation = state.frame_snapshot();
        let navigation_settings = state.navigation_settings_snapshot();
        let render = state.render_snapshot();
        let patch_lab = state.patch_lab.read_model();
        let summary = state.summary();
        let revision = summary.revision;
        drop(state);
        self.assets.lock_mut().replace_cloned(assets);
        self.primary_asset.set_neq(primary_asset);
        self.installed_primary_scene
            .set_neq(installed_primary_scene);
        self.animation_clip_selection
            .set_neq(animation_clip_selection);
        self.authored_assets
            .lock_mut()
            .replace_cloned(authored.assets);
        self.authored_entities
            .lock_mut()
            .replace_cloned(authored.entities);
        self.diagnostics.lock_mut().replace_cloned(diagnostics);
        self.presentation.set_neq(presentation);
        self.navigation.set_neq(navigation);
        self.navigation_settings.set_neq(navigation_settings);
        self.render.set_neq(render);
        self.patch_lab.set_neq(patch_lab);
        self.summary.set_neq(summary);
        revision
    }

    pub fn frame_snapshot(&self) -> AppFrameSnapshot {
        self.lock_state().frame_snapshot()
    }

    /// Low-rate diagnostics projection kept separate from the render-frame
    /// snapshot so a nonempty diagnostic log is not cloned every frame.
    pub fn navigation_diagnostics_snapshot(&self) -> Vec<String> {
        self.lock_state().navigation.diagnostics.0.clone()
    }

    /// Inspect the high-rate navigation error fence without cloning the
    /// diagnostic log. Input adapters use this around one atomic reducer step
    /// and only copy the final message when that step rejected its intent.
    pub fn navigation_diagnostic_count(&self) -> usize {
        self.lock_state().navigation.diagnostics.0.len()
    }

    pub fn last_navigation_diagnostic(&self) -> Option<String> {
        self.lock_state().navigation.diagnostics.0.last().cloned()
    }

    pub fn summary_snapshot(&self) -> AppSummary {
        self.summary.get_cloned()
    }

    /// Last navigation projection published through the low-rate UI commit
    /// fence. Render and input adapters must use [`Self::frame_snapshot`]
    /// instead so UI throttling cannot delay semantic integration.
    pub fn navigation_snapshot(&self) -> AppFrameSnapshot {
        self.navigation.get_cloned()
    }

    pub fn navigation_settings_snapshot(&self) -> AppNavigationSettingsSnapshot {
        self.navigation_settings.get()
    }

    pub fn render_snapshot(&self) -> AppRenderSnapshot {
        self.render.get()
    }

    pub fn patch_lab_snapshot(&self) -> PatchLabReadModel {
        self.patch_lab.get_cloned()
    }

    pub fn asset_snapshot(&self) -> Vec<AssetReadModel> {
        self.assets.lock_ref().to_vec()
    }

    pub fn primary_asset_snapshot(&self) -> Option<PrimaryAssetReadModel> {
        self.primary_asset.get_cloned()
    }

    pub fn installed_primary_scene_snapshot(&self) -> Option<InstalledPrimarySceneReadModel> {
        self.installed_primary_scene.get_cloned()
    }

    pub fn animation_clip_selection_snapshot(&self) -> AnimationClipSelectionReadModel {
        self.animation_clip_selection.get_cloned()
    }

    pub fn authored_scene_snapshot(&self) -> AuthoredSceneReadModel {
        self.lock_state().authored_scene_read_model()
    }

    pub fn diagnostic_snapshot(&self) -> Vec<DiagnosticReadModel> {
        self.diagnostics.lock_ref().to_vec()
    }

    pub fn presentation_snapshot(&self) -> Option<PresentationReadModel> {
        self.presentation.get_cloned()
    }

    /// Resolve the active cue against resident renderer nodes without
    /// publishing signals or mutating application state. The app revision,
    /// active cue, and authored materialization are sampled under one lock.
    pub fn extract_active_presentation_scene(
        &self,
        bindings: &[PackedPresentationLayerBinding],
    ) -> Result<ActivePresentationSceneReadModel, PresentationSceneReadError> {
        let state = self.lock_state();
        let snapshot = state
            .active_presentation
            .as_ref()
            .ok_or(PresentationSceneReadError::NoActivePresentation)?;
        let extraction = extract_packed_presentation_scene(
            snapshot,
            bindings,
            &state.authored_entities,
        )?;
        Ok(ActivePresentationSceneReadModel {
            revision: state.revision,
            authored_projection_revision: state.authored_projection_revision,
            cue_id: snapshot.cue_id,
            scene_id: snapshot.scene_id,
            nodes: extraction.nodes,
            unmatched_authored_entities: extraction.unmatched_authored_entities,
        })
    }

    /// Presence remains a high-rate snapshot lane rather than an automatic UI
    /// signal. A view may sample it on the same throttle as camera overlays.
    pub fn presence_snapshot(&self) -> Vec<PeerPresenceReadModel> {
        self.lock_state().presence_read_models()
    }

    pub fn summary_signal(&self) -> MutableSignalCloned<AppSummary> {
        self.summary.signal_cloned()
    }

    /// Low-rate FRP navigation projection. The summary revision changes only
    /// after this value and all collection projections have been published.
    pub fn navigation_signal(&self) -> MutableSignalCloned<AppFrameSnapshot> {
        self.navigation.signal_cloned()
    }

    /// Low-rate FRP navigation preferences published before the summary
    /// revision fence. Views emit complete replacement intents through the
    /// reducer instead of mutating individual control signals.
    pub fn navigation_settings_signal(
        &self,
    ) -> MutableSignalCloned<AppNavigationSettingsSnapshot> {
        self.navigation_settings.signal_cloned()
    }

    /// Low-rate FRP render projection published before the summary commit
    /// fence. UI adapters queue semantic changes; they cannot mutate it.
    pub fn render_signal(&self) -> MutableSignalCloned<AppRenderSnapshot> {
        self.render.signal_cloned()
    }

    pub fn patch_lab_signal(&self) -> MutableSignalCloned<PatchLabReadModel> {
        self.patch_lab.signal_cloned()
    }

    pub fn asset_signal_vec(&self) -> MutableSignalVec<AssetReadModel> {
        self.assets.signal_vec_cloned()
    }

    pub fn primary_asset_signal(&self) -> MutableSignalCloned<Option<PrimaryAssetReadModel>> {
        self.primary_asset.signal_cloned()
    }

    pub fn installed_primary_scene_signal(
        &self,
    ) -> MutableSignalCloned<Option<InstalledPrimarySceneReadModel>> {
        self.installed_primary_scene.signal_cloned()
    }

    pub fn animation_clip_selection_signal(
        &self,
    ) -> MutableSignalCloned<AnimationClipSelectionReadModel> {
        self.animation_clip_selection.signal_cloned()
    }

    pub fn authored_asset_signal_vec(&self) -> MutableSignalVec<AssetDescriptor> {
        self.authored_assets.signal_vec_cloned()
    }

    pub fn authored_entity_signal_vec(&self) -> MutableSignalVec<AuthoredEntityReadModel> {
        self.authored_entities.signal_vec_cloned()
    }

    pub fn diagnostic_signal_vec(&self) -> MutableSignalVec<DiagnosticReadModel> {
        self.diagnostics.signal_vec_cloned()
    }

    pub fn presentation_signal(&self) -> MutableSignalCloned<Option<PresentationReadModel>> {
        self.presentation.signal_cloned()
    }

    fn lock_state(&self) -> MutexGuard<'_, AppState> {
        self.state
            .lock()
            .expect("Hyperscope app reducer mutex poisoned")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReduceError {
    InvalidTime,
    InvalidAnimationClock,
    InvalidPrimarySceneInstallMetadata(PrimarySceneInstallMetadataError),
    InvalidNavigationSettings(&'static str),
    InvalidRenderSettings(&'static str),
    InputSequenceExhausted,
    AnimationClipJobExhausted,
    PatchLabJobExhausted,
    FutureEffectInput,
    FuturePresentationInput,
    FutureAnimationInput,
    FutureNavigationSettingsInput,
    FutureRenderSettingsInput,
    Navigation(&'static str),
    NavigationState(String),
    Presentation(String),
    PresentationAnimation(String),
    NoPresentation,
    UnknownPresentationAnimationAsset(AssetId),
    NoInstalledPrimaryScene,
    UnknownAnimationClip(u32),
    Wire(String),
    UnknownAsset(AssetId),
}

impl fmt::Display for ReduceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTime => {
                formatter.write_str("application time must be finite and monotonic")
            }
            Self::InvalidAnimationClock => formatter
                .write_str("animation time and speed must remain finite"),
            Self::InvalidPrimarySceneInstallMetadata(error) => {
                write!(formatter, "primary scene installation failed validation: {error}")
            }
            Self::InvalidNavigationSettings(message) => {
                write!(formatter, "navigation settings failed validation: {message}")
            }
            Self::InvalidRenderSettings(message) => {
                write!(formatter, "render settings failed validation: {message}")
            }
            Self::InputSequenceExhausted => {
                formatter.write_str("local semantic input sequence is exhausted")
            }
            Self::AnimationClipJobExhausted => {
                formatter.write_str("animation clip selection job sequence is exhausted")
            }
            Self::PatchLabJobExhausted => {
                formatter.write_str("patch-lab job sequence is exhausted")
            }
            Self::FutureEffectInput => formatter.write_str(
                "effect-producing input cannot be scheduled beyond the current app frame",
            ),
            Self::FuturePresentationInput => formatter
                .write_str("presentation input cannot be scheduled beyond the current app frame"),
            Self::FutureAnimationInput => formatter
                .write_str("animation input cannot be scheduled beyond the current app frame"),
            Self::FutureNavigationSettingsInput => formatter.write_str(
                "navigation-settings input cannot be scheduled beyond the current app frame",
            ),
            Self::FutureRenderSettingsInput => formatter.write_str(
                "render-settings input cannot be scheduled beyond the current app frame",
            ),
            Self::Navigation(message) => write!(formatter, "navigation event failed: {message}"),
            Self::NavigationState(message) => {
                write!(formatter, "navigation synchronization failed: {message}")
            }
            Self::Presentation(message) => {
                write!(formatter, "presentation event failed: {message}")
            }
            Self::PresentationAnimation(message) => {
                write!(formatter, "presentation animation failed: {message}")
            }
            Self::NoPresentation => formatter.write_str("no presentation is loaded"),
            Self::UnknownPresentationAnimationAsset(asset_id) => write!(
                formatter,
                "presentation has no animation asset {asset_id}",
            ),
            Self::NoInstalledPrimaryScene => {
                formatter.write_str("no renderer-installed primary scene is available")
            }
            Self::UnknownAnimationClip(index) => {
                write!(formatter, "installed primary scene has no animation clip {index}")
            }
            Self::Wire(message) => write!(formatter, "wire value failed validation: {message}"),
            Self::UnknownAsset(asset) => write!(formatter, "unknown asset {asset}"),
        }
    }
}

impl Error for ReduceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidPrimarySceneInstallMetadata(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscape::{CameraRig, NavigationFrame};
    use hyperscape_protocol::{
        AuthoredCommand, CameraPresence, EntityId, MessageHeader, MessageId, ProtocolVersion,
        WireTransform, CURRENT_PROTOCOL_VERSION,
    };

    fn asset(id: u128, uri: &str) -> AssetDescriptor {
        AssetDescriptor {
            id: AssetId::from_u128(id).unwrap(),
            uri: uri.into(),
            media_type: Some("model/gltf-binary".into()),
            content_digest: None,
        }
    }

    fn request(id: u128) -> RequestId {
        RequestId::from_u128(id).unwrap()
    }

    fn authored(sequence: u64, command: AuthoredCommand) -> AuthoredEnvelope {
        AuthoredEnvelope {
            header: MessageHeader {
                version: CURRENT_PROTOCOL_VERSION,
                message_id: MessageId::from_u128(u128::from(sequence) + 1_000).unwrap(),
                sender: PeerId::from_u128(2_000).unwrap(),
                sequence,
            },
            command,
        }
    }

    fn transform(x: f64) -> WireTransform {
        WireTransform {
            translation: [x, x + 1.0, x + 2.0],
            rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }

    fn selection_identity_for(asset: u128, entity: u128) -> AssetEntityId {
        AssetEntityId::new(
            AssetId::from_u128(asset).unwrap(),
            EntityId::from_u128(entity).unwrap(),
        )
        .unwrap()
    }

    fn selection_identity() -> AssetEntityId {
        selection_identity_for(0x6000, 0x7000)
    }

    #[test]
    fn session_node_identities_are_asset_scoped_injective_and_recognizable() {
        let first_asset = AssetId::from_u128(0x6000).unwrap();
        let second_asset = AssetId::from_u128(0x6001).unwrap();
        let first = session_node_identity(first_asset, 0);
        let later = session_node_identity(first_asset, u32::MAX);
        let other_asset = session_node_identity(second_asset, 0);

        assert_eq!(first.asset, first_asset);
        assert_eq!(other_asset.asset, second_asset);
        assert_eq!(
            first.entity.as_uuid().as_u128(),
            SESSION_NODE_ENTITY_PREFIX | 1
        );
        assert_eq!(
            later.entity.as_uuid().as_u128(),
            SESSION_NODE_ENTITY_PREFIX | (u128::from(u32::MAX) + 1),
        );
        assert_ne!(first.entity, later.entity);
        assert_eq!(first.entity, other_asset.entity);
        assert_ne!(first, other_asset);
    }

    #[test]
    fn primary_scene_install_metadata_requires_canonical_renderer_facts() {
        let metadata = PrimarySceneInstallMetadata {
            num_vertices: 796,
            num_faces: 984,
            animation_clips: vec![
                AnimationClipDescriptor {
                    index: 0,
                    name: "idle".to_owned(),
                    time_min_seconds: 0.0,
                    time_max_seconds: 1.5,
                },
                AnimationClipDescriptor {
                    index: 1,
                    name: String::new(),
                    time_min_seconds: 2.0,
                    time_max_seconds: 2.0,
                },
            ],
        };
        assert_eq!(metadata.clone().validate(), Ok(metadata));
        assert_eq!(
            PrimarySceneInstallMetadata {
                num_vertices: 0,
                num_faces: 984,
                animation_clips: Vec::new(),
            }
            .validate(),
            Err(PrimarySceneInstallMetadataError::EmptyGeometry),
        );
        assert_eq!(
            PrimarySceneInstallMetadata {
                num_vertices: 3,
                num_faces: 1,
                animation_clips: vec![AnimationClipDescriptor {
                    index: 1,
                    name: "misindexed".to_owned(),
                    time_min_seconds: 0.0,
                    time_max_seconds: 1.0,
                }],
            }
            .validate(),
            Err(PrimarySceneInstallMetadataError::NonCanonicalClipIndex {
                expected: 0,
                actual: 1,
            }),
        );
        assert_eq!(
            PrimarySceneInstallMetadata {
                num_vertices: 3,
                num_faces: 1,
                animation_clips: vec![AnimationClipDescriptor {
                    index: 0,
                    name: "broken".to_owned(),
                    time_min_seconds: 1.0,
                    time_max_seconds: 0.0,
                }],
            }
            .validate(),
            Err(PrimarySceneInstallMetadataError::InvalidClipRange { index: 0 }),
        );
    }

    fn presentation_fixture() -> Presentation {
        Presentation::from_json(hyperscape::HACKER_NIGHT_PRESENTATION_JSON)
        .unwrap()
    }

    #[test]
    fn render_settings_are_atomic_validated_and_revision_fenced() {
        let store = AppStore::default();
        assert_eq!(store.render_snapshot().settings, RenderSettings::default());
        let settings = RenderSettings {
            style: RenderStyle::Stretch,
            resolution_level: 6,
            tessellation: PresentationTessellation {
                density: 237.5,
                screen_attenuation: false,
                min_pixels_per_subdivision: 63.5,
            },
            atlas_exponent: 9,
            max_face_edge_ratio: 4,
        };

        let commit = store
            .dispatch(AppEvent::Input(Timed {
                sequence: 1,
                at_seconds: 0.0,
                value: SemanticAction::SetRenderSettings(settings),
            }))
            .unwrap();
        let render = store.render_snapshot();
        assert_eq!(render.revision, commit.revision);
        assert_eq!(render.revision, store.summary_snapshot().revision);
        assert_eq!(render.settings, settings);

        let before = render;
        let invalid = RenderSettings {
            tessellation: PresentationTessellation {
                density: f64::NAN,
                ..settings.tessellation
            },
            ..settings
        };
        assert_eq!(
            store.dispatch(AppEvent::Input(Timed {
                sequence: 2,
                at_seconds: 0.0,
                value: SemanticAction::SetRenderSettings(invalid),
            })),
            Err(ReduceError::InvalidRenderSettings(
                "tessellation density must be finite and in [1,500]",
            )),
        );
        assert_eq!(store.render_snapshot(), before);
        assert_eq!(store.summary_snapshot().revision, before.revision);

        assert_eq!(
            store.dispatch(AppEvent::Input(Timed {
                sequence: 3,
                at_seconds: 1.0,
                value: SemanticAction::SetRenderSettings(settings),
            })),
            Err(ReduceError::FutureRenderSettingsInput),
        );
        assert_eq!(store.render_snapshot(), before);
    }

    #[test]
    fn navigation_settings_are_atomic_validated_and_revision_fenced() {
        let store = AppStore::default();
        assert_eq!(
            store.navigation_settings_snapshot().settings,
            NavigationSettings::default(),
        );
        let settings = NavigationSettings {
            transition_seconds: 2.5,
            surface_walk: hyperscape::SurfaceWalkControls {
                speed_octave_steps: 100.0,
                body_scale_octave_steps: -50.0,
                eye_height_octave_steps: 25.0,
                smoothing_seconds: 0.45,
                tangent_pull_fraction: 0.25,
                ..Default::default()
            },
        };
        let (_, commit) = store
            .dispatch_semantic(SemanticAction::SetNavigationSettings(settings))
            .unwrap();
        let snapshot = store.navigation_settings_snapshot();
        assert_eq!(snapshot.revision, commit.revision);
        assert_eq!(snapshot.revision, store.summary_snapshot().revision);
        assert_eq!(snapshot.settings, settings);

        let before = snapshot;
        let invalid = NavigationSettings {
            transition_seconds: f64::NAN,
            ..settings
        };
        assert_eq!(
            store.dispatch_semantic(SemanticAction::SetNavigationSettings(invalid)),
            Err(ReduceError::InvalidNavigationSettings(
                "navigation transition duration must be finite and in [0,5]",
            )),
        );
        assert_eq!(store.navigation_settings_snapshot(), before);

        assert_eq!(
            store.dispatch(AppEvent::Input(Timed {
                sequence: 9,
                at_seconds: 1.0,
                value: SemanticAction::SetNavigationSettings(settings),
            })),
            Err(ReduceError::FutureNavigationSettingsInput),
        );
        assert_eq!(store.navigation_settings_snapshot(), before);
    }

    #[test]
    fn cue_activation_atomically_updates_only_authored_render_policy() {
        let store = AppStore::default();
        store
            .dispatch(AppEvent::PresentationLoaded(presentation_fixture()))
            .unwrap();
        let session = RenderSettings {
            style: RenderStyle::Stretch,
            resolution_level: 6,
            tessellation: PresentationTessellation {
                density: 12.0,
                screen_attenuation: false,
                min_pixels_per_subdivision: 64.0,
            },
            atlas_exponent: 9,
            max_face_edge_ratio: 4,
        };
        store
            .dispatch(AppEvent::Input(Timed {
                sequence: 1,
                at_seconds: 0.0,
                value: SemanticAction::SetRenderSettings(session),
            }))
            .unwrap();
        let commit = store
            .dispatch(AppEvent::Input(Timed {
                sequence: 2,
                at_seconds: 0.0,
                value: SemanticAction::Present(PresentationAction::Start),
            }))
            .unwrap();

        let active = store.presentation_snapshot().unwrap().active.unwrap();
        let render = store.render_snapshot();
        assert_eq!(render.revision, commit.revision);
        assert_eq!(render.settings.style, active.render_style);
        assert_eq!(render.settings.tessellation, active.tessellation);
        assert_eq!(render.settings.resolution_level, session.resolution_level);
        assert_eq!(render.settings.atlas_exponent, session.atlas_exponent);
        assert_eq!(
            render.settings.max_face_edge_ratio,
            session.max_face_edge_ratio,
        );
    }

    #[test]
    fn animation_playback_is_reducer_owned_and_cue_initialized() {
        let store = AppStore::default();
        assert!(store.summary_snapshot().animation_playing);
        assert_eq!(store.summary_snapshot().animation_time_seconds, 0.0);
        assert_eq!(store.summary_snapshot().animation_speed, 1.0);

        store
            .dispatch(AppEvent::Input(Timed {
                sequence: 1,
                at_seconds: 0.0,
                value: SemanticAction::Animate(AnimationAction::SetPlaying(false)),
            }))
            .unwrap();
        assert!(!store.summary_snapshot().animation_playing);

        store
            .dispatch(AppEvent::Input(Timed {
                sequence: 2,
                at_seconds: 0.0,
                value: SemanticAction::Animate(AnimationAction::TogglePlaying),
            }))
            .unwrap();
        assert!(store.summary_snapshot().animation_playing);

        let rejected = store.dispatch(AppEvent::Input(Timed {
            sequence: 3,
            at_seconds: 1.0,
            value: SemanticAction::Animate(AnimationAction::SetPlaying(false)),
        }));
        assert_eq!(rejected, Err(ReduceError::FutureAnimationInput));
        assert!(store.summary_snapshot().animation_playing);

        let mut presentation = presentation_fixture();
        presentation.cues[0].animations[0].playing = false;
        presentation.cues[0].animations[0].time_seconds = 0.375;
        presentation.cues[0].animations[0].speed = -0.5;
        store
            .dispatch(AppEvent::PresentationLoaded(presentation))
            .unwrap();
        dispatch_presentation(&store, 4, 0.0, PresentationAction::Start).unwrap();
        let summary = store.summary_snapshot();
        assert!(!summary.animation_playing);
        assert_eq!(summary.animation_time_seconds, 0.375);
        assert_eq!(summary.animation_speed, -0.5);
    }

    #[test]
    fn animation_clock_is_atomic_and_frame_partition_invariant() {
        fn clock_after(deltas: &[f64]) -> AnimationClock {
            let store = AppStore::default();
            store
                .dispatch(AppEvent::Input(Timed {
                    sequence: 1,
                    at_seconds: 0.0,
                    value: SemanticAction::Animate(AnimationAction::SetClock(
                        AnimationClock {
                            playing: true,
                            time_seconds: 2.0,
                            speed: -0.5,
                        },
                    )),
                }))
                .unwrap();
            let mut elapsed = 0.0;
            for delta_seconds in deltas {
                elapsed += delta_seconds;
                store
                    .dispatch(AppEvent::Frame(FrameTick {
                        elapsed_seconds: elapsed,
                        delta_seconds: *delta_seconds,
                    }))
                    .unwrap();
            }
            store.frame_snapshot().animation
        }

        assert_eq!(clock_after(&[1.0]), clock_after(&[0.25, 0.25, 0.5]));
        assert_eq!(clock_after(&[1.0]).time_seconds, 1.5);
        assert_eq!(
            AnimationClock {
                time_seconds: -0.25,
                ..AnimationClock::default()
            }
            .clip_time(3.0, 2.0),
            Some(4.75),
        );
        assert_eq!(AnimationClock::default().clip_time(0.0, 0.0), None);

        let store = AppStore::default();
        let before = store.frame_snapshot().animation;
        let rejected = store.dispatch(AppEvent::Input(Timed {
            sequence: 1,
            at_seconds: 0.0,
            value: SemanticAction::Animate(AnimationAction::SetSpeed(f64::NAN)),
        }));
        assert_eq!(rejected, Err(ReduceError::InvalidAnimationClock));
        assert_eq!(store.frame_snapshot().animation, before);
    }

    fn dispatch_presentation(
        store: &AppStore,
        sequence: u64,
        at_seconds: f64,
        action: PresentationAction,
    ) -> Result<AppCommit, ReduceError> {
        store.dispatch(AppEvent::Input(Timed {
            sequence,
            at_seconds,
            value: SemanticAction::Present(action),
        }))
    }

    fn dispatch_request(
        store: &AppStore,
        sequence: u64,
        request_id: RequestId,
        asset: AssetDescriptor,
    ) -> AppCommit {
        dispatch_scoped_request(
            store,
            sequence,
            request_id,
            asset,
            AssetLoadScope::Asset,
        )
    }

    fn dispatch_scoped_request(
        store: &AppStore,
        sequence: u64,
        request_id: RequestId,
        asset: AssetDescriptor,
        scope: AssetLoadScope,
    ) -> AppCommit {
        store
            .dispatch(AppEvent::Input(Timed {
                sequence,
                at_seconds: 0.0,
                value: SemanticAction::RequestAsset {
                    request_id,
                    asset,
                    scope,
                },
            }))
            .unwrap()
    }

    fn install_metadata() -> PrimarySceneInstallMetadata {
        PrimarySceneInstallMetadata {
            num_vertices: 796,
            num_faces: 984,
            animation_clips: vec![
                AnimationClipDescriptor {
                    index: 0,
                    name: "gallop".to_owned(),
                    time_min_seconds: 0.0,
                    time_max_seconds: 1.5,
                },
                AnimationClipDescriptor {
                    index: 1,
                    name: "turn".to_owned(),
                    time_min_seconds: 2.0,
                    time_max_seconds: 3.0,
                },
            ],
        }
    }

    #[test]
    fn reducer_emits_effect_then_rejects_a_stale_completion() {
        let store = AppStore::default();
        let horse = asset(1, "horse.glb");
        let first = request(10);
        let second = request(11);
        let first_commit = dispatch_request(&store, 1, first, horse.clone());
        assert_eq!(
            first_commit.effects,
            vec![AppEffect::FetchAsset {
                request_id: first,
                asset: horse.clone(),
            }]
        );
        let replacement = dispatch_request(&store, 2, second, horse.clone());
        assert_eq!(
            replacement.effects,
            vec![
                AppEffect::CancelAssetLoad {
                    request_id: first,
                    asset_id: horse.id,
                },
                AppEffect::FetchAsset {
                    request_id: second,
                    asset: horse.clone(),
                },
            ]
        );

        let stale = store
            .dispatch(AppEvent::EffectCompleted(EffectCompletion::AssetLoad(
                AssetLoadCompletion {
                    request_id: first,
                    asset_id: horse.id,
                    outcome: AssetLoadOutcome::Loaded {
                        byte_length: 100,
                        content_digest: None,
                        metadata: AssetMetadata::default(),
                    },
                },
            )))
            .unwrap();
        assert_eq!(stale.disposition, CommitDisposition::IgnoredStale);
        assert_eq!(
            store.asset_snapshot()[0].status,
            AssetStatus::Loading { request_id: second }
        );
        assert_eq!(
            store.diagnostic_snapshot()[0].code,
            "stale_effect_completion"
        );
    }

    #[test]
    fn primary_scene_requests_cancel_across_asset_ids_and_only_latest_can_settle() {
        let store = AppStore::default();
        let horse = asset(1, "horse.glb");
        let chess = asset(2, "chess.glb");
        let first = request(10);
        let second = request(11);

        dispatch_scoped_request(
            &store,
            1,
            first,
            horse.clone(),
            AssetLoadScope::PrimaryScene,
        );
        let replacement = dispatch_scoped_request(
            &store,
            2,
            second,
            chess.clone(),
            AssetLoadScope::PrimaryScene,
        );
        assert_eq!(
            replacement.effects,
            vec![
                AppEffect::CancelAssetLoad {
                    request_id: first,
                    asset_id: horse.id,
                },
                AppEffect::FetchAsset {
                    request_id: second,
                    asset: chess.clone(),
                },
            ]
        );
        assert_eq!(
            store.summary_snapshot().loading_primary_scene_asset,
            Some(chess.id)
        );
        assert_eq!(
            store.summary_snapshot().loading_primary_scene_request,
            Some(second)
        );

        let stale = store
            .dispatch(AppEvent::EffectCompleted(EffectCompletion::AssetLoad(
                AssetLoadCompletion {
                    request_id: first,
                    asset_id: horse.id,
                    outcome: AssetLoadOutcome::Loaded {
                        byte_length: 100,
                        content_digest: None,
                        metadata: AssetMetadata::default(),
                    },
                },
            )))
            .unwrap();
        assert_eq!(stale.disposition, CommitDisposition::IgnoredStale);
        assert_eq!(store.primary_asset_snapshot(), None);
        assert_eq!(
            store.summary_snapshot().loading_primary_scene_request,
            Some(second)
        );

        let applied = store
            .dispatch(AppEvent::EffectCompleted(EffectCompletion::AssetLoad(
                AssetLoadCompletion {
                    request_id: second,
                    asset_id: chess.id,
                    outcome: AssetLoadOutcome::Loaded {
                        byte_length: 200,
                        content_digest: None,
                        metadata: AssetMetadata::default(),
                    },
                },
            )))
            .unwrap();
        assert_eq!(applied.disposition, CommitDisposition::Applied);
        assert_eq!(
            applied.effects,
            vec![AppEffect::InstallPrimaryScene {
                request_id: second,
                asset_id: chess.id,
            }]
        );
        assert_eq!(store.summary_snapshot().loading_primary_scene_asset, None);
        assert_eq!(store.summary_snapshot().ready_primary_asset, Some(chess.id));
        assert_eq!(store.summary_snapshot().ready_primary_request, Some(second));
        assert_eq!(
            store.summary_snapshot().installing_primary_scene_asset,
            Some(chess.id)
        );
        assert_eq!(store.installed_primary_scene_snapshot(), None);
        assert_eq!(
            store.primary_asset_snapshot(),
            Some(PrimaryAssetReadModel {
                request_id: second,
                descriptor: chess.clone(),
                byte_length: 200,
                content_digest: None,
                metadata: AssetMetadata::default(),
            })
        );

        let assets = store.asset_snapshot();
        assert_eq!(assets[0].status, AssetStatus::Cancelled);
        assert_eq!(
            assets[1].status,
            AssetStatus::Ready {
                byte_length: 200,
                content_digest: None,
                metadata: AssetMetadata::default(),
            }
        );

        let installed = store
            .dispatch(AppEvent::EffectCompleted(
                EffectCompletion::PrimarySceneInstall(PrimarySceneInstallCompletion {
                    request_id: second,
                    asset_id: chess.id,
                    outcome: PrimarySceneInstallOutcome::Installed(install_metadata()),
                }),
            ))
            .unwrap();
        assert_eq!(installed.disposition, CommitDisposition::Applied);
        assert_eq!(
            store.summary_snapshot().installed_primary_scene_asset,
            Some(chess.id)
        );
        assert_eq!(
            store.installed_primary_scene_snapshot(),
            Some(InstalledPrimarySceneReadModel {
                asset: store.primary_asset_snapshot().unwrap(),
                install: install_metadata(),
            })
        );
    }

    #[test]
    fn failed_primary_asset_replacement_preserves_the_ready_candidate() {
        let store = AppStore::default();
        let horse = asset(1, "horse.glb");
        let chess = asset(2, "chess.glb");
        let horse_request = request(10);
        let chess_request = request(11);

        dispatch_scoped_request(
            &store,
            1,
            horse_request,
            horse.clone(),
            AssetLoadScope::PrimaryScene,
        );
        store
            .dispatch(AppEvent::EffectCompleted(EffectCompletion::AssetLoad(
                AssetLoadCompletion {
                    request_id: horse_request,
                    asset_id: horse.id,
                    outcome: AssetLoadOutcome::Loaded {
                        byte_length: 100,
                        content_digest: Some([7; 32]),
                        metadata: AssetMetadata::default(),
                    },
                },
            )))
            .unwrap();
        let resident = store.primary_asset_snapshot().unwrap();

        let replacement = dispatch_scoped_request(
            &store,
            2,
            chess_request,
            chess.clone(),
            AssetLoadScope::PrimaryScene,
        );
        assert_eq!(
            replacement.effects,
            vec![
                AppEffect::CancelPrimarySceneInstall {
                    request_id: horse_request,
                    asset_id: horse.id,
                },
                AppEffect::FetchAsset {
                    request_id: chess_request,
                    asset: chess.clone(),
                },
            ]
        );
        assert_eq!(store.primary_asset_snapshot(), Some(resident.clone()));
        store
            .dispatch(AppEvent::EffectCompleted(EffectCompletion::AssetLoad(
                AssetLoadCompletion {
                    request_id: chess_request,
                    asset_id: chess.id,
                    outcome: AssetLoadOutcome::Failed {
                        code: "parse".to_owned(),
                        message: "invalid glTF".to_owned(),
                        retryable: false,
                    },
                },
            )))
            .unwrap();

        assert_eq!(store.primary_asset_snapshot(), Some(resident));
        assert_eq!(store.summary_snapshot().loading_primary_scene_asset, None);
        assert_eq!(store.summary_snapshot().ready_primary_asset, Some(horse.id));
        assert_eq!(
            store.summary_snapshot().ready_primary_request,
            Some(horse_request)
        );
        assert_eq!(
            store.summary_snapshot().installing_primary_scene_request,
            None
        );
    }

    #[test]
    fn animation_clip_jobs_are_catalog_validated_cancellable_and_stale_safe() {
        let store = AppStore::default();
        let before = store.summary_snapshot().revision;
        assert_eq!(
            store.dispatch_semantic(SemanticAction::Animate(
                AnimationAction::SelectClip(0),
            )),
            Err(ReduceError::NoInstalledPrimaryScene),
        );
        assert_eq!(store.summary_snapshot().revision, before);

        let horse = asset(1, "horse.glb");
        let request_id = request(10);
        dispatch_scoped_request(
            &store,
            1,
            request_id,
            horse.clone(),
            AssetLoadScope::PrimaryScene,
        );
        store
            .dispatch(AppEvent::EffectCompleted(EffectCompletion::AssetLoad(
                AssetLoadCompletion {
                    request_id,
                    asset_id: horse.id,
                    outcome: AssetLoadOutcome::Loaded {
                        byte_length: 100,
                        content_digest: None,
                        metadata: AssetMetadata::default(),
                    },
                },
            )))
            .unwrap();
        store
            .dispatch(AppEvent::EffectCompleted(
                EffectCompletion::PrimarySceneInstall(PrimarySceneInstallCompletion {
                    request_id,
                    asset_id: horse.id,
                    outcome: PrimarySceneInstallOutcome::Installed(install_metadata()),
                }),
            ))
            .unwrap();
        assert_eq!(store.summary_snapshot().active_animation_clip, Some(0));
        assert_eq!(
            store.frame_snapshot().active_animation_clip,
            Some(AnimationClipFrameSnapshot {
                scene_request_id: request_id,
                asset_id: horse.id,
                clip_index: 0,
                time_min_seconds: 0.0,
                time_max_seconds: 1.5,
                sample_time_seconds: 0.0,
            }),
        );
        let installed_revision = store.summary_snapshot().revision;
        assert_eq!(
            store.dispatch_semantic(SemanticAction::Animate(
                AnimationAction::SelectClip(99),
            )),
            Err(ReduceError::UnknownAnimationClip(99)),
        );
        assert_eq!(store.summary_snapshot().revision, installed_revision);

        let (_, first) = store
            .dispatch_semantic(SemanticAction::Animate(AnimationAction::SelectClip(1)))
            .unwrap();
        assert_eq!(
            first.effects,
            vec![AppEffect::SelectAnimationClip {
                job_id: 0,
                scene_request_id: request_id,
                asset_id: horse.id,
                clip_index: 1,
            }]
        );
        assert_eq!(store.summary_snapshot().pending_animation_clip_job, Some(0));
        assert_eq!(
            store
                .frame_snapshot()
                .active_animation_clip
                .unwrap()
                .clip_index,
            0,
            "pending renderer work must not change the sampled active clip",
        );
        let (_, duplicate) = store
            .dispatch_semantic(SemanticAction::Animate(AnimationAction::SelectClip(1)))
            .unwrap();
        assert!(duplicate.effects.is_empty());

        let (_, back_to_active) = store
            .dispatch_semantic(SemanticAction::Animate(AnimationAction::SelectClip(0)))
            .unwrap();
        assert_eq!(
            back_to_active.effects,
            vec![AppEffect::CancelAnimationClipSelection {
                job_id: 0,
                scene_request_id: request_id,
                asset_id: horse.id,
                clip_index: 1,
            }]
        );
        assert_eq!(store.summary_snapshot().pending_animation_clip_job, None);

        let (_, second) = store
            .dispatch_semantic(SemanticAction::Animate(AnimationAction::SelectClip(1)))
            .unwrap();
        assert_eq!(
            second.effects,
            vec![AppEffect::SelectAnimationClip {
                job_id: 1,
                scene_request_id: request_id,
                asset_id: horse.id,
                clip_index: 1,
            }]
        );
        let stale = store
            .dispatch(AppEvent::EffectCompleted(
                EffectCompletion::AnimationClipSelection(AnimationClipSelectionCompletion {
                    job_id: 0,
                    scene_request_id: request_id,
                    asset_id: horse.id,
                    clip_index: 1,
                    outcome: AnimationClipSelectionOutcome::Selected,
                }),
            ))
            .unwrap();
        assert_eq!(stale.disposition, CommitDisposition::IgnoredStale);
        assert_eq!(store.summary_snapshot().pending_animation_clip_job, Some(1));

        store
            .dispatch(AppEvent::EffectCompleted(
                EffectCompletion::AnimationClipSelection(AnimationClipSelectionCompletion {
                    job_id: 1,
                    scene_request_id: request_id,
                    asset_id: horse.id,
                    clip_index: 1,
                    outcome: AnimationClipSelectionOutcome::Failed {
                        code: "worker".to_owned(),
                        message: "clip upload failed".to_owned(),
                        retryable: true,
                    },
                }),
            ))
            .unwrap();
        assert_eq!(store.summary_snapshot().active_animation_clip, Some(0));
        assert_eq!(store.summary_snapshot().pending_animation_clip_job, None);
        assert_eq!(
            store
                .frame_snapshot()
                .active_animation_clip
                .unwrap()
                .clip_index,
            0,
        );

        let (_, third) = store
            .dispatch_semantic(SemanticAction::Animate(AnimationAction::SelectClip(1)))
            .unwrap();
        assert_eq!(
            third.effects,
            vec![AppEffect::SelectAnimationClip {
                job_id: 2,
                scene_request_id: request_id,
                asset_id: horse.id,
                clip_index: 1,
            }]
        );
        store
            .dispatch(AppEvent::EffectCompleted(
                EffectCompletion::AnimationClipSelection(AnimationClipSelectionCompletion {
                    job_id: 2,
                    scene_request_id: request_id,
                    asset_id: horse.id,
                    clip_index: 1,
                    outcome: AnimationClipSelectionOutcome::Selected,
                }),
            ))
            .unwrap();
        let selection = store.animation_clip_selection_snapshot();
        assert_eq!(selection.active.unwrap().clip.index, 1);
        assert_eq!(selection.pending, None);
        assert_eq!(
            store.frame_snapshot().active_animation_clip,
            Some(AnimationClipFrameSnapshot {
                scene_request_id: request_id,
                asset_id: horse.id,
                clip_index: 1,
                time_min_seconds: 2.0,
                time_max_seconds: 3.0,
                sample_time_seconds: 2.0,
            }),
        );
        store
            .dispatch_semantic(SemanticAction::Animate(AnimationAction::SetClock(
                AnimationClock {
                    playing: true,
                    time_seconds: -0.25,
                    speed: -1.0,
                },
            )))
            .unwrap();
        assert_eq!(
            store
                .frame_snapshot()
                .active_animation_clip
                .unwrap()
                .sample_time_seconds,
            2.75,
            "the installed clip frame must retain reverse Euclidean wrapping",
        );
    }

    #[test]
    fn presentation_animation_residency_defers_identity_and_dispatches_exact_clip_jobs() {
        let store = AppStore::default();
        store
            .dispatch(AppEvent::PresentationLoaded(presentation_fixture()))
            .unwrap();
        dispatch_presentation(&store, 1, 0.0, PresentationAction::Start).unwrap();

        let resident = asset(0x991, "cached-horse.glb");
        let scene_request = request(0x992);
        dispatch_scoped_request(
            &store,
            2,
            scene_request,
            resident.clone(),
            AssetLoadScope::PrimaryScene,
        );
        store
            .dispatch(AppEvent::EffectCompleted(EffectCompletion::AssetLoad(
                AssetLoadCompletion {
                    request_id: scene_request,
                    asset_id: resident.id,
                    outcome: AssetLoadOutcome::Loaded {
                        byte_length: 100,
                        content_digest: None,
                        metadata: AssetMetadata::default(),
                    },
                },
            )))
            .unwrap();
        store
            .dispatch(AppEvent::EffectCompleted(
                EffectCompletion::PrimarySceneInstall(PrimarySceneInstallCompletion {
                    request_id: scene_request,
                    asset_id: resident.id,
                    outcome: PrimarySceneInstallOutcome::Installed(
                        PrimarySceneInstallMetadata {
                            num_vertices: 796,
                            num_faces: 984,
                            animation_clips: vec![
                                AnimationClipDescriptor {
                                    index: 0,
                                    name: "idle".to_owned(),
                                    time_min_seconds: 0.0,
                                    time_max_seconds: 1.0,
                                },
                                AnimationClipDescriptor {
                                    index: 1,
                                    name: "horse_A_".to_owned(),
                                    time_min_seconds: 2.0,
                                    time_max_seconds: 3.5,
                                },
                            ],
                        },
                    ),
                }),
            ))
            .unwrap();

        let presentation_asset_id = AssetId::new(
            Uuid::parse_str("a0000000-0000-4000-8000-000000000001").unwrap(),
        )
        .unwrap();
        assert_ne!(presentation_asset_id, resident.id);
        let binding = PresentationAnimationResidencyBinding {
            presentation_asset_id,
            scene_request_id: scene_request,
            resident_asset_id: resident.id,
        };
        let bound = store
            .dispatch(AppEvent::PresentationAnimationResidencyChanged(Some(
                binding,
            )))
            .unwrap();
        assert_eq!(
            bound.effects,
            vec![AppEffect::SelectAnimationClip {
                job_id: 0,
                scene_request_id: scene_request,
                asset_id: resident.id,
                clip_index: 1,
            }],
        );
        assert_eq!(
            store
                .presentation_snapshot()
                .unwrap()
                .animation_residency,
            Some(binding),
        );
        assert_eq!(store.summary_snapshot().pending_animation_clip, Some(1));

        // The next cue uses a different layer instance of the same authored
        // horse asset. Asset-level residency remains valid and the already
        // pending exact clip job is not duplicated.
        let next = dispatch_presentation(&store, 3, 0.0, PresentationAction::Advance).unwrap();
        assert!(next.effects.is_empty());
        assert_eq!(store.summary_snapshot().pending_animation_clip, Some(1));
    }

    #[test]
    fn rejected_presentation_animation_binding_is_atomic() {
        let store = AppStore::default();
        store
            .dispatch(AppEvent::PresentationLoaded(presentation_fixture()))
            .unwrap();
        let before = store.summary_snapshot().revision;
        let presentation_asset_id = AssetId::new(
            Uuid::parse_str("a0000000-0000-4000-8000-000000000001").unwrap(),
        )
        .unwrap();
        assert_eq!(
            store.dispatch(AppEvent::PresentationAnimationResidencyChanged(Some(
                PresentationAnimationResidencyBinding {
                    presentation_asset_id,
                    scene_request_id: request(0x993),
                    resident_asset_id: AssetId::from_u128(0x994).unwrap(),
                },
            ))),
            Err(ReduceError::NoInstalledPrimaryScene),
        );
        assert_eq!(store.summary_snapshot().revision, before);
        assert_eq!(
            store
                .presentation_snapshot()
                .unwrap()
                .animation_residency,
            None,
        );
    }

    #[test]
    fn failed_or_invalid_install_preserves_the_resident_scene_atomically() {
        let store = AppStore::default();
        let horse = asset(1, "horse.glb");
        let chess = asset(2, "chess.glb");
        let horse_request = request(10);
        let chess_request = request(11);

        dispatch_scoped_request(
            &store,
            1,
            horse_request,
            horse.clone(),
            AssetLoadScope::PrimaryScene,
        );
        store
            .dispatch(AppEvent::EffectCompleted(EffectCompletion::AssetLoad(
                AssetLoadCompletion {
                    request_id: horse_request,
                    asset_id: horse.id,
                    outcome: AssetLoadOutcome::Loaded {
                        byte_length: 100,
                        content_digest: None,
                        metadata: AssetMetadata::default(),
                    },
                },
            )))
            .unwrap();
        store
            .dispatch(AppEvent::EffectCompleted(
                EffectCompletion::PrimarySceneInstall(PrimarySceneInstallCompletion {
                    request_id: horse_request,
                    asset_id: horse.id,
                    outcome: PrimarySceneInstallOutcome::Installed(install_metadata()),
                }),
            ))
            .unwrap();
        let resident = store.installed_primary_scene_snapshot().unwrap();

        dispatch_scoped_request(
            &store,
            2,
            chess_request,
            chess.clone(),
            AssetLoadScope::PrimaryScene,
        );
        store
            .dispatch(AppEvent::EffectCompleted(EffectCompletion::AssetLoad(
                AssetLoadCompletion {
                    request_id: chess_request,
                    asset_id: chess.id,
                    outcome: AssetLoadOutcome::Loaded {
                        byte_length: 200,
                        content_digest: None,
                        metadata: AssetMetadata::default(),
                    },
                },
            )))
            .unwrap();
        let revision_before_invalid = store.summary_snapshot().revision;
        let invalid = store.dispatch(AppEvent::EffectCompleted(
            EffectCompletion::PrimarySceneInstall(PrimarySceneInstallCompletion {
                request_id: chess_request,
                asset_id: chess.id,
                outcome: PrimarySceneInstallOutcome::Installed(PrimarySceneInstallMetadata {
                    num_vertices: 0,
                    num_faces: 1,
                    animation_clips: Vec::new(),
                }),
            }),
        ));
        assert_eq!(
            invalid,
            Err(ReduceError::InvalidPrimarySceneInstallMetadata(
                PrimarySceneInstallMetadataError::EmptyGeometry,
            ))
        );
        assert_eq!(store.summary_snapshot().revision, revision_before_invalid);
        assert_eq!(
            store.installed_primary_scene_snapshot(),
            Some(resident.clone())
        );

        store
            .dispatch(AppEvent::EffectCompleted(
                EffectCompletion::PrimarySceneInstall(PrimarySceneInstallCompletion {
                    request_id: chess_request,
                    asset_id: chess.id,
                    outcome: PrimarySceneInstallOutcome::Failed {
                        code: "gpu_upload".to_owned(),
                        message: "allocation rejected".to_owned(),
                        retryable: true,
                    },
                }),
            ))
            .unwrap();
        assert_eq!(
            store.installed_primary_scene_snapshot(),
            Some(resident.clone())
        );
        assert_eq!(
            store.summary_snapshot().installed_primary_scene_asset,
            Some(horse.id)
        );
        assert_eq!(
            store.diagnostic_snapshot().last().unwrap().code,
            "primary_scene_install_failed"
        );

        let stale_invalid = store
            .dispatch(AppEvent::EffectCompleted(
                EffectCompletion::PrimarySceneInstall(PrimarySceneInstallCompletion {
                    request_id: chess_request,
                    asset_id: chess.id,
                    outcome: PrimarySceneInstallOutcome::Installed(
                        PrimarySceneInstallMetadata {
                            num_vertices: 0,
                            num_faces: 0,
                            animation_clips: Vec::new(),
                        },
                    ),
                }),
            ))
            .unwrap();
        assert_eq!(stale_invalid.disposition, CommitDisposition::IgnoredStale);
        assert_eq!(store.installed_primary_scene_snapshot(), Some(resident));
        assert_eq!(
            store.diagnostic_snapshot().last().unwrap().code,
            "stale_primary_scene_install_completion"
        );
    }

    #[test]
    fn per_asset_requests_remain_parallel_for_presentation_composition() {
        let store = AppStore::default();
        let horse = asset(1, "horse.glb");
        let chess = asset(2, "chess.glb");

        let first = dispatch_request(&store, 1, request(10), horse.clone());
        let second = dispatch_request(&store, 2, request(11), chess.clone());
        assert_eq!(first.effects.len(), 1);
        assert_eq!(
            second.effects,
            vec![AppEffect::FetchAsset {
                request_id: request(11),
                asset: chess,
            }]
        );
        assert_eq!(store.summary_snapshot().loading_assets, 2);
        assert_eq!(store.summary_snapshot().loading_primary_scene_asset, None);
    }

    #[test]
    fn navigation_replay_is_cadence_independent_through_the_app_boundary() {
        fn run(frames: &[f64]) -> CameraRig {
            let store = AppStore::default();
            store
                .dispatch(AppEvent::Input(Timed {
                    sequence: 1,
                    at_seconds: 0.25,
                    value: SemanticAction::Navigate(NavigationAction::ApplyFrame(
                        NavigationFrame {
                            translation: [2.0, 0.0, 0.0],
                            ..NavigationFrame::default()
                        },
                    )),
                }))
                .unwrap();
            let mut elapsed = 0.0;
            for delta in frames {
                elapsed += delta;
                store
                    .dispatch(AppEvent::Frame(FrameTick {
                        elapsed_seconds: elapsed,
                        delta_seconds: *delta,
                    }))
                    .unwrap();
            }
            store.frame_snapshot().camera
        }

        assert_eq!(run(&[1.0]), run(&[0.1; 10]));
    }

    #[test]
    fn local_navigation_matches_controller_queue_semantics_before_and_after_tick() {
        let store = AppStore::default();
        let mut controller = NavigationController::default();
        let action = NavigationAction::SetPreset(NavigationPreset::Fly);

        let (app_sequence, commit) = store.dispatch_navigation(action.clone()).unwrap();
        let controller_sequence = controller.push(action).unwrap();
        assert_eq!(app_sequence, controller_sequence);
        assert!(!commit.published_ui);

        let queued = store.frame_snapshot();
        assert_eq!(queued.pending_navigation_actions, controller.queue.len());
        assert_eq!(queued.last_applied_navigation_sequence, None);
        assert_eq!(queued.navigation_preset, controller.runtime.preset);

        store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: 0.0,
                delta_seconds: 0.0,
            }))
            .unwrap();
        controller.tick(0.0).unwrap();
        let applied = store.frame_snapshot();
        assert_eq!(applied.pending_navigation_actions, controller.queue.len());
        assert_eq!(
            applied.last_applied_navigation_sequence,
            controller.runtime.last_applied_sequence
        );
        assert_eq!(applied.navigation_preset, controller.runtime.preset);
        assert_eq!(applied.camera, controller.camera);
        assert_eq!(applied.focus, controller.focus);
    }

    #[test]
    fn direct_semantic_dispatch_allocates_atomically_without_stealing_navigation_order() {
        let store = AppStore::default();
        let first_settings = RenderSettings {
            resolution_level: 1,
            ..RenderSettings::default()
        };
        let (first_sequence, first_commit) = store
            .dispatch_semantic(SemanticAction::SetRenderSettings(first_settings))
            .unwrap();
        assert_eq!(first_sequence, 0);
        assert!(first_commit.published_ui);

        store
            .dispatch(AppEvent::Input(Timed {
                sequence: 7,
                at_seconds: 0.0,
                value: SemanticAction::Animate(AnimationAction::SetPlaying(false)),
            }))
            .unwrap();
        let (next_sequence, _) = store
            .dispatch_semantic(SemanticAction::Animate(AnimationAction::TogglePlaying))
            .unwrap();
        assert_eq!(next_sequence, 8);

        let invalid = RenderSettings {
            tessellation: PresentationTessellation {
                density: f64::NAN,
                ..PresentationTessellation::default()
            },
            ..RenderSettings::default()
        };
        assert!(matches!(
            store.dispatch_semantic(SemanticAction::SetRenderSettings(invalid)),
            Err(ReduceError::InvalidRenderSettings(_)),
        ));
        let (after_rejection, _) = store
            .dispatch_semantic(SemanticAction::SetRenderSettings(RenderSettings::default()))
            .unwrap();
        assert_eq!(after_rejection, 9);

        let (navigation_sequence, navigation_commit) = store
            .dispatch_semantic(SemanticAction::Navigate(NavigationAction::SetPreset(
                NavigationPreset::Fly,
            )))
            .unwrap();
        assert_eq!(navigation_sequence, 0);
        assert!(!navigation_commit.published_ui);
    }

    #[test]
    fn direct_semantic_sequence_exhaustion_is_explicit() {
        let store = AppStore::default();
        store
            .dispatch(AppEvent::Input(Timed {
                sequence: u64::MAX,
                at_seconds: 0.0,
                value: SemanticAction::SetRenderSettings(RenderSettings::default()),
            }))
            .unwrap();

        assert_eq!(
            store.dispatch_semantic(SemanticAction::Animate(AnimationAction::TogglePlaying)),
            Err(ReduceError::InputSequenceExhausted),
        );
        assert_eq!(store.summary_snapshot().revision, 1);
    }

    #[test]
    fn app_navigation_pole_consumes_input_without_splitting_focus_and_reflection() {
        let store = AppStore::default();
        let camera = CameraRig::default();
        let focus = FocusNavigation {
            sphere: FocusSphere::new(camera.eye, 2.0).unwrap(),
            ..FocusNavigation::default()
        };
        store
            .dispatch(AppEvent::NavigationSynchronized(
                NavigationSynchronization {
                    camera,
                    focus: focus.clone(),
                },
            ))
            .unwrap();
        let before = store.frame_snapshot();

        let (sequence, queued) = store
            .dispatch_navigation(NavigationAction::SetInversionEnabled(true))
            .unwrap();
        assert_eq!(sequence, 0);
        assert!(!queued.published_ui);
        store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: 0.0,
                delta_seconds: 0.0,
            }))
            .unwrap();

        let after = store.frame_snapshot();
        assert_eq!(after.camera, before.camera);
        assert_eq!(after.focus, before.focus);
        assert_eq!(after.reflection, before.reflection);
        assert_eq!(after.pending_navigation_actions, 0);
        assert_eq!(after.last_applied_navigation_sequence, Some(sequence));
        assert!(store.navigation_diagnostics_snapshot()[0]
            .contains("camera transport reached a spherical-reflection pole"));
    }

    fn apply_navigation_now(store: &AppStore, action: NavigationAction) -> u64 {
        let (sequence, _) = store.dispatch_navigation(action).unwrap();
        let elapsed_seconds = store.frame_snapshot().elapsed_seconds;
        store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds,
                delta_seconds: 0.0,
            }))
            .unwrap();
        sequence
    }

    fn selected_focus_action(source_pivot: [f64; 3]) -> NavigationAction {
        NavigationAction::AnchorFocus {
            identity: selection_identity(),
            source_bound: FocusSphere::new([0.0; 3], 2.0).unwrap(),
            source_pivot,
            margin: 1.0,
            duration_seconds: 0.0,
            easing: hyperscape::TransitionEasing::SmootherStep,
        }
    }

    #[test]
    fn frame_snapshot_projects_selection_in_identity_and_sphere_charts() {
        let store = AppStore::default();
        let anchor_sequence = apply_navigation_now(&store, selected_focus_action([4.0, 0.0, 0.0]));
        let identity = store.frame_snapshot();
        let selected = identity.selected_focus.unwrap();
        assert_eq!(selected.identity, selection_identity());
        assert_eq!(selected.source_pivot, [4.0, 0.0, 0.0]);
        assert_eq!(selected.output_pivot, Some([4.0, 0.0, 0.0]));
        assert_eq!(selected.output_radius, Some(2.0));

        let inversion_sequence =
            apply_navigation_now(&store, NavigationAction::SetInversionEnabled(true));
        let reflected = store.frame_snapshot();
        let selected = reflected.selected_focus.unwrap();
        assert_eq!(selected.source_pivot, [4.0, 0.0, 0.0]);
        assert_eq!(selected.output_pivot, Some([1.0, 0.0, 0.0]));
        assert_eq!(selected.output_radius, Some(0.5));
        assert_eq!(anchor_sequence, 0);
        assert_eq!(inversion_sequence, 1);
        assert_eq!(reflected.last_applied_navigation_sequence, Some(1));
    }

    #[test]
    fn selection_identity_commits_on_the_app_frame_and_keeps_asset_scope() {
        let store = AppStore::default();
        let source_bound = FocusSphere::new([0.0; 3], 2.0).unwrap();
        let action = |identity| NavigationAction::AnchorFocus {
            identity,
            source_bound,
            source_pivot: [1.0, 0.0, 0.0],
            margin: 1.1,
            duration_seconds: 0.0,
            easing: hyperscape::TransitionEasing::SmootherStep,
        };

        let first = selection_identity_for(0x6000, 0x7000);
        assert_eq!(store.dispatch_navigation(action(first)).unwrap().0, 0);
        let pending = store.frame_snapshot();
        assert_eq!(pending.pending_navigation_actions, 1);
        assert_eq!(pending.selected_focus, None);
        store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: 0.0,
                delta_seconds: 0.0,
            }))
            .unwrap();
        assert_eq!(
            store.frame_snapshot().selected_focus.unwrap().identity,
            first
        );

        let second_asset = selection_identity_for(0x6001, 0x7000);
        assert_eq!(
            store.dispatch_navigation(action(second_asset)).unwrap().0,
            1
        );
        store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: 0.0,
                delta_seconds: 0.0,
            }))
            .unwrap();
        assert_eq!(
            store.frame_snapshot().selected_focus.unwrap().identity,
            second_asset,
        );
        assert_ne!(first, second_asset);
    }

    #[test]
    fn selection_at_reflection_pole_preserves_source_and_clears_output() {
        let store = AppStore::default();
        apply_navigation_now(&store, selected_focus_action([0.0; 3]));
        apply_navigation_now(&store, NavigationAction::SetInversionEnabled(true));

        let frame = store.frame_snapshot();
        let selected = frame.selected_focus.unwrap();
        assert_eq!(selected.source_pivot, [0.0; 3]);
        assert_eq!(
            selected.source_bound,
            FocusSphere::new([0.0; 3], 2.0).unwrap()
        );
        assert_eq!(selected.output_pivot, None);
        assert_eq!(selected.output_radius, None);
        assert!(frame.focus.inversion_enabled);
        assert!(matches!(frame.reflection, SphereReflectionState::Sphere(_)));
    }

    #[test]
    fn nonfinite_projected_radius_is_not_exposed_to_adapters() {
        let mut focus = FocusNavigation::default();
        focus
            .anchor_to_pivot_with_easing(
                selection_identity(),
                FocusSphere::new([0.0; 3], 2.0).unwrap(),
                [2.0e-154, 0.0, 0.0],
                1.0,
                0.0,
                hyperscape::TransitionEasing::SmootherStep,
            )
            .unwrap();

        let selected = SelectedFocusSnapshot::from_navigation(
            &focus,
            SphereReflectionState::Sphere(focus.sphere),
        )
        .unwrap();
        assert_eq!(selected.output_pivot, None);
        assert_eq!(selected.output_radius, None);
    }

    #[test]
    fn underflowed_projected_radius_is_not_exposed_to_adapters() {
        let mut focus = FocusNavigation::default();
        focus
            .anchor_to_pivot_with_easing(
                selection_identity(),
                FocusSphere::new([0.0; 3], f64::from_bits(1)).unwrap(),
                [1.0e154, 0.0, 0.0],
                1.0,
                0.0,
                hyperscape::TransitionEasing::SmootherStep,
            )
            .unwrap();

        let selected = SelectedFocusSnapshot::from_navigation(
            &focus,
            SphereReflectionState::Sphere(focus.sphere),
        )
        .unwrap();
        assert_eq!(selected.output_pivot, None);
        assert_eq!(selected.output_radius, None);
    }

    #[test]
    fn detach_focus_clears_selection_without_resetting_chart() {
        let store = AppStore::default();
        apply_navigation_now(&store, selected_focus_action([4.0, 0.0, 0.0]));
        apply_navigation_now(&store, NavigationAction::SetInversionEnabled(true));
        let before = store.frame_snapshot();

        apply_navigation_now(&store, NavigationAction::DetachFocus);
        let detached = store.frame_snapshot();

        assert_eq!(detached.selected_focus, None);
        assert_eq!(detached.focus.sphere, before.focus.sphere);
        assert_eq!(detached.focus.focus_enabled, before.focus.focus_enabled);
        assert_eq!(
            detached.focus.inversion_enabled,
            before.focus.inversion_enabled
        );
        assert_eq!(detached.reflection, before.reflection);
    }

    #[test]
    fn presentation_and_local_navigation_share_sequence_and_integration_authority() {
        let store = AppStore::default();
        store
            .dispatch(AppEvent::PresentationLoaded(presentation_fixture()))
            .unwrap();
        dispatch_presentation(&store, 1, 0.0, PresentationAction::Start).unwrap();

        let after_start = store.frame_snapshot();
        let presentation_sequence = after_start
            .last_applied_navigation_sequence
            .expect("cue activation applies navigation actions");
        let (local_sequence, _) = store
            .dispatch_navigation(NavigationAction::SetPreset(NavigationPreset::Fly))
            .unwrap();
        assert_eq!(local_sequence, presentation_sequence + 1);
        let queued = store.frame_snapshot();
        assert_eq!(queued.pending_navigation_actions, 1);
        assert_eq!(
            queued.last_applied_navigation_sequence,
            Some(presentation_sequence)
        );

        // Cue activation is also a navigation integration boundary: it must
        // apply its own zero-time actions and therefore drains an already-due
        // local action in the same sequence order as NavigationController.
        dispatch_presentation(&store, 2, 0.0, PresentationAction::Advance).unwrap();
        let after_advance = store.frame_snapshot();
        assert_eq!(after_advance.pending_navigation_actions, 0);
        assert_eq!(after_advance.navigation_preset, NavigationPreset::Fly);
        let advanced_sequence = after_advance
            .last_applied_navigation_sequence
            .expect("cue advance applies navigation actions");
        assert!(advanced_sequence > local_sequence);
        let (post_presentation_sequence, _) = store
            .dispatch_navigation(NavigationAction::ToggleInversion)
            .unwrap();
        assert_eq!(post_presentation_sequence, advanced_sequence + 1);

        store
            .dispatch(AppEvent::NavigationSynchronized(
                NavigationSynchronization {
                    camera: CameraRig::default(),
                    focus: FocusNavigation::default(),
                },
            ))
            .unwrap();
        let (resynchronized_sequence, _) = store
            .dispatch_navigation(NavigationAction::SetFocusEnabled(true))
            .unwrap();
        assert_eq!(resynchronized_sequence, 0);
        assert_eq!(store.frame_snapshot().pending_navigation_actions, 1);
        assert_eq!(
            store.frame_snapshot().last_applied_navigation_sequence,
            None
        );
    }

    #[test]
    fn navigation_synchronization_is_validated_atomic_and_clock_preserving() {
        let store = AppStore::default();
        store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: 2.0,
                delta_seconds: 2.0,
            }))
            .unwrap();
        let camera = CameraRig {
            eye: [1.0, 2.0, 4.0],
            semantic_target: Some([1.0, 2.0, 0.0]),
            control_distance: 4.0,
            lens: hyperscape::PerspectiveLens {
                vertical_fov_radians: 1.25,
                near: 0.002,
                far: 25_000.0,
            },
            ..CameraRig::default()
        };
        let focus = FocusNavigation {
            sphere: FocusSphere::new([0.25, -0.5, 0.75], 1.5).unwrap(),
            focus_enabled: true,
            inversion_enabled: true,
            focus_coordinate: 0.4,
            angular_aperture: 0.08,
            ..FocusNavigation::default()
        };

        let commit = store
            .dispatch(AppEvent::NavigationSynchronized(
                NavigationSynchronization {
                    camera,
                    focus: focus.clone(),
                },
            ))
            .unwrap();
        assert!(!commit.published_ui);
        let synchronized = store.frame_snapshot();
        assert_eq!(synchronized.elapsed_seconds, 2.0);
        assert_eq!(synchronized.camera, camera);
        assert_eq!(synchronized.focus, focus);
        assert_eq!(
            synchronized.reflection,
            SphereReflectionState::Sphere(focus.sphere)
        );
        assert_eq!(synchronized.camera_transition_remaining, None);

        let revision = synchronized.revision;
        let mut invalid_focus = focus;
        invalid_focus.sphere.radius = -1.0;
        assert!(store
            .dispatch(AppEvent::NavigationSynchronized(
                NavigationSynchronization {
                    camera,
                    focus: invalid_focus,
                },
            ))
            .is_err());
        assert_eq!(store.frame_snapshot(), synchronized);
        assert_eq!(store.frame_snapshot().revision, revision);
    }

    #[test]
    fn presentation_load_and_cue_activation_publish_one_coherent_read_model() {
        let store = AppStore::default();
        let fixture = presentation_fixture();
        let presentation_id = fixture.id;
        let presentation_assets = fixture.assets.clone();
        let cue_count = fixture.cues.len();
        let first_cue = fixture.cues[0].id;

        store
            .dispatch(AppEvent::PresentationLoaded(fixture))
            .unwrap();
        assert!(store.summary_snapshot().presentation_loaded);
        assert_eq!(store.summary_snapshot().active_cue, None);
        let loaded = store.presentation_snapshot().unwrap();
        assert_eq!(loaded.presentation_id, presentation_id);
        assert_eq!(loaded.cue_count, cue_count);
        assert_eq!(loaded.assets, presentation_assets);
        assert_eq!(loaded.active, None);

        dispatch_presentation(&store, 1, 0.0, PresentationAction::Start).unwrap();
        let summary = store.summary_snapshot();
        let active = store.presentation_snapshot().unwrap();
        assert_eq!(summary.active_cue, Some(first_cue));
        assert_eq!(summary.revision, store.frame_snapshot().revision);
        assert_eq!(active.active.unwrap().cue_id, first_cue);
    }

    #[test]
    fn active_presentation_scene_read_is_coherent_and_nonmutating() {
        let store = AppStore::default();
        store
            .dispatch(AppEvent::PresentationLoaded(presentation_fixture()))
            .unwrap();
        dispatch_presentation(&store, 1, 0.0, PresentationAction::Start).unwrap();
        let before_frame = store.frame_snapshot();
        let before_presentation = store.presentation_snapshot();
        let active = before_presentation
            .as_ref()
            .and_then(|presentation| presentation.active.as_ref())
            .unwrap();
        let identity = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0,
        ];
        let bindings = active
            .layers
            .iter()
            .enumerate()
            .map(|(index, layer)| PackedPresentationLayerBinding {
                layer: layer.id,
                asset: AssetId::new(layer.asset).unwrap(),
                nodes: vec![hyperscape::PackedNodeSource {
                    packed_node: index as u32 + 7,
                    source_node: index as u32,
                    entity: None,
                    source_world: identity,
                }],
            })
            .collect::<Vec<_>>();

        let extraction = store.extract_active_presentation_scene(&bindings).unwrap();

        assert_eq!(extraction.revision, before_frame.revision);
        assert_eq!(extraction.cue_id, active.cue_id);
        assert_eq!(extraction.scene_id, active.scene_id);
        assert_eq!(extraction.nodes.len(), active.layers.len());
        for (node, layer) in extraction.nodes.iter().zip(&active.layers) {
            assert_eq!(node.layer, layer.id);
            assert_eq!(node.asset.as_uuid(), layer.asset);
            assert_eq!(node.visible, layer.visible);
            assert_eq!(
                node.opacity,
                if layer.visible {
                    layer.opacity as f32
                } else {
                    0.0
                }
            );
        }
        assert_eq!(store.frame_snapshot(), before_frame);
        assert_eq!(store.presentation_snapshot(), before_presentation);

        let mut missing = bindings;
        missing.pop();
        assert!(matches!(
            store.extract_active_presentation_scene(&missing),
            Err(PresentationSceneReadError::Extraction(
                PackedPresentationSceneError::MissingLayerBinding(_)
            ))
        ));
        assert_eq!(store.frame_snapshot(), before_frame);
    }

    #[test]
    fn manual_aim_policy_preempts_pending_presentation_target_without_stopping_pose() {
        let store = AppStore::default();
        let sphere = FocusSphere::new([0.0; 3], 3.0).unwrap();
        let focus = FocusNavigation {
            sphere,
            inversion_enabled: true,
            ..FocusNavigation::default()
        };
        let camera = CameraRig {
            eye: [0.0, 0.0, 6.0],
            semantic_target: None,
            control_distance: 6.0,
            ..CameraRig::default()
        };
        store
            .dispatch(AppEvent::NavigationSynchronized(
                NavigationSynchronization { camera, focus },
            ))
            .unwrap();
        let fixture = presentation_fixture();
        let horse_cue = fixture
            .cues
            .iter()
            .find(|cue| {
                cue.id == Uuid::parse_str("e0000000-0000-4000-8000-000000000001").unwrap()
            })
            .unwrap();
        let horse_cue_id = horse_cue.id;
        let expected = fixture
            .views
            .iter()
            .find(|view| view.id == horse_cue.view)
            .unwrap()
            .camera
            .to_camera_rig()
            .unwrap();
        store
            .dispatch(AppEvent::PresentationLoaded(fixture))
            .unwrap();
        dispatch_presentation(
            &store,
            0,
            0.0,
            PresentationAction::JumpToCue(horse_cue_id),
        )
        .unwrap();
        store
            .dispatch_navigation(NavigationAction::SetSemanticTargetEnabled(false))
            .unwrap();
        store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: 0.7,
                delta_seconds: 0.7,
            }))
            .unwrap();

        let settled = store.frame_snapshot();
        assert_eq!(settled.camera.semantic_target, None);
        for (actual, expected) in settled.camera.eye.into_iter().zip(expected.eye) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        assert!((settled.camera.orientation.dot(expected.orientation).abs() - 1.0).abs() < 1.0e-12);
        assert_eq!(settled.camera.lens, expected.lens);
    }

    #[test]
    fn future_aim_policy_does_not_preempt_a_presentation_target_early() {
        let store = AppStore::default();
        let sphere = FocusSphere::new([0.0; 3], 3.0).unwrap();
        store
            .dispatch(AppEvent::NavigationSynchronized(
                NavigationSynchronization {
                    camera: CameraRig {
                        eye: [0.0, 0.0, 6.0],
                        semantic_target: None,
                        control_distance: 6.0,
                        ..CameraRig::default()
                    },
                    focus: FocusNavigation {
                        sphere,
                        inversion_enabled: true,
                        ..FocusNavigation::default()
                    },
                },
            ))
            .unwrap();
        store
            .dispatch(AppEvent::PresentationLoaded(presentation_fixture()))
            .unwrap();
        dispatch_presentation(&store, 0, 0.0, PresentationAction::Start).unwrap();
        let sequence = store
            .frame_snapshot()
            .last_applied_navigation_sequence
            .unwrap()
            + 1;
        store
            .dispatch(AppEvent::Input(Timed {
                sequence,
                at_seconds: 1.0,
                value: SemanticAction::Navigate(
                    NavigationAction::SetSemanticTargetEnabled(false),
                ),
            }))
            .unwrap();

        store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: 0.7,
                delta_seconds: 0.7,
            }))
            .unwrap();
        assert_eq!(store.frame_snapshot().camera.semantic_target, Some([0.0; 3]));

        store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: 1.0,
                delta_seconds: 0.3,
            }))
            .unwrap();
        let after_due = store.frame_snapshot();
        assert_eq!(after_due.camera.semantic_target, None);
        assert_eq!(after_due.last_applied_navigation_sequence, Some(sequence));
    }

    #[test]
    fn presentation_transition_is_cadence_independent_through_the_app_boundary() {
        fn run(partition: &[f64]) -> (AppFrameSnapshot, Uuid) {
            let store = AppStore::default();
            let fixture = presentation_fixture();
            let destination = fixture.cues[5].id;
            store
                .dispatch(AppEvent::PresentationLoaded(fixture))
                .unwrap();
            dispatch_presentation(&store, 1, 0.0, PresentationAction::Start).unwrap();
            store
                .dispatch(AppEvent::Frame(FrameTick {
                    elapsed_seconds: 0.7,
                    delta_seconds: 0.7,
                }))
                .unwrap();
            dispatch_presentation(&store, 2, 0.7, PresentationAction::JumpToCue(destination))
                .unwrap();
            let mut elapsed = 0.7;
            for delta in partition {
                elapsed += delta;
                store
                    .dispatch(AppEvent::Frame(FrameTick {
                        elapsed_seconds: elapsed,
                        delta_seconds: *delta,
                    }))
                    .unwrap();
            }
            (store.frame_snapshot(), destination)
        }

        let (single, destination) = run(&[1.2]);
        let (partitioned, partitioned_destination) = run(&[0.1; 12]);
        assert_eq!(destination, partitioned_destination);
        for (single, partitioned) in single.camera.eye.into_iter().zip(partitioned.camera.eye) {
            assert!((single - partitioned).abs() < 1.0e-9);
        }
        assert!(
            (single
                .camera
                .orientation
                .dot(partitioned.camera.orientation)
                .abs()
                - 1.0)
                .abs()
                < 1.0e-9
        );
        assert!((single.focus.sphere.radius - partitioned.focus.sphere.radius).abs() < 1.0e-9);
    }

    #[test]
    fn rejected_presentation_action_is_atomic() {
        let store = AppStore::default();
        store
            .dispatch(AppEvent::PresentationLoaded(presentation_fixture()))
            .unwrap();
        dispatch_presentation(&store, 1, 0.0, PresentationAction::Start).unwrap();
        let revision = store.frame_snapshot().revision;
        let camera = store.frame_snapshot().camera;
        let active = store.presentation_snapshot();

        let unknown = Uuid::parse_str("f0000000-0000-4000-8000-000000000099").unwrap();
        let error = dispatch_presentation(&store, 2, 0.0, PresentationAction::JumpToCue(unknown))
            .unwrap_err();
        assert!(error.to_string().contains("unknown presentation cue"));
        assert_eq!(store.frame_snapshot().revision, revision);
        assert_eq!(store.frame_snapshot().camera, camera);
        assert_eq!(store.presentation_snapshot(), active);
    }

    #[test]
    fn invalid_or_future_presentation_input_cannot_partially_mutate_state() {
        let store = AppStore::default();
        let mut invalid = presentation_fixture();
        invalid.title.clear();
        assert!(store
            .dispatch(AppEvent::PresentationLoaded(invalid))
            .is_err());
        assert_eq!(store.frame_snapshot().revision, 0);
        assert!(!store.summary_snapshot().presentation_loaded);

        store
            .dispatch(AppEvent::PresentationLoaded(presentation_fixture()))
            .unwrap();
        let revision = store.frame_snapshot().revision;
        assert_eq!(
            dispatch_presentation(&store, 1, 1.0, PresentationAction::Start),
            Err(ReduceError::FuturePresentationInput)
        );
        assert_eq!(store.frame_snapshot().revision, revision);
        assert_eq!(store.summary_snapshot().active_cue, None);
    }

    #[test]
    fn high_rate_events_wait_for_an_explicit_ui_flush() {
        let store = AppStore::default();
        assert_eq!(store.navigation_snapshot().elapsed_seconds, 0.0);
        let presence = PresenceEnvelope {
            header: MessageHeader {
                version: CURRENT_PROTOCOL_VERSION,
                message_id: MessageId::from_u128(20).unwrap(),
                sender: PeerId::from_u128(21).unwrap(),
                sequence: 1,
            },
            presence: EphemeralPresence {
                ttl_millis: 100,
                camera: Some(CameraPresence {
                    eye: [0.0, 0.0, 3.0],
                    forward: [0.0, 0.0, -1.0],
                    up: [0.0, 1.0, 0.0],
                }),
                selection: Vec::new(),
                focus: None,
                active_cue: None,
                animation_seconds: None,
            },
        };
        let commit = store
            .dispatch(AppEvent::RemotePresence(ReceivedPresence {
                envelope: presence,
                received_at_seconds: 0.0,
            }))
            .unwrap();
        assert!(!commit.published_ui);
        assert_eq!(store.summary_snapshot().active_peers, 0);
        assert_eq!(store.presence_snapshot().len(), 1);
        store.flush_read_models();
        assert_eq!(store.summary_snapshot().active_peers, 1);

        store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: 0.2,
                delta_seconds: 0.2,
            }))
            .unwrap();
        assert_eq!(store.frame_snapshot().elapsed_seconds, 0.2);
        assert_eq!(store.navigation_snapshot().elapsed_seconds, 0.0);
        assert_eq!(store.summary_snapshot().active_peers, 1);
        store.flush_read_models();
        assert_eq!(store.navigation_snapshot().elapsed_seconds, 0.2);
        assert_eq!(store.summary_snapshot().active_peers, 0);
        assert!(store.presence_snapshot().is_empty());
    }

    #[test]
    fn future_effect_input_is_rejected_without_partial_state() {
        let store = AppStore::default();
        let result = store.dispatch(AppEvent::Input(Timed {
            sequence: 1,
            at_seconds: 1.0,
            value: SemanticAction::RequestAsset {
                request_id: request(25),
                asset: asset(26, "future.glb"),
                scope: AssetLoadScope::PrimaryScene,
            },
        }));
        assert_eq!(result, Err(ReduceError::FutureEffectInput));
        assert_eq!(store.frame_snapshot().revision, 0);
        assert!(store.asset_snapshot().is_empty());
    }

    #[test]
    fn authored_revisions_materialize_sorted_assets_and_entity_transforms() {
        let store = AppStore::default();
        let asset_a = asset(90, "a.glb");
        let asset_b = asset(91, "b.glb");
        let entity_a = EntityId::from_u128(92).unwrap();
        let entity_b = EntityId::from_u128(93).unwrap();
        let first = store
            .dispatch(AppEvent::AuthoredRevision(AuthoredRevision {
                projection_revision: 7,
                commands: vec![
                    authored(
                        0,
                        AuthoredCommand::UpsertAsset {
                            asset: asset_b.clone(),
                        },
                    ),
                    authored(
                        1,
                        AuthoredCommand::SetEntityTransform {
                            entity: entity_b,
                            transform: transform(2.0),
                        },
                    ),
                    authored(
                        2,
                        AuthoredCommand::UpsertAsset {
                            asset: asset_a.clone(),
                        },
                    ),
                    authored(
                        3,
                        AuthoredCommand::SetEntityTransform {
                            entity: entity_a,
                            transform: transform(1.0),
                        },
                    ),
                ],
            }))
            .unwrap();
        assert_eq!(first.disposition, CommitDisposition::Applied);
        assert!(first.published_ui);

        let scene = store.authored_scene_snapshot();
        assert_eq!(scene.projection_revision, Some(7));
        assert_eq!(
            scene.assets.iter().map(|asset| asset.id).collect::<Vec<_>>(),
            vec![asset_a.id, asset_b.id],
        );
        assert_eq!(
            scene
                .entities
                .iter()
                .map(|entity| entity.entity)
                .collect::<Vec<_>>(),
            vec![entity_a, entity_b],
        );
        assert_eq!(scene.entities[0].transform, transform(1.0));
        assert_eq!(store.summary_snapshot().authored_assets, 2);
        assert_eq!(store.summary_snapshot().authored_entities, 2);
        assert_eq!(store.summary_snapshot().revision, first.revision);

        let mut replacement_asset = asset_a.clone();
        replacement_asset.uri = "a-v2.glb".into();
        store
            .dispatch(AppEvent::AuthoredRevision(AuthoredRevision {
                projection_revision: 8,
                commands: vec![
                    authored(
                        4,
                        AuthoredCommand::UpsertAsset {
                            asset: replacement_asset,
                        },
                    ),
                    authored(
                        5,
                        AuthoredCommand::SetEntityTransform {
                            entity: entity_a,
                            transform: transform(4.0),
                        },
                    ),
                    authored(6, AuthoredCommand::RemoveEntity { entity: entity_b }),
                ],
            }))
            .unwrap();
        let scene = store.authored_scene_snapshot();
        assert_eq!(scene.projection_revision, Some(8));
        assert_eq!(scene.assets[0].uri, "a-v2.glb");
        assert_eq!(scene.entities.len(), 1);
        assert_eq!(scene.entities[0].entity, entity_a);
        assert_eq!(scene.entities[0].transform, transform(4.0));

        let stale = store
            .dispatch(AppEvent::AuthoredRevision(AuthoredRevision {
                projection_revision: 8,
                commands: vec![authored(
                    7,
                    AuthoredCommand::RemoveEntity { entity: entity_a },
                )],
            }))
            .unwrap();
        assert_eq!(stale.disposition, CommitDisposition::IgnoredStale);
        assert_eq!(store.authored_scene_snapshot().entities[0].entity, entity_a);
    }

    #[test]
    fn invalid_authored_revision_is_atomic() {
        let store = AppStore::default();
        let retained_entity = EntityId::from_u128(29).unwrap();
        store
            .dispatch(AppEvent::AuthoredRevision(AuthoredRevision {
                projection_revision: 1,
                commands: vec![authored(
                    0,
                    AuthoredCommand::SetEntityTransform {
                        entity: retained_entity,
                        transform: transform(1.0),
                    },
                )],
            }))
            .unwrap();
        let revision_before = store.frame_snapshot().revision;
        let scene_before = store.authored_scene_snapshot();
        let invalid: AuthoredEnvelope = hyperscape_protocol::AuthoredEnvelope {
            header: MessageHeader {
                version: ProtocolVersion { major: 9, minor: 0 },
                message_id: MessageId::from_u128(30).unwrap(),
                sender: PeerId::from_u128(31).unwrap(),
                sequence: 1,
            },
            command: hyperscape_protocol::AuthoredCommand::UpsertAsset {
                asset: asset(32, "scene.glb"),
            },
        };
        assert!(store
            .dispatch(AppEvent::AuthoredRevision(AuthoredRevision {
                projection_revision: 2,
                commands: vec![
                    authored(
                        1,
                        AuthoredCommand::SetEntityTransform {
                            entity: retained_entity,
                            transform: transform(9.0),
                        },
                    ),
                    invalid,
                ],
            }))
            .is_err());
        assert_eq!(store.frame_snapshot().revision, revision_before);
        assert_eq!(store.authored_scene_snapshot(), scene_before);
    }

    #[test]
    fn signalvec_snapshots_are_key_sorted_and_summary_is_the_commit_fence() {
        let store = AppStore::default();
        dispatch_request(&store, 1, request(40), asset(2, "b.glb"));
        dispatch_request(&store, 2, request(41), asset(1, "a.glb"));
        let assets = store.asset_snapshot();
        assert_eq!(
            assets
                .iter()
                .map(|asset| asset.descriptor.id)
                .collect::<Vec<_>>(),
            vec![
                AssetId::from_u128(1).unwrap(),
                AssetId::from_u128(2).unwrap()
            ]
        );
        assert_eq!(
            store.summary_snapshot().revision,
            store.frame_snapshot().revision
        );
    }
}
