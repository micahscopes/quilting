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
mod settings;

#[cfg(feature = "replay")]
pub use replay::*;
pub use settings::*;

use futures_signals::signal::{Mutable, MutableSignalCloned};
use futures_signals::signal_vec::{MutableSignalVec, MutableVec};
use hyperscape::{
    CameraRig, FocusNavigation, FocusSphere, NavigationAction, NavigationController,
    NavigationPreset, Presentation, PresentationRuntime, PresentationSnapshot,
    ScheduledNavigationAction, SphereReflectionState,
};
use hyperscape_protocol::{
    AssetDescriptor, AssetId, AuthoredEnvelope, EphemeralPresence, PeerId, PresenceEnvelope,
    RequestId,
};
use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};
use uuid::Uuid;

const MAX_DIAGNOSTICS: usize = 256;

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
    RequestAsset {
        request_id: RequestId,
        asset: AssetDescriptor,
    },
    CancelAsset(AssetId),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectCompletion {
    AssetLoad(AssetLoadCompletion),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AppEvent {
    Input(Timed<SemanticAction>),
    PresentationLoaded(Presentation),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticReadModel {
    pub revision: u64,
    pub code: &'static str,
    pub message: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AppSummary {
    /// This is a commit fence: asset and diagnostic SignalVec replacements are
    /// published before this revision changes.
    pub revision: u64,
    pub authored_projection_revision: Option<u64>,
    pub assets: usize,
    pub loading_assets: usize,
    pub active_peers: usize,
    pub diagnostics: usize,
    pub presentation_loaded: bool,
    pub active_cue: Option<Uuid>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AppFrameSnapshot {
    pub revision: u64,
    pub elapsed_seconds: f64,
    pub navigation_preset: NavigationPreset,
    pub pending_navigation_actions: usize,
    pub last_applied_navigation_sequence: Option<u64>,
    pub camera: CameraRig,
    pub focus: FocusNavigation,
    pub reflection: SphereReflectionState,
    pub camera_transition_remaining: Option<f64>,
    pub surface_anchor_transition_remaining: Option<f64>,
    pub surface_anchor_hop_height: Option<f64>,
}

/// Low-rate FRP projection. Camera/focus motion stays in
/// [`AppFrameSnapshot`], so ticking a transition does not clone cue assets and
/// layers into every render frame.
#[derive(Debug, Clone, PartialEq)]
pub struct PresentationReadModel {
    pub presentation_id: Uuid,
    pub title: String,
    pub cue_count: usize,
    pub active: Option<PresentationSnapshot>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PeerPresenceReadModel {
    pub peer: PeerId,
    pub sequence: u64,
    pub expires_at_seconds: f64,
    pub presence: EphemeralPresence,
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
#[derive(Debug, Clone, Default)]
pub struct AppState {
    revision: u64,
    frame_elapsed_seconds: f64,
    navigation: NavigationController,
    presentation: Option<PresentationRuntime>,
    active_presentation: Option<PresentationSnapshot>,
    assets: BTreeMap<AssetId, AssetRecord>,
    presence: BTreeMap<PeerId, PresenceRecord>,
    authored_projection_revision: Option<u64>,
    diagnostics: VecDeque<DiagnosticReadModel>,
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
            .runtime
            .surface_anchor_transition
            .map(|transition| (transition.duration_seconds - transition.elapsed_seconds).max(0.0));
        AppFrameSnapshot {
            revision: self.revision,
            elapsed_seconds: self.frame_elapsed_seconds,
            navigation_preset: self.navigation.runtime.preset,
            pending_navigation_actions: self.navigation.queue.len(),
            last_applied_navigation_sequence: self.navigation.runtime.last_applied_sequence,
            camera: self.navigation.camera,
            focus: self.navigation.focus.clone(),
            reflection: self.navigation.runtime.reflection,
            camera_transition_remaining,
            surface_anchor_transition_remaining,
            surface_anchor_hop_height: self
                .navigation
                .runtime
                .surface_anchor_transition
                .map(|transition| transition.hop_height),
        }
    }

    pub fn reduce(&mut self, event: AppEvent) -> Result<AppCommit, ReduceError> {
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
                        self.apply_presentation_action(action)?;
                    }
                    SemanticAction::RequestAsset { request_id, asset } => {
                        if !input_may_run_now {
                            return Err(ReduceError::FutureEffectInput);
                        }
                        request_id
                            .validate()
                            .map_err(|error| ReduceError::Wire(error.to_string()))?;
                        asset
                            .validate()
                            .map_err(|error| ReduceError::Wire(error.to_string()))?;
                        if let Some(AssetRecord {
                            status:
                                AssetStatus::Loading {
                                    request_id: previous_request,
                                },
                            ..
                        }) = self.assets.get(&asset.id)
                        {
                            effects.push(AppEffect::CancelAssetLoad {
                                request_id: *previous_request,
                                asset_id: asset.id,
                            });
                        }
                        self.assets.insert(
                            asset.id,
                            AssetRecord {
                                descriptor: asset.clone(),
                                status: AssetStatus::Loading { request_id },
                            },
                        );
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
                self.navigation
                    .advance_to(frame.elapsed_seconds)
                    .map_err(ReduceError::Navigation)?;
                if let Some(presentation) = self.presentation.as_mut() {
                    presentation.reconcile_navigation(&mut self.navigation);
                }
                self.frame_elapsed_seconds = frame.elapsed_seconds;
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
                    let record = self
                        .assets
                        .get_mut(&completion.asset_id)
                        .expect("active request implies asset record");
                    record.status = match completion.outcome {
                        AssetLoadOutcome::Loaded {
                            byte_length,
                            content_digest,
                        } => AssetStatus::Ready {
                            byte_length,
                            content_digest,
                        },
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
                    self.authored_projection_revision = Some(revision.projection_revision);
                }
            }
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

    fn apply_presentation_action(&mut self, action: PresentationAction) -> Result<(), ReduceError> {
        if action == PresentationAction::Clear {
            self.presentation = None;
            self.active_presentation = None;
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
        self.presentation = Some(presentation);
        self.navigation = navigation;
        self.active_presentation = Some(snapshot);
        Ok(())
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
        AppSummary {
            revision: self.revision,
            authored_projection_revision: self.authored_projection_revision,
            assets: self.assets.len(),
            loading_assets: self
                .assets
                .values()
                .filter(|record| matches!(record.status, AssetStatus::Loading { .. }))
                .count(),
            active_peers: self.presence.len(),
            diagnostics: self.diagnostics.len(),
            presentation_loaded: self.presentation.is_some(),
            active_cue: self
                .active_presentation
                .as_ref()
                .map(|snapshot| snapshot.cue_id),
        }
    }

    fn presentation_read_model(&self) -> Option<PresentationReadModel> {
        self.presentation.as_ref().map(|runtime| {
            let presentation = runtime.presentation();
            PresentationReadModel {
                presentation_id: presentation.id,
                title: presentation.title.clone(),
                cue_count: presentation.cues.len(),
                active: self.active_presentation.clone(),
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
}

/// Thread-safe reducer host with futures-signals projections. Signal vectors
/// are projections only; callers cannot mutate the reducer through them.
#[derive(Clone)]
pub struct AppStore {
    state: Arc<Mutex<AppState>>,
    summary: Mutable<AppSummary>,
    assets: MutableVec<AssetReadModel>,
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
        let assets = state.asset_read_models();
        let diagnostics = state.diagnostic_read_models();
        let presentation = state.presentation_read_model();
        Self {
            state: Arc::new(Mutex::new(state)),
            summary: Mutable::new(summary),
            assets: MutableVec::new_with_values(assets),
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
        let diagnostics = state.diagnostic_read_models();
        let presentation = state.presentation_read_model();
        let summary = state.summary();
        let revision = summary.revision;
        drop(state);
        self.assets.lock_mut().replace_cloned(assets);
        self.diagnostics.lock_mut().replace_cloned(diagnostics);
        self.presentation.set_neq(presentation);
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

    pub fn summary_snapshot(&self) -> AppSummary {
        self.summary.get_cloned()
    }

    pub fn asset_snapshot(&self) -> Vec<AssetReadModel> {
        self.assets.lock_ref().to_vec()
    }

    pub fn diagnostic_snapshot(&self) -> Vec<DiagnosticReadModel> {
        self.diagnostics.lock_ref().to_vec()
    }

    pub fn presentation_snapshot(&self) -> Option<PresentationReadModel> {
        self.presentation.get_cloned()
    }

    /// Presence remains a high-rate snapshot lane rather than an automatic UI
    /// signal. A view may sample it on the same throttle as camera overlays.
    pub fn presence_snapshot(&self) -> Vec<PeerPresenceReadModel> {
        self.lock_state().presence_read_models()
    }

    pub fn summary_signal(&self) -> MutableSignalCloned<AppSummary> {
        self.summary.signal_cloned()
    }

    pub fn asset_signal_vec(&self) -> MutableSignalVec<AssetReadModel> {
        self.assets.signal_vec_cloned()
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
    FutureEffectInput,
    FuturePresentationInput,
    Navigation(&'static str),
    NavigationState(String),
    Presentation(String),
    NoPresentation,
    Wire(String),
    UnknownAsset(AssetId),
}

impl fmt::Display for ReduceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTime => {
                formatter.write_str("application time must be finite and monotonic")
            }
            Self::FutureEffectInput => formatter.write_str(
                "effect-producing input cannot be scheduled beyond the current app frame",
            ),
            Self::FuturePresentationInput => formatter
                .write_str("presentation input cannot be scheduled beyond the current app frame"),
            Self::Navigation(message) => write!(formatter, "navigation event failed: {message}"),
            Self::NavigationState(message) => {
                write!(formatter, "navigation synchronization failed: {message}")
            }
            Self::Presentation(message) => {
                write!(formatter, "presentation event failed: {message}")
            }
            Self::NoPresentation => formatter.write_str("no presentation is loaded"),
            Self::Wire(message) => write!(formatter, "wire value failed validation: {message}"),
            Self::UnknownAsset(asset) => write!(formatter, "unknown asset {asset}"),
        }
    }
}

impl Error for ReduceError {}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscape::{CameraRig, NavigationFrame};
    use hyperscape_protocol::{
        CameraPresence, MessageHeader, MessageId, ProtocolVersion, CURRENT_PROTOCOL_VERSION,
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

    fn presentation_fixture() -> Presentation {
        Presentation::from_json(include_str!(
            "../../../examples/hacker-night.presentation.json"
        ))
        .unwrap()
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
        store
            .dispatch(AppEvent::Input(Timed {
                sequence,
                at_seconds: 0.0,
                value: SemanticAction::RequestAsset { request_id, asset },
            }))
            .unwrap()
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
        let first_cue = fixture.cues[0].id;

        store
            .dispatch(AppEvent::PresentationLoaded(fixture))
            .unwrap();
        assert!(store.summary_snapshot().presentation_loaded);
        assert_eq!(store.summary_snapshot().active_cue, None);
        let loaded = store.presentation_snapshot().unwrap();
        assert_eq!(loaded.presentation_id, presentation_id);
        assert_eq!(loaded.cue_count, 6);
        assert_eq!(loaded.active, None);

        dispatch_presentation(&store, 1, 0.0, PresentationAction::Start).unwrap();
        let summary = store.summary_snapshot();
        let active = store.presentation_snapshot().unwrap();
        assert_eq!(summary.active_cue, Some(first_cue));
        assert_eq!(summary.revision, store.frame_snapshot().revision);
        assert_eq!(active.active.unwrap().cue_id, first_cue);
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
        assert_eq!(store.summary_snapshot().active_peers, 1);
        store.flush_read_models();
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
            },
        }));
        assert_eq!(result, Err(ReduceError::FutureEffectInput));
        assert_eq!(store.frame_snapshot().revision, 0);
        assert!(store.asset_snapshot().is_empty());
    }

    #[test]
    fn invalid_authored_revision_is_atomic() {
        let store = AppStore::default();
        let revision_before = store.frame_snapshot().revision;
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
                projection_revision: 1,
                commands: vec![invalid],
            }))
            .is_err());
        assert_eq!(store.frame_snapshot().revision, revision_before);
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
