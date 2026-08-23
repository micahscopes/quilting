//! Application authority above Hyperscape's semantic runtime and below any
//! browser, Blender, or persistence adapter.
//!
//! The boundary is `AppEvent -> reducer -> AppCommit + AppEffect`. Effects are
//! values; an adapter performs them and returns an [`EffectCompletion`] event.
//! High-rate frame/navigation/presence events update authoritative state but do
//! not publish UI signals automatically. A view adapter calls
//! [`AppStore::flush_read_models`] at its chosen cadence.

use futures_signals::signal::{Mutable, MutableSignalCloned};
use futures_signals::signal_vec::{MutableSignalVec, MutableVec};
use hyperscape::{
    CameraRig, FocusNavigation, NavigationAction, NavigationController, ScheduledNavigationAction,
};
use hyperscape_protocol::{
    AssetDescriptor, AssetId, AuthoredEnvelope, EphemeralPresence, PeerId, PresenceEnvelope,
    RequestId,
};
use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

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
    RequestAsset {
        request_id: RequestId,
        asset: AssetDescriptor,
    },
    CancelAsset(AssetId),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTick {
    pub elapsed_seconds: f64,
    pub delta_seconds: f64,
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct AppFrameSnapshot {
    pub revision: u64,
    pub elapsed_seconds: f64,
    pub camera: CameraRig,
    pub focus: FocusNavigation,
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
        AppFrameSnapshot {
            revision: self.revision,
            elapsed_seconds: self.frame_elapsed_seconds,
            camera: self.navigation.camera,
            focus: self.navigation.focus.clone(),
        }
    }

    pub fn reduce(&mut self, event: AppEvent) -> Result<AppCommit, ReduceError> {
        let mut effects = Vec::new();
        let mut disposition = CommitDisposition::Applied;
        let mut publish_ui = true;

        match event {
            AppEvent::Input(timed) => {
                timed.validate_time()?;
                let effect_may_run_now = timed.at_seconds <= self.frame_elapsed_seconds;
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
                        self.navigation
                            .advance_to(self.frame_elapsed_seconds)
                            .map_err(ReduceError::Navigation)?;
                        publish_ui = false;
                    }
                    SemanticAction::RequestAsset { request_id, asset } => {
                        if !effect_may_run_now {
                            return Err(ReduceError::FutureEffectInput);
                        }
                        request_id
                            .validate()
                            .map_err(|error| ReduceError::Wire(error.to_string()))?;
                        asset
                            .validate()
                            .map_err(|error| ReduceError::Wire(error.to_string()))?;
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
                        if !effect_may_run_now {
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
        }
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
        Self {
            state: Arc::new(Mutex::new(state)),
            summary: Mutable::new(summary),
            assets: MutableVec::new_with_values(assets),
            diagnostics: MutableVec::new_with_values(diagnostics),
        }
    }

    pub fn dispatch(&self, event: AppEvent) -> Result<AppCommit, ReduceError> {
        let commit = self.lock_state().reduce(event)?;
        if commit.published_ui {
            self.flush_read_models();
        }
        Ok(commit)
    }

    /// Publish a coherent read-model batch after a high-rate event burst. The
    /// summary is set last and its revision acts as the consumer commit fence.
    pub fn flush_read_models(&self) -> u64 {
        let state = self.lock_state();
        let assets = state.asset_read_models();
        let diagnostics = state.diagnostic_read_models();
        let summary = state.summary();
        let revision = summary.revision;
        drop(state);
        self.assets.lock_mut().replace_cloned(assets);
        self.diagnostics.lock_mut().replace_cloned(diagnostics);
        self.summary.set_neq(summary);
        revision
    }

    pub fn frame_snapshot(&self) -> AppFrameSnapshot {
        self.lock_state().frame_snapshot()
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
    Navigation(&'static str),
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
            Self::Navigation(message) => write!(formatter, "navigation event failed: {message}"),
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
        dispatch_request(&store, 2, second, horse.clone());

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
