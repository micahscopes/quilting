//! Transport-independent admission for the direct local Blender peer.
//!
//! This is an arrival-ordered, single-writer demo path. Both lanes reject
//! delayed sender-local sequences before dispatch; presence remains ephemeral,
//! while multi-writer authored convergence remains the responsibility of
//! `hyperscape-hhhs`.

use crate::{AppCommit, AppEvent, AppStore, AuthoredRevision, CommitDisposition, ReceivedPresence};
use hyperscape_protocol::{
    AssetEntityId, AuthoredEnvelope, AuthoringLeaseClaim, LeaseId, LocalPeerEnvelope, MessageId,
    PeerId, PresenceEnvelope, WireError, MAX_AUTHORING_LEASES_PER_PRESENCE,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

pub const DEFAULT_LOCAL_PEER_MESSAGE_MEMORY: usize = 4_096;

/// Process-local lease identity retained across low-rate presence refreshes.
/// Selection desire comes from application state; the platform supplies only
/// peer/message identity and transports the resulting envelope.
#[derive(Debug, Default)]
pub struct LocalAuthoringLeaseController {
    owner: Option<PeerId>,
    claims: BTreeMap<AssetEntityId, LeaseId>,
}

impl LocalAuthoringLeaseController {
    pub fn synchronize(
        &mut self,
        owner: PeerId,
        targets: impl IntoIterator<Item = AssetEntityId>,
        acquisition_seed: MessageId,
    ) -> Result<Vec<AuthoringLeaseClaim>, WireError> {
        owner.validate()?;
        acquisition_seed.validate()?;
        let targets = targets.into_iter().collect::<BTreeSet<_>>();
        if targets.len() > MAX_AUTHORING_LEASES_PER_PRESENCE {
            return Err(WireError::InvalidValue(
                "local presence has too many authoring lease targets",
            ));
        }
        for target in &targets {
            target.validate()?;
        }
        if self.owner != Some(owner) {
            self.owner = Some(owner);
            self.claims.clear();
        }
        self.claims.retain(|target, _| targets.contains(target));
        let mut used = self.claims.values().copied().collect::<BTreeSet<_>>();
        let mut candidate = acquisition_seed.as_uuid().as_u128();
        for target in targets {
            if self.claims.contains_key(&target) {
                continue;
            }
            let lease_id = loop {
                if candidate == 0 {
                    candidate = 1;
                }
                let lease_id = LeaseId::from_u128(candidate)?;
                candidate = candidate.wrapping_add(1);
                if used.insert(lease_id) {
                    break lease_id;
                }
            };
            self.claims.insert(target, lease_id);
        }
        Ok(self
            .claims
            .iter()
            .map(|(&target, &lease_id)| AuthoringLeaseClaim { lease_id, target })
            .collect())
    }

    pub fn clear(&mut self) {
        self.owner = None;
        self.claims.clear();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalPeerLane {
    Authored,
    Presence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalPeerDisposition {
    Applied,
    IgnoredStale,
    IgnoredDuplicate,
    IgnoredEcho,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalPeerReceipt {
    pub lane: LocalPeerLane,
    pub disposition: LocalPeerDisposition,
    pub projection_revision: Option<u64>,
    pub commit: Option<AppCommit>,
}

/// Result of checking authored deduplication, echo, and sender-sequence policy
/// before an application or durable-history write begins.
pub enum LocalAuthoredPreparation<'a> {
    Immediate(LocalPeerReceipt),
    Pending(PendingLocalAuthored<'a>),
}

/// Exclusive authored-ingress reservation.
///
/// The mutable ingress borrow remains held across an asynchronous durability
/// write. Dropping this value without completing it records nothing, so a
/// failed write can be retried with the same envelope.
pub struct PendingLocalAuthored<'a> {
    ingress: &'a mut LocalPeerIngress,
    envelope: AuthoredEnvelope,
}

impl PendingLocalAuthored<'_> {
    pub fn envelope(&self) -> &AuthoredEnvelope {
        &self.envelope
    }

    /// Record an envelope only after the caller's authoritative admission has
    /// succeeded. `commit` may be absent when durable history succeeded but a
    /// rebuildable AppStore projection must be repaired.
    pub fn complete(
        self,
        projection_revision: u64,
        commit: Option<AppCommit>,
    ) -> LocalPeerReceipt {
        let header = self.envelope.header;
        self.ingress
            .seen_authored
            .insert(header.message_id);
        self.ingress
            .authored_sender_sequences
            .observe(header.sender, header.sequence);
        let disposition = commit
            .as_ref()
            .map_or(LocalPeerDisposition::Applied, commit_disposition);
        LocalPeerReceipt {
            lane: LocalPeerLane::Authored,
            disposition,
            projection_revision: Some(projection_revision),
            commit,
        }
    }
}

/// Admission policy shared by any local WebSocket, HTTP, IPC, or test adapter.
///
/// A transport decodes one [`LocalPeerEnvelope`], then calls [`Self::accept`].
/// Only the resulting application event can mutate authored or presence state.
#[derive(Debug, Clone)]
pub struct LocalPeerIngress {
    seen_authored: BoundedMessageMemory,
    seen_presence: BoundedMessageMemory,
    local_authored_echoes: BoundedMessageMemory,
    local_presence_echoes: BoundedMessageMemory,
    authored_sender_sequences: BoundedSenderSequences,
    presence_sender_sequences: BoundedSenderSequences,
}

impl Default for LocalPeerIngress {
    fn default() -> Self {
        Self::new(DEFAULT_LOCAL_PEER_MESSAGE_MEMORY)
            .expect("the default local peer message capacity is positive")
    }
}

impl LocalPeerIngress {
    pub fn new(message_capacity: usize) -> Result<Self, LocalPeerIngressError> {
        if message_capacity == 0 {
            return Err(LocalPeerIngressError::InvalidCapacity);
        }
        Ok(Self {
            seen_authored: BoundedMessageMemory::new(message_capacity),
            seen_presence: BoundedMessageMemory::new(message_capacity),
            local_authored_echoes: BoundedMessageMemory::new(message_capacity),
            local_presence_echoes: BoundedMessageMemory::new(message_capacity),
            authored_sender_sequences: BoundedSenderSequences::new(message_capacity),
            presence_sender_sequences: BoundedSenderSequences::new(message_capacity),
        })
    }

    /// Remember one locally applied authored message before it is sent. A
    /// relay echo is consumed without dispatching a second authored revision.
    pub fn record_local_authored(
        &mut self,
        envelope: &AuthoredEnvelope,
    ) -> Result<(), LocalPeerIngressError> {
        envelope
            .validate()
            .map_err(|error| LocalPeerIngressError::InvalidEnvelope(error.to_string()))?;
        self.local_authored_echoes
            .insert(envelope.header.message_id);
        self.authored_sender_sequences
            .observe(envelope.header.sender, envelope.header.sequence);
        Ok(())
    }

    /// Validate and remember one outbound local presence sample. Its relay
    /// echo is consumed without admitting this process as its own remote peer.
    pub fn record_local_presence(
        &mut self,
        envelope: &PresenceEnvelope,
    ) -> Result<(), LocalPeerIngressError> {
        envelope
            .validate()
            .map_err(|error| LocalPeerIngressError::InvalidEnvelope(error.to_string()))?;
        self.local_presence_echoes
            .insert(envelope.header.message_id);
        self.presence_sender_sequences
            .observe(envelope.header.sender, envelope.header.sequence);
        Ok(())
    }

    /// Check authored ingress policy and reserve this single-writer ingress
    /// until the returned pending value is completed or dropped.
    pub fn prepare_authored(
        &mut self,
        envelope: AuthoredEnvelope,
    ) -> Result<LocalAuthoredPreparation<'_>, LocalPeerIngressError> {
        envelope
            .validate()
            .map_err(|error| LocalPeerIngressError::InvalidEnvelope(error.to_string()))?;
        Ok(self.prepare_validated_authored(envelope))
    }

    fn prepare_validated_authored(
        &mut self,
        envelope: AuthoredEnvelope,
    ) -> LocalAuthoredPreparation<'_> {
        let header = envelope.header;
        let message_id = header.message_id;
        if self.local_authored_echoes.remove(message_id) {
            self.seen_authored.insert(message_id);
            self.authored_sender_sequences
                .observe(header.sender, header.sequence);
            return LocalAuthoredPreparation::Immediate(LocalPeerReceipt {
                lane: LocalPeerLane::Authored,
                disposition: LocalPeerDisposition::IgnoredEcho,
                projection_revision: None,
                commit: None,
            });
        }
        if self.seen_authored.contains(message_id) {
            return LocalAuthoredPreparation::Immediate(LocalPeerReceipt {
                lane: LocalPeerLane::Authored,
                disposition: LocalPeerDisposition::IgnoredDuplicate,
                projection_revision: None,
                commit: None,
            });
        }
        if self
            .authored_sender_sequences
            .is_stale(header.sender, header.sequence)
        {
            self.seen_authored.insert(message_id);
            return LocalAuthoredPreparation::Immediate(LocalPeerReceipt {
                lane: LocalPeerLane::Authored,
                disposition: LocalPeerDisposition::IgnoredStale,
                projection_revision: None,
                commit: None,
            });
        }
        LocalAuthoredPreparation::Pending(PendingLocalAuthored {
            ingress: self,
            envelope,
        })
    }

    pub fn accept(
        &mut self,
        store: &AppStore,
        envelope: LocalPeerEnvelope,
        received_at_seconds: f64,
    ) -> Result<LocalPeerReceipt, LocalPeerIngressError> {
        envelope
            .validate()
            .map_err(|error| LocalPeerIngressError::InvalidEnvelope(error.to_string()))?;
        match envelope {
            LocalPeerEnvelope::Authored(envelope) => {
                let pending = match self.prepare_validated_authored(envelope) {
                    LocalAuthoredPreparation::Immediate(receipt) => return Ok(receipt),
                    LocalAuthoredPreparation::Pending(pending) => pending,
                };
                let projection_revision = store
                    .authored_scene_snapshot()
                    .projection_revision
                    .map_or(Ok(0), |revision| {
                        revision
                            .checked_add(1)
                            .ok_or(LocalPeerIngressError::ProjectionRevisionOverflow)
                    })?;
                let commit = store
                    .dispatch(AppEvent::AuthoredRevision(AuthoredRevision {
                        projection_revision,
                        commands: vec![pending.envelope().clone()],
                    }))
                    .map_err(|error| LocalPeerIngressError::Application(error.to_string()))?;
                Ok(pending.complete(projection_revision, Some(commit)))
            }
            LocalPeerEnvelope::Presence(envelope) => {
                let header = envelope.header;
                let message_id = header.message_id;
                envelope
                    .presence
                    .expires_at_seconds(received_at_seconds)
                    .map_err(|error| LocalPeerIngressError::InvalidEnvelope(error.to_string()))?;
                if self.local_presence_echoes.remove(message_id) {
                    self.seen_presence.insert(message_id);
                    self.presence_sender_sequences
                        .observe(header.sender, header.sequence);
                    return Ok(LocalPeerReceipt {
                        lane: LocalPeerLane::Presence,
                        disposition: LocalPeerDisposition::IgnoredEcho,
                        projection_revision: None,
                        commit: None,
                    });
                }
                if self.seen_presence.contains(message_id) {
                    return Ok(LocalPeerReceipt {
                        lane: LocalPeerLane::Presence,
                        disposition: LocalPeerDisposition::IgnoredDuplicate,
                        projection_revision: None,
                        commit: None,
                    });
                }
                if self
                    .presence_sender_sequences
                    .is_stale(header.sender, header.sequence)
                {
                    self.seen_presence.insert(message_id);
                    return Ok(LocalPeerReceipt {
                        lane: LocalPeerLane::Presence,
                        disposition: LocalPeerDisposition::IgnoredStale,
                        projection_revision: None,
                        commit: None,
                    });
                }
                let commit = store
                    .dispatch(AppEvent::RemotePresence(ReceivedPresence {
                        envelope,
                        received_at_seconds,
                    }))
                    .map_err(|error| LocalPeerIngressError::Application(error.to_string()))?;
                self.seen_presence.insert(message_id);
                self.presence_sender_sequences
                    .observe(header.sender, header.sequence);
                let disposition = commit_disposition(&commit);
                Ok(LocalPeerReceipt {
                    lane: LocalPeerLane::Presence,
                    disposition,
                    projection_revision: None,
                    commit: Some(commit),
                })
            }
        }
    }
}

fn commit_disposition(commit: &AppCommit) -> LocalPeerDisposition {
    match commit.disposition {
        CommitDisposition::Applied => LocalPeerDisposition::Applied,
        CommitDisposition::IgnoredStale => LocalPeerDisposition::IgnoredStale,
    }
}

#[derive(Debug, Clone)]
struct BoundedMessageMemory {
    capacity: usize,
    ordered: VecDeque<MessageId>,
    known: BTreeSet<MessageId>,
}

impl BoundedMessageMemory {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            ordered: VecDeque::new(),
            known: BTreeSet::new(),
        }
    }

    fn contains(&self, message: MessageId) -> bool {
        self.known.contains(&message)
    }

    fn insert(&mut self, message: MessageId) {
        if !self.known.insert(message) {
            return;
        }
        if self.ordered.len() == self.capacity {
            if let Some(expired) = self.ordered.pop_front() {
                self.known.remove(&expired);
            }
        }
        self.ordered.push_back(message);
    }

    fn remove(&mut self, message: MessageId) -> bool {
        if !self.known.remove(&message) {
            return false;
        }
        self.ordered.retain(|candidate| *candidate != message);
        true
    }
}

#[derive(Debug, Clone)]
struct BoundedSenderSequences {
    capacity: usize,
    ordered: VecDeque<PeerId>,
    latest: BTreeMap<PeerId, u64>,
}

impl BoundedSenderSequences {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            ordered: VecDeque::new(),
            latest: BTreeMap::new(),
        }
    }

    fn is_stale(&self, sender: PeerId, sequence: u64) -> bool {
        self.latest
            .get(&sender)
            .is_some_and(|current| *current >= sequence)
    }

    fn observe(&mut self, sender: PeerId, sequence: u64) {
        if let Some(current) = self.latest.get_mut(&sender) {
            *current = (*current).max(sequence);
            return;
        }
        if self.ordered.len() == self.capacity {
            if let Some(expired) = self.ordered.pop_front() {
                self.latest.remove(&expired);
            }
        }
        self.ordered.push_back(sender);
        self.latest.insert(sender, sequence);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalPeerIngressError {
    InvalidCapacity,
    InvalidEnvelope(String),
    ProjectionRevisionOverflow,
    Application(String),
}

impl fmt::Display for LocalPeerIngressError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCapacity => {
                formatter.write_str("local peer message memory must be positive")
            }
            Self::InvalidEnvelope(message) => {
                write!(formatter, "local peer envelope is invalid: {message}")
            }
            Self::ProjectionRevisionOverflow => {
                formatter.write_str("local peer projection revision overflow")
            }
            Self::Application(message) => {
                write!(formatter, "local peer admission failed: {message}")
            }
        }
    }
}

impl Error for LocalPeerIngressError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AppEvent, AppStore, FrameTick};
    use hyperscape_protocol::{
        AssetId, AuthoredCommand, EntityId, EphemeralPresence, MessageHeader, PresenceEnvelope,
        WireTransform, CURRENT_PROTOCOL_VERSION,
    };

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
            scale: [1.0; 3],
        }
    }

    #[test]
    fn deduplicates_authored_echoes_and_keeps_presence_ephemeral() {
        let store = AppStore::default();
        let mut ingress = LocalPeerIngress::new(8).unwrap();
        let entity = EntityId::from_u128(94).unwrap();
        let first = authored(
            10,
            AuthoredCommand::SetEntityTransform {
                entity,
                transform: transform(1.0),
            },
        );
        let receipt = ingress
            .accept(&store, LocalPeerEnvelope::Authored(first.clone()), 0.0)
            .unwrap();
        assert_eq!(receipt.disposition, LocalPeerDisposition::Applied);
        assert_eq!(receipt.projection_revision, Some(0));
        let revision_after_first = store.frame_snapshot().revision;

        let duplicate = ingress
            .accept(&store, LocalPeerEnvelope::Authored(first), 0.0)
            .unwrap();
        assert_eq!(
            duplicate.disposition,
            LocalPeerDisposition::IgnoredDuplicate
        );
        assert_eq!(duplicate.commit, None);

        let distinct_stale = authored(
            9,
            AuthoredCommand::SetEntityTransform {
                entity,
                transform: transform(9.0),
            },
        );
        assert_eq!(
            ingress
                .accept(&store, LocalPeerEnvelope::Authored(distinct_stale), 0.0)
                .unwrap()
                .disposition,
            LocalPeerDisposition::IgnoredStale,
        );

        let local = authored(
            11,
            AuthoredCommand::SetEntityTransform {
                entity,
                transform: transform(2.0),
            },
        );
        ingress.record_local_authored(&local).unwrap();
        assert_eq!(
            ingress
                .accept(&store, LocalPeerEnvelope::Authored(local), 0.0)
                .unwrap()
                .disposition,
            LocalPeerDisposition::IgnoredEcho,
        );
        assert_eq!(store.frame_snapshot().revision, revision_after_first);

        let presence = PresenceEnvelope {
            header: MessageHeader {
                version: CURRENT_PROTOCOL_VERSION,
                message_id: MessageId::from_u128(95).unwrap(),
                sender: PeerId::from_u128(96).unwrap(),
                sequence: 1,
            },
            presence: EphemeralPresence {
                ttl_millis: 100,
                camera: None,
                selection: vec![entity],
                authoring_leases: Vec::new(),
                focus: None,
                active_cue: None,
                animation_seconds: None,
            },
        };
        let presence_receipt = ingress
            .accept(&store, LocalPeerEnvelope::Presence(presence.clone()), 2.0)
            .unwrap();
        assert_eq!(presence_receipt.lane, LocalPeerLane::Presence);
        assert_eq!(presence_receipt.disposition, LocalPeerDisposition::Applied);
        assert_eq!(store.presence_snapshot().len(), 1);
        assert_eq!(store.authored_scene_snapshot().projection_revision, Some(0));
        assert_eq!(
            ingress
                .accept(&store, LocalPeerEnvelope::Presence(presence.clone()), 2.01)
                .unwrap()
                .disposition,
            LocalPeerDisposition::IgnoredDuplicate,
        );
        store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: 2.2,
                delta_seconds: 2.2,
            }))
            .unwrap();
        assert!(store.presence_snapshot().is_empty());

        let local_presence = PresenceEnvelope {
            header: MessageHeader {
                version: CURRENT_PROTOCOL_VERSION,
                message_id: MessageId::from_u128(97).unwrap(),
                sender: PeerId::from_u128(98).unwrap(),
                sequence: 1,
            },
            ..presence
        };
        ingress.record_local_presence(&local_presence).unwrap();
        let revision = store.frame_snapshot().revision;
        let receipt = ingress
            .accept(
                &store,
                LocalPeerEnvelope::Presence(local_presence.clone()),
                2.3,
            )
            .unwrap();
        assert_eq!(receipt.disposition, LocalPeerDisposition::IgnoredEcho);
        assert_eq!(receipt.commit, None);
        assert_eq!(store.frame_snapshot().revision, revision);
        assert!(store.presence_snapshot().is_empty());
        assert_eq!(
            ingress
                .accept(
                    &store,
                    LocalPeerEnvelope::Presence(local_presence),
                    2.31,
                )
                .unwrap()
                .disposition,
            LocalPeerDisposition::IgnoredDuplicate,
        );
    }

    #[test]
    fn invalid_frame_does_not_poison_retry_admission() {
        let store = AppStore::default();
        let mut ingress = LocalPeerIngress::new(1).unwrap();
        let entity = EntityId::from_u128(98).unwrap();
        let mut invalid = authored(
            30,
            AuthoredCommand::SetEntityTransform {
                entity,
                transform: transform(1.0),
            },
        );
        let AuthoredCommand::SetEntityTransform {
            transform: invalid_transform,
            ..
        } = &mut invalid.command
        else {
            unreachable!("the fixture command is a transform")
        };
        invalid_transform.scale[0] = f64::NAN;
        assert!(matches!(
            ingress.accept(&store, LocalPeerEnvelope::Authored(invalid), 0.0),
            Err(LocalPeerIngressError::InvalidEnvelope(_)),
        ));
        assert_eq!(store.frame_snapshot().revision, 0);
        assert_eq!(store.authored_scene_snapshot().projection_revision, None);

        let corrected = authored(
            30,
            AuthoredCommand::SetEntityTransform {
                entity,
                transform: transform(1.0),
            },
        );
        assert_eq!(
            ingress
                .accept(&store, LocalPeerEnvelope::Authored(corrected), 0.0)
                .unwrap()
                .disposition,
            LocalPeerDisposition::Applied,
        );
    }

    #[test]
    fn pending_authored_reservation_records_only_after_completion() {
        let mut ingress = LocalPeerIngress::new(2).unwrap();
        let envelope = authored(
            31,
            AuthoredCommand::SetEntityTransform {
                entity: EntityId::from_u128(0x3131).unwrap(),
                transform: transform(3.0),
            },
        );

        let pending = match ingress.prepare_authored(envelope.clone()).unwrap() {
            LocalAuthoredPreparation::Pending(pending) => pending,
            LocalAuthoredPreparation::Immediate(receipt) => {
                panic!("new envelope was unexpectedly ignored: {receipt:?}")
            }
        };
        drop(pending);

        let pending = match ingress.prepare_authored(envelope.clone()).unwrap() {
            LocalAuthoredPreparation::Pending(pending) => pending,
            LocalAuthoredPreparation::Immediate(receipt) => {
                panic!("dropped reservation poisoned retry: {receipt:?}")
            }
        };
        let receipt = pending.complete(7, None);
        assert_eq!(receipt.disposition, LocalPeerDisposition::Applied);
        assert_eq!(receipt.projection_revision, Some(7));
        assert_eq!(receipt.commit, None);

        let duplicate = match ingress.prepare_authored(envelope).unwrap() {
            LocalAuthoredPreparation::Immediate(receipt) => receipt,
            LocalAuthoredPreparation::Pending(_) => panic!("completed envelope was not recorded"),
        };
        assert_eq!(duplicate.disposition, LocalPeerDisposition::IgnoredDuplicate);
    }

    #[test]
    fn bounded_id_eviction_retains_sender_sequence_staleness() {
        let store = AppStore::default();
        let mut ingress = LocalPeerIngress::new(1).unwrap();
        let entity = EntityId::from_u128(99).unwrap();
        let first = authored(
            40,
            AuthoredCommand::SetEntityTransform {
                entity,
                transform: transform(1.0),
            },
        );
        let second = authored(
            41,
            AuthoredCommand::SetEntityTransform {
                entity,
                transform: transform(2.0),
            },
        );
        for envelope in [first.clone(), second] {
            assert_eq!(
                ingress
                    .accept(&store, LocalPeerEnvelope::Authored(envelope), 0.0)
                    .unwrap()
                    .disposition,
                LocalPeerDisposition::Applied,
            );
        }
        assert_eq!(
            ingress
                .accept(&store, LocalPeerEnvelope::Authored(first), 0.0)
                .unwrap()
                .disposition,
            LocalPeerDisposition::IgnoredStale,
        );
        assert_eq!(
            store.authored_scene_snapshot().entities[0].transform,
            transform(2.0),
        );
    }

    #[test]
    fn delayed_presence_is_rejected_before_it_can_refresh_an_older_pose() {
        let store = AppStore::default();
        let mut ingress = LocalPeerIngress::new(2).unwrap();
        let sender = PeerId::from_u128(120).unwrap();
        let entity = EntityId::from_u128(121).unwrap();
        let presence = |message: u128, sequence, ttl_millis| PresenceEnvelope {
            header: MessageHeader {
                version: CURRENT_PROTOCOL_VERSION,
                message_id: MessageId::from_u128(message).unwrap(),
                sender,
                sequence,
            },
            presence: EphemeralPresence {
                ttl_millis,
                camera: None,
                selection: vec![entity],
                authoring_leases: Vec::new(),
                focus: None,
                active_cue: None,
                animation_seconds: Some(sequence as f64),
            },
        };

        let newest = presence(122, 10, 100);
        assert_eq!(
            ingress
                .accept(&store, LocalPeerEnvelope::Presence(newest), 2.0)
                .unwrap()
                .disposition,
            LocalPeerDisposition::Applied,
        );
        let revision = store.frame_snapshot().revision;
        let delayed = presence(123, 9, 60_000);
        let receipt = ingress
            .accept(&store, LocalPeerEnvelope::Presence(delayed), 2.09)
            .unwrap();
        assert_eq!(receipt.disposition, LocalPeerDisposition::IgnoredStale);
        assert_eq!(receipt.commit, None);
        assert_eq!(store.frame_snapshot().revision, revision);
        let retained = &store.presence_snapshot()[0];
        assert_eq!(retained.sequence, 10);
        assert_eq!(retained.expires_at_seconds, 2.1);

        store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: 2.11,
                delta_seconds: 2.11,
            }))
            .unwrap();
        assert!(store.presence_snapshot().is_empty());
    }

    #[test]
    fn local_presence_sequence_fences_distinct_delayed_echoes() {
        let store = AppStore::default();
        let mut ingress = LocalPeerIngress::new(2).unwrap();
        let sender = PeerId::from_u128(124).unwrap();
        let make_presence = |message: u128, sequence| PresenceEnvelope {
            header: MessageHeader {
                version: CURRENT_PROTOCOL_VERSION,
                message_id: MessageId::from_u128(message).unwrap(),
                sender,
                sequence,
            },
            presence: EphemeralPresence {
                ttl_millis: 100,
                camera: None,
                selection: Vec::new(),
                authoring_leases: Vec::new(),
                focus: None,
                active_cue: None,
                animation_seconds: None,
            },
        };
        let outbound = make_presence(125, 8);
        ingress.record_local_presence(&outbound).unwrap();

        let delayed = make_presence(126, 7);
        assert_eq!(
            ingress
                .accept(&store, LocalPeerEnvelope::Presence(delayed), 3.0)
                .unwrap()
                .disposition,
            LocalPeerDisposition::IgnoredStale,
        );
        assert!(store.presence_snapshot().is_empty());
    }

    #[test]
    fn local_authoring_claims_refresh_release_and_change_with_owner() {
        let mut leases = LocalAuthoringLeaseController::default();
        let first_owner = PeerId::from_u128(200).unwrap();
        let second_owner = PeerId::from_u128(201).unwrap();
        let first_target = AssetEntityId::new(
            AssetId::from_u128(202).unwrap(),
            EntityId::from_u128(203).unwrap(),
        )
        .unwrap();
        let second_target = AssetEntityId::new(
            AssetId::from_u128(202).unwrap(),
            EntityId::from_u128(204).unwrap(),
        )
        .unwrap();

        let first = leases
            .synchronize(
                first_owner,
                [second_target, first_target],
                MessageId::from_u128(205).unwrap(),
            )
            .unwrap();
        let refreshed = leases
            .synchronize(
                first_owner,
                [first_target, second_target],
                MessageId::from_u128(206).unwrap(),
            )
            .unwrap();
        assert_eq!(refreshed, first);
        assert_eq!(first[0].target, first_target);
        assert_eq!(first[1].target, second_target);

        let retained = leases
            .synchronize(
                first_owner,
                [second_target],
                MessageId::from_u128(207).unwrap(),
            )
            .unwrap();
        assert_eq!(retained, vec![first[1]]);
        let reacquired = leases
            .synchronize(
                first_owner,
                [first_target, second_target],
                MessageId::from_u128(208).unwrap(),
            )
            .unwrap();
        assert_ne!(reacquired[0].lease_id, first[0].lease_id);
        assert_eq!(reacquired[1], first[1]);

        assert!(leases
            .synchronize(
                first_owner,
                [],
                MessageId::from_u128(209).unwrap(),
            )
            .unwrap()
            .is_empty());
        let new_owner = leases
            .synchronize(
                second_owner,
                [second_target],
                MessageId::from_u128(210).unwrap(),
            )
            .unwrap();
        assert_ne!(new_owner[0].lease_id, first[1].lease_id);

        let excessive = (1..=MAX_AUTHORING_LEASES_PER_PRESENCE + 1).map(|value| {
            AssetEntityId::new(
                AssetId::from_u128(211).unwrap(),
                EntityId::from_u128(value as u128).unwrap(),
            )
            .unwrap()
        });
        assert_eq!(
            leases.synchronize(
                second_owner,
                excessive,
                MessageId::from_u128(212).unwrap(),
            ),
            Err(WireError::InvalidValue(
                "local presence has too many authoring lease targets"
            )),
        );
        assert_eq!(
            leases
                .synchronize(
                    second_owner,
                    [second_target],
                    MessageId::from_u128(213).unwrap(),
                )
                .unwrap(),
            new_owner,
        );
    }

    #[test]
    fn sequential_single_writer_delivery_converges_without_echo_reapplication() {
        let first_store = AppStore::default();
        let second_store = AppStore::default();
        let mut first_ingress = LocalPeerIngress::default();
        let mut second_ingress = LocalPeerIngress::default();
        let entity = EntityId::from_u128(97).unwrap();
        for (sequence, value) in [(20, 3.0), (21, 7.0)] {
            let envelope = authored(
                sequence,
                AuthoredCommand::SetEntityTransform {
                    entity,
                    transform: transform(value),
                },
            );
            first_ingress
                .accept(
                    &first_store,
                    LocalPeerEnvelope::Authored(envelope.clone()),
                    0.0,
                )
                .unwrap();
            first_ingress.record_local_authored(&envelope).unwrap();
            second_ingress
                .accept(
                    &second_store,
                    LocalPeerEnvelope::Authored(envelope.clone()),
                    0.0,
                )
                .unwrap();
            assert_eq!(
                first_ingress
                    .accept(&first_store, LocalPeerEnvelope::Authored(envelope), 0.0,)
                    .unwrap()
                    .disposition,
                LocalPeerDisposition::IgnoredEcho,
            );
        }
        assert_eq!(
            first_store.authored_scene_snapshot(),
            second_store.authored_scene_snapshot(),
        );
        assert_eq!(
            first_store.authored_scene_snapshot().entities[0].transform,
            transform(7.0),
        );
    }
}
