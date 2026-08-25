//! Transport-independent admission for the direct local Blender peer.
//!
//! This is an arrival-ordered, single-writer demo path. Multi-writer authored
//! convergence remains the responsibility of `hyperscape-hhhs`.

use crate::{AppCommit, AppEvent, AppStore, AuthoredRevision, CommitDisposition, ReceivedPresence};
use hyperscape_protocol::{AuthoredEnvelope, LocalPeerEnvelope, MessageId, PeerId};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

pub const DEFAULT_LOCAL_PEER_MESSAGE_MEMORY: usize = 4_096;

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

/// Admission policy shared by any local WebSocket, HTTP, IPC, or test adapter.
///
/// A transport decodes one [`LocalPeerEnvelope`], then calls [`Self::accept`].
/// Only the resulting application event can mutate authored or presence state.
#[derive(Debug, Clone)]
pub struct LocalPeerIngress {
    seen_authored: BoundedMessageMemory,
    local_authored_echoes: BoundedMessageMemory,
    authored_sender_sequences: BoundedSenderSequences,
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
            local_authored_echoes: BoundedMessageMemory::new(message_capacity),
            authored_sender_sequences: BoundedSenderSequences::new(message_capacity),
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
                let header = envelope.header;
                let message_id = header.message_id;
                if self.local_authored_echoes.remove(message_id) {
                    self.seen_authored.insert(message_id);
                    self.authored_sender_sequences
                        .observe(header.sender, header.sequence);
                    return Ok(LocalPeerReceipt {
                        lane: LocalPeerLane::Authored,
                        disposition: LocalPeerDisposition::IgnoredEcho,
                        projection_revision: None,
                        commit: None,
                    });
                }
                if self.seen_authored.contains(message_id) {
                    return Ok(LocalPeerReceipt {
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
                    return Ok(LocalPeerReceipt {
                        lane: LocalPeerLane::Authored,
                        disposition: LocalPeerDisposition::IgnoredStale,
                        projection_revision: None,
                        commit: None,
                    });
                }
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
                        commands: vec![envelope],
                    }))
                    .map_err(|error| LocalPeerIngressError::Application(error.to_string()))?;
                self.seen_authored.insert(message_id);
                self.authored_sender_sequences
                    .observe(header.sender, header.sequence);
                let disposition = commit_disposition(&commit);
                Ok(LocalPeerReceipt {
                    lane: LocalPeerLane::Authored,
                    disposition,
                    projection_revision: Some(projection_revision),
                    commit: Some(commit),
                })
            }
            LocalPeerEnvelope::Presence(envelope) => {
                let commit = store
                    .dispatch(AppEvent::RemotePresence(ReceivedPresence {
                        envelope,
                        received_at_seconds,
                    }))
                    .map_err(|error| LocalPeerIngressError::Application(error.to_string()))?;
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
        AuthoredCommand, EntityId, EphemeralPresence, MessageHeader, PresenceEnvelope,
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
                .accept(&store, LocalPeerEnvelope::Presence(presence), 2.01)
                .unwrap()
                .disposition,
            LocalPeerDisposition::IgnoredStale,
        );
        store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: 2.2,
                delta_seconds: 2.2,
            }))
            .unwrap();
        assert!(store.presence_snapshot().is_empty());
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
