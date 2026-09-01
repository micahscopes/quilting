//! Optional behavior shadow between [`hyperscope_app::AppStore`] authored
//! revisions and [`hyperscape_hhhs`] durable authored history.
//!
//! [`AuthoredHhhsShadow::dispatch`] always commits the AppStore first. HHHS is
//! only a diagnostic observer in this checkpoint: an HHHS failure is returned
//! inside [`AuthoredShadowDispatch`] and never rejects or rolls back AppStore.
//! The default [`MemoryDurability`] host is process-local and is not browser
//! persistence.
//!
//! Portable HHHS archives recover authored scene history. A separate
//! [`AuthoredShadowCheckpoint`] binds the source-local projection cursor to an
//! exact project/history/state horizon. Imported archives intentionally start
//! without that cursor; local restart supplies the checksummed checkpoint and
//! aligns a fresh AppStore through [`AuthoredHhhsShadow::align_store`].
//!
//! One AppStore revision may contain several envelopes, while
//! [`DurableProject::admit`] persists one envelope at a time. A failure can
//! therefore leave a durable prefix. The observer reports that prefix, becomes
//! poisoned, and deliberately has no retry API: replaying the whole revision
//! would create duplicate HHHS entries. Promoting this path to application
//! authority requires an atomic or resumable/idempotent batch protocol first.
//!
//! [`DurableAuthoredCoordinator`] is the narrow authority path for the common
//! one-envelope revision. It atomically persists that envelope and a
//! receiver-local source cursor before compare-and-dispatching the rebuildable
//! AppStore projection. It deliberately rejects multi-command revisions rather
//! than inheriting the diagnostic shadow's partial-prefix failure mode. After
//! restart, [`DurableAuthoredCoordinator::restore_store`] installs canonical
//! HHHS materialization into a fresh AppStore without fabricating commands.
//!
//! This crate accepts local [`AuthoredRevision`] values and typed inbound
//! [`ReplicaRecord`] values. Camera motion, frame clocks, presence, selection,
//! animation, renderer state, and GPU resources are not representable at this
//! durable authored boundary.
//!
//! The coordinator must be the exclusive path for authored revisions on its
//! AppStore. Other AppStore event kinds may still be dispatched normally. A
//! directly dispatched authored revision changes the AppStore baseline without
//! HHHS; the next coordinated dispatch detects that drift, skips mirroring, and
//! poisons the observer. Authored writers must therefore be serialized around
//! this diagnostic boundary.

#![forbid(unsafe_code)]

use hhhs::{Digest, EntryHash};
use hhhs_replica::{AsyncTransactionSink, ReplicaRecord};
use hyperscape_hhhs::{
    decode_authored, AdapterError, ApplyRecordWithCheckpointReport, DurableProject,
    MemoryDurability, ProjectId, ProjectState, ProjectionKey, RecordRefusal,
};
use hyperscape_protocol::{
    AssetDescriptor, AssetId, AuthoredCommand, AuthoredEnvelope, EntityId, LocalPeerEnvelope,
    PresenceEnvelope, WireTransform,
};
use hyperscope_app::{
    AppCommit, AppEvent, AppStore, AuthoredEntityReadModel, AuthoredProjectionDispatchError,
    AuthoredProjectionSnapshot, AuthoredRevision, AuthoredSceneReadModel, CommitDisposition,
    LocalAuthoredPreparation, LocalPeerIngress, LocalPeerIngressError, LocalPeerReceipt, ReduceError,
    DEFAULT_LOCAL_PEER_MESSAGE_MEMORY,
};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

/// Frozen domain for the source-local projection checkpoint codec.
pub const AUTHORED_SHADOW_CHECKPOINT_DOMAIN: &[u8] =
    b"hyperscope authored source checkpoint v0.1\0";
pub const AUTHORED_SHADOW_CHECKPOINT_VERSION: (u16, u16) = (0, 1);
const CHECKPOINT_CHECKSUM_BYTES: usize = 32;
const CHECKPOINT_BODY_BYTES: usize = 4 + 16 + 1 + 8 + 8 + 32 + 32;
const CHECKPOINT_BYTES: usize =
    AUTHORED_SHADOW_CHECKPOINT_DOMAIN.len() + CHECKPOINT_BODY_BYTES + CHECKPOINT_CHECKSUM_BYTES;
/// Receiver-local HHHS checkpoint key for the durable-first source cursor.
pub const DURABLE_AUTHORED_CURSOR_NAME: &str = "hyperscope/authored-source-cursor";
/// Cursor bytes contain one little-endian `u64` projection revision.
pub const DURABLE_AUTHORED_CURSOR_SCHEMA: u32 = 1;

pub fn durable_authored_cursor_key() -> ProjectionKey {
    ProjectionKey::new(
        DURABLE_AUTHORED_CURSOR_NAME,
        DURABLE_AUTHORED_CURSOR_SCHEMA,
    )
    .expect("the static durable authored cursor key is valid")
}

/// A successful diagnostic observation after the authoritative AppStore
/// dispatch completed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthoredShadowObservation {
    /// The exact vector order was admitted and the independent sequential
    /// projection matched HHHS materialization.
    Matched {
        projection_revision: u64,
        command_count: usize,
        history_len: usize,
        history_root: [u8; 32],
        state_root: [u8; 32],
    },
    /// AppStore rejected a stale projection revision. The observer performed
    /// no HHHS admission.
    IgnoredStale {
        projection_revision: u64,
        observed_projection_revision: Option<u64>,
        history_len: usize,
    },
}

/// The permanent fault that poisoned one observer instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthoredShadowFault {
    pub projection_revision: u64,
    /// Zero-based envelope index. `commands.len()` means final
    /// materialization/parity validation failed after all admissions.
    pub failed_command_index: usize,
    /// Envelopes from this revision already published to HHHS.
    pub admitted_prefix: usize,
    pub history_len: usize,
    pub reason: String,
}

/// A diagnostic shadow failure. Neither variant changes the already-committed
/// AppStore result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthoredShadowError {
    Fault(AuthoredShadowFault),
    Poisoned {
        original: AuthoredShadowFault,
        skipped_projection_revision: u64,
    },
}

impl fmt::Display for AuthoredShadowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fault(fault) => write!(
                formatter,
                "authored HHHS shadow faulted at projection {} command {} after admitting {}: {}",
                fault.projection_revision,
                fault.failed_command_index,
                fault.admitted_prefix,
                fault.reason
            ),
            Self::Poisoned {
                original,
                skipped_projection_revision,
            } => write!(
                formatter,
                "authored HHHS shadow is poisoned by projection {}; skipped projection {}",
                original.projection_revision, skipped_projection_revision
            ),
        }
    }
}

impl Error for AuthoredShadowError {}

/// Construction failure for an observer baseline.
#[derive(Debug)]
pub enum AuthoredShadowInitError {
    Adapter(AdapterError),
    NonEmptyBaseline,
    CheckpointProject {
        expected: ProjectId,
        actual: ProjectId,
    },
    CheckpointHistoryLength {
        expected: u64,
        actual: u64,
    },
    CheckpointHistoryRoot,
    CheckpointStateRoot,
    WrongCheckpointDomain,
    WrongCheckpointVersion {
        major: u16,
        minor: u16,
    },
    MalformedCheckpoint,
    CheckpointChecksum,
    FaultedCheckpoint(AuthoredShadowFault),
    StoreBaseline {
        expected: Option<u64>,
        actual: Option<u64>,
    },
    Store(ReduceError),
}

impl fmt::Display for AuthoredShadowInitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Adapter(error) => error.fmt(formatter),
            Self::NonEmptyBaseline => formatter.write_str(
                "from_project requires empty HHHS history; use from_imported_project or from_project_checkpoint for a materialized baseline",
            ),
            Self::CheckpointProject { expected, actual } => write!(
                formatter,
                "authored checkpoint belongs to project {}, expected {}",
                actual.as_uuid(),
                expected.as_uuid()
            ),
            Self::CheckpointHistoryLength { expected, actual } => write!(
                formatter,
                "authored checkpoint expected {expected} history entries, found {actual}"
            ),
            Self::CheckpointHistoryRoot => {
                formatter.write_str("authored checkpoint history root does not match HHHS")
            }
            Self::CheckpointStateRoot => {
                formatter.write_str("authored checkpoint state root does not match HHHS")
            }
            Self::WrongCheckpointDomain => {
                formatter.write_str("authored checkpoint domain is unsupported")
            }
            Self::WrongCheckpointVersion { major, minor } => write!(
                formatter,
                "authored checkpoint version {major}.{minor} is unsupported"
            ),
            Self::MalformedCheckpoint => {
                formatter.write_str("authored checkpoint is truncated or malformed")
            }
            Self::CheckpointChecksum => {
                formatter.write_str("authored checkpoint checksum does not match its contents")
            }
            Self::FaultedCheckpoint(fault) => write!(
                formatter,
                "authored checkpoint is unavailable after projection {} faulted with a durable prefix of {} commands",
                fault.projection_revision, fault.admitted_prefix
            ),
            Self::StoreBaseline { expected, actual } => write!(
                formatter,
                "AppStore authored baseline {actual:?} does not match checkpoint {expected:?}"
            ),
            Self::Store(error) => error.fmt(formatter),
        }
    }
}

impl Error for AuthoredShadowInitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Adapter(error) => Some(error),
            Self::Store(error) => Some(error),
            Self::NonEmptyBaseline
            | Self::CheckpointProject { .. }
            | Self::CheckpointHistoryLength { .. }
            | Self::CheckpointHistoryRoot
            | Self::CheckpointStateRoot
            | Self::WrongCheckpointDomain
            | Self::WrongCheckpointVersion { .. }
            | Self::MalformedCheckpoint
            | Self::CheckpointChecksum
            | Self::FaultedCheckpoint(_)
            | Self::StoreBaseline { .. } => None,
        }
    }
}

impl From<AdapterError> for AuthoredShadowInitError {
    fn from(error: AdapterError) -> Self {
        Self::Adapter(error)
    }
}

impl From<ReduceError> for AuthoredShadowInitError {
    fn from(error: ReduceError) -> Self {
        Self::Store(error)
    }
}

/// Source-local replay cursor paired with an exact durable HHHS horizon.
///
/// The roots and project identity prevent attaching a cursor to the wrong
/// history. The cursor is deliberately not an HHHS authored command: it is a
/// local ingest/echo-suppression checkpoint and may be absent after importing
/// an archive from another peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthoredShadowCheckpoint {
    pub project_id: ProjectId,
    pub projection_revision: Option<u64>,
    pub history_len: u64,
    pub history_root: [u8; 32],
    pub state_root: [u8; 32],
}

impl AuthoredShadowCheckpoint {
    /// Encode a fixed-width source-local checkpoint. The checksum detects
    /// corruption but is not an authority signature.
    pub fn encode(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(CHECKPOINT_BYTES);
        bytes.extend_from_slice(AUTHORED_SHADOW_CHECKPOINT_DOMAIN);
        bytes.extend_from_slice(&AUTHORED_SHADOW_CHECKPOINT_VERSION.0.to_le_bytes());
        bytes.extend_from_slice(&AUTHORED_SHADOW_CHECKPOINT_VERSION.1.to_le_bytes());
        bytes.extend_from_slice(self.project_id.as_uuid().as_bytes());
        bytes.push(u8::from(self.projection_revision.is_some()));
        bytes.extend_from_slice(&self.projection_revision.unwrap_or_default().to_le_bytes());
        bytes.extend_from_slice(&self.history_len.to_le_bytes());
        bytes.extend_from_slice(&self.history_root);
        bytes.extend_from_slice(&self.state_root);
        let checksum = Digest::of(&bytes);
        bytes.extend_from_slice(checksum.as_bytes());
        debug_assert_eq!(bytes.len(), CHECKPOINT_BYTES);
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, AuthoredShadowInitError> {
        if bytes.len() != CHECKPOINT_BYTES {
            return Err(AuthoredShadowInitError::MalformedCheckpoint);
        }
        if !bytes.starts_with(AUTHORED_SHADOW_CHECKPOINT_DOMAIN) {
            return Err(AuthoredShadowInitError::WrongCheckpointDomain);
        }
        let checksum_offset = bytes.len() - CHECKPOINT_CHECKSUM_BYTES;
        if Digest::of(&bytes[..checksum_offset]).as_bytes() != &bytes[checksum_offset..] {
            return Err(AuthoredShadowInitError::CheckpointChecksum);
        }
        let mut cursor = AUTHORED_SHADOW_CHECKPOINT_DOMAIN.len();
        let major = u16::from_le_bytes(checkpoint_array(bytes, &mut cursor)?);
        let minor = u16::from_le_bytes(checkpoint_array(bytes, &mut cursor)?);
        if (major, minor) != AUTHORED_SHADOW_CHECKPOINT_VERSION {
            return Err(AuthoredShadowInitError::WrongCheckpointVersion { major, minor });
        }
        let project_id =
            ProjectId::from_u128(u128::from_be_bytes(checkpoint_array(bytes, &mut cursor)?))?;
        let has_revision = checkpoint_array::<1>(bytes, &mut cursor)?[0];
        let raw_revision = u64::from_le_bytes(checkpoint_array(bytes, &mut cursor)?);
        let projection_revision = match (has_revision, raw_revision) {
            (0, 0) => None,
            (1, revision) => Some(revision),
            _ => return Err(AuthoredShadowInitError::MalformedCheckpoint),
        };
        let history_len = u64::from_le_bytes(checkpoint_array(bytes, &mut cursor)?);
        let history_root = checkpoint_array(bytes, &mut cursor)?;
        let state_root = checkpoint_array(bytes, &mut cursor)?;
        if cursor != checksum_offset {
            return Err(AuthoredShadowInitError::MalformedCheckpoint);
        }
        Ok(Self {
            project_id,
            projection_revision,
            history_len,
            history_root,
            state_root,
        })
    }
}

fn checkpoint_array<const N: usize>(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<[u8; N], AuthoredShadowInitError> {
    let end = cursor
        .checked_add(N)
        .filter(|end| *end <= bytes.len())
        .ok_or(AuthoredShadowInitError::MalformedCheckpoint)?;
    let value = bytes[*cursor..end]
        .try_into()
        .map_err(|_| AuthoredShadowInitError::MalformedCheckpoint)?;
    *cursor = end;
    Ok(value)
}

/// Both outcomes of one coordinated dispatch. A successful outer `Result`
/// means AppStore accepted the event as valid; inspect `shadow` separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthoredShadowDispatch {
    pub app: AppCommit,
    pub shadow: Result<AuthoredShadowObservation, AuthoredShadowError>,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct SequentialProjection {
    assets: BTreeMap<AssetId, AssetDescriptor>,
    entity_transforms: BTreeMap<EntityId, WireTransform>,
}

impl SequentialProjection {
    fn from_state(state: &ProjectState) -> Self {
        Self {
            assets: state.assets.clone(),
            entity_transforms: state.entity_transforms.clone(),
        }
    }

    fn apply(&mut self, envelope: &AuthoredEnvelope) {
        match &envelope.command {
            AuthoredCommand::UpsertAsset { asset } => {
                self.assets.insert(asset.id, asset.clone());
            }
            AuthoredCommand::SetEntityTransform { entity, transform } => {
                self.entity_transforms.insert(*entity, *transform);
            }
            AuthoredCommand::RemoveEntity { entity } => {
                self.entity_transforms.remove(entity);
            }
        }
    }

    fn matches(&self, state: &ProjectState) -> bool {
        self.assets == state.assets && self.entity_transforms == state.entity_transforms
    }

    fn matches_app(&self, scene: &AuthoredSceneReadModel) -> bool {
        scene.assets.len() == self.assets.len()
            && scene
                .assets
                .iter()
                .all(|asset| self.assets.get(&asset.id) == Some(asset))
            && scene.entities.len() == self.entity_transforms.len()
            && scene.entities.iter().all(|entity| {
                self.entity_transforms.get(&entity.entity) == Some(&entity.transform)
            })
    }

    fn snapshot(&self, projection_revision: u64) -> AuthoredProjectionSnapshot {
        AuthoredProjectionSnapshot {
            projection_revision,
            assets: self.assets.values().cloned().collect(),
            entities: self
                .entity_transforms
                .iter()
                .map(|(&entity, &transform)| AuthoredEntityReadModel { entity, transform })
                .collect(),
        }
    }
}

/// Permanent evidence that HHHS committed an authored envelope but its
/// rebuildable AppStore projection could not advance coherently afterward.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableAuthoredFault {
    pub projection_revision: u64,
    pub durable_entry: EntryHash,
    pub history_len: usize,
    pub reason: String,
}

#[derive(Debug, thiserror::Error)]
pub enum DurableAuthoredInitError {
    #[error(transparent)]
    Adapter(#[from] AdapterError),
    #[error("durable authored history has no local source cursor")]
    MissingCursor,
    #[error("empty durable authored history has a source cursor")]
    UnexpectedCursor,
    #[error("durable authored source cursor failed its integrity check")]
    CorruptCursor,
    #[error("durable authored source cursor is not at the current history root")]
    CursorHistoryRoot,
    #[error("durable authored source cursor bytes are malformed")]
    MalformedCursor,
}

#[derive(Debug, thiserror::Error)]
pub enum DurableAuthoredDispatchError {
    #[error(
        "durable authored projection {projection_revision} contains {command_count} commands; exactly one is supported"
    )]
    UnsupportedCommandCount {
        projection_revision: u64,
        command_count: usize,
    },
    #[error(
        "AppStore authored projection revision {actual:?} does not match durable cursor {expected:?}"
    )]
    StoreProjectionRevision {
        expected: Option<u64>,
        actual: Option<u64>,
    },
    #[error("AppStore authored projection content does not match durable history")]
    StoreProjectionContent,
    #[error(transparent)]
    Store(#[from] AuthoredProjectionDispatchError),
    #[error(transparent)]
    Adapter(#[from] AdapterError),
    #[error(
        "durable authored coordinator is poisoned by projection {}; skipped projection {skipped_projection_revision}",
        original.projection_revision
    )]
    Poisoned {
        original: DurableAuthoredFault,
        skipped_projection_revision: u64,
    },
}

/// Failure to align a rebuildable AppStore projection with recovered HHHS
/// authority.
#[derive(Debug, thiserror::Error)]
pub enum DurableAuthoredRestoreError {
    #[error(
        "durable authored coordinator is poisoned by projection {}",
        original.projection_revision
    )]
    Poisoned { original: DurableAuthoredFault },
    #[error(
        "AppStore authored projection revision {actual:?} does not match durable cursor {expected:?}"
    )]
    StoreProjectionRevision {
        expected: Option<u64>,
        actual: Option<u64>,
    },
    #[error("AppStore authored projection content does not match durable history")]
    StoreProjectionContent,
    #[error(transparent)]
    Store(#[from] ReduceError),
}

/// Result of one durable-first authored dispatch. The record in `Applied` is
/// safe to announce because external persistence and local HHHS publication
/// both completed before this value was returned. `app` remains a separate
/// rebuildable-projection outcome so an AppStore race cannot hide a committed
/// record from the carrier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurableAuthoredObservation {
    Applied {
        projection_revision: u64,
        record: Box<ReplicaRecord>,
        history_len: usize,
    },
    IgnoredStale {
        projection_revision: u64,
        observed_projection_revision: Option<u64>,
        history_len: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableAuthoredDispatch {
    pub app: Result<AppCommit, DurableAuthoredFault>,
    pub durable: DurableAuthoredObservation,
}

/// One transport-neutral local peer admission. Presence and ignored authored
/// frames have no durable result; a newly admitted authored frame carries the
/// announceable HHHS record and the separate AppStore projection outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableLocalPeerDispatch {
    pub peer: LocalPeerReceipt,
    pub durable: Option<DurableAuthoredDispatch>,
}

/// Durable outcome of offering one typed HHHS record from a carrier.
///
/// Only `Applied` advances the receiver-local projection cursor. Duplicate,
/// deferred, and refused records leave both the cursor and AppStore unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurableCarrierObservation {
    Applied {
        projection_revision: u64,
        entry: EntryHash,
        history_len: usize,
    },
    AlreadyPresent {
        entry: EntryHash,
        history_len: usize,
    },
    Deferred {
        entry: EntryHash,
        missing: Vec<EntryHash>,
        history_len: usize,
    },
    Refused {
        entry: EntryHash,
        reason: RecordRefusal,
        history_len: usize,
    },
}

/// One carrier admission with its independent rebuildable-projection result.
/// `app` is present only when a newly durable record required projection
/// synchronization. An applied carrier record remains committed even if that
/// synchronization detects a concurrent AppStore bypass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableCarrierDispatch {
    pub app: Option<Result<AppCommit, DurableAuthoredFault>>,
    pub durable: DurableCarrierObservation,
}

#[derive(Debug, thiserror::Error)]
pub enum DurableLocalPeerError {
    #[error(transparent)]
    Peer(#[from] LocalPeerIngressError),
    #[error("local peer authored projection revision overflow")]
    ProjectionRevisionOverflow,
    #[error(transparent)]
    Durable(#[from] DurableAuthoredDispatchError),
}

#[derive(Debug, thiserror::Error)]
pub enum DurableCarrierError {
    #[error("inbound carrier projection revision overflow")]
    ProjectionRevisionOverflow,
    #[error(
        "AppStore authored projection revision {actual:?} does not match durable cursor {expected:?}"
    )]
    StoreProjectionRevision {
        expected: Option<u64>,
        actual: Option<u64>,
    },
    #[error("AppStore authored projection content does not match durable history")]
    StoreProjectionContent,
    #[error(transparent)]
    Peer(#[from] LocalPeerIngressError),
    #[error(transparent)]
    Adapter(#[from] AdapterError),
    #[error(
        "durable authored coordinator is poisoned by projection {}; skipped carrier entry {skipped_entry:?}",
        original.projection_revision
    )]
    Poisoned {
        original: DurableAuthoredFault,
        skipped_entry: EntryHash,
    },
}

/// Failure to construct the single-writer authored session from its durable
/// project and recovered peer-ingress memory.
#[derive(Debug, thiserror::Error)]
pub enum DurableAuthoredSessionInitError {
    #[error(transparent)]
    Adapter(#[from] AdapterError),
    #[error(transparent)]
    Coordinator(#[from] DurableAuthoredInitError),
    #[error(transparent)]
    Peer(#[from] LocalPeerIngressError),
    #[error("durable authored source cursor revision overflow during archive adoption")]
    ProjectionRevisionOverflow,
}

/// One-envelope HHHS-authoritative ingress with an AppStore projection.
///
/// Unlike [`AuthoredHhhsShadow`], persistence completes before AppStore is
/// allowed to publish. Multi-command revisions are rejected before either
/// side changes; adding them requires an explicit atomic batch payload rather
/// than a loop of individually durable entries.
pub struct DurableAuthoredCoordinator<D = MemoryDurability> {
    project: DurableProject<D>,
    expected: SequentialProjection,
    observed_projection_revision: Option<u64>,
    fault: Option<DurableAuthoredFault>,
}

/// One restart-safe, single-writer authored ingress boundary.
///
/// The coordinator owns durable HHHS authority and the ingress owns bounded
/// echo/deduplication policy. Keeping them together ensures restart recovery
/// cannot pair an advanced durable cursor with empty authored message memory.
/// Presence remains ephemeral and is never reconstructed from HHHS.
pub struct DurableAuthoredSession<D = MemoryDurability> {
    coordinator: DurableAuthoredCoordinator<D>,
    ingress: LocalPeerIngress,
}

impl DurableAuthoredSession<MemoryDurability> {
    pub fn new(project_id: ProjectId) -> Result<Self, DurableAuthoredSessionInitError> {
        Self::from_project(DurableProject::new(project_id)?)
    }

    pub fn recover(
        project_id: ProjectId,
        durable_bytes: Vec<u8>,
    ) -> Result<Self, DurableAuthoredSessionInitError> {
        Self::from_project(DurableProject::recover(project_id, durable_bytes)?)
    }
}

impl<D> DurableAuthoredSession<D> {
    pub fn from_project(
        project: DurableProject<D>,
    ) -> Result<Self, DurableAuthoredSessionInitError> {
        Self::from_project_with_message_capacity(project, DEFAULT_LOCAL_PEER_MESSAGE_MEMORY)
    }

    pub fn from_project_with_message_capacity(
        project: DurableProject<D>,
        message_capacity: usize,
    ) -> Result<Self, DurableAuthoredSessionInitError> {
        let authored_history = project.authored_history()?;
        let coordinator = DurableAuthoredCoordinator::from_project(project)?;
        let ingress =
            LocalPeerIngress::from_authored_history(message_capacity, &authored_history)?;
        Ok(Self {
            coordinator,
            ingress,
        })
    }

    pub fn project_id(&self) -> ProjectId {
        self.coordinator.project_id()
    }

    pub fn history_len(&self) -> usize {
        self.coordinator.history_len()
    }

    pub fn project_state(&self) -> Result<ProjectState, AdapterError> {
        self.coordinator.project_state()
    }

    pub fn observed_projection_revision(&self) -> Option<u64> {
        self.coordinator.observed_projection_revision()
    }

    pub fn fault(&self) -> Option<&DurableAuthoredFault> {
        self.coordinator.fault()
    }

    pub fn durability(&self) -> &D {
        self.coordinator.durability()
    }

    pub fn durability_mut(&mut self) -> &mut D {
        self.coordinator.durability_mut()
    }

    pub fn restore_store(
        &self,
        store: &AppStore,
    ) -> Result<Option<AppCommit>, DurableAuthoredRestoreError> {
        self.coordinator.restore_store(store)
    }

    pub fn record_local_presence(
        &mut self,
        envelope: &PresenceEnvelope,
    ) -> Result<(), LocalPeerIngressError> {
        self.ingress.record_local_presence(envelope)
    }
}

impl<D> DurableAuthoredSession<D>
where
    D: AsyncTransactionSink,
{
    /// Adopt a project produced by an explicit portable-archive import.
    ///
    /// Portable archives intentionally omit receiver-local projection state.
    /// A fresh non-empty import starts its local cursor at revision zero. If
    /// importing extended a locally durable history, its prior cursor must
    /// prove an exact history prefix and advances once. An already-current
    /// cursor is an exact zero-write retry. Ordinary restart should continue
    /// to use [`Self::from_project`], which never repairs cursor state.
    pub async fn from_imported_project(
        project: DurableProject<D>,
    ) -> Result<Self, DurableAuthoredSessionInitError> {
        Self::from_imported_project_with_message_capacity(
            project,
            DEFAULT_LOCAL_PEER_MESSAGE_MEMORY,
        )
        .await
    }

    pub async fn from_imported_project_with_message_capacity(
        mut project: DurableProject<D>,
        message_capacity: usize,
    ) -> Result<Self, DurableAuthoredSessionInitError> {
        let state = project.state()?;
        let checkpoint = project.projection_checkpoint(&durable_authored_cursor_key());
        let adopted_revision = match (project.history_len(), checkpoint) {
            (0, None) => None,
            (0, Some(_)) => return Err(DurableAuthoredInitError::UnexpectedCursor.into()),
            (_, None) => Some(0),
            (_, Some(checkpoint)) => {
                if !checkpoint.is_intact() {
                    return Err(DurableAuthoredInitError::CorruptCursor.into());
                }
                let bytes: [u8; 8] = checkpoint
                    .bytes()
                    .try_into()
                    .map_err(|_| DurableAuthoredInitError::MalformedCursor)?;
                if checkpoint.history_root.as_bytes() == &state.history_root {
                    None
                } else {
                    if !project.projection_checkpoint_matches_history_prefix(&checkpoint) {
                        return Err(DurableAuthoredInitError::CursorHistoryRoot.into());
                    }
                    Some(
                        u64::from_le_bytes(bytes)
                            .checked_add(1)
                            .ok_or(DurableAuthoredSessionInitError::ProjectionRevisionOverflow)?,
                    )
                }
            }
        };

        if let Some(revision) = adopted_revision {
            project
                .persist_projection_checkpoint(
                    durable_authored_cursor_key(),
                    revision.to_le_bytes().to_vec(),
                )
                .await?;
        }
        Self::from_project_with_message_capacity(project, message_capacity)
    }

    /// Admit an authored or remote-presence peer envelope through the paired
    /// restart-safe ingress. A locally originated authored envelope should
    /// pass through this method before transport; its relay echo is then an
    /// ordinary zero-write duplicate. Locally originated presence should use
    /// [`Self::record_local_presence`] before transport instead.
    pub async fn accept_local_peer(
        &mut self,
        store: &AppStore,
        envelope: LocalPeerEnvelope,
        received_at_seconds: f64,
    ) -> Result<DurableLocalPeerDispatch, DurableLocalPeerError> {
        self.coordinator
            .accept_local_peer(
                &mut self.ingress,
                store,
                envelope,
                received_at_seconds,
            )
            .await
    }

    /// Consume one typed record from a carrier through the same durable
    /// single-writer boundary as local authoring.
    pub async fn accept_replica_record(
        &mut self,
        store: &AppStore,
        record: ReplicaRecord,
    ) -> Result<DurableCarrierDispatch, DurableCarrierError> {
        self.coordinator
            .accept_replica_record(&mut self.ingress, store, record)
            .await
    }
}

impl DurableAuthoredCoordinator<MemoryDurability> {
    pub fn new(project_id: ProjectId) -> Result<Self, DurableAuthoredInitError> {
        Self::from_project(DurableProject::new(project_id)?)
    }

    pub fn recover(
        project_id: ProjectId,
        durable_bytes: Vec<u8>,
    ) -> Result<Self, DurableAuthoredInitError> {
        Self::from_project(DurableProject::recover(project_id, durable_bytes)?)
    }
}

impl<D> DurableAuthoredCoordinator<D> {
    pub fn from_project(project: DurableProject<D>) -> Result<Self, DurableAuthoredInitError> {
        let state = project.state()?;
        let checkpoint = project.projection_checkpoint(&durable_authored_cursor_key());
        let observed_projection_revision = match (project.history_len(), checkpoint) {
            (0, None) => None,
            (0, Some(_)) => return Err(DurableAuthoredInitError::UnexpectedCursor),
            (_, None) => return Err(DurableAuthoredInitError::MissingCursor),
            (_, Some(checkpoint)) => {
                if !checkpoint.is_intact() {
                    return Err(DurableAuthoredInitError::CorruptCursor);
                }
                if checkpoint.history_root.as_bytes() != &state.history_root {
                    return Err(DurableAuthoredInitError::CursorHistoryRoot);
                }
                let bytes: [u8; 8] = checkpoint
                    .bytes()
                    .try_into()
                    .map_err(|_| DurableAuthoredInitError::MalformedCursor)?;
                Some(u64::from_le_bytes(bytes))
            }
        };
        Ok(Self {
            project,
            expected: SequentialProjection::from_state(&state),
            observed_projection_revision,
            fault: None,
        })
    }

    pub fn project_id(&self) -> ProjectId {
        self.project.project_id()
    }

    pub fn history_len(&self) -> usize {
        self.project.history_len()
    }

    pub fn project_state(&self) -> Result<ProjectState, AdapterError> {
        self.project.state()
    }

    pub fn observed_projection_revision(&self) -> Option<u64> {
        self.observed_projection_revision
    }

    pub fn fault(&self) -> Option<&DurableAuthoredFault> {
        self.fault.as_ref()
    }

    pub fn durability(&self) -> &D {
        self.project.durability()
    }

    pub fn durability_mut(&mut self) -> &mut D {
        self.project.durability_mut()
    }

    /// Restore recovered HHHS materialization into an empty AppStore authored
    /// lane without inventing authored commands or adding durable history.
    ///
    /// Exact parity is an idempotent no-op. A non-empty divergent AppStore is
    /// rejected rather than overwritten. The reducer validates and installs
    /// the canonical snapshot atomically under the AppStore lock.
    pub fn restore_store(
        &self,
        store: &AppStore,
    ) -> Result<Option<AppCommit>, DurableAuthoredRestoreError> {
        if let Some(original) = &self.fault {
            return Err(DurableAuthoredRestoreError::Poisoned {
                original: original.clone(),
            });
        }

        let baseline = store.authored_scene_snapshot();
        if baseline.projection_revision == self.observed_projection_revision
            && self.expected.matches_app(&baseline)
        {
            return Ok(None);
        }
        if baseline.projection_revision.is_some()
            || !baseline.assets.is_empty()
            || !baseline.entities.is_empty()
        {
            if baseline.projection_revision != self.observed_projection_revision {
                return Err(DurableAuthoredRestoreError::StoreProjectionRevision {
                    expected: self.observed_projection_revision,
                    actual: baseline.projection_revision,
                });
            }
            return Err(DurableAuthoredRestoreError::StoreProjectionContent);
        }

        let Some(projection_revision) = self.observed_projection_revision else {
            return Err(DurableAuthoredRestoreError::StoreProjectionContent);
        };
        let commit = store.dispatch(AppEvent::AuthoredProjectionRestored(
            self.expected.snapshot(projection_revision),
        ))?;
        let restored = store.authored_scene_snapshot();
        if restored.projection_revision != self.observed_projection_revision {
            return Err(DurableAuthoredRestoreError::StoreProjectionRevision {
                expected: self.observed_projection_revision,
                actual: restored.projection_revision,
            });
        }
        if commit.disposition != CommitDisposition::Applied || !self.expected.matches_app(&restored)
        {
            return Err(DurableAuthoredRestoreError::StoreProjectionContent);
        }
        Ok(Some(commit))
    }
}

impl<D> DurableAuthoredCoordinator<D>
where
    D: AsyncTransactionSink,
{
    /// Persist one carrier record and the receiver-local cursor atomically,
    /// then replace the AppStore authored read model with canonical HHHS
    /// materialization. Remote commands are never replayed in arrival order.
    pub async fn accept_replica_record(
        &mut self,
        ingress: &mut LocalPeerIngress,
        store: &AppStore,
        record: ReplicaRecord,
    ) -> Result<DurableCarrierDispatch, DurableCarrierError> {
        let entry = record.entry_hash();
        let envelope = decode_authored(self.project_id(), &record.entry().payload)?;
        if let Some(original) = &self.fault {
            return Err(DurableCarrierError::Poisoned {
                original: original.clone(),
                skipped_entry: entry,
            });
        }

        let app_baseline = store.authored_scene_snapshot();
        if app_baseline.projection_revision != self.observed_projection_revision {
            return Err(DurableCarrierError::StoreProjectionRevision {
                expected: self.observed_projection_revision,
                actual: app_baseline.projection_revision,
            });
        }
        if !self.expected.matches_app(&app_baseline) {
            return Err(DurableCarrierError::StoreProjectionContent);
        }

        let projection_revision = self
            .observed_projection_revision
            .map_or(Some(0), |revision| revision.checked_add(1))
            .ok_or(DurableCarrierError::ProjectionRevisionOverflow)?;
        let admission = self
            .project
            .apply_record_with_projection_checkpoint(
                record,
                durable_authored_cursor_key(),
                projection_revision.to_le_bytes().to_vec(),
            )
            .await?;

        match admission {
            ApplyRecordWithCheckpointReport::AlreadyPresent { entry } => {
                ingress.observe_durable_authored(&envelope)?;
                Ok(DurableCarrierDispatch {
                    app: None,
                    durable: DurableCarrierObservation::AlreadyPresent {
                        entry,
                        history_len: self.project.history_len(),
                    },
                })
            }
            ApplyRecordWithCheckpointReport::Deferred { entry, missing } => {
                Ok(DurableCarrierDispatch {
                    app: None,
                    durable: DurableCarrierObservation::Deferred {
                        entry,
                        missing,
                        history_len: self.project.history_len(),
                    },
                })
            }
            ApplyRecordWithCheckpointReport::Refused { entry, reason } => {
                Ok(DurableCarrierDispatch {
                    app: None,
                    durable: DurableCarrierObservation::Refused {
                        entry,
                        reason,
                        history_len: self.project.history_len(),
                    },
                })
            }
            ApplyRecordWithCheckpointReport::Applied { entry } => {
                ingress.observe_durable_authored(&envelope)?;
                self.observed_projection_revision = Some(projection_revision);
                let state = match self.project.state() {
                    Ok(state) => state,
                    Err(error) => {
                        return Ok(self.faulted_carrier_dispatch(
                            projection_revision,
                            entry,
                            error.to_string(),
                        ));
                    }
                };
                self.expected = SequentialProjection::from_state(&state);

                let cursor = self
                    .project
                    .projection_checkpoint(&durable_authored_cursor_key());
                let cursor_matches = cursor.is_some_and(|cursor| {
                    cursor.is_intact()
                        && cursor.history_root.as_bytes() == &state.history_root
                        && cursor.bytes() == projection_revision.to_le_bytes()
                });
                if !cursor_matches {
                    return Ok(self.faulted_carrier_dispatch(
                        projection_revision,
                        entry,
                        "receiver-local source cursor did not commit at the inbound history horizon"
                            .into(),
                    ));
                }

                let app = match store.synchronize_authored_projection_if_current(
                    app_baseline.projection_revision,
                    self.expected.snapshot(projection_revision),
                ) {
                    Ok(app) => app,
                    Err(error) => {
                        return Ok(self.faulted_carrier_dispatch(
                            projection_revision,
                            entry,
                            error.to_string(),
                        ));
                    }
                };
                if app.disposition != CommitDisposition::Applied {
                    return Ok(self.faulted_carrier_dispatch(
                        projection_revision,
                        entry,
                        "AppStore ignored a projection already committed to HHHS".into(),
                    ));
                }
                if !self.expected.matches_app(&store.authored_scene_snapshot()) {
                    return Ok(self.faulted_carrier_dispatch(
                        projection_revision,
                        entry,
                        "AppStore projection diverged after inbound carrier admission".into(),
                    ));
                }

                Ok(DurableCarrierDispatch {
                    app: Some(Ok(app)),
                    durable: DurableCarrierObservation::Applied {
                        projection_revision,
                        entry,
                        history_len: self.project.history_len(),
                    },
                })
            }
        }
    }

    /// Admit one local Blender/browser peer frame without making ephemeral
    /// presence durable.
    ///
    /// Authored deduplication is reserved across the asynchronous HHHS write.
    /// A pre-commit failure drops that reservation, so the exact envelope can
    /// retry. Once HHHS succeeds, ingress memory advances even if AppStore must
    /// be rebuilt; the returned durable record remains safe to announce.
    pub async fn accept_local_peer(
        &mut self,
        ingress: &mut LocalPeerIngress,
        store: &AppStore,
        envelope: LocalPeerEnvelope,
        received_at_seconds: f64,
    ) -> Result<DurableLocalPeerDispatch, DurableLocalPeerError> {
        let authored = match envelope {
            LocalPeerEnvelope::Presence(presence) => {
                let peer = ingress.accept(
                    store,
                    LocalPeerEnvelope::Presence(presence),
                    received_at_seconds,
                )?;
                return Ok(DurableLocalPeerDispatch {
                    peer,
                    durable: None,
                });
            }
            LocalPeerEnvelope::Authored(authored) => authored,
        };
        let pending = match ingress.prepare_authored(authored)? {
            LocalAuthoredPreparation::Immediate(peer) => {
                return Ok(DurableLocalPeerDispatch {
                    peer,
                    durable: None,
                });
            }
            LocalAuthoredPreparation::Pending(pending) => pending,
        };
        let projection_revision = self
            .observed_projection_revision
            .map_or(Some(0), |revision| revision.checked_add(1))
            .ok_or(DurableLocalPeerError::ProjectionRevisionOverflow)?;
        let durable = self
            .dispatch(
                store,
                AuthoredRevision {
                    projection_revision,
                    commands: vec![pending.envelope().clone()],
                },
            )
            .await?;
        let app_commit = durable.app.as_ref().ok().cloned();
        let peer = pending.complete(projection_revision, app_commit);
        Ok(DurableLocalPeerDispatch {
            peer,
            durable: Some(durable),
        })
    }

    pub async fn dispatch(
        &mut self,
        store: &AppStore,
        revision: AuthoredRevision,
    ) -> Result<DurableAuthoredDispatch, DurableAuthoredDispatchError> {
        if let Some(original) = &self.fault {
            return Err(DurableAuthoredDispatchError::Poisoned {
                original: original.clone(),
                skipped_projection_revision: revision.projection_revision,
            });
        }

        let app_baseline = store.authored_scene_snapshot();
        if app_baseline.projection_revision != self.observed_projection_revision {
            return Err(DurableAuthoredDispatchError::StoreProjectionRevision {
                expected: self.observed_projection_revision,
                actual: app_baseline.projection_revision,
            });
        }
        if !self.expected.matches_app(&app_baseline) {
            return Err(DurableAuthoredDispatchError::StoreProjectionContent);
        }

        if self
            .observed_projection_revision
            .is_some_and(|current| revision.projection_revision <= current)
        {
            let app = store
                .dispatch_authored_if_current(self.observed_projection_revision, revision.clone())
                .map_err(DurableAuthoredDispatchError::Store)?;
            return Ok(DurableAuthoredDispatch {
                app: Ok(app),
                durable: DurableAuthoredObservation::IgnoredStale {
                    projection_revision: revision.projection_revision,
                    observed_projection_revision: self.observed_projection_revision,
                    history_len: self.project.history_len(),
                },
            });
        }

        if revision.commands.len() != 1 {
            return Err(DurableAuthoredDispatchError::UnsupportedCommandCount {
                projection_revision: revision.projection_revision,
                command_count: revision.commands.len(),
            });
        }

        let envelope = &revision.commands[0];
        let mut next_expected = self.expected.clone();
        next_expected.apply(envelope);
        let record = self
            .project
            .admit_with_projection_checkpoint(
                envelope,
                durable_authored_cursor_key(),
                revision.projection_revision.to_le_bytes().to_vec(),
            )
            .await?;

        self.expected = next_expected;
        self.observed_projection_revision = Some(revision.projection_revision);
        let state = match self.project.state() {
            Ok(state) => state,
            Err(error) => {
                return Ok(self.faulted_dispatch(
                    revision.projection_revision,
                    record,
                    error.to_string(),
                ));
            }
        };
        if !self.expected.matches(&state) {
            return Ok(self.faulted_dispatch(
                revision.projection_revision,
                record,
                "HHHS materialization diverged from the sequential authored projection".into(),
            ));
        }

        let cursor = self
            .project
            .projection_checkpoint(&durable_authored_cursor_key());
        let cursor_matches = cursor.is_some_and(|cursor| {
            cursor.is_intact()
                && cursor.history_root.as_bytes() == &state.history_root
                && cursor.bytes() == revision.projection_revision.to_le_bytes()
        });
        if !cursor_matches {
            return Ok(self.faulted_dispatch(
                revision.projection_revision,
                record,
                "receiver-local source cursor did not commit at the authored history horizon"
                    .into(),
            ));
        }

        let app = match store.dispatch_authored_if_current(
            app_baseline.projection_revision,
            revision.clone(),
        ) {
            Ok(app) => app,
            Err(error) => {
                return Ok(self.faulted_dispatch(
                    revision.projection_revision,
                    record,
                    error.to_string(),
                ));
            }
        };
        if app.disposition != CommitDisposition::Applied {
            return Ok(self.faulted_dispatch(
                revision.projection_revision,
                record,
                "AppStore ignored a revision already committed to HHHS".into(),
            ));
        }
        if !self.expected.matches_app(&store.authored_scene_snapshot()) {
            return Ok(self.faulted_dispatch(
                revision.projection_revision,
                record,
                "AppStore projection diverged after durable authored dispatch".into(),
            ));
        }

        Ok(DurableAuthoredDispatch {
            app: Ok(app),
            durable: DurableAuthoredObservation::Applied {
                projection_revision: revision.projection_revision,
                record: Box::new(record),
                history_len: self.project.history_len(),
            },
        })
    }

    fn faulted_dispatch(
        &mut self,
        projection_revision: u64,
        record: ReplicaRecord,
        reason: String,
    ) -> DurableAuthoredDispatch {
        let fault = DurableAuthoredFault {
            projection_revision,
            durable_entry: record.entry_hash(),
            history_len: self.project.history_len(),
            reason,
        };
        self.fault = Some(fault.clone());
        DurableAuthoredDispatch {
            app: Err(fault),
            durable: DurableAuthoredObservation::Applied {
                projection_revision,
                record: Box::new(record),
                history_len: self.project.history_len(),
            },
        }
    }

    fn faulted_carrier_dispatch(
        &mut self,
        projection_revision: u64,
        entry: EntryHash,
        reason: String,
    ) -> DurableCarrierDispatch {
        let fault = DurableAuthoredFault {
            projection_revision,
            durable_entry: entry,
            history_len: self.project.history_len(),
            reason,
        };
        self.fault = Some(fault.clone());
        DurableCarrierDispatch {
            app: Some(Err(fault)),
            durable: DurableCarrierObservation::Applied {
                projection_revision,
                entry,
                history_len: self.project.history_len(),
            },
        }
    }
}

/// Explicit, opt-in coordinator for authored-revision behavior shadowing.
///
/// Commands are mirrored in their `AuthoredRevision::commands` vector order;
/// sender sequence numbers are metadata, not a sorting authority. Each local
/// admission consequently observes the preceding command in that vector.
/// Concurrent peer history belongs at [`DurableAuthoredSession::accept_replica_record`],
/// outside this local arrival-ordered diagnostic observer.
pub struct AuthoredHhhsShadow<D = MemoryDurability> {
    project: DurableProject<D>,
    expected: SequentialProjection,
    observed_projection_revision: Option<u64>,
    fault: Option<AuthoredShadowFault>,
}

impl AuthoredHhhsShadow<MemoryDurability> {
    /// Construct the process-local diagnostic checkpoint. This is not a
    /// durable browser store.
    pub fn new(project_id: ProjectId) -> Result<Self, AuthoredShadowInitError> {
        Self::from_project(DurableProject::new(project_id)?)
    }

    /// Import a portable HHHS archive as a new local projection source. The
    /// materialized scene baseline is retained, while the source-local
    /// projection cursor intentionally starts absent.
    pub async fn import_archive(bytes: &[u8]) -> Result<Self, AuthoredShadowInitError> {
        Self::from_imported_project(DurableProject::import_archive(bytes).await?)
    }
}

impl<D> AuthoredHhhsShadow<D> {
    /// Attach a newly constructed empty durable project.
    ///
    /// Recovered history must use [`Self::from_project_checkpoint`]; history
    /// imported from another source uses [`Self::from_imported_project`].
    pub fn from_project(project: DurableProject<D>) -> Result<Self, AuthoredShadowInitError> {
        let state = project.state()?;
        if project.history_len() != 0
            || !state.assets.is_empty()
            || !state.entity_transforms.is_empty()
        {
            return Err(AuthoredShadowInitError::NonEmptyBaseline);
        }
        Ok(Self {
            project,
            expected: SequentialProjection::default(),
            observed_projection_revision: None,
            fault: None,
        })
    }

    /// Attach imported history without claiming a source-local projection
    /// cursor. A fresh AppStore therefore aligns at `None`; the first new
    /// revision establishes the local source cursor.
    pub fn from_imported_project(
        project: DurableProject<D>,
    ) -> Result<Self, AuthoredShadowInitError> {
        Self::from_materialized_project(project, None)
    }

    /// Recover both durable history and its source-local projection cursor.
    /// The checkpoint must name this exact project horizon.
    pub fn from_project_checkpoint(
        project: DurableProject<D>,
        checkpoint: AuthoredShadowCheckpoint,
    ) -> Result<Self, AuthoredShadowInitError> {
        let state = project.state()?;
        if checkpoint.project_id != project.project_id() {
            return Err(AuthoredShadowInitError::CheckpointProject {
                expected: project.project_id(),
                actual: checkpoint.project_id,
            });
        }
        let history_len = u64::try_from(project.history_len())
            .expect("supported Rust targets have at most 64-bit usize");
        if checkpoint.history_len != history_len {
            return Err(AuthoredShadowInitError::CheckpointHistoryLength {
                expected: checkpoint.history_len,
                actual: history_len,
            });
        }
        if checkpoint.history_root != state.history_root {
            return Err(AuthoredShadowInitError::CheckpointHistoryRoot);
        }
        if checkpoint.state_root != state.state_root {
            return Err(AuthoredShadowInitError::CheckpointStateRoot);
        }
        Ok(Self {
            project,
            expected: SequentialProjection::from_state(&state),
            observed_projection_revision: checkpoint.projection_revision,
            fault: None,
        })
    }

    fn from_materialized_project(
        project: DurableProject<D>,
        projection_revision: Option<u64>,
    ) -> Result<Self, AuthoredShadowInitError> {
        let state = project.state()?;
        Ok(Self {
            project,
            expected: SequentialProjection::from_state(&state),
            observed_projection_revision: projection_revision,
            fault: None,
        })
    }

    pub fn project_id(&self) -> ProjectId {
        self.project.project_id()
    }

    pub fn history_len(&self) -> usize {
        self.project.history_len()
    }

    pub fn project_state(&self) -> Result<ProjectState, AdapterError> {
        self.project.state()
    }

    pub fn observed_projection_revision(&self) -> Option<u64> {
        self.observed_projection_revision
    }

    pub fn fault(&self) -> Option<&AuthoredShadowFault> {
        self.fault.as_ref()
    }

    pub fn checkpoint(&self) -> Result<AuthoredShadowCheckpoint, AuthoredShadowInitError> {
        if let Some(fault) = &self.fault {
            return Err(AuthoredShadowInitError::FaultedCheckpoint(fault.clone()));
        }
        let state = self.project.state()?;
        Ok(AuthoredShadowCheckpoint {
            project_id: self.project.project_id(),
            projection_revision: self.observed_projection_revision,
            history_len: u64::try_from(self.project.history_len())
                .expect("supported Rust targets have at most 64-bit usize"),
            history_root: state.history_root,
            state_root: state.state_root,
        })
    }

    pub fn export_archive(&self) -> Result<Vec<u8>, AdapterError> {
        self.project.export_archive()
    }

    /// Align a fresh/recovered AppStore with this source-local cursor through
    /// the existing reducer event. No authored command is generated or added
    /// to HHHS. Exact alignment is a zero-revision no-op.
    pub fn align_store(
        &self,
        store: &AppStore,
    ) -> Result<Option<AppCommit>, AuthoredShadowInitError> {
        if let Some(fault) = &self.fault {
            return Err(AuthoredShadowInitError::FaultedCheckpoint(fault.clone()));
        }
        let actual = store.summary_snapshot().authored_projection_revision;
        let expected = self.observed_projection_revision;
        if actual == expected {
            return Ok(None);
        }
        if actual.is_none() {
            if let Some(projection_revision) = expected {
                let commit = store.dispatch(AppEvent::AuthoredRevision(AuthoredRevision {
                    projection_revision,
                    commands: Vec::new(),
                }))?;
                return Ok(Some(commit));
            }
        }
        Err(AuthoredShadowInitError::StoreBaseline { expected, actual })
    }
}

impl<D> AuthoredHhhsShadow<D>
where
    D: AsyncTransactionSink,
{
    /// Dispatch an authored revision to AppStore, then observe it with HHHS.
    ///
    /// The synchronous AppStore dispatch and UI publication happen before the
    /// first await. Invalid revisions return [`ReduceError`] and never reach
    /// HHHS. A valid AppStore commit is never converted into an outer error by
    /// the diagnostic observer.
    pub async fn dispatch(
        &mut self,
        store: &AppStore,
        revision: AuthoredRevision,
    ) -> Result<AuthoredShadowDispatch, ReduceError> {
        // The coordinator must be the exclusive authored-revision path for its
        // AppStore. Reading this immediately before dispatch makes a direct
        // bypass visible without requiring HHHS state inside AppStore.
        let app_baseline = store.summary_snapshot().authored_projection_revision;
        let app = store.dispatch(AppEvent::AuthoredRevision(revision.clone()))?;
        let shadow = if let Some(original) = &self.fault {
            Err(AuthoredShadowError::Poisoned {
                original: original.clone(),
                skipped_projection_revision: revision.projection_revision,
            })
        } else if app_baseline != self.observed_projection_revision {
            Err(self.poison(
                revision.projection_revision,
                0,
                0,
                format!(
                    "AppStore authored baseline {app_baseline:?} bypassed shadow baseline {:?}",
                    self.observed_projection_revision
                ),
            ))
        } else {
            self.observe_committed(&revision, app.disposition).await
        };
        Ok(AuthoredShadowDispatch { app, shadow })
    }

    async fn observe_committed(
        &mut self,
        revision: &AuthoredRevision,
        disposition: CommitDisposition,
    ) -> Result<AuthoredShadowObservation, AuthoredShadowError> {
        if let Some(original) = &self.fault {
            return Err(AuthoredShadowError::Poisoned {
                original: original.clone(),
                skipped_projection_revision: revision.projection_revision,
            });
        }

        if disposition == CommitDisposition::IgnoredStale {
            return Ok(AuthoredShadowObservation::IgnoredStale {
                projection_revision: revision.projection_revision,
                observed_projection_revision: self.observed_projection_revision,
                history_len: self.project.history_len(),
            });
        }

        if self
            .observed_projection_revision
            .is_some_and(|current| revision.projection_revision <= current)
        {
            return Err(self.poison(
                revision.projection_revision,
                0,
                0,
                format!(
                    "AppStore applied non-increasing projection revision after shadow revision {:?}",
                    self.observed_projection_revision
                ),
            ));
        }

        for (index, envelope) in revision.commands.iter().enumerate() {
            if let Err(error) = self.project.admit(envelope).await {
                return Err(self.poison(
                    revision.projection_revision,
                    index,
                    index,
                    error.to_string(),
                ));
            }
            self.expected.apply(envelope);
        }

        let state = match self.project.state() {
            Ok(state) => state,
            Err(error) => {
                return Err(self.poison(
                    revision.projection_revision,
                    revision.commands.len(),
                    revision.commands.len(),
                    error.to_string(),
                ));
            }
        };
        if !self.expected.matches(&state) {
            return Err(self.poison(
                revision.projection_revision,
                revision.commands.len(),
                revision.commands.len(),
                "HHHS materialization diverged from the sequential authored projection".into(),
            ));
        }

        self.observed_projection_revision = Some(revision.projection_revision);
        Ok(AuthoredShadowObservation::Matched {
            projection_revision: revision.projection_revision,
            command_count: revision.commands.len(),
            history_len: self.project.history_len(),
            history_root: state.history_root,
            state_root: state.state_root,
        })
    }

    fn poison(
        &mut self,
        projection_revision: u64,
        failed_command_index: usize,
        admitted_prefix: usize,
        reason: String,
    ) -> AuthoredShadowError {
        let fault = AuthoredShadowFault {
            projection_revision,
            failed_command_index,
            admitted_prefix,
            history_len: self.project.history_len(),
            reason,
        };
        self.fault = Some(fault.clone());
        AuthoredShadowError::Fault(fault)
    }
}
