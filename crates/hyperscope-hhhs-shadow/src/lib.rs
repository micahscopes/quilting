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
//! This crate accepts only [`AuthoredRevision`]. Camera motion, frame clocks,
//! presence, selection, animation, renderer state, and GPU resources are not
//! representable at this boundary.
//!
//! The coordinator must be the exclusive path for authored revisions on its
//! AppStore. Other AppStore event kinds may still be dispatched normally. A
//! directly dispatched authored revision changes the AppStore baseline without
//! HHHS; the next coordinated dispatch detects that drift, skips mirroring, and
//! poisons the observer. Authored writers must therefore be serialized around
//! this diagnostic boundary.

#![forbid(unsafe_code)]

use hhhs::Digest;
use hhhs_replica::AsyncTransactionSink;
use hyperscape_hhhs::{AdapterError, DurableProject, MemoryDurability, ProjectId, ProjectState};
use hyperscape_protocol::{
    AssetDescriptor, AssetId, AuthoredCommand, AuthoredEnvelope, EntityId, WireTransform,
};
use hyperscope_app::{
    AppCommit, AppEvent, AppStore, AuthoredRevision, CommitDisposition, ReduceError,
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
}

/// Explicit, opt-in coordinator for authored-revision behavior shadowing.
///
/// Commands are mirrored in their `AuthoredRevision::commands` vector order;
/// sender sequence numbers are metadata, not a sorting authority. Each local
/// admission consequently observes the preceding command in that vector.
/// Concurrent peer history belongs at `DurableProject::apply_records`, outside
/// this local AppStore observer.
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
