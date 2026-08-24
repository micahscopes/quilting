//! Optional behavior shadow between [`hyperscope_app::AppStore`] authored
//! revisions and [`hyperscape_hhhs`] durable authored history.
//!
//! [`AuthoredHhhsShadow::dispatch`] always commits the AppStore first. HHHS is
//! only a diagnostic observer in this checkpoint: an HHHS failure is returned
//! inside [`AuthoredShadowDispatch`] and never rejects or rolls back AppStore.
//! The default [`MemoryDurability`] host is process-local and is not browser
//! persistence.
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
}

impl fmt::Display for AuthoredShadowInitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Adapter(error) => error.fmt(formatter),
            Self::NonEmptyBaseline => formatter.write_str(
                "authored shadow requires an empty HHHS project; projection-baseline import is not implemented",
            ),
        }
    }
}

impl Error for AuthoredShadowInitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Adapter(error) => Some(error),
            Self::NonEmptyBaseline => None,
        }
    }
}

impl From<AdapterError> for AuthoredShadowInitError {
    fn from(error: AdapterError) -> Self {
        Self::Adapter(error)
    }
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
}

impl<D> AuthoredHhhsShadow<D> {
    /// Attach an already-constructed durable project. The project must be
    /// empty because this checkpoint has no projection-baseline import yet.
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
