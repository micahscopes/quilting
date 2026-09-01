//! Renderer-authority policy for device-resident LOD.
//!
//! Browser and native adapters report only semantic evidence: whether WebGPU
//! is the visible presenter, whether a dispatch was accepted, and whether the
//! incumbent recovered from a complete snapshot. This reducer owns authority
//! transitions and stale-dispatch rejection without owning GPU resources.

use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebGpuLodAuthorityPhase {
    AwaitingPresentation,
    AwaitingDeviceEpoch,
    DeviceResident,
    IncumbentRecovery,
    Incumbent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebGpuLodAuthorityReason {
    PresentationObserved,
    DeviceEpochAccepted,
    DevicePrefixAccepted,
    PresentationRetired,
    DeviceDispatchRejected,
    IncumbentRecovered,
    SparseIncumbentRejected,
    StaleDeviceCompletion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebGpuLodAuthorityDisposition {
    Observed,
    DispatchStarted,
    DeviceAuthority,
    IncumbentRequired,
    IncumbentRecovered,
    IgnoredStale,
    Unchanged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebGpuLodDispatch {
    pub token: u64,
    pub complete_scene: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebGpuLodAuthoritySnapshot {
    pub revision: u64,
    pub phase: WebGpuLodAuthorityPhase,
    pub presentation_authoritative: bool,
    pub active: bool,
    pub incumbent_reset_pending: bool,
    pub pending_dispatch: Option<WebGpuLodDispatch>,
    pub activations: u64,
    pub dispatches: u64,
    pub full_scene_dispatches: u64,
    pub fallback_transitions: u64,
    pub incumbent_recoveries: u64,
    pub stale_device_completions: u64,
    pub sparse_incumbent_rejections: u64,
    pub last_reason: Option<WebGpuLodAuthorityReason>,
}

impl WebGpuLodAuthoritySnapshot {
    /// The visible device may suppress incumbent classification only after an
    /// accepted complete-scene dispatch in the current presentation epoch.
    pub const fn suppress_incumbent(self) -> bool {
        self.active && self.presentation_authoritative
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebGpuLodAuthorityTransition {
    pub disposition: WebGpuLodAuthorityDisposition,
    pub dispatch: Option<WebGpuLodDispatch>,
    pub snapshot: WebGpuLodAuthoritySnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebGpuLodAuthorityError {
    DispatchAlreadyPending,
    NoDispatchPending,
    DispatchTokenMismatch { expected: u64, received: u64 },
    DispatchTokenExhausted,
    RevisionExhausted,
}

impl fmt::Display for WebGpuLodAuthorityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DispatchAlreadyPending => {
                formatter.write_str("a WebGPU LOD dispatch is already pending")
            }
            Self::NoDispatchPending => formatter.write_str("no WebGPU LOD dispatch is pending"),
            Self::DispatchTokenMismatch { expected, received } => write!(
                formatter,
                "WebGPU LOD dispatch token mismatch: expected {expected}, received {received}",
            ),
            Self::DispatchTokenExhausted => {
                formatter.write_str("WebGPU LOD dispatch token space is exhausted")
            }
            Self::RevisionExhausted => {
                formatter.write_str("WebGPU LOD authority revision space is exhausted")
            }
        }
    }
}

impl Error for WebGpuLodAuthorityError {}

#[derive(Debug, Clone)]
pub struct WebGpuLodAuthority {
    revision: u64,
    presentation_epoch: u64,
    next_dispatch_token: Option<u64>,
    phase: WebGpuLodAuthorityPhase,
    presentation_authoritative: bool,
    active: bool,
    incumbent_reset_pending: bool,
    pending_dispatch: Option<PendingWebGpuLodDispatch>,
    activations: u64,
    dispatches: u64,
    full_scene_dispatches: u64,
    fallback_transitions: u64,
    incumbent_recoveries: u64,
    stale_device_completions: u64,
    sparse_incumbent_rejections: u64,
    last_reason: Option<WebGpuLodAuthorityReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingWebGpuLodDispatch {
    dispatch: WebGpuLodDispatch,
    presentation_epoch: u64,
}

impl Default for WebGpuLodAuthority {
    fn default() -> Self {
        Self {
            revision: 0,
            presentation_epoch: 0,
            next_dispatch_token: Some(1),
            phase: WebGpuLodAuthorityPhase::AwaitingPresentation,
            presentation_authoritative: false,
            active: false,
            incumbent_reset_pending: false,
            pending_dispatch: None,
            activations: 0,
            dispatches: 0,
            full_scene_dispatches: 0,
            fallback_transitions: 0,
            incumbent_recoveries: 0,
            stale_device_completions: 0,
            sparse_incumbent_rejections: 0,
            last_reason: None,
        }
    }
}

impl WebGpuLodAuthority {
    pub fn snapshot(&self) -> WebGpuLodAuthoritySnapshot {
        WebGpuLodAuthoritySnapshot {
            revision: self.revision,
            phase: self.phase,
            presentation_authoritative: self.presentation_authoritative,
            active: self.active,
            incumbent_reset_pending: self.incumbent_reset_pending,
            pending_dispatch: self.pending_dispatch.map(|pending| pending.dispatch),
            activations: self.activations,
            dispatches: self.dispatches,
            full_scene_dispatches: self.full_scene_dispatches,
            fallback_transitions: self.fallback_transitions,
            incumbent_recoveries: self.incumbent_recoveries,
            stale_device_completions: self.stale_device_completions,
            sparse_incumbent_rejections: self.sparse_incumbent_rejections,
            last_reason: self.last_reason,
        }
    }

    pub fn observe_presentation(
        &mut self,
        authoritative: bool,
    ) -> Result<WebGpuLodAuthorityTransition, WebGpuLodAuthorityError> {
        if self.presentation_authoritative == authoritative {
            return Ok(self.transition(WebGpuLodAuthorityDisposition::Unchanged, None));
        }
        let next_revision = self.next_revision()?;
        let next_presentation_epoch = self
            .presentation_epoch
            .checked_add(1)
            .ok_or(WebGpuLodAuthorityError::RevisionExhausted)?;
        self.revision = next_revision;
        self.presentation_epoch = next_presentation_epoch;
        self.presentation_authoritative = authoritative;
        let disposition = if authoritative {
            if !self.active {
                self.phase = WebGpuLodAuthorityPhase::AwaitingDeviceEpoch;
            }
            self.last_reason = Some(WebGpuLodAuthorityReason::PresentationObserved);
            WebGpuLodAuthorityDisposition::Observed
        } else if self.active {
            self.retire_device(WebGpuLodAuthorityReason::PresentationRetired);
            WebGpuLodAuthorityDisposition::IncumbentRequired
        } else {
            self.phase = if self.incumbent_reset_pending {
                WebGpuLodAuthorityPhase::IncumbentRecovery
            } else {
                WebGpuLodAuthorityPhase::Incumbent
            };
            self.last_reason = Some(WebGpuLodAuthorityReason::PresentationRetired);
            WebGpuLodAuthorityDisposition::Observed
        };
        Ok(self.transition(disposition, None))
    }

    /// Begin one device dispatch. Initial activation and explicit scene/view
    /// changes require a complete epoch. Once a complete epoch is active, the
    /// adapter may refresh a topology-closed animated prefix without retiring
    /// the retained suffix. Completion still cannot activate device authority
    /// until presentation is independently observed as authoritative.
    pub fn begin_dispatch(
        &mut self,
        complete_scene_required: bool,
    ) -> Result<WebGpuLodAuthorityTransition, WebGpuLodAuthorityError> {
        if self.pending_dispatch.is_some() {
            return Err(WebGpuLodAuthorityError::DispatchAlreadyPending);
        }
        let next_revision = self.next_revision()?;
        let token = self
            .next_dispatch_token
            .ok_or(WebGpuLodAuthorityError::DispatchTokenExhausted)?;
        self.revision = next_revision;
        self.next_dispatch_token = token.checked_add(1);
        let dispatch = WebGpuLodDispatch {
            token,
            complete_scene: complete_scene_required
                || (self.presentation_authoritative && !self.active),
        };
        self.pending_dispatch = Some(PendingWebGpuLodDispatch {
            dispatch,
            presentation_epoch: self.presentation_epoch,
        });
        self.dispatches = self.dispatches.saturating_add(1);
        if dispatch.complete_scene {
            self.full_scene_dispatches = self.full_scene_dispatches.saturating_add(1);
        }
        Ok(self.transition(
            WebGpuLodAuthorityDisposition::DispatchStarted,
            Some(dispatch),
        ))
    }

    pub fn complete_dispatch(
        &mut self,
        token: u64,
        accepted: bool,
    ) -> Result<WebGpuLodAuthorityTransition, WebGpuLodAuthorityError> {
        let pending = self
            .pending_dispatch
            .ok_or(WebGpuLodAuthorityError::NoDispatchPending)?;
        if pending.dispatch.token != token {
            return Err(WebGpuLodAuthorityError::DispatchTokenMismatch {
                expected: pending.dispatch.token,
                received: token,
            });
        }
        self.revision = self.next_revision()?;
        self.pending_dispatch = None;
        if pending.presentation_epoch != self.presentation_epoch {
            self.stale_device_completions = self.stale_device_completions.saturating_add(1);
            self.last_reason = Some(WebGpuLodAuthorityReason::StaleDeviceCompletion);
            return Ok(self.transition(WebGpuLodAuthorityDisposition::IgnoredStale, None));
        }
        if accepted && pending.dispatch.complete_scene && self.presentation_authoritative {
            if !self.active {
                self.activations = self.activations.saturating_add(1);
            }
            self.active = true;
            self.phase = WebGpuLodAuthorityPhase::DeviceResident;
            self.incumbent_reset_pending = true;
            self.last_reason = Some(WebGpuLodAuthorityReason::DeviceEpochAccepted);
            return Ok(self.transition(WebGpuLodAuthorityDisposition::DeviceAuthority, None));
        }
        if accepted && self.active && self.presentation_authoritative {
            self.phase = WebGpuLodAuthorityPhase::DeviceResident;
            self.last_reason = Some(WebGpuLodAuthorityReason::DevicePrefixAccepted);
            return Ok(self.transition(WebGpuLodAuthorityDisposition::DeviceAuthority, None));
        }
        if !accepted && self.active {
            self.retire_device(WebGpuLodAuthorityReason::DeviceDispatchRejected);
            return Ok(self.transition(WebGpuLodAuthorityDisposition::IncumbentRequired, None));
        }
        Ok(self.transition(WebGpuLodAuthorityDisposition::Unchanged, None))
    }

    pub fn complete_incumbent_recovery(
        &mut self,
        full_snapshot: bool,
    ) -> Result<WebGpuLodAuthorityTransition, WebGpuLodAuthorityError> {
        self.revision = self.next_revision()?;
        if self.active || !self.incumbent_reset_pending {
            return Ok(self.transition(WebGpuLodAuthorityDisposition::Unchanged, None));
        }
        if !full_snapshot {
            self.sparse_incumbent_rejections = self.sparse_incumbent_rejections.saturating_add(1);
            self.last_reason = Some(WebGpuLodAuthorityReason::SparseIncumbentRejected);
            return Ok(self.transition(WebGpuLodAuthorityDisposition::IncumbentRequired, None));
        }
        self.incumbent_reset_pending = false;
        self.phase = WebGpuLodAuthorityPhase::Incumbent;
        self.incumbent_recoveries = self.incumbent_recoveries.saturating_add(1);
        self.last_reason = Some(WebGpuLodAuthorityReason::IncumbentRecovered);
        Ok(self.transition(WebGpuLodAuthorityDisposition::IncumbentRecovered, None))
    }

    fn retire_device(&mut self, reason: WebGpuLodAuthorityReason) {
        self.active = false;
        self.phase = WebGpuLodAuthorityPhase::IncumbentRecovery;
        self.incumbent_reset_pending = true;
        self.fallback_transitions = self.fallback_transitions.saturating_add(1);
        self.last_reason = Some(reason);
    }

    fn next_revision(&self) -> Result<u64, WebGpuLodAuthorityError> {
        self.revision
            .checked_add(1)
            .ok_or(WebGpuLodAuthorityError::RevisionExhausted)
    }

    fn transition(
        &self,
        disposition: WebGpuLodAuthorityDisposition,
        dispatch: Option<WebGpuLodDispatch>,
    ) -> WebGpuLodAuthorityTransition {
        WebGpuLodAuthorityTransition {
            disposition,
            dispatch,
            snapshot: self.snapshot(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn begin(authority: &mut WebGpuLodAuthority) -> WebGpuLodDispatch {
        authority
            .begin_dispatch(false)
            .expect("dispatch starts")
            .dispatch
            .expect("dispatch receipt")
    }

    #[test]
    fn device_authority_requires_visible_complete_epoch() {
        let mut authority = WebGpuLodAuthority::default();
        let shadow = begin(&mut authority);
        assert!(!shadow.complete_scene);
        let completed = authority
            .complete_dispatch(shadow.token, true)
            .expect("shadow completion is admitted");
        assert!(!completed.snapshot.active);

        authority
            .observe_presentation(true)
            .expect("presentation observation");
        let visible = begin(&mut authority);
        assert!(visible.complete_scene);
        let completed = authority
            .complete_dispatch(visible.token, true)
            .expect("visible completion is admitted");
        assert_eq!(
            completed.disposition,
            WebGpuLodAuthorityDisposition::DeviceAuthority
        );
        assert!(completed.snapshot.suppress_incumbent());
        assert!(completed.snapshot.incumbent_reset_pending);
    }

    #[test]
    fn presentation_retirement_requires_complete_incumbent_recovery() {
        let mut authority = WebGpuLodAuthority::default();
        authority.observe_presentation(true).unwrap();
        let dispatch = begin(&mut authority);
        authority.complete_dispatch(dispatch.token, true).unwrap();

        let retired = authority.observe_presentation(false).unwrap();
        assert_eq!(
            retired.disposition,
            WebGpuLodAuthorityDisposition::IncumbentRequired
        );
        assert!(!retired.snapshot.active);
        assert!(retired.snapshot.incumbent_reset_pending);

        let sparse = authority.complete_incumbent_recovery(false).unwrap();
        assert_eq!(
            sparse.disposition,
            WebGpuLodAuthorityDisposition::IncumbentRequired
        );
        assert!(sparse.snapshot.incumbent_reset_pending);
        let full = authority.complete_incumbent_recovery(true).unwrap();
        assert_eq!(
            full.disposition,
            WebGpuLodAuthorityDisposition::IncumbentRecovered
        );
        assert!(!full.snapshot.incumbent_reset_pending);
        assert_eq!(full.snapshot.phase, WebGpuLodAuthorityPhase::Incumbent);
    }

    #[test]
    fn complete_recovery_dispatch_prewarms_without_seizing_authority() {
        let mut authority = WebGpuLodAuthority::default();
        let recovery = authority
            .begin_dispatch(true)
            .unwrap()
            .dispatch
            .unwrap();
        assert!(recovery.complete_scene);
        let completed = authority.complete_dispatch(recovery.token, true).unwrap();
        assert_eq!(
            completed.disposition,
            WebGpuLodAuthorityDisposition::Unchanged
        );
        assert!(!completed.snapshot.active);
        assert!(!completed.snapshot.presentation_authoritative);
    }

    #[test]
    fn presentation_epoch_rejects_stale_device_completion() {
        let mut authority = WebGpuLodAuthority::default();
        authority.observe_presentation(true).unwrap();
        let dispatch = begin(&mut authority);
        authority.observe_presentation(false).unwrap();
        let stale = authority.complete_dispatch(dispatch.token, true).unwrap();
        assert_eq!(
            stale.disposition,
            WebGpuLodAuthorityDisposition::IgnoredStale
        );
        assert!(!stale.snapshot.active);
        assert_eq!(stale.snapshot.stale_device_completions, 1);
    }

    #[test]
    fn rejected_refresh_retires_existing_device_authority() {
        let mut authority = WebGpuLodAuthority::default();
        authority.observe_presentation(true).unwrap();
        let first = begin(&mut authority);
        authority.complete_dispatch(first.token, true).unwrap();
        let refresh = begin(&mut authority);
        let rejected = authority.complete_dispatch(refresh.token, false).unwrap();
        assert_eq!(
            rejected.disposition,
            WebGpuLodAuthorityDisposition::IncumbentRequired
        );
        assert!(!rejected.snapshot.active);
        assert_eq!(rejected.snapshot.fallback_transitions, 1);
    }

    #[test]
    fn active_device_accepts_a_partial_refresh_without_reactivation() {
        let mut authority = WebGpuLodAuthority::default();
        authority.observe_presentation(true).unwrap();
        let initial = begin(&mut authority);
        assert!(initial.complete_scene);
        authority.complete_dispatch(initial.token, true).unwrap();

        let refresh = begin(&mut authority);
        assert!(!refresh.complete_scene);
        let completed = authority.complete_dispatch(refresh.token, true).unwrap();
        assert_eq!(
            completed.disposition,
            WebGpuLodAuthorityDisposition::DeviceAuthority
        );
        assert!(completed.snapshot.active);
        assert_eq!(completed.snapshot.activations, 1);
        assert_eq!(completed.snapshot.dispatches, 2);
        assert_eq!(completed.snapshot.full_scene_dispatches, 1);
        assert_eq!(
            completed.snapshot.last_reason,
            Some(WebGpuLodAuthorityReason::DevicePrefixAccepted)
        );
    }

    #[test]
    fn failed_counter_advances_are_mutation_atomic() {
        let mut authority = WebGpuLodAuthority {
            presentation_epoch: u64::MAX,
            ..WebGpuLodAuthority::default()
        };
        let before = authority.snapshot();
        assert_eq!(
            authority.observe_presentation(true),
            Err(WebGpuLodAuthorityError::RevisionExhausted)
        );
        assert_eq!(authority.snapshot(), before);

        let mut authority = WebGpuLodAuthority {
            next_dispatch_token: None,
            ..WebGpuLodAuthority::default()
        };
        let before = authority.snapshot();
        assert_eq!(
            authority.begin_dispatch(false),
            Err(WebGpuLodAuthorityError::DispatchTokenExhausted)
        );
        assert_eq!(authority.snapshot(), before);

        let mut authority = WebGpuLodAuthority {
            revision: u64::MAX,
            ..WebGpuLodAuthority::default()
        };
        let before = authority.snapshot();
        assert_eq!(
            authority.begin_dispatch(false),
            Err(WebGpuLodAuthorityError::RevisionExhausted)
        );
        assert_eq!(authority.snapshot(), before);
    }
}
