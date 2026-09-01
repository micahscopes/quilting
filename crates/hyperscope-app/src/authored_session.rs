//! Reducer-side lifecycle for optional durable authored collaboration.
//!
//! Project identity and proposal authority are semantic application state.
//! IndexedDB handles, bearer credentials, relay URLs, and HHHS objects remain
//! platform resources executed from the typed effects below.

use hyperscape_protocol::ProjectId;
use std::fmt;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(rename_all = "snake_case"))]
pub enum AuthoredProposalRole {
    #[default]
    Replica,
    AdmissionAuthority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
pub struct AuthoredSessionIntent {
    pub project_id: ProjectId,
    pub proposal_role: AuthoredProposalRole,
}

impl AuthoredSessionIntent {
    pub fn validate(self) -> Result<(), AuthoredSessionError> {
        self.project_id
            .validate()
            .map_err(|error| AuthoredSessionError::InvalidProject(error.to_string()))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(tag = "status", rename_all = "snake_case"))]
pub enum AuthoredSessionStatus {
    #[default]
    Disabled,
    Opening {
        job_id: u64,
        intent: AuthoredSessionIntent,
    },
    Active {
        intent: AuthoredSessionIntent,
        history_len: u64,
        projection_revision: Option<u64>,
        restored_projection: bool,
    },
    Failed {
        intent: AuthoredSessionIntent,
        code: String,
        message: String,
        retryable: bool,
    },
}

impl AuthoredSessionStatus {
    pub fn is_disabled(&self) -> bool {
        matches!(self, Self::Disabled)
    }

    pub fn intent(&self) -> Option<AuthoredSessionIntent> {
        match self {
            Self::Disabled => None,
            Self::Opening { intent, .. }
            | Self::Active { intent, .. }
            | Self::Failed { intent, .. } => Some(*intent),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthoredSessionReadModel {
    pub revision: u64,
    pub status: AuthoredSessionStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(tag = "type", rename_all = "snake_case"))]
pub enum AuthoredSessionEffect {
    Open {
        job_id: u64,
        intent: AuthoredSessionIntent,
    },
    CancelOpen {
        job_id: u64,
        project_id: ProjectId,
    },
    Close {
        project_id: ProjectId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "replay", serde(tag = "status", rename_all = "snake_case"))]
pub enum AuthoredSessionOpenOutcome {
    Opened {
        history_len: u64,
        projection_revision: Option<u64>,
        restored_projection: bool,
    },
    Failed {
        code: String,
        message: String,
        retryable: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "replay", derive(serde::Serialize, serde::Deserialize))]
pub struct AuthoredSessionCompletion {
    pub job_id: u64,
    pub project_id: ProjectId,
    pub outcome: AuthoredSessionOpenOutcome,
}

#[derive(Debug, Clone)]
pub(crate) struct AuthoredSessionRuntime {
    next_job_id: Option<u64>,
    status: AuthoredSessionStatus,
}

impl Default for AuthoredSessionRuntime {
    fn default() -> Self {
        Self {
            next_job_id: Some(0),
            status: AuthoredSessionStatus::Disabled,
        }
    }
}

impl AuthoredSessionRuntime {
    pub(crate) fn status(&self) -> &AuthoredSessionStatus {
        &self.status
    }

    pub(crate) fn set_intent(
        &mut self,
        intent: Option<AuthoredSessionIntent>,
    ) -> Result<Vec<AuthoredSessionEffect>, AuthoredSessionError> {
        if let Some(intent) = intent {
            intent.validate()?;
        }
        let already_settled = match (&self.status, intent) {
            (AuthoredSessionStatus::Disabled, None) => true,
            (
                AuthoredSessionStatus::Opening {
                    intent: current, ..
                }
                | AuthoredSessionStatus::Active {
                    intent: current, ..
                },
                Some(requested),
            ) => *current == requested,
            _ => false,
        };
        if already_settled {
            return Ok(Vec::new());
        }
        let mut effects = self.stop_effects();
        let Some(intent) = intent else {
            self.status = AuthoredSessionStatus::Disabled;
            return Ok(effects);
        };
        let job_id = self
            .next_job_id
            .ok_or(AuthoredSessionError::JobIdentityExhausted)?;
        self.next_job_id = job_id.checked_add(1);
        self.status = AuthoredSessionStatus::Opening { job_id, intent };
        effects.push(AuthoredSessionEffect::Open { job_id, intent });
        Ok(effects)
    }

    pub(crate) fn complete(
        &mut self,
        completion: AuthoredSessionCompletion,
    ) -> Result<bool, AuthoredSessionError> {
        completion
            .project_id
            .validate()
            .map_err(|error| AuthoredSessionError::InvalidProject(error.to_string()))?;
        let AuthoredSessionStatus::Opening { job_id, intent } = self.status.clone() else {
            return Ok(false);
        };
        if completion.job_id != job_id || completion.project_id != intent.project_id {
            return Ok(false);
        }
        self.status = match completion.outcome {
            AuthoredSessionOpenOutcome::Opened {
                history_len,
                projection_revision,
                restored_projection,
            } => AuthoredSessionStatus::Active {
                intent,
                history_len,
                projection_revision,
                restored_projection,
            },
            AuthoredSessionOpenOutcome::Failed {
                code,
                message,
                retryable,
            } => {
                if code.trim().is_empty() || message.trim().is_empty() {
                    return Err(AuthoredSessionError::InvalidFailure);
                }
                AuthoredSessionStatus::Failed {
                    intent,
                    code,
                    message,
                    retryable,
                }
            }
        };
        Ok(true)
    }

    fn stop_effects(&self) -> Vec<AuthoredSessionEffect> {
        match &self.status {
            AuthoredSessionStatus::Opening { job_id, intent } => {
                vec![AuthoredSessionEffect::CancelOpen {
                    job_id: *job_id,
                    project_id: intent.project_id,
                }]
            }
            AuthoredSessionStatus::Active { intent, .. } => {
                vec![AuthoredSessionEffect::Close {
                    project_id: intent.project_id,
                }]
            }
            AuthoredSessionStatus::Disabled | AuthoredSessionStatus::Failed { .. } => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthoredSessionError {
    InvalidProject(String),
    JobIdentityExhausted,
    InvalidFailure,
}

impl fmt::Display for AuthoredSessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProject(message) => {
                write!(
                    formatter,
                    "authored collaboration project is invalid: {message}"
                )
            }
            Self::JobIdentityExhausted => {
                formatter.write_str("authored collaboration job identity space is exhausted")
            }
            Self::InvalidFailure => formatter
                .write_str("authored collaboration failure requires nonempty code and message"),
        }
    }
}

impl std::error::Error for AuthoredSessionError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn intent(project: u128, proposal_role: AuthoredProposalRole) -> AuthoredSessionIntent {
        AuthoredSessionIntent {
            project_id: ProjectId::from_u128(project).unwrap(),
            proposal_role,
        }
    }

    #[test]
    fn opening_replacement_and_close_emit_exact_resource_lifecycle() {
        let mut runtime = AuthoredSessionRuntime::default();
        let first = intent(1, AuthoredProposalRole::Replica);
        assert_eq!(
            runtime.set_intent(Some(first)).unwrap(),
            vec![AuthoredSessionEffect::Open {
                job_id: 0,
                intent: first,
            }]
        );
        let second = intent(2, AuthoredProposalRole::AdmissionAuthority);
        assert_eq!(
            runtime.set_intent(Some(second)).unwrap(),
            vec![
                AuthoredSessionEffect::CancelOpen {
                    job_id: 0,
                    project_id: first.project_id,
                },
                AuthoredSessionEffect::Open {
                    job_id: 1,
                    intent: second,
                },
            ]
        );
        assert_eq!(
            runtime.set_intent(None).unwrap(),
            vec![AuthoredSessionEffect::CancelOpen {
                job_id: 1,
                project_id: second.project_id,
            }]
        );
        assert_eq!(runtime.status(), &AuthoredSessionStatus::Disabled);
    }

    #[test]
    fn matching_open_or_active_intent_is_idempotent_but_failure_can_retry() {
        let mut runtime = AuthoredSessionRuntime::default();
        let active = intent(6, AuthoredProposalRole::Replica);
        runtime.set_intent(Some(active)).unwrap();
        assert!(runtime.set_intent(Some(active)).unwrap().is_empty());
        runtime
            .complete(AuthoredSessionCompletion {
                job_id: 0,
                project_id: active.project_id,
                outcome: AuthoredSessionOpenOutcome::Opened {
                    history_len: 0,
                    projection_revision: None,
                    restored_projection: false,
                },
            })
            .unwrap();
        assert!(runtime.set_intent(Some(active)).unwrap().is_empty());

        let failed = intent(7, AuthoredProposalRole::AdmissionAuthority);
        runtime.set_intent(Some(failed)).unwrap();
        runtime
            .complete(AuthoredSessionCompletion {
                job_id: 1,
                project_id: failed.project_id,
                outcome: AuthoredSessionOpenOutcome::Failed {
                    code: "unavailable".into(),
                    message: "browser storage is unavailable".into(),
                    retryable: true,
                },
            })
            .unwrap();
        assert_eq!(
            runtime.set_intent(Some(failed)).unwrap(),
            vec![AuthoredSessionEffect::Open {
                job_id: 2,
                intent: failed,
            }]
        );
    }

    #[test]
    fn stale_completion_cannot_replace_the_current_session() {
        let mut runtime = AuthoredSessionRuntime::default();
        let first = intent(3, AuthoredProposalRole::Replica);
        runtime.set_intent(Some(first)).unwrap();
        let second = intent(4, AuthoredProposalRole::AdmissionAuthority);
        runtime.set_intent(Some(second)).unwrap();
        assert!(!runtime
            .complete(AuthoredSessionCompletion {
                job_id: 0,
                project_id: first.project_id,
                outcome: AuthoredSessionOpenOutcome::Opened {
                    history_len: 99,
                    projection_revision: Some(9),
                    restored_projection: true,
                },
            })
            .unwrap());
        assert_eq!(
            runtime.status(),
            &AuthoredSessionStatus::Opening {
                job_id: 1,
                intent: second,
            }
        );
    }

    #[test]
    fn matching_completion_activates_and_close_releases_the_project() {
        let mut runtime = AuthoredSessionRuntime::default();
        let intent = intent(5, AuthoredProposalRole::Replica);
        runtime.set_intent(Some(intent)).unwrap();
        assert!(runtime
            .complete(AuthoredSessionCompletion {
                job_id: 0,
                project_id: intent.project_id,
                outcome: AuthoredSessionOpenOutcome::Opened {
                    history_len: 7,
                    projection_revision: Some(3),
                    restored_projection: true,
                },
            })
            .unwrap());
        assert_eq!(
            runtime.status(),
            &AuthoredSessionStatus::Active {
                intent,
                history_len: 7,
                projection_revision: Some(3),
                restored_projection: true,
            }
        );
        assert_eq!(
            runtime.set_intent(None).unwrap(),
            vec![AuthoredSessionEffect::Close {
                project_id: intent.project_id,
            }]
        );
    }
}
