//! Rust-owned projection of optional durable authored collaboration.
//!
//! The project and proposal role become semantic only when the platform open
//! callback enters `hyperscope-app`. Relay URLs, bearer credentials,
//! IndexedDB handles, and HHHS resources deliberately do not appear here.

use hyperscope_app::{
    AuthoredProposalRole, AuthoredSessionIntent, AuthoredSessionReadModel, AuthoredSessionStatus,
};

#[cfg(all(feature = "csr", target_arch = "wasm32"))]
mod csr;
#[cfg(all(feature = "csr", target_arch = "wasm32"))]
pub use csr::mount_authored_session_control;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthoredSessionPhase {
    Disabled,
    Opening,
    Active,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthoredSessionViewModel {
    pub revision: u64,
    pub phase: AuthoredSessionPhase,
    pub intent: Option<AuthoredSessionIntent>,
    pub inputs_locked: bool,
    pub primary_label: &'static str,
    pub primary_disabled: bool,
    pub status_label: String,
    pub status_is_error: bool,
}

impl AuthoredSessionViewModel {
    pub fn project_id(&self) -> Option<String> {
        self.intent.map(|intent| intent.project_id.to_string())
    }

    pub fn proposal_role(&self) -> AuthoredProposalRole {
        self.intent
            .map(|intent| intent.proposal_role)
            .unwrap_or_default()
    }
}

pub fn proposal_role_wire_name(role: AuthoredProposalRole) -> &'static str {
    match role {
        AuthoredProposalRole::Replica => "replica",
        AuthoredProposalRole::AdmissionAuthority => "admission_authority",
    }
}

pub fn proposal_role_label(role: AuthoredProposalRole) -> &'static str {
    match role {
        AuthoredProposalRole::Replica => "Replica",
        AuthoredProposalRole::AdmissionAuthority => "Admission authority",
    }
}

pub fn project_authored_session(snapshot: &AuthoredSessionReadModel) -> AuthoredSessionViewModel {
    let (phase, intent, inputs_locked, primary_label, primary_disabled, status_label, error) =
        match &snapshot.status {
            AuthoredSessionStatus::Disabled => (
                AuthoredSessionPhase::Disabled,
                None,
                false,
                "Connect",
                false,
                "Durable authored history is disabled".to_owned(),
                false,
            ),
            AuthoredSessionStatus::Opening { intent, .. } => (
                AuthoredSessionPhase::Opening,
                Some(*intent),
                true,
                "Opening…",
                true,
                format!(
                    "Opening project {} as {}",
                    intent.project_id,
                    proposal_role_label(intent.proposal_role),
                ),
                false,
            ),
            AuthoredSessionStatus::Active {
                intent,
                history_len,
                projection_revision,
                restored_projection,
            } => {
                let projection = projection_revision.map_or_else(
                    || "no authored projection".to_owned(),
                    |revision| format!("projection revision {revision}"),
                );
                let restored = if *restored_projection {
                    " · restored"
                } else {
                    ""
                };
                (
                    AuthoredSessionPhase::Active,
                    Some(*intent),
                    true,
                    "Disconnect",
                    false,
                    format!(
                        "Active as {} · {history_len} record{} · {projection}{restored}",
                        proposal_role_label(intent.proposal_role),
                        if *history_len == 1 { "" } else { "s" },
                    ),
                    false,
                )
            }
            AuthoredSessionStatus::Failed {
                intent,
                code,
                message,
                retryable,
            } => (
                AuthoredSessionPhase::Failed,
                Some(*intent),
                false,
                if *retryable { "Retry" } else { "Connect" },
                false,
                format!("{code}: {message}"),
                true,
            ),
        };
    AuthoredSessionViewModel {
        revision: snapshot.revision,
        phase,
        intent,
        inputs_locked,
        primary_label,
        primary_disabled,
        status_label,
        status_is_error: error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscope_app::AuthoredSessionOpenOutcome;
    use hyperscape_protocol::ProjectId;

    fn intent(role: AuthoredProposalRole) -> AuthoredSessionIntent {
        AuthoredSessionIntent {
            project_id: ProjectId::from_u128(0x42).unwrap(),
            proposal_role: role,
        }
    }

    #[test]
    fn disabled_projection_has_no_synthetic_project_or_platform_resource() {
        let view = project_authored_session(&AuthoredSessionReadModel {
            revision: 7,
            status: AuthoredSessionStatus::Disabled,
        });
        assert_eq!(view.revision, 7);
        assert_eq!(view.phase, AuthoredSessionPhase::Disabled);
        assert_eq!(view.intent, None);
        assert_eq!(view.project_id(), None);
        assert_eq!(view.proposal_role(), AuthoredProposalRole::Replica);
        assert!(!view.inputs_locked);
        assert_eq!(view.primary_label, "Connect");
    }

    #[test]
    fn opening_and_active_views_preserve_explicit_authority_role() {
        let authority = intent(AuthoredProposalRole::AdmissionAuthority);
        let opening = project_authored_session(&AuthoredSessionReadModel {
            revision: 8,
            status: AuthoredSessionStatus::Opening {
                job_id: 3,
                intent: authority,
            },
        });
        assert_eq!(opening.phase, AuthoredSessionPhase::Opening);
        assert!(opening.inputs_locked);
        assert!(opening.primary_disabled);
        assert!(opening.status_label.contains("Admission authority"));

        let active = project_authored_session(&AuthoredSessionReadModel {
            revision: 9,
            status: AuthoredSessionStatus::Active {
                intent: authority,
                history_len: 2,
                projection_revision: Some(11),
                restored_projection: true,
            },
        });
        assert_eq!(active.phase, AuthoredSessionPhase::Active);
        assert_eq!(active.primary_label, "Disconnect");
        assert!(!active.primary_disabled);
        assert_eq!(active.project_id(), Some(authority.project_id.to_string()),);
        assert!(active.status_label.contains("2 records"));
        assert!(active.status_label.contains("projection revision 11"));
        assert!(active.status_label.contains("restored"));
    }

    #[test]
    fn failed_projection_is_retryable_without_hiding_the_exact_fault() {
        let replica = intent(AuthoredProposalRole::Replica);
        let outcome = AuthoredSessionOpenOutcome::Failed {
            code: "indexed_db_open".to_owned(),
            message: "storage denied".to_owned(),
            retryable: true,
        };
        let AuthoredSessionOpenOutcome::Failed {
            code,
            message,
            retryable,
        } = outcome
        else {
            unreachable!()
        };
        let failed = project_authored_session(&AuthoredSessionReadModel {
            revision: 10,
            status: AuthoredSessionStatus::Failed {
                intent: replica,
                code,
                message,
                retryable,
            },
        });
        assert_eq!(failed.phase, AuthoredSessionPhase::Failed);
        assert_eq!(failed.primary_label, "Retry");
        assert!(!failed.inputs_locked);
        assert!(failed.status_is_error);
        assert_eq!(failed.status_label, "indexed_db_open: storage denied");
    }

    #[test]
    fn carrier_role_wire_values_are_explicit_and_stable() {
        assert_eq!(
            proposal_role_wire_name(AuthoredProposalRole::Replica),
            "replica",
        );
        assert_eq!(
            proposal_role_wire_name(AuthoredProposalRole::AdmissionAuthority),
            "admission_authority",
        );
    }
}
