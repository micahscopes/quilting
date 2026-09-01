//! Feature-gated browser wire projection for reducer commits and effects.
//!
//! The application owns the meaning and exact serialized shape of its effects.
//! Platform adapters may execute or serialize these values, but must not
//! independently reinterpret the reducer vocabulary.

use serde::Serialize;

use crate::{
    AppCommit, AppEffect, AuthoredProposalRole, AuthoredSessionEffect, CommitDisposition,
    PatchLabEffect, PatchLabEffects,
};

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppCommitWire {
    revision: String,
    disposition: &'static str,
    published_ui: bool,
    effects: Vec<AppEffectWire>,
}

impl From<&AppCommit> for AppCommitWire {
    fn from(commit: &AppCommit) -> Self {
        Self {
            revision: commit.revision.to_string(),
            disposition: match commit.disposition {
                CommitDisposition::Applied => "applied",
                CommitDisposition::IgnoredStale => "ignored_stale",
            },
            published_ui: commit.published_ui,
            effects: commit.effects.iter().map(AppEffectWire::from).collect(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AppEffectWire {
    FetchAsset {
        request_id: String,
        asset_id: String,
        uri: String,
    },
    CancelAssetLoad {
        request_id: String,
        asset_id: String,
    },
    InstallPrimaryScene {
        request_id: String,
        asset_id: String,
    },
    CancelPrimarySceneInstall {
        request_id: String,
        asset_id: String,
    },
    SelectAnimationClip {
        job_id: String,
        scene_request_id: String,
        asset_id: String,
        clip_index: u32,
    },
    CancelAnimationClipSelection {
        job_id: String,
        scene_request_id: String,
        asset_id: String,
        clip_index: u32,
    },
    PatchLab {
        effect: PatchLabEffectWire,
    },
    OpenAuthoredSession {
        job_id: String,
        project_id: String,
        proposal_role: &'static str,
    },
    CancelAuthoredSessionOpen {
        job_id: String,
        project_id: String,
    },
    CloseAuthoredSession {
        project_id: String,
    },
}

impl From<&AppEffect> for AppEffectWire {
    fn from(effect: &AppEffect) -> Self {
        match effect {
            AppEffect::FetchAsset { request_id, asset } => Self::FetchAsset {
                request_id: request_id.to_string(),
                asset_id: asset.id.to_string(),
                uri: asset.uri.clone(),
            },
            AppEffect::CancelAssetLoad {
                request_id,
                asset_id,
            } => Self::CancelAssetLoad {
                request_id: request_id.to_string(),
                asset_id: asset_id.to_string(),
            },
            AppEffect::InstallPrimaryScene {
                request_id,
                asset_id,
            } => Self::InstallPrimaryScene {
                request_id: request_id.to_string(),
                asset_id: asset_id.to_string(),
            },
            AppEffect::CancelPrimarySceneInstall {
                request_id,
                asset_id,
            } => Self::CancelPrimarySceneInstall {
                request_id: request_id.to_string(),
                asset_id: asset_id.to_string(),
            },
            AppEffect::SelectAnimationClip {
                job_id,
                scene_request_id,
                asset_id,
                clip_index,
            } => Self::SelectAnimationClip {
                job_id: job_id.to_string(),
                scene_request_id: scene_request_id.to_string(),
                asset_id: asset_id.to_string(),
                clip_index: *clip_index,
            },
            AppEffect::CancelAnimationClipSelection {
                job_id,
                scene_request_id,
                asset_id,
                clip_index,
            } => Self::CancelAnimationClipSelection {
                job_id: job_id.to_string(),
                scene_request_id: scene_request_id.to_string(),
                asset_id: asset_id.to_string(),
                clip_index: *clip_index,
            },
            AppEffect::PatchLab(effect) => Self::PatchLab {
                effect: effect.into(),
            },
            AppEffect::AuthoredSession(effect) => match effect {
                AuthoredSessionEffect::Open { job_id, intent } => Self::OpenAuthoredSession {
                    job_id: job_id.to_string(),
                    project_id: intent.project_id.to_string(),
                    proposal_role: match intent.proposal_role {
                        AuthoredProposalRole::Replica => "replica",
                        AuthoredProposalRole::AdmissionAuthority => "admission_authority",
                    },
                },
                AuthoredSessionEffect::CancelOpen { job_id, project_id } => {
                    Self::CancelAuthoredSessionOpen {
                        job_id: job_id.to_string(),
                        project_id: project_id.to_string(),
                    }
                }
                AuthoredSessionEffect::Close { project_id } => Self::CloseAuthoredSession {
                    project_id: project_id.to_string(),
                },
            },
        }
    }
}

pub fn app_effects_wire(effects: &[AppEffect]) -> Vec<AppEffectWire> {
    effects.iter().map(Into::into).collect()
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PatchLabEffectWire {
    BuildGeometry {
        job_id: String,
        shape: &'static str,
        grid: u8,
        bend_percent: u8,
    },
    CancelGeometry {
        job_id: String,
    },
    DiscardGeometry {
        geometry_job_id: String,
    },
    EvaluateLod {
        job_id: String,
        geometry_job_id: String,
        field: &'static str,
        phase_microradians: u32,
        min_exponent: u8,
        max_exponent: u8,
        manual_edge_exponents: [u8; 3],
        atlas_exponent: u8,
        max_face_edge_ratio: u8,
    },
    CancelLod {
        job_id: String,
        geometry_job_id: String,
    },
}

impl From<&PatchLabEffect> for PatchLabEffectWire {
    fn from(effect: &PatchLabEffect) -> Self {
        match effect {
            PatchLabEffect::BuildGeometry { job_id, geometry } => Self::BuildGeometry {
                job_id: job_id.to_string(),
                shape: geometry.shape.wire_name(),
                grid: geometry.grid,
                bend_percent: geometry.bend_percent,
            },
            PatchLabEffect::CancelGeometry { job_id } => Self::CancelGeometry {
                job_id: job_id.to_string(),
            },
            PatchLabEffect::DiscardGeometry { geometry_job_id } => Self::DiscardGeometry {
                geometry_job_id: geometry_job_id.to_string(),
            },
            PatchLabEffect::EvaluateLod {
                job_id,
                geometry_job_id,
                parameters,
            } => Self::EvaluateLod {
                job_id: job_id.to_string(),
                geometry_job_id: geometry_job_id.to_string(),
                field: parameters.field.wire_name(),
                phase_microradians: parameters.phase_microradians,
                min_exponent: parameters.min_exponent,
                max_exponent: parameters.max_exponent,
                manual_edge_exponents: parameters.manual_edge_exponents,
                atlas_exponent: parameters.atlas_exponent,
                max_face_edge_ratio: parameters.max_face_edge_ratio,
            },
            PatchLabEffect::CancelLod {
                job_id,
                geometry_job_id,
            } => Self::CancelLod {
                job_id: job_id.to_string(),
                geometry_job_id: geometry_job_id.to_string(),
            },
        }
    }
}

pub fn patch_lab_effects_wire(effects: &PatchLabEffects) -> Vec<PatchLabEffectWire> {
    effects.as_slice().iter().map(Into::into).collect()
}

#[cfg(all(test, feature = "replay"))]
mod tests {
    use super::*;
    use crate::{PatchLabField, PatchLabLodParameters};
    use serde_json::json;

    #[test]
    fn commit_wire_preserves_string_counters_and_tagged_effect_shapes() {
        let commit = AppCommit {
            revision: u64::MAX,
            effects: vec![AppEffect::PatchLab(PatchLabEffect::EvaluateLod {
                job_id: u64::MAX - 1,
                geometry_job_id: 7,
                parameters: PatchLabLodParameters {
                    field: PatchLabField::ManualEdges,
                    phase_microradians: 125_000,
                    min_exponent: 1,
                    max_exponent: 6,
                    manual_edge_exponents: [2, 3, 4],
                    atlas_exponent: 8,
                    max_face_edge_ratio: 4,
                },
            })],
            disposition: CommitDisposition::Applied,
            published_ui: false,
        };

        assert_eq!(
            serde_json::to_value(AppCommitWire::from(&commit)).unwrap(),
            json!({
                "revision": "18446744073709551615",
                "disposition": "applied",
                "publishedUi": false,
                "effects": [{
                    "type": "patch_lab",
                    "effect": {
                        "type": "evaluate_lod",
                        "job_id": "18446744073709551614",
                        "geometry_job_id": "7",
                        "field": "manual_edges",
                        "phase_microradians": 125000,
                        "min_exponent": 1,
                        "max_exponent": 6,
                        "manual_edge_exponents": [2, 3, 4],
                        "atlas_exponent": 8,
                        "max_face_edge_ratio": 4
                    }
                }]
            })
        );
    }
}
