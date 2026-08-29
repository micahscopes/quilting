//! Animation playback and installed-clip controls for browser views.
//!
//! Playback remains application state and toggling remains a semantic action.
//! Clip choices come only from the renderer-installed scene projection, and
//! both controls dispatch semantic actions directly through the application
//! reducer. Browser callbacks receive committed platform effects, never raw
//! control intent.

use hyperscope_app::{
    AnimationAction, AnimationClipSelectionReadModel, AppEffect, AppStore, AppSummary,
    InstalledPrimarySceneReadModel, ReduceError, SemanticAction,
};

#[cfg(all(feature = "csr", target_arch = "wasm32"))]
mod csr;
#[cfg(all(feature = "csr", target_arch = "wasm32"))]
pub use csr::{mount_animation_clip_control, mount_animation_control};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimationControlViewModel {
    pub playing: bool,
    pub action_label: &'static str,
    pub state_label: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnimationControlCommit {
    pub sequence: u64,
    pub revision: u64,
    pub playing: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimationClipOptionViewModel {
    pub index: u32,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimationClipControlViewModel {
    pub options: Vec<AnimationClipOptionViewModel>,
    pub selected_index: Option<u32>,
    pub pending_index: Option<u32>,
    pub disabled: bool,
    pub status_label: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimationClipJobEffect {
    pub job_id: u64,
    pub scene_request_id: String,
    pub asset_id: String,
    pub clip_index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimationClipControlCommit {
    pub sequence: u64,
    pub revision: u64,
    pub requested_index: u32,
    pub selection: Option<AnimationClipJobEffect>,
    pub cancellations: Vec<AnimationClipJobEffect>,
}

/// Commit the control's atomic toggle intent through the application reducer.
/// Keeping this operation framework-neutral makes the Leptos component a thin
/// event/view adapter and gives native tests the same boundary as WASM.
pub fn toggle_animation_playback(store: &AppStore) -> Result<AnimationControlCommit, ReduceError> {
    let (sequence, commit) =
        store.dispatch_semantic(SemanticAction::Animate(AnimationAction::TogglePlaying))?;
    Ok(AnimationControlCommit {
        sequence,
        revision: commit.revision,
        playing: store.frame_snapshot().animation.playing,
    })
}

pub fn project_animation_control(summary: &AppSummary) -> AnimationControlViewModel {
    if summary.animation_playing {
        AnimationControlViewModel {
            playing: true,
            action_label: "Pause animation",
            state_label: "Animation playing",
        }
    } else {
        AnimationControlViewModel {
            playing: false,
            action_label: "Play animation",
            state_label: "Animation paused",
        }
    }
}

/// Derive one coherent selector from the installed catalog and the committed
/// active/pending selection. A pending choice is shown immediately, but the
/// active clip changes only after its renderer completion event commits.
pub fn project_animation_clip_control(
    installed: Option<&InstalledPrimarySceneReadModel>,
    selection: &AnimationClipSelectionReadModel,
) -> AnimationClipControlViewModel {
    let options = installed
        .into_iter()
        .flat_map(|scene| scene.install.animation_clips.iter())
        .map(|clip| AnimationClipOptionViewModel {
            index: clip.index,
            label: if clip.name.is_empty() {
                format!("Animation {}", clip.index)
            } else {
                clip.name.clone()
            },
        })
        .collect::<Vec<_>>();
    let active_index = selection.active.as_ref().map(|active| active.clip.index);
    let pending_index = selection.pending.as_ref().map(|pending| pending.clip.index);
    let selected_index = pending_index.or(active_index);
    let selected_label = selected_index.and_then(|index| {
        options
            .iter()
            .find(|option| option.index == index)
            .map(|option| option.label.as_str())
    });
    let status_label = match (pending_index, active_index, selected_label) {
        (Some(_), _, Some(label)) => format!("Switching to {label}"),
        (None, Some(_), Some(label)) => format!("Active animation: {label}"),
        _ if installed.is_none() => "No renderer scene installed".to_owned(),
        _ => "No animation clips".to_owned(),
    };
    AnimationClipControlViewModel {
        disabled: options.is_empty(),
        options,
        selected_index,
        pending_index,
        status_label,
    }
}

/// Commit a clip choice and retain the exact renderer selection/cancellation
/// effects allocated by the reducer. The web adapter must execute these facts;
/// it cannot infer a job identity from the selected index.
pub fn select_animation_clip(
    store: &AppStore,
    index: u32,
) -> Result<AnimationClipControlCommit, ReduceError> {
    let (sequence, commit) =
        store.dispatch_semantic(SemanticAction::Animate(AnimationAction::SelectClip(index)))?;
    let mut selection = None;
    let mut cancellations = Vec::new();
    for effect in &commit.effects {
        let (destination, effect) = match effect {
            AppEffect::SelectAnimationClip {
                job_id,
                scene_request_id,
                asset_id,
                clip_index,
            } => (
                &mut selection,
                AnimationClipJobEffect {
                    job_id: *job_id,
                    scene_request_id: scene_request_id.to_string(),
                    asset_id: asset_id.to_string(),
                    clip_index: *clip_index,
                },
            ),
            AppEffect::CancelAnimationClipSelection {
                job_id,
                scene_request_id,
                asset_id,
                clip_index,
            } => {
                cancellations.push(AnimationClipJobEffect {
                    job_id: *job_id,
                    scene_request_id: scene_request_id.to_string(),
                    asset_id: asset_id.to_string(),
                    clip_index: *clip_index,
                });
                continue;
            }
            _ => continue,
        };
        *destination = Some(effect);
    }
    Ok(AnimationClipControlCommit {
        sequence,
        revision: commit.revision,
        requested_index: index,
        selection,
        cancellations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscape_protocol::{AssetDescriptor, AssetId, RequestId};
    use hyperscope_app::{
        AnimationClipDescriptor, AnimationClipSelectionCompletion, AnimationClipSelectionOutcome,
        AssetLoadCompletion, AssetLoadOutcome, AssetLoadScope, AssetMetadata, EffectCompletion,
        PrimarySceneInstallCompletion, PrimarySceneInstallMetadata, PrimarySceneInstallOutcome,
    };

    fn installed_store() -> (AppStore, RequestId, AssetId) {
        let store = AppStore::default();
        let request_id = RequestId::from_u128(10).unwrap();
        let asset_id = AssetId::from_u128(20).unwrap();
        store
            .dispatch_semantic(SemanticAction::RequestAsset {
                request_id,
                asset: AssetDescriptor {
                    id: asset_id,
                    uri: "horse.glb".to_owned(),
                    media_type: Some("model/gltf-binary".to_owned()),
                    content_digest: None,
                },
                scope: AssetLoadScope::PrimaryScene,
            })
            .unwrap();
        store
            .dispatch(hyperscope_app::AppEvent::EffectCompleted(
                EffectCompletion::AssetLoad(AssetLoadCompletion {
                    request_id,
                    asset_id,
                    outcome: AssetLoadOutcome::Loaded {
                        byte_length: 42,
                        content_digest: None,
                        metadata: AssetMetadata::default(),
                    },
                }),
            ))
            .unwrap();
        store
            .dispatch(hyperscope_app::AppEvent::EffectCompleted(
                EffectCompletion::PrimarySceneInstall(PrimarySceneInstallCompletion {
                    request_id,
                    asset_id,
                    outcome: PrimarySceneInstallOutcome::Installed(PrimarySceneInstallMetadata {
                        num_vertices: 796,
                        num_faces: 984,
                        animation_clips: vec![
                            AnimationClipDescriptor {
                                index: 0,
                                name: "Gallop".to_owned(),
                                time_min_seconds: 0.0,
                                time_max_seconds: 1.5,
                            },
                            AnimationClipDescriptor {
                                index: 1,
                                name: String::new(),
                                time_min_seconds: 2.0,
                                time_max_seconds: 3.0,
                            },
                        ],
                    }),
                }),
            ))
            .unwrap();
        (store, request_id, asset_id)
    }

    #[test]
    fn projection_exposes_playback_state_and_inverse_action() {
        let mut summary = AppSummary::default();
        summary.animation_playing = true;
        assert_eq!(
            project_animation_control(&summary),
            AnimationControlViewModel {
                playing: true,
                action_label: "Pause animation",
                state_label: "Animation playing",
            }
        );

        summary.animation_playing = false;
        assert_eq!(
            project_animation_control(&summary),
            AnimationControlViewModel {
                playing: false,
                action_label: "Play animation",
                state_label: "Animation paused",
            }
        );
    }

    #[test]
    fn toggle_commits_through_the_application_sequence_authority() {
        let store = AppStore::default();

        assert_eq!(
            toggle_animation_playback(&store).unwrap(),
            AnimationControlCommit {
                sequence: 0,
                revision: 1,
                playing: false,
            },
        );
        assert_eq!(
            toggle_animation_playback(&store).unwrap(),
            AnimationControlCommit {
                sequence: 1,
                revision: 2,
                playing: true,
            },
        );
        assert!(store.summary_snapshot().animation_playing);
    }

    #[test]
    fn clip_projection_uses_the_installed_catalog_and_pending_intent() {
        let (store, _, _) = installed_store();
        assert_eq!(
            project_animation_clip_control(
                store.installed_primary_scene_snapshot().as_ref(),
                &store.animation_clip_selection_snapshot(),
            ),
            AnimationClipControlViewModel {
                options: vec![
                    AnimationClipOptionViewModel {
                        index: 0,
                        label: "Gallop".to_owned(),
                    },
                    AnimationClipOptionViewModel {
                        index: 1,
                        label: "Animation 1".to_owned(),
                    },
                ],
                selected_index: Some(0),
                pending_index: None,
                disabled: false,
                status_label: "Active animation: Gallop".to_owned(),
            },
        );

        select_animation_clip(&store, 1).unwrap();
        assert_eq!(
            project_animation_clip_control(
                store.installed_primary_scene_snapshot().as_ref(),
                &store.animation_clip_selection_snapshot(),
            )
            .status_label,
            "Switching to Animation 1",
        );
    }

    #[test]
    fn clip_control_returns_exact_selection_and_cancellation_effects() {
        let (store, request_id, asset_id) = installed_store();
        let selected = select_animation_clip(&store, 1).unwrap();
        assert_eq!(selected.requested_index, 1);
        assert_eq!(selected.cancellations, Vec::new());
        assert_eq!(
            selected.selection,
            Some(AnimationClipJobEffect {
                job_id: 0,
                scene_request_id: request_id.to_string(),
                asset_id: asset_id.to_string(),
                clip_index: 1,
            }),
        );

        let canceled = select_animation_clip(&store, 0).unwrap();
        assert_eq!(canceled.selection, None);
        assert_eq!(canceled.cancellations, vec![selected.selection.unwrap()]);
        assert_eq!(store.animation_clip_selection_snapshot().pending, None);
    }

    #[test]
    fn a_matching_completion_updates_the_control_projection() {
        let (store, request_id, asset_id) = installed_store();
        let selected = select_animation_clip(&store, 1).unwrap();
        store
            .dispatch(hyperscope_app::AppEvent::EffectCompleted(
                EffectCompletion::AnimationClipSelection(AnimationClipSelectionCompletion {
                    job_id: selected.selection.unwrap().job_id,
                    scene_request_id: request_id,
                    asset_id,
                    clip_index: 1,
                    outcome: AnimationClipSelectionOutcome::Selected,
                }),
            ))
            .unwrap();
        let projected = project_animation_clip_control(
            store.installed_primary_scene_snapshot().as_ref(),
            &store.animation_clip_selection_snapshot(),
        );
        assert_eq!(projected.selected_index, Some(1));
        assert_eq!(projected.pending_index, None);
        assert_eq!(projected.status_label, "Active animation: Animation 1");
    }
}
