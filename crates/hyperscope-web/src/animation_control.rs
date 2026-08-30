//! Animation playback and installed-clip controls for browser views.
//!
//! Playback remains application state and toggling remains a semantic action.
//! Clip choices come only from the renderer-installed scene projection, and
//! both controls dispatch semantic actions directly through the application
//! reducer. Browser callbacks receive committed platform effects, never raw
//! control intent.

use hyperscope_app::{
    AnimationAction, AnimationClipSelectionReadModel, AppAnimationSnapshot, AppStore,
    InstalledPrimarySceneReadModel, ReduceError, SemanticAction,
};
pub use hyperscope_app::{
    AnimationClipJobEffect, AnimationClipRequest as AnimationClipControlCommit,
};

#[cfg(all(feature = "csr", target_arch = "wasm32"))]
mod csr;
#[cfg(all(feature = "csr", target_arch = "wasm32"))]
pub use csr::{mount_animation_clip_control, mount_animation_control, mount_animation_timeline};

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

#[derive(Debug, Clone, PartialEq)]
pub struct AnimationTimelineViewModel {
    pub revision: u64,
    pub sample_time_seconds: f64,
    pub minimum_seconds: f64,
    pub maximum_seconds: f64,
    pub disabled: bool,
    pub status_label: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimationTimelineCommit {
    pub sequence: u64,
    pub revision: u64,
    pub sample_time_seconds: f64,
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

pub fn project_animation_control(snapshot: &AppAnimationSnapshot) -> AnimationControlViewModel {
    if snapshot.clock.playing {
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

pub fn project_animation_timeline(snapshot: &AppAnimationSnapshot) -> AnimationTimelineViewModel {
    let Some(clip) = snapshot.active_clip else {
        return AnimationTimelineViewModel {
            revision: snapshot.revision,
            sample_time_seconds: 0.0,
            minimum_seconds: 0.0,
            maximum_seconds: 1.0,
            disabled: true,
            status_label: "No active animation".to_owned(),
        };
    };
    AnimationTimelineViewModel {
        revision: snapshot.revision,
        sample_time_seconds: clip.sample_time_seconds,
        minimum_seconds: clip.time_min_seconds,
        maximum_seconds: clip.time_max_seconds,
        disabled: snapshot.clock.playing,
        status_label: if snapshot.clock.playing {
            "Pause animation to seek".to_owned()
        } else {
            format!("Animation time {:.2} seconds", clip.sample_time_seconds)
        },
    }
}

/// Seek in authored clip time. The reducer continues to own its unwrapped,
/// clip-relative clock; the view cannot seek while playback is advancing,
/// matching the incumbent browser interaction contract.
pub fn seek_animation_timeline(
    store: &AppStore,
    sample_time_seconds: f64,
) -> Result<AnimationTimelineCommit, AnimationTimelineError> {
    let frame = store.frame_snapshot();
    if frame.animation.playing {
        return Err(AnimationTimelineError::Playing);
    }
    let clip = frame
        .active_animation_clip
        .ok_or(AnimationTimelineError::NoActiveClip)?;
    if !sample_time_seconds.is_finite()
        || sample_time_seconds < clip.time_min_seconds
        || sample_time_seconds > clip.time_max_seconds
    {
        return Err(AnimationTimelineError::InvalidSampleTime);
    }
    let (sequence, commit) = store
        .dispatch_semantic(SemanticAction::Animate(AnimationAction::Seek(
            sample_time_seconds - clip.time_min_seconds,
        )))
        .map_err(AnimationTimelineError::Reduce)?;
    let committed = store
        .frame_snapshot()
        .active_animation_clip
        .ok_or(AnimationTimelineError::NoActiveClip)?;
    Ok(AnimationTimelineCommit {
        sequence,
        revision: commit.revision,
        sample_time_seconds: committed.sample_time_seconds,
    })
}

#[derive(Debug)]
pub enum AnimationTimelineError {
    NoActiveClip,
    Playing,
    InvalidSampleTime,
    Reduce(ReduceError),
}

impl std::fmt::Display for AnimationTimelineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoActiveClip => formatter.write_str("no active animation clip is installed"),
            Self::Playing => formatter.write_str("pause animation before seeking"),
            Self::InvalidSampleTime => {
                formatter.write_str("animation sample time is outside the active clip")
            }
            Self::Reduce(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AnimationTimelineError {}

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
    store.request_animation_clip(index)
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
        let store = AppStore::default();
        assert_eq!(
            project_animation_control(&store.animation_snapshot()),
            AnimationControlViewModel {
                playing: true,
                action_label: "Pause animation",
                state_label: "Animation playing",
            }
        );

        toggle_animation_playback(&store).unwrap();
        assert_eq!(
            project_animation_control(&store.animation_snapshot()),
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
    fn timeline_projects_authored_clip_time_and_seeks_only_while_paused() {
        let (store, _, _) = installed_store();
        let playing = project_animation_timeline(&store.animation_snapshot());
        assert_eq!(playing.minimum_seconds, 0.0);
        assert_eq!(playing.maximum_seconds, 1.5);
        assert!(playing.disabled);
        assert!(matches!(
            seek_animation_timeline(&store, 0.75),
            Err(AnimationTimelineError::Playing),
        ));

        toggle_animation_playback(&store).unwrap();
        let paused = project_animation_timeline(&store.animation_snapshot());
        assert!(!paused.disabled);
        let committed = seek_animation_timeline(&store, 0.75).unwrap();
        assert_eq!(committed.sample_time_seconds, 0.75);
        assert_eq!(store.animation_snapshot().clock.time_seconds, 0.75);
    }

    #[test]
    fn timeline_rejects_invalid_samples_without_mutating_the_clock() {
        let (store, _, _) = installed_store();
        toggle_animation_playback(&store).unwrap();
        let before = store.frame_snapshot();
        for invalid in [-0.1, 1.6, f64::NAN] {
            assert!(matches!(
                seek_animation_timeline(&store, invalid),
                Err(AnimationTimelineError::InvalidSampleTime),
            ));
        }
        assert_eq!(store.frame_snapshot(), before);
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
                scene_request_id: request_id,
                asset_id,
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
