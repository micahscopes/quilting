//! Read-only animation-playback projection for browser views.
//!
//! Playback remains application state and toggling remains a semantic action.
//! This module only derives accessible display state from the committed
//! low-rate [`hyperscope_app::AppSummary`].

use hyperscope_app::{AnimationAction, AppStore, AppSummary, ReduceError, SemanticAction};

#[cfg(all(feature = "csr", target_arch = "wasm32"))]
mod csr;
#[cfg(all(feature = "csr", target_arch = "wasm32"))]
pub use csr::mount_animation_control;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
