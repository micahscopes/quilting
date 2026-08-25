//! Read-only animation-playback projection for browser views.
//!
//! Playback remains application state and toggling remains a semantic action.
//! This module only derives accessible display state from the committed
//! low-rate [`hyperscope_app::AppSummary`].

use hyperscope_app::AppSummary;

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
}
