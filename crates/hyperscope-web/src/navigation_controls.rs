//! Rust-owned semantic navigation-settings projection for browser views.
//!
//! High-rate camera and surface motion stay on the direct frame path. This
//! module owns only the low-rate focus-transition and surface-walk preference
//! packet, emitting one complete replacement through `AppStore` per edit.

use crate::controls::{numeric_control_domain, NumericControlViewDomain};
use hyperscope_app::{
    AppNavigationSettingsSnapshot, AppStore, NavigationSettings, ReduceError, SemanticAction,
};

#[cfg(all(feature = "csr", target_arch = "wasm32"))]
mod csr;
#[cfg(all(feature = "csr", target_arch = "wasm32"))]
pub use csr::mount_navigation_controls;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationControlIntent {
    settings: NavigationSettings,
}

impl NavigationControlIntent {
    pub fn settings(self) -> NavigationSettings {
        self.settings
    }

    pub fn with_transition_seconds(mut self, value: f64) -> Self {
        self.settings.transition_seconds = value;
        self
    }

    pub fn with_smoothing_seconds(mut self, value: f64) -> Self {
        self.settings.surface_walk.smoothing_seconds = value;
        self
    }

    pub fn with_tangent_pull_fraction(mut self, value: f64) -> Self {
        self.settings.surface_walk.tangent_pull_fraction = value;
        self
    }

    pub fn with_speed_octave_steps(mut self, value: f64) -> Self {
        self.settings.surface_walk.speed_octave_steps = value;
        self
    }

    pub fn with_body_scale_octave_steps(mut self, value: f64) -> Self {
        self.settings.surface_walk.body_scale_octave_steps = value;
        self
    }

    pub fn with_eye_height_octave_steps(mut self, value: f64) -> Self {
        self.settings.surface_walk.eye_height_octave_steps = value;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationControlCommit {
    pub sequence: u64,
    pub revision: u64,
    pub value: NavigationControlIntent,
}

pub fn set_navigation_controls(
    store: &AppStore,
    intent: NavigationControlIntent,
) -> Result<NavigationControlCommit, NavigationControlError> {
    let settings = intent
        .settings
        .validate()
        .map_err(NavigationControlError::InvalidSettings)?;
    let (sequence, commit) = store
        .dispatch_semantic(SemanticAction::SetNavigationSettings(settings))
        .map_err(NavigationControlError::Reduce)?;
    Ok(NavigationControlCommit {
        sequence,
        revision: commit.revision,
        value: project_navigation_controls(&store.navigation_settings_snapshot()).value,
    })
}

#[derive(Debug)]
pub enum NavigationControlError {
    InvalidSettings(&'static str),
    Reduce(ReduceError),
}

impl std::fmt::Display for NavigationControlError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSettings(error) => formatter.write_str(error),
            Self::Reduce(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for NavigationControlError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationControlsViewModel {
    pub revision: u64,
    pub value: NavigationControlIntent,
    pub transition: NumericControlViewDomain,
    pub smoothing: NumericControlViewDomain,
    pub tangent_pull: NumericControlViewDomain,
    pub speed: NumericControlViewDomain,
    pub body_scale: NumericControlViewDomain,
    pub eye_height: NumericControlViewDomain,
}

fn scaled_domain(key: &str, scale: f64) -> NumericControlViewDomain {
    let domain = numeric_control_domain(key);
    NumericControlViewDomain {
        minimum: domain.minimum * scale,
        maximum: domain.maximum * scale,
        integral: domain.integral && scale == 1.0,
        step: domain.step * scale,
    }
}

pub fn project_navigation_controls(
    snapshot: &AppNavigationSettingsSnapshot,
) -> NavigationControlsViewModel {
    NavigationControlsViewModel {
        revision: snapshot.revision,
        value: NavigationControlIntent {
            settings: snapshot.settings,
        },
        transition: scaled_domain("interp", 0.01),
        smoothing: scaled_domain("walksmooth", 0.01),
        tangent_pull: scaled_domain("walkalign", 0.01),
        speed: scaled_domain("walkspeed", 1.0),
        body_scale: scaled_domain("walkscale", 1.0),
        eye_height: scaled_domain("walkheight", 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_uses_semantic_units_and_preserves_hidden_walk_policy() {
        let mut settings = NavigationSettings::default();
        settings.surface_walk.fast_multiplier = 4.25;
        let view = project_navigation_controls(&AppNavigationSettingsSnapshot {
            revision: 41,
            settings,
        });
        assert_eq!(view.revision, 41);
        assert_eq!(view.transition.minimum, 0.05);
        assert_eq!(view.transition.maximum, 5.0);
        assert_eq!(view.transition.step, 0.01);
        assert_eq!(view.smoothing.maximum, 1.5);
        assert_eq!(view.tangent_pull.maximum, 1.0);
        assert_eq!(view.speed.minimum, -400.0);
        assert_eq!(view.body_scale.minimum, -800.0);
        assert_eq!(view.eye_height.maximum, 400.0);
        assert_eq!(
            view.value
                .with_speed_octave_steps(125.0)
                .settings()
                .surface_walk
                .fast_multiplier,
            4.25,
        );
    }

    #[test]
    fn complete_control_edit_dispatches_through_the_application_reducer() {
        let store = AppStore::default();
        let intent = project_navigation_controls(&store.navigation_settings_snapshot())
            .value
            .with_transition_seconds(2.25)
            .with_smoothing_seconds(0.4)
            .with_tangent_pull_fraction(0.35)
            .with_speed_octave_steps(100.0)
            .with_body_scale_octave_steps(-50.0)
            .with_eye_height_octave_steps(25.0);
        let committed = set_navigation_controls(&store, intent).unwrap();
        assert_eq!(committed.sequence, 0);
        assert_eq!(committed.revision, 1);
        assert_eq!(committed.value, intent);
        assert_eq!(
            store.navigation_settings_snapshot().settings,
            intent.settings()
        );
    }

    #[test]
    fn invalid_control_edit_is_atomic() {
        let store = AppStore::default();
        let before = store.navigation_settings_snapshot();
        let invalid = project_navigation_controls(&before)
            .value
            .with_tangent_pull_fraction(1.5);
        assert!(matches!(
            set_navigation_controls(&store, invalid),
            Err(NavigationControlError::InvalidSettings(_)),
        ));
        assert_eq!(store.navigation_settings_snapshot(), before);
    }
}
