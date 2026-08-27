//! Read-only navigation and focus projection for browser views.
//!
//! The renderer consumes the immediate frame snapshot. This view observes the
//! same state only when the application publishes its throttled UI commit
//! fence, so a reactive sidebar can never become part of frame integration.

use hyperscope_app::AppFrameSnapshot;

#[cfg(all(feature = "csr", target_arch = "wasm32"))]
mod csr;
#[cfg(all(feature = "csr", target_arch = "wasm32"))]
pub use csr::mount_navigation_status;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationStatusViewModel {
    pub revision: u64,
    pub anchored: bool,
    pub focus_enabled: bool,
    pub inversion_enabled: bool,
    pub sphere_radius: f64,
    pub focus_coordinate: f64,
    pub angular_aperture: f64,
    pub vertical_fov_degrees: f64,
}

impl NavigationStatusViewModel {
    pub fn anchor_label(self) -> &'static str {
        if self.anchored {
            "Selected-object anchor"
        } else {
            "Free focus sphere"
        }
    }

    pub fn chart_label(self) -> &'static str {
        if self.inversion_enabled {
            "inverted chart"
        } else {
            "ordinary chart"
        }
    }

    pub fn focus_label(self) -> &'static str {
        if self.focus_enabled {
            "focus active"
        } else {
            "focus inactive"
        }
    }
}

pub fn project_navigation_status(frame: &AppFrameSnapshot) -> NavigationStatusViewModel {
    NavigationStatusViewModel {
        revision: frame.revision,
        anchored: frame.selected_focus.is_some(),
        focus_enabled: frame.focus.focus_enabled,
        inversion_enabled: frame.focus.inversion_enabled,
        sphere_radius: frame.focus.sphere.radius,
        focus_coordinate: frame.focus.focus_coordinate,
        angular_aperture: frame.focus.angular_aperture,
        vertical_fov_degrees: frame.camera.lens.vertical_fov_radians.to_degrees(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscape::{FocusNavigation, FocusSphere};

    #[test]
    fn projection_distinguishes_free_and_anchored_focus_without_owning_state() {
        let mut frame = hyperscope_app::AppStore::default().frame_snapshot();
        frame.focus = FocusNavigation {
            sphere: FocusSphere::new([1.0, 2.0, 3.0], 2.5).unwrap(),
            focus_enabled: true,
            inversion_enabled: true,
            focus_coordinate: 0.4,
            angular_aperture: 0.08,
            ..FocusNavigation::default()
        };
        let free = project_navigation_status(&frame);
        assert_eq!(free.anchor_label(), "Free focus sphere");
        assert_eq!(free.chart_label(), "inverted chart");
        assert_eq!(free.focus_label(), "focus active");
        assert_eq!(free.sphere_radius, 2.5);
        assert_eq!(free.focus_coordinate, 0.4);
        assert_eq!(free.angular_aperture, 0.08);

        let anchored = NavigationStatusViewModel {
            anchored: true,
            ..free
        };
        assert_eq!(anchored.anchor_label(), "Selected-object anchor");
    }
}
