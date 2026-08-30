//! Rust-owned semantic camera-lens control for browser views.
//!
//! The application camera remains authoritative. This module projects its
//! perspective lens and queues one device-neutral navigation action per edit;
//! browser code only integrates and applies the accepted Rust projection.

use crate::controls::{numeric_control_domain, NumericControlViewDomain};
use hyperscape::{CameraError, NavigationAction, PerspectiveLens};
use hyperscope_app::{AppFrameSnapshot, AppStore, ReduceError};

#[cfg(all(feature = "csr", target_arch = "wasm32"))]
mod csr;
#[cfg(all(feature = "csr", target_arch = "wasm32"))]
pub use csr::mount_camera_lens_control;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraLensControlViewModel {
    pub revision: u64,
    pub vertical_fov_degrees: f64,
    pub domain: NumericControlViewDomain,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraLensQueueReceipt {
    pub sequence: u64,
    pub queue_revision: u64,
    pub requested_lens: PerspectiveLens,
}

pub fn project_camera_lens_control(frame: &AppFrameSnapshot) -> CameraLensControlViewModel {
    CameraLensControlViewModel {
        revision: frame.revision,
        vertical_fov_degrees: frame.camera.lens.vertical_fov_radians.to_degrees(),
        domain: numeric_control_domain("fov"),
    }
}

pub fn queue_camera_lens_control(
    store: &AppStore,
    vertical_fov_degrees: f64,
) -> Result<CameraLensQueueReceipt, CameraLensControlError> {
    let domain = numeric_control_domain("fov");
    if !vertical_fov_degrees.is_finite()
        || vertical_fov_degrees < domain.minimum
        || vertical_fov_degrees > domain.maximum
        || (domain.integral && vertical_fov_degrees.fract() != 0.0)
    {
        return Err(CameraLensControlError::InvalidFieldOfView);
    }
    let current = store.frame_snapshot().camera.lens;
    let requested_lens = PerspectiveLens {
        vertical_fov_radians: vertical_fov_degrees.to_radians(),
        ..current
    }
    .validate()
    .map_err(CameraLensControlError::InvalidLens)?;
    let (sequence, commit) = store
        .dispatch_navigation(NavigationAction::SetPerspectiveLens(requested_lens))
        .map_err(CameraLensControlError::Reduce)?;
    Ok(CameraLensQueueReceipt {
        sequence,
        queue_revision: commit.revision,
        requested_lens,
    })
}

#[derive(Debug)]
pub enum CameraLensControlError {
    InvalidFieldOfView,
    InvalidLens(CameraError),
    Reduce(ReduceError),
}

impl std::fmt::Display for CameraLensControlError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFieldOfView => {
                formatter.write_str("camera field of view is outside the control domain")
            }
            Self::InvalidLens(error) => error.fmt(formatter),
            Self::Reduce(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CameraLensControlError {}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscope_app::{AppEvent, FrameTick};

    #[test]
    fn projection_uses_the_canonical_route_domain() {
        let store = AppStore::default();
        let view = project_camera_lens_control(&store.frame_snapshot());
        assert_eq!(view.domain.minimum, 35.0);
        assert_eq!(view.domain.maximum, 110.0);
        assert!(view.domain.integral);
        assert_eq!(view.domain.step, 1.0);
        assert!((view.vertical_fov_degrees - 60.0).abs() <= 1.0e-12);
    }

    #[test]
    fn lens_edit_queues_through_application_navigation_and_preserves_clip_planes() {
        let store = AppStore::default();
        let before = store.frame_snapshot().camera.lens;
        let queued = queue_camera_lens_control(&store, 93.0).unwrap();
        assert_eq!(queued.sequence, 0);
        assert_eq!(queued.queue_revision, 1);
        assert_eq!(queued.requested_lens.near, before.near);
        assert_eq!(queued.requested_lens.far, before.far);
        assert_eq!(store.frame_snapshot().pending_navigation_actions, 1);

        store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds: 0.0,
                delta_seconds: 0.0,
            }))
            .unwrap();
        let committed = store.frame_snapshot().camera.lens;
        assert!((committed.vertical_fov_radians.to_degrees() - 93.0).abs() <= 1.0e-12);
        assert_eq!(committed.near, before.near);
        assert_eq!(committed.far, before.far);
    }

    #[test]
    fn invalid_ui_value_cannot_enter_the_navigation_queue() {
        let store = AppStore::default();
        let before = store.frame_snapshot();
        for invalid in [34.0, 110.5, 111.0, f64::NAN] {
            assert!(matches!(
                queue_camera_lens_control(&store, invalid),
                Err(CameraLensControlError::InvalidFieldOfView),
            ));
        }
        assert_eq!(store.frame_snapshot(), before);
    }
}
