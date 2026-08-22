//! Backend-neutral focus, camera, and navigation state.
//!
//! Browser, HID, gamepad, replay, and network adapters should translate their
//! inputs into the semantic values defined here. They must not integrate a
//! second camera or focus sphere of their own.

mod camera;
mod focus;

pub use camera::{
    map_space_mouse_axes, CameraBasis, CameraError, CameraRig, NavigationAxes, NavigationFrame,
    NavigationPreset, PerspectiveLens, SpaceMouseMapping, SphereReflectionState,
};
pub use focus::{FocusAnchor, FocusNavigation, FocusSphere, FocusSphereTransition};
