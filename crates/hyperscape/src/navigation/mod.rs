//! Backend-neutral focus, camera, and navigation state.
//!
//! Browser, HID, gamepad, replay, and network adapters should translate their
//! inputs into the semantic values defined here. They must not integrate a
//! second camera or focus sphere of their own.

pub(crate) mod action;
mod camera;
mod focus;
mod surface_anchor;
mod surface_walk;
mod surface_walk_runtime;
mod transition;

pub use action::{
    NavigationAction, NavigationActionQueue, NavigationController, NavigationRuntime,
    ScheduledNavigationAction,
};
pub use camera::{
    map_space_mouse_axes, map_space_mouse_camera, CameraBasis, CameraError, CameraRig,
    MappedSpaceMouseFrame, NavigationAxes, NavigationFrame, NavigationPreset, PerspectiveLens,
    ReflectionTransport, SpaceMouseCameraInput, SpaceMouseInputError, SpaceMouseMapping,
    SphereReflectionState, TurntableFrame,
};
pub use focus::{FocusAnchor, FocusNavigation, FocusSphere, FocusSphereTransition};
pub use surface_anchor::{
    AnimatedSurfaceAnchor, SurfaceAnchorError, SurfaceAnchoredCameraFrame, SurfaceRelativeCamera,
    SurfaceTangentFrame,
};
pub use surface_walk::{
    compose_surface_relative_forward, decompose_surface_relative_forward,
    scale_relative_near_plane, SurfaceRelativeView, SurfaceWalkContactFrame, SurfaceWalkController,
    SurfaceWalkControls, SurfaceWalkError, SurfaceWalkFrame, SurfaceWalkInput, SurfaceWalkMetrics,
    SurfaceWalkMotion,
};
pub use surface_walk_runtime::{
    SurfaceWalkAttachRequest, SurfaceWalkRecoveryRequest, SurfaceWalkReflectionTransport,
    SurfaceWalkRuntime, SurfaceWalkRuntimeError, SurfaceWalkStepRequest, SurfaceWalkUpdate,
};
pub use transition::{
    CameraTransition, SurfaceAnchorTarget, SurfaceAnchorTransition, TransitionEasing,
};
