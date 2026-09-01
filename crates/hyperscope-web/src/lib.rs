//! Optional browser-host adapters for Hyperscope.
//!
//! The crate has no default features and is not a current application or
//! renderer dependency. Browser durability can therefore mature without
//! changing the running Hyperscope behavior.

#![forbid(unsafe_code)]

#[cfg(any(
    feature = "camera-controls",
    feature = "navigation-controls",
    feature = "patch-lab",
    feature = "render-controls"
))]
mod controls;

#[cfg(feature = "camera-controls")]
pub mod camera_controls;

#[cfg(all(feature = "csr", target_arch = "wasm32"))]
mod effect_js;

#[cfg(feature = "animation-control")]
pub mod animation_control;

#[cfg(feature = "asset-credits")]
pub mod asset_credits;

#[cfg(feature = "durable-history")]
pub mod durable_history;

#[cfg(feature = "local-peer-relay")]
pub mod local_peer_relay;

#[cfg(feature = "interaction-status")]
pub mod interaction_status;

#[cfg(feature = "navigation-status")]
pub mod navigation_status;

#[cfg(feature = "navigation-controls")]
pub mod navigation_controls;

#[cfg(feature = "presentation-card")]
pub mod presentation_card;

#[cfg(feature = "patch-lab")]
pub mod patch_lab;

#[cfg(feature = "render-controls")]
pub mod render_controls;
