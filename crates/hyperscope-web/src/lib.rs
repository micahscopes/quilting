//! Optional browser-host adapters for Hyperscope.
//!
//! The crate has no default features and is not a current application or
//! renderer dependency. Browser durability can therefore mature without
//! changing the running Hyperscope behavior.

#![forbid(unsafe_code)]

#[cfg(any(feature = "patch-lab", feature = "render-controls"))]
mod controls;

#[cfg(feature = "animation-control")]
pub mod animation_control;

#[cfg(feature = "asset-credits")]
pub mod asset_credits;

#[cfg(feature = "durable-history")]
pub mod durable_history;

#[cfg(feature = "local-peer-relay")]
pub mod local_peer_relay;

#[cfg(feature = "navigation-status")]
pub mod navigation_status;

#[cfg(feature = "presentation-card")]
pub mod presentation_card;

#[cfg(feature = "patch-lab")]
pub mod patch_lab;

#[cfg(feature = "render-controls")]
pub mod render_controls;
