//! Optional browser-host adapters for Hyperscope.
//!
//! The crate has no default features and is not a current application or
//! renderer dependency. Browser durability can therefore mature without
//! changing the running Hyperscope behavior.

#![forbid(unsafe_code)]

#[cfg(feature = "durable-history")]
pub mod durable_history;
