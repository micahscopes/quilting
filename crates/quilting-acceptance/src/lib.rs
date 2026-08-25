//! Workspace-level acceptance boundary for composed artifacts.
//!
//! Published library crates keep their fixtures package-local. Tests that
//! intentionally exercise a root-level Blender artifact across multiple
//! crates live here instead of making those libraries depend on repository
//! layout.
