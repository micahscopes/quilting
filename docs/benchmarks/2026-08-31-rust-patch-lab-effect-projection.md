# Shared Patch Lab effect projection — 2026-08-31

`PatchLabEffects` is now the sole Rust interpretation of nested Patch Lab jobs
inside an `AppCommit`. Both the dedicated Patch Lab Leptos controls and the
render controls delegate to it rather than independently matching
`AppEffect::PatchLab`.

This is a consolidation cut, not a browser cutover. The WASM/browser paths
still receive generic commits and are the next consumers to move onto the
typed projection. `node scripts/smoke-patch-lab-effect-projection.mjs` rejects
duplicate parsing in `hyperscope-web` without invoking Cargo, Trunk, or
Binaryen.
