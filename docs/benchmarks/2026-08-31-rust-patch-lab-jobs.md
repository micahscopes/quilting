# Rust Patch Lab job boundary — 2026-08-31

## Outcome

Patch Lab control edits and geometry/LOD completions now return typed Rust job
receipts. `AppStore::set_patch_lab_session` and
`AppStore::complete_patch_lab` share `PatchLabEffects`, the same projection
already used by both Leptos control surfaces.

The ordinary WASM/browser paths use `requestPatchLab`, the four
`finishPatchLab*` completion methods, and `drainPatchLabEffects`. Render-setting
synchronization carries a dedicated `patchLabEffects` list. JavaScript no
longer unwraps or filters top-level `AppCommit.effects`; it switches only on
the typed Patch Lab job kind in order to execute platform geometry and LOD
work.

## Rollback and frame safety

The generic `dispatchPatchLab`, completion methods, and `drainAdapterEffects`
remain available as rollback/generated-WASM seams. The typed quiet-frame drain
restores and rejects its queue if a future reducer emits a non-Patch-Lab frame
effect, preventing silent loss when this contract evolves.

## Lightweight verification

`node scripts/smoke-patch-lab-job-boundary.mjs` requires the typed
AppStore/WASM/browser path, rejects generic browser parsing, and parses the
inline browser module. The shared-projection and render-settings source smokes
cover the adjacent Rust and settings boundaries. Native/wasm32/live checks
remain deferred while unrelated builds occupy the machine; no Trunk or
Binaryen process is launched by these oracles.
