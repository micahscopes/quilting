# Rust primary-scene install state — 2026-08-31

## Outcome

The typed primary-scene install completion now carries the compact installed
scene and animation clip-selection read models in addition to obsolete clip
job identities. Both are sampled after the reducer commit from Rust-owned
state.

The browser uses these fields to validate the installed asset and compare clip
residency for successful, failed, and stale completions. It no longer performs
a second WASM crossing to serialize the complete application snapshot after a
primary installation result.

The WASM adapter also centralizes conversion of
`InstalledPrimarySceneReadModel`, `AnimationClipSelectionReadModel`, and
`AnimationClipDescriptor`; the full snapshot and typed receipts share those
converters instead of maintaining parallel serialization logic.

## Verification

Rust coverage checks that a replacement completion returns the new installed
asset, its initial active clip, no pending clip, and the exact cancellation for
the old scene. `node scripts/smoke-primary-scene-install-boundary.mjs` requires
the compact AppStore/WASM/browser fields and rejects full application snapshots
from ordinary install success and failure adapters. The underlying native
scene-replacement test passed under a one-job, lowest-priority Cargo run before
the receipt assertions were extended; rerunning those assertions and wasm32 is
deferred while unrelated Rust/Trunk work occupies the machine.
