# Rust animation-runtime read model — 2026-08-31

## Outcome

Hyperscope now exposes one atomic, low-rate Rust animation-runtime read model:
application revision, renderer-local presentation residency, active/pending
clip state, and the playback clock.

The renderer adapter continues to use the allocation-free installed-animation
sample packet for each applied pose. It reads the compact model at cue, clip,
clock synchronization, URL restoration, and paused-seek boundaries. These
paths no longer
serialize cue composition, assets, authored scene state, navigation, Patch Lab,
jobs, or diagnostics through the complete application snapshot.

## Verification

Application coverage checks the model revision, exact residency, and pending
clip selected by the same binding commit. The presentation source smoke
requires the Rust/WASM read-model boundary and rejects complete application
snapshots from ordinary presentation animation adaptation. Neighboring typed
boundary smokes and inline JavaScript syntax pass. Native and wasm32 checks
remain deferred while an unrelated ESP release compiler occupies the machine.
