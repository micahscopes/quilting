# Rust animation clip completion — 2026-08-31

## Outcome

Renderer clip-selection completions now use a typed application receipt.
`AppStore::complete_animation_clip_selection` requires the completion to be
terminal—emitting no follow-up platform jobs—and returns the compact
`AnimationClipSelectionReadModel` together with the commit disposition.

The WASM boundary exposes `finishAnimationClipSelected` and
`finishAnimationClipSelectionFailed`. Ordinary browser adaptation consumes
their compact `selection` projection directly. It no longer serializes the
generic commit and then crosses WASM a second time to serialize the entire
application snapshot merely to compare active and pending clip residency.

## Compatibility and verification

The older `completeAnimationClipSelected` and
`completeAnimationClipSelectionFailed` methods remain as rollback/generated
binding seams. `node scripts/smoke-animation-clip-boundary.mjs` requires the
typed request and completion ports, rejects the full-snapshot completion path,
and parses the browser module. Existing Rust coverage now exercises both stale
and accepted completions through the typed port. Compiled and live gates remain
deferred while unrelated release builds occupy the machine.
