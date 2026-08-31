# Rust animation clip request state — 2026-08-31

## Outcome

`AnimationClipRequest` now carries the compact
`AnimationClipSelectionReadModel` that Rust already computes while validating
the requested clip. The WASM receipt serializes that model as `state`, next to
the exact selection and cancellation jobs.

Ordinary browser clip requests use this receipt to compare renderer residency
for pending, no-op, and repair cases. They no longer call the full application
snapshot after every request. Combined with the typed completion receipt, an
ordinary clip switch now crosses the WASM boundary only with clip-specific
state and job identities.

## Verification

Rust coverage requires the request state to distinguish active and pending
clips across selection, duplicate, and cancellation-only requests.
`node scripts/smoke-animation-clip-boundary.mjs` requires the compact state on
the AppStore, WASM, and browser sides and rejects `refreshAppShadowSnapshot`
from both ordinary request and completion adapters. Compiled/live gates remain
deferred while unrelated release builds occupy the machine.
