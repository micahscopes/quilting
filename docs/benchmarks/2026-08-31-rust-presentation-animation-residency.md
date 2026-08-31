# Rust presentation animation residency — 2026-08-31

## Outcome

The renderer-local binding between an authored presentation asset and the
currently resident primary scene now uses a typed application receipt.
`AppStore::set_presentation_animation_residency` returns the commit plus the
exact animation selection and cancellation jobs chosen while resolving the
active cue. The binding remains process-local evidence and never enters
authored or HHHS state.

This closes a concrete adapter hole: the browser previously passed a generic
`AppCommit` to code expecting `{ effect, cancellations }`, so a clip selection
emitted while establishing residency could be silently skipped. The ordinary
path now calls `setPresentationAnimationResidency` and forwards its typed job
fields into the same renderer adaptation used by cue transitions.

## Compatibility and verification

The older `bindPresentationAnimationResidency` and
`clearPresentationAnimationResidency` methods remain as generic rollback and
generated-binding seams. `node scripts/smoke-presentation-dispatch-boundary.mjs`
requires the typed AppStore/WASM/browser path and rejects generic residency
adaptation; the broader presentation source oracle checks the same call site.
Compiled and live gates remain deferred while unrelated builds occupy the
machine, and this cut starts no Trunk, Cargo, or Binaryen process.
