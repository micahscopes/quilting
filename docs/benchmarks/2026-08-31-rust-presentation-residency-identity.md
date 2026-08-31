# Rust presentation residency identity — 2026-08-31

## Outcome

Presentation animation residency is now bound atomically inside AppStore. The
adapter supplies only the authored presentation asset ID; Rust reads the exact
renderer-installed scene request and asset identities it already owns, admits
the binding through the reducer, publishes read models, and returns the
resulting residency, active cue, and renderer clip jobs.

This removes a browser read/echo cycle in which JavaScript serialized the
complete application, extracted process-local scene identities, sent them back
to Rust, then serialized the complete application again to validate the
result. The explicit three-ID setter remains available as a rollback and
adapter-testing seam.

## Verification

The application residency test exercises the atomic installed-scene port and
checks its exact binding, cue, and clip-selection job. The presentation source
smoke requires the AppStore and WASM ports and rejects the legacy identity
round-trip and full snapshots from the ordinary browser binding path. Inline
JavaScript and neighboring typed boundary smokes pass. The focused native
residency test passes; the wasm32 adapter check remains pending.
