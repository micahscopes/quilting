# Rust presentation cue state — 2026-08-31

## Outcome

Every typed presentation action now returns the active cue selected by that
same reducer commit together with the exact animation clip job receipt. The
browser no longer performs a second full application snapshot to discover the
cue after Rust-authoritative API, keyboard, or Leptos presentation-card input.

The Leptos card serializes only the committed `PresentationSnapshot` into its
platform callback. Browser code applies that cue to renderer resources and
updates its diagnostic cache without interpreting reducer effects or
reconstructing presentation state.

The explicitly sequenced generic `present` API remains as the shadow/replay
rollback seam and still samples the complete projection for parity comparison.

## Verification

The focused native `hyperscope-web` presentation-card test executes with the
`presentation-card` feature and proves that advancing returns cue index 1 and
the exact renderer clip job. The presentation boundary smoke requires compact
cue state in AppStore, WASM, Leptos CSR, and browser adapters; it rejects a full
application snapshot from the ordinary Rust-authority card path. Neighboring
typed boundary smokes and inline JavaScript syntax pass. A wasm32 CSR check is
deferred until system load returns below the safe build threshold.
