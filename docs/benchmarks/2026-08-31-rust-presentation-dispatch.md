# Rust presentation dispatch boundary — 2026-08-31

## Outcome

Presentation navigation and its animation-clip renderer jobs now cross one
typed application boundary. `AppStore::dispatch_presentation` commits the cue
action through the reducer and returns the Rust-allocated sequence, complete
commit, selection job, and cancellation jobs in one `PresentationDispatch`.

`hyperscope-web` delegates directly to that port. Its CSR projection converts
typed `RequestId` and `AssetId` values only at the final JavaScript boundary.
The ordinary browser path calls `requestPresentation` and executes those typed
jobs without filtering or interpreting `AppCommit.effects`.

## Authority and rollback

The reducer remains the only owner of cue, clock, navigation, animation
residency, and job transitions. The browser still performs the physical
renderer clip switch and returns the exact job identity through the existing
completion boundary; it does not decide which job should run.

The generic `dispatchPresentation` method and explicitly sequenced `present`
method remain as rollback/replay seams. They are no longer used by ordinary
application-authority presentation input. Existing `presentimpl` and
`animclipimpl` switches remain unchanged; this cut does not promote defaults.

## Verification policy

`node scripts/smoke-presentation-dispatch-boundary.mjs` is a zero-build oracle.
It requires the typed Rust/AppStore/Leptos/WASM path, rejects effect-array
parsing in `hyperscope-web` and the browser adapter, and parses the inline
browser module. It deliberately does not regenerate bindings, launch Trunk, or
invoke Binaryen. Focused Rust, wasm32, and live-browser gates remain required
before any authority-default promotion.
