# Rust animation-clock model gate — 2026-08-27

## Invariant

Animation transport is application state, not a browser RAF side effect. The
primary clock contains playing intent, an unwrapped scene time, and signed
finite speed. Frame events advance it from their explicit delta; renderers map
that clock into the resident clip range before evaluating a pose.

`SetClock` restores all three fields atomically. Invalid time or speed leaves
the preceding application revision and clock unchanged. Presentation cue
activation installs its animation directive in the same transaction as cue and
navigation state.

## Evidence

- Replay schema `hyperscope-app-replay/0.17` records time and speed and rejects
  clock-only actions when reading a 0.16 script instead of guessing defaults.
- A one-second frame and the partition `0.25 + 0.25 + 0.5` produce the same
  clock. Starting from time `2.0` at speed `-0.5` produces time `1.5`.
- A non-finite speed is rejected with the complete clock unchanged.
- Cue activation was tested with paused playback, time `0.375`, and speed
  `-0.5`; all three values committed together.
- 73 all-feature `hyperscope-app` tests, strict no-dependency Clippy, strict
  Rustdoc, and the WASM32 `quilting-wasm` check with `leptos-ui` passed.
- The deterministic presentation, navigation, and orchestration fingerprints
  are respectively:
  - `fnv1a-128-json:a9e80c56ba5cbb219b41405c70f84e21`
  - `fnv1a-128-json:79d5531484ba00aee56d6666bab3cda9`
  - `fnv1a-128-json:8607b176a788e02120433b9a8e56d74a`

This checkpoint models authority only. The next measured slice exposes a
compact high-rate clock through generated WASM, shadows the incumbent browser
clock, and keeps JavaScript authority until cadence and clip-wrapping parity are
proven.
