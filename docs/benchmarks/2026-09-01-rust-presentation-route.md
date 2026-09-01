# Rust-authoritative presentation route packet

Date: 2026-09-01

## Outcome

Presentation startup now receives one typed Rust packet containing the enable
state and optional stable cue UUID. The browser still has to inspect the raw
enable bit before WASM in order to allocate the optional presentation worker,
but Rust verifies that bootstrap decision before presentation state is used.
Cue restoration consumes the Rust-normalized UUID rather than `initParams.cue`.

A cue without `presentation=1` is now a route diagnostic. Such a link formerly
looked specific while silently running the non-presentation client. Invalid or
nil cue UUIDs likewise cannot produce the typed startup packet. Resolving a
valid UUID against the loaded manifest remains the existing Rust application
action, so unknown-but-well-formed cue IDs still fail visibly at the correct
boundary.

## Evidence

- `cargo test -p hyperscope-app`: 141 passed.
- `cargo check -p quilting-wasm --target wasm32-unknown-unknown --features leptos-ui,durable-history`: passed.
- development WASM rebuilt with `HYPERSCOPE_WASM_PROFILE=dev` and
  `HYPERSCOPE_WASM_OPT=0`.
- `node scripts/smoke-rust-presentation-route.mjs`: passed.
- `node scripts/smoke-rust-patch-lab-route.mjs`: passed as a neighboring route
  regression gate.
- `node scripts/smoke-hyperscope-presentation.mjs`: passed all eight cues plus
  linked-cue and rejection behavior.
- the extracted browser module parses under `node --check`.
- `git diff --check`: passed.
