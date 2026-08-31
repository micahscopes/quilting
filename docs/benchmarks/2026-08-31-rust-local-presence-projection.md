# Rust local-presence projection — 2026-08-31

## Boundary

Ephemeral presence remains non-durable platform traffic, but its semantic
payload should describe committed application state. The browser previously
rebuilt camera basis, focus sphere, stable selection, active cue, and animation
time from several incumbent caches even in full Rust-authority mode.

`AppStore::local_presence_snapshot` now projects one protocol-valid
`EphemeralPresence` directly from the committed navigation, selection,
presentation, and animation state. JavaScript retains sender identity,
sequence, TTL policy, deduplication, retry, and transport. The browser-owned
sample remains behind the complete `js|shadow` rollback gate.

## Evidence

The focused native test passes and covers default camera/animation state,
stable selected identity, committed focus/inversion, active cue, and TTL
rejection. The source oracle freezes the complete authority gate, typed WASM
port, retained rollback path, and inline-module syntax. The WebGL2-only wasm32
adapter check also passes:

```text
cargo check --target wasm32-unknown-unknown -p quilting-wasm --features leptos-ui
Finished `dev` profile ... in 12.60s
```

Trunk, wasm-pack, binding generation, and wasm-opt were not run.
