# Rust frame-delta authority — 2026-08-31

## Boundary

The browser owns RAF timestamps and therefore supplies a measured delta. It
does not need to own the application's elapsed epoch. Previously the live lane
accumulated an absolute time in JavaScript and round-tripped both values into
Rust on every frame, despite `AppState` retaining and validating the same
epoch.

The Rust-authority frame port now accepts only the platform delta. `AppStore`
derives absolute application time and reduces the frame within one lock scope,
so a future worker-capable adapter cannot race between reading and advancing
the epoch. The explicit elapsed-plus-delta port remains for deterministic
replay and `js|shadow` evidence. Ephemeral presence samples the committed epoch
through a low-rate allocation-free getter; JavaScript retains no elapsed-time
mirror.

## Evidence

`scripts/smoke-rust-frame-delta-authority.mjs` freezes the ordinary RAF and
selection-event split lanes, the atomic Rust operation, the retained
rollback port, and inline-module syntax. The focused atomic reducer test passes.
The checked-in generated-WASM app-shadow oracle also passes, proving that the
absolute-time rollback port retained its incumbent behavior. Finally, the
WebGL2-only wasm32 adapter check passes:

```text
cargo check --target wasm32-unknown-unknown -p quilting-wasm --features leptos-ui
Finished `dev` profile ... in 7.09s
```

Trunk, wasm-pack, binding generation, and wasm-opt were not run.
