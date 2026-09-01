# Hyperscope HHHS integration consolidation

Date: 2026-09-01

The crate formerly named `hyperscope-hhhs-shadow` is now
`hyperscope-hhhs`. The old name described its first diagnostic type but became
misleading after the crate acquired the durable-first coordinator, restartable
peer ingress, canonical AppStore restore, and browser durability lifecycle.

No compatibility facade or third integration crate was added. The directory,
package, Rust crate identifier, dependencies, tests, and current documentation
moved together. `AuthoredHhhsShadow` remains available inside the renamed crate
for its explicit AppStore-first diagnostic use; `DurableAuthoredSession` is the
normal authoritative path.

The ownership boundary remains:

- `hyperscape-hhhs`: application-neutral authored payloads, materialization,
  archives, and durable Replica hosting;
- `hyperscope-hhhs`: AppStore coordination, durable source cursors, peer
  ingress, and canonical restore;
- `hyperscope-web`: browser durability and UI adapters;
- `quilting-wasm`: thin generated browser bindings.

Verification on HHHS candidate `7575510`:

- `cargo test -p hyperscope-hhhs`: 25 tests passed;
- `cargo test -p hyperscope-web --features durable-history`: 9 tests passed;
- `quilting-wasm` with `leptos-ui,durable-history` passed its
  `wasm32-unknown-unknown` check;
- downstream-only strict Clippy passed for the renamed integration graph.
