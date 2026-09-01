# HHHS read-only reactive adoption

Date: 2026-09-01

Candidate: `3968ae1dd1bd5b4a86f8aaa426d955a334cf9ecb`

`DurableProject` once exposed `state_stream` and `state_signal_vec` by cloning
the storage also owned by its durable writer. The exclusive-host migration
removed that unsafe ownership shape. HHHS now provides
`ReplicaReadHandle<MemoryStorage>`, so both FRP adapters are restored without
restoring a second writer, Replica authority, repair access, secrets, policy,
or the durability sink.

The downstream integration proof establishes:

- an empty stream waits without polling;
- one persist-before-publish admission wakes it with the committed
  materialized row;
- a failed persistence publishes neither history nor a reactive revision;
- the fenced owner refuses new read handles, streams, and signal vectors;
- authoritative reopen creates a replacement stream whose initial revision
  is the exact durable state.

Verification:

- `cargo test -p hyperscape-hhhs`: 28 tests passed;
- `cargo test -p hyperscope-hhhs`: 25 tests passed;
- `cargo test -p hyperscope-web --features durable-history`: 9 tests passed;
- downstream-only strict Clippy passed;
- `hyperscope-web` with `csr,durable-history` and `quilting-wasm` with
  `leptos-ui,durable-history` both passed `wasm32-unknown-unknown` checks.

This is the low-rate durable/materialized state lane. Per-frame camera,
pointer, SpaceMouse, animation, LOD, and render data do not belong in it.
