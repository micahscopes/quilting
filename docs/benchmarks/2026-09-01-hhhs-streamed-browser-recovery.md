# HHHS streamed browser recovery checkpoint

Date: 2026-09-01

Candidate: `87cdecf5ab3ef39084f931c5fad5cf424a26576d`

This is a downstream compatibility and memory-ownership checkpoint, not an
HHHS release qualification or a claim about the eventual `v0.4.5` tag.

## Adopted path

Ordinary Hyperscope IndexedDB restart now uses
`IndexedDbReplicaLog::open_with_memory_storage`. HHHS cursor-validates one
transaction row at a time into the live `MemoryStorage`; the returned browser
log deliberately retains no second vector of decoded transaction bodies.
`hyperscape-hhhs` accepts that exact sink/store pair through a narrow
trusted-local constructor, validates every public authored payload and project
identity, and builds the replica around the recovered store without replaying
the journal into another store.

The retained Rust-side recovery shape is therefore canonical `MemoryStorage`
plus at most the current decoded IndexedDB row, rather than:

1. the browser log's complete decoded transaction vector;
2. a clone of that vector in `hyperscope-web`; and
3. a separately reconstructed `MemoryStorage`.

No numerical peak-memory claim is made here. The browser implementation may
retain its own cursor/request buffers, and canonical history necessarily stays
resident in `MemoryStorage`.

## Legacy boundary

The old Hyperscope row namespace still receives a deliberately separate
inspection path. When legacy rows exist, downstream recovery retains the HHHS
journal long enough to compare both representations byte for byte or migrate
the exact sequence. It then reconstructs one store from that inspected
journal. An empty legacy namespace takes the streamed path immediately.

This preserves the existing non-destructive migration refusal: disagreeing
legacy and HHHS-owned logs are never merged or silently preferred. Schema
creation is now performed before the legacy read so a completely fresh origin
can select the streamed path without an initial failed object-store lookup.

## Verification

- `cargo test -p hyperscape-hhhs -p hyperscope-hhhs`: 55 passed.
- `cargo test -p hyperscope-web --features durable-history`: 9 passed.
- `cargo test -p hyperscope-app`: 142 passed.
- `cargo check -p hyperscope-web --target wasm32-unknown-unknown --features durable-history`: passed.
- `cargo check -p quilting-wasm --target wasm32-unknown-unknown --features leptos-ui,durable-history`: passed.
- `cargo clippy -p hyperscape-hhhs -p hyperscope-hhhs -p hyperscope-web --features hyperscope-web/durable-history --no-deps -- -D warnings`: passed.
- The browser test target compiled successfully through `wasm-pack test
  --headless --chrome crates/hyperscope-web --features durable-history`.
- Browser execution did not begin: local `wasm-pack 0.13.1` attempted to fetch
  obsolete ChromeDriver `114.0.5735.90` from a missing upstream URL after
  detecting ChromeDriver version `152.0.7977.64`.

Focused tests prove that direct-store recovery preserves materialized state and
roots, continues durable writes, refuses foreign-project payloads, and that an
ordinary IndexedDB restart reports `retains_decoded_transactions() == false`
with an empty inspection journal.
