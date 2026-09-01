# Ephemeral authoring-lease foundation — 2026-09-01

The v0.1 presence schema now admits an optional `authoring_leases` collection.
Each `AuthoringLeaseClaim` has a stable `LeaseId` and an `AssetEntityId`; the
containing envelope supplies its peer, sender-local sequence, and
receipt-relative TTL. Existing presence fixtures remain byte-for-byte canonical
because an empty collection is omitted.

The contract is deliberately conservative:

- claims are advisory editing coordination, never authorization;
- a peer may claim a target at most once per sample, and lease IDs are unique;
- one sample carries at most 256 claims;
- the latest accepted presence replaces that peer's entire claim set;
- omission releases a claim and TTL expiry removes it;
- one live claim resolves to `Held`;
- two or more live claims resolve to sorted `Contended` state with no invented
  arrival-order winner; and
- neither claims nor their derived read models have an HHHS admission path.

`AppStore::authoring_lease_snapshot` derives this view under the same state lock
as peer presence. A native test admits the same two peers in opposite orders,
proves identical contention, expires one holder, and releases the other by
omission. The Python codec validates and canonically reconstructs the new Rust
fixture at
`crates/hyperscape-protocol/fixtures/presence-authoring-lease-v0.1.json`.

CPU-only evidence:

```text
cargo test -p hyperscape-protocol --lib                 # 8 passed
cargo test -p hyperscope-app --lib                      # 123 passed
cargo test -p hyperscope-app --features replay --lib    # 157 passed
cargo test -p hyperscape-hhhs --lib                     # compiled; 0 tests
python3 -m unittest discover -s tools/blender_hyperscape/tests -p 'test_*.py'
                                                        # 36 passed
cargo check -p quilting-wasm --target wasm32-unknown-unknown \
  --features leptos-ui,webgpu-backend                   # passed
```

No browser, renderer, WebGPU device, server, relay process, or Blender process
was started. Blender acquisition/refresh/release and edit gating are the next
lease slice; the protocol foundation does not silently change current authoring
behavior.
