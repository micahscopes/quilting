# Rust-authoritative browser authoring leases — 2026-09-01

The application presence read now atomically returns both its protocol-facing
entity selection and the selected `AssetEntityId` used for advisory authoring.
This closes the information loss where browser presence retained only a local
entity UUID and therefore could not safely coordinate two composed assets with
the same node identity.

`LocalAuthoringLeaseController` is process-local Rust session state. It:

- retains a lease ID while a peer continues to desire the same target;
- emits targets in deterministic asset/entity order;
- omits deselected targets immediately;
- allocates a fresh ID on reacquisition;
- resets claims if the platform peer identity changes; and
- enforces the protocol's 256-claim bound before mutation.

The authoritative `HyperscopeAppShadow` encoder samples camera, focus,
selection, cue, animation time, and authoring targets from Rust, synchronizes
the controller, validates the complete `PresenceEnvelope`, and returns exact
JSON. JavaScript supplies only a fresh message ID, sender-local peer/sequence,
TTL, and carrier delivery. Its explicit JS/shadow rollback still emits no lease
claims because that path has only entity-local selection and must not invent
asset scope.

This is coordination, not authorization: the encoder has no authored-command
admission privilege, no HHHS path, and no durable checkpoint.

CPU/compile-only evidence:

```text
cargo test -p hyperscope-app --lib                       # 124 passed
cargo test -p hyperscope-app --features replay --lib     # 158 passed
cargo check -p quilting-wasm --target wasm32-unknown-unknown \
  --features leptos-ui,webgpu-backend                   # passed
node scripts/smoke-rust-local-presence-projection.mjs   # passed
```

No browser, renderer, WebGPU adapter/device, server, relay, or Blender process
was started. Generated-WASM/browser runtime and real Blender contention tests
remain part of the deferred graphics-runtime gate.
