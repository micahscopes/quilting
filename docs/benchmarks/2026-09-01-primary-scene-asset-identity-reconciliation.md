# Primary-scene asset identity reconciliation — 2026-09-01

Renderer residency and durable authoring identity are now joined explicitly
before a decoded primary scene can mutate renderer state.

The Rust application policy preserves three separate facts:

- `resident_asset_id` identifies the process-local fetch/install job;
- `authoring_asset_id` is present only when the runtime catalog or embedded
  Hyperscape payload declares durable identity; and
- `interaction_asset_id` scopes selection and authoring leases, falling back
  to residency only for a legacy session asset.

An embedded `extras.hyperscape.asset_id` upgrades a dropped or otherwise
session-loaded GLB to its durable Blender identity without relabeling the
renderer job. A presentation/catalog declaration remains valid for a legacy
GLB with no embedded ID. If both durable declarations are present, they must
match; a disagreement fails before model upload rather than creating two lease
namespaces for the same nodes.

A legacy Hyperscape payload that has stable node UUIDs but no asset UUID no
longer promotes those nodes by pairing them with a session residency ID. It
falls back to synthetic, explicitly non-durable session node identities and
therefore cannot enter authoring leases or durable history accidentally.

`load_gltf_data` now returns the validated embedded asset UUID next to stable
node UUIDs. The WASM application boundary performs the typed reconciliation,
and the browser consumes only its returned resident/authoring/interaction
roles. Secondary presentation assets pass through the same mismatch gate.

CPU and compile-only evidence:

```text
cargo test -p hyperscope-app --lib                       # 126 passed
cargo test -p hyperscope-app --features replay --lib     # 160 passed
cargo clippy -p hyperscope-app --lib --no-deps -- \
  -D warnings                                            # passed
cargo check -p quilting-wasm --target wasm32-unknown-unknown \
  --features leptos-ui,webgpu-backend                   # passed
node scripts/smoke-primary-scene-install-boundary.mjs   # passed
git diff --check                                        # passed
```

No browser, renderer, WebGPU adapter/device, server, relay, or Blender process
was started. Live presentation/drag-drop identity checks remain deferred while
another project is deliberately stressing the machine's WebGPU contexts.
