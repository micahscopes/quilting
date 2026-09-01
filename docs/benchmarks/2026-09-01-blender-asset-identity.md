# Blender/glTF asset identity — 2026-09-01

Hyperscape glTF interchange v0.1 now carries an optional durable `asset_id` at
`extras.hyperscape.asset_id`. The field is an `AssetId` in Rust and a validated,
non-nil UUID at the Blender codec boundary. Omission remains valid so existing
0.1 files continue to load and reserialize without inventing an identity.

The Blender extension persists the identity in scene settings, exposes a UUID
generator in the Hyperscape scene panel, emits it on export, and restores it on
import. The editable demo source assigns the same stable asset UUID used by the
presentation manifest. This makes its stable node UUIDs usable as complete
`AssetEntityId` lease targets rather than guessing asset scope.

The embedded identity is intentionally not yet treated as the runtime asset
request key. A composed-scene loader must explicitly reconcile the requested
asset identity and embedded authoring identity; this slice does not silently
choose precedence or rewrite either one.

CPU-only evidence:

```text
python3 -m unittest discover -s tools/blender_hyperscape/tests -p 'test_*.py'
                                                        # 37 passed
cargo test -p quilting-gltf --lib                       # 31 passed
cargo test -p quilting-acceptance --test hyperscape_blender_demo
                                                        # 2 passed
cargo check -p quilting-wasm --target wasm32-unknown-unknown \
  --features leptos-ui                                  # passed
python3 -m compileall -q tools/blender_hyperscape       # passed
rustfmt --edition 2021 --check crates/quilting-gltf/src/hyperscape.rs
                                                        # passed
```

No browser, renderer, WebGPU device, server, relay process, or Blender process
was started. The real Blender round-trip test source now asserts asset identity,
but executing it and regenerating the checked-in `.blend`/`.glb` examples are
deferred while graphics contexts are externally contended.
