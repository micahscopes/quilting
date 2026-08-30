# Rust focus UI and WebGPU render contract — 2026-08-30

## Outcome

The Fuzzy Vision sidebar section now has a Leptos view backed by the same
`AppStore::render_signal` and reducer as the existing Rust render controls.
Every edit replaces one complete `RenderSettings` value, so mounting the focus
section separately does not create another state authority. The browser
controls remain an explicit fallback under `renderstateimpl=js|shadow` and if
either Rust view fails to mount.

The shared `quilting-core` render contract now carries an optional, fully
resolved `FocusPostprocessPacket` instead of a boolean. It includes the field
mode, blur radius and strength, focus coordinate, bandwidth, normalization
range, Gaussian/Kawase pass counts, and Kawase offset. Contract validation
rejects non-finite and out-of-range values before backend dispatch.

The incumbent WebGL2 pipeline consumes the exact packet. WebGPU extraction and
frame evidence carry the same packet, but WebGPU still explicitly rejects
focus-enabled frames. This cut establishes semantic parity and does **not**
claim that visible WebGPU focus composition is implemented.

## Rollback and compatibility

- `renderstateimpl=js|shadow|rust` remains available.
- Both Rust focus and render islands must mount before the browser controls are
  hidden; partial cutover falls back atomically.
- Existing URL units and defaults are preserved, including one Gaussian pass,
  three Kawase passes, and a 1.5 Kawase offset.
- No browser tab was opened, reloaded, or otherwise disturbed.

## Verification

- `cargo test -p quilting-core`: 248 unit and 15 integration tests passed.
- `cargo test -p hyperscope-app --all-features`: 114 passed.
- `cargo test -p hyperscope-web --all-features`: 38 library and 3 binary tests
  passed.
- `cargo test -p quilting-webgpu --test native_lod`: 2 passed.
- `cargo check --target wasm32-unknown-unknown -p quilting-wasm --features leptos-ui,webgpu-backend --tests`: passed.
- Strict no-dependency Clippy passed for the new code after allowing only the
  pre-existing crate-wide lint categories in `quilting-core`,
  `quilting-wasm`, and `quilting-webgpu`; `hyperscope-app` and
  `hyperscope-web` passed without category allowances.
- The inline Hyperscope ES module passed `node --input-type=module --check`.

## Next measured cut

Implement WebGPU focus composition from `FocusPostprocessPacket`, preserve the
current WebGL2 result as the visual oracle, and admit focus-enabled WebGPU frame
evidence only after deterministic image comparisons pass.
