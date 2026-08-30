# Rust navigation-settings authority cut

Date: 2026-08-30

## Scope

The focus-transition duration and five device-independent surface-walk
preferences now cross the browser boundary as one `NavigationSettings` value.
`hyperscope-app` validates and commits the complete value atomically, publishes
it behind the application revision fence, and exposes the committed projection
through `HyperscopeAppShadow`.

Raw HID axis mappings, sensitivity, window-focus policy, and SpaceMouse profile
remain browser-adapter concerns. They are intentionally absent from the shared
application packet.

The URL lane is explicit and rollback-safe:

- `navstateimpl=js` keeps the incumbent browser signals authoritative (default).
- `navstateimpl=shadow` sends the same packet to Rust and records divergence.
- `navstateimpl=rust` applies the committed Rust packet back to the six browser
  projections in one signal batch.

## Verification

- `cargo test -p hyperscope-app settings --lib`: 23 passed.
- `cargo clippy -p hyperscope-app --all-features --no-deps -- -D warnings`:
  passed.
- Exact `wasm32-unknown-unknown` check with `leptos-ui,webgpu-backend`: passed.
- Route/browser smoke: passed with 88 Rust control specifications.
- Direct Node/WASM boundary probe: exact six-field round-trip; an invalid
  transition duration left the prior revision and value unchanged.
- Inline browser module syntax parse: passed.

Live Chromium comparison and authority evidence remain pending. The default is
therefore still `js`; no release route is silently promoted by this cut.

## Architectural consequence

Navigation preferences now follow the same reducer/store/WASM/thin-adapter
shape as render settings. This gives the packet the stable reducer identity
needed by replay and a future HHHS/Blender session without making browser HID
policy part of shared scene state. The replay schema bump is still pending.
The WebGL2 and WebGPU renderers consume the same application semantics; this
cut adds no backend-specific navigation policy.
