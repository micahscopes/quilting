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

The browser synchronization port is now idempotent and Rust-owned as well.
`AppStore::synchronize_navigation_settings` validates and compares the complete
packet under the reducer lock, allocates a local input sequence only when the
packet changed, and routes that change through `SetNavigationSettings`. An
unchanged projection advances neither the application revision nor the FRP
commit fence. The WASM adapter returns the disposition, optional sequence and
commit, exact Rust projection, and typed match result; JavaScript no longer
owns epsilon comparison, the decision to dispatch, or input-sequence
allocation. The older explicitly sequenced WASM setter remains available as a
compatibility seam, but the browser no longer calls it.

The URL lane is explicit and rollback-safe:

- `navstateimpl=js` keeps the incumbent browser signals authoritative (default).
- `navstateimpl=shadow` sends the same packet to Rust and records divergence.
- `navstateimpl=rust` applies the committed Rust packet back to the six browser
  projections in one signal batch. It also mounts a Leptos CSR control island
  over `AppStore::navigation_settings_signal`; each edit reads the reducer's
  current complete packet, commits one replacement, and exposes only the
  committed result to the browser adapter. The incumbent HTML controls remain
  intact and visible if the Rust view cannot mount.

## Verification

- `cargo test -p hyperscope-app settings --lib`: 23 passed.
- `cargo clippy -p hyperscope-app --all-features --no-deps -- -D warnings`:
  passed.
- Exact `wasm32-unknown-unknown` check with `leptos-ui,webgpu-backend`: passed.
- Route/browser smoke: passed with 88 Rust control specifications.
- Direct Node/WASM boundary probe: exact six-field round-trip; an invalid
  transition duration left the prior revision and value unchanged.
- Replay 0.23: exact nested-settings JSON round-trip, deterministic committed
  state, and explicit 0.22 rejection without mutation.
- Inline browser module syntax parse: passed.
- `hyperscope-web --all-features`: 36 native tests passed, including complete
  control dispatch, semantic-unit projection, preservation of non-UI walk
  policy, and atomic rejection.
- Strict `hyperscope-web --all-features` Clippy: passed.
- Two focused native synchronization tests pass: changed/unchanged sequencing,
  idempotent revisions, exact projection, and invalid-input atomicity.
- `node scripts/smoke-navigation-settings-boundary.mjs`: passed. This
  zero-build oracle also parses the inline browser module and rejects a return
  of browser-owned equality, snapshot admission, or sequence allocation.

Live Chromium comparison and authority evidence remain pending. The default is
therefore still `js`; no release route is silently promoted by this cut.

## Architectural consequence

Navigation preferences now follow the same reducer/store/WASM/thin-adapter
shape as render settings, including a Rust-owned FRP browser view. Replay 0.23
now gives the packet a durable semantic identity suitable for a future
HHHS/Blender session without making browser HID policy part of shared scene
state. The WebGL2 and WebGPU renderers consume the same application semantics;
this cut adds no backend-specific navigation policy.
