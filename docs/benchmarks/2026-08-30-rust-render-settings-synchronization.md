# Rust render-settings synchronization boundary — 2026-08-30

## Outcome

The browser no longer decides whether a backend-neutral render packet differs
from Rust state. `AppStore::synchronize_render_settings` validates and compares
the complete `RenderSettings` value under the reducer lock, allocates a local
input sequence only for a change, and routes that change through
`SetRenderSettings`. An identical projection advances no revision, publishes
no FRP value, and emits no effect.

The WASM port accepts one typed camel-case packet and returns a disposition,
optional Rust-owned sequence and commit, an exact match result, and the
committed render projection. JavaScript forwards the packet, mirrors the
projection, and executes effects from the returned commit. It no longer owns
render/focus equality, snapshot admission, or settings input sequencing.

## Patch Lab effect preservation

Render settings are not a purely passive value: changing the atlas exponent or
maximum face-edge ratio can normalize Patch Lab controls, cancel superseded
work, and request a new LOD evaluation. The synchronization port deliberately
enters through the existing reducer instead of mutating the render projection.
The focused Patch Lab test proves that the exact reconciled `EvaluateLod`
effect remains attached to the returned commit.

## Rollback

- `renderstateimpl=js|shadow|rust` is unchanged; `js` remains the default.
- The incumbent browser controls and WebGL2 renderer remain available.
- The older explicitly sequenced `setRenderSettings` WASM method remains as a
  compatibility seam, but Hyperscope browser code no longer calls it.
- Generated bindings and live Chromium evidence remain deferred to the
  explicitly authorized low-CPU build lane.

## Verification

- Four native render-settings tests pass: the existing atomic revision fence,
  explicit cutover default, changed/unchanged synchronization, Rust sequence
  ownership, exact projection, and rejection without mutation.
- The focused active-Patch-Lab atlas/grading test passes with the exact LOD
  evaluation effect carried by the synchronization commit.
- `node scripts/smoke-render-settings-boundary.mjs`: passed. The zero-build
  oracle parses the inline browser module and rejects browser-owned equality,
  snapshot admission, explicit sequence allocation, or the legacy setter.
- The exact `wasm32-unknown-unknown` check with
  `leptos-ui,webgpu-backend` and tests passed at one idle-priority Cargo job;
  no bindings, Trunk server, or optimizer were invoked.
