# Rust-authoritative conformal chart packet

Date: 2026-08-30

## Outcome

The Rust navigation snapshot now carries the exact packed Möbius coefficients
for its committed reflection chart. In the Rust authority lane, the browser
validates that finite 16-scalar packet and passes it to the renderer without
reconstructing the transform from focus controls. The incumbent JavaScript
`computeMobius` function remains only for the explicit rollback lane.

This closes a split-authority gap: Rust already chose the inversion sphere and
transported the camera, but JavaScript independently rebuilt the chart used by
rendering and LOD. Camera transport, selection projection, WebGL2, and WebGPU
can now share one accepted conformal state and one coefficient convention.

## Shared contract

`quilting-core::Mobius::coefficients_f32` is the canonical backend-neutral
packing: `[a, b, c, d]`, with every quaternion stored as `[w, x, y, z]`.
Hyperscape scene extraction uses the same method, removing its private copy of
the renderer packing convention. `SphereReflectionState::mobius` bridges the
Hyperscape navigation chart to Quilting's conformal primitive without moving
renderer or browser concerns into navigation.

The browser rejects missing, non-finite, incorrectly sized, or semantically
inconsistent Rust chart packets before mutating compatibility signals. The
scoped projection fence still prevents a second browser camera transport.

## Verification

- canonical identity and non-origin sphere-reflection coefficient tests pass;
- exact Möbius point evaluation agrees with Hyperscape's reflection transport;
- the 88-spec Hyperscope route/source oracle passes;
- the complete inline ES module passes `node --check`;
- the exact `wasm32-unknown-unknown` check for `quilting-wasm` with
  `leptos-ui,webgpu-backend` passes; and
- no browser was launched, reloaded, or controlled. Interactive WebGL2/WebGPU
  image parity remains a separate user-triggered promotion gate.
