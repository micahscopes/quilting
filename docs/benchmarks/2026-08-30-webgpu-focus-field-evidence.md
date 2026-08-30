# WebGPU focus-field evidence — 2026-08-30

## Outcome

The retained WebGPU focus target now has an explicit diagnostic readback for
its RGBA16F raw PBR MRT. `FocusRawFieldImage` decodes row padding and half
floats, reports covered texels, and derives finite covered-channel ranges for
Möbius stretch, logarithmic view depth, and normalized S3 geodesic radius.

This copy/map boundary is intentionally separate from both rendering and focus
composition. The production frame performs no readback; evidence collection
must be requested explicitly after submission.

The native conformance fixture stages the raw field after a complete focus
frame when a device exists, checks nonempty coverage, and verifies all three
semantic channels remain finite in `[0, 1]`. A pure decoder test covers row
padding and channel-range behavior on hosts without a graphics adapter.

## Verification

- Focus schedule/readback unit tests: 3 passed.
- Native conformance test binary: compiled.
- Strict crate-local Clippy: passed.
- `quilting-wasm` with `leptos-ui,webgpu-backend` on
  `wasm32-unknown-unknown`: passed.
- No browser tab or user-run server was touched.

## Next measured cut

Record raw-field and final-image signatures from the browser WebGPU device and
the incumbent WebGL2 renderer for the same extracted frame. Capability stays
off until tolerances, origins, coverage, and expected precision differences
are documented.
