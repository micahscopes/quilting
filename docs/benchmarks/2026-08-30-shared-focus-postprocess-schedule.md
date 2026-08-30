# Shared focus-postprocess schedule — 2026-08-30

## Outcome

The focus pipeline's backend-neutral execution policy now lives in
`quilting-core::focus_postprocess`. It freezes:

- the 2x JFA downsample rule;
- descending power-of-two propagation plus two unit cleanup passes;
- ping-pong source/destination parity;
- the spheroidal dense-field JFA bypass;
- the three 0°, 60°, and 120° directional Gaussian passes;
- total-blur strength normalization as quality passes increase;
- Kawase pass offsets; and
- final output ownership.

The incumbent `fuzzy-vision` WebGL2 implementation consumes these shared
types and constants. Its previous private schedule was removed. This is a
semantic consolidation only: shader programs and graphics resources remain
backend-owned, and no visible WebGL behavior was intentionally changed.

WebGPU can now lower the same immutable schedule into `wgpu` render passes
without copying the old parity and pass-order logic. The shared schedule is
validated from the exact `FocusPostprocessPacket` established by the preceding
cut.

## Verification

- `cargo test -p quilting-core focus_postprocess --lib`: 6 focused tests passed.
- `cargo test -p fuzzy-vision`: 13 tests passed.
- Full core, WASM feature, and strict changed-code Clippy gates are recorded
  with the commit after final verification.
- No browser tab or user-run server was touched.

## Next measured cut

Add the WGSL focus pass family and retained WebGPU texture/pipeline residency,
then lower `FocusPostprocessSchedule` into an offscreen image-evidence frame.
The presentation path remains on WebGL2 until that image comparison passes.
