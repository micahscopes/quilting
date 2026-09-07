# Shared arc controls: implementation and remaining gates

September 7, 2026. This is a partial delivery checkpoint, not a claim that the
controllable triangular and quad Patch views are complete.

## Shared Fe behavior

`quilting_demo_view::handles` now owns pure screen-space picking: a pixel radius,
finite positive homogeneous depth, nearest-center selection and stable depth/ID
tie breaking. Both `arc_curve_explorer` and the existing `clifford_patch_lab`
use it. Neither application delegates the picking algorithm to JavaScript.
The patch viewer's old radius was in normalized screen coordinates, which made
its effective pixel size depend on the viewport. Both now request 15 pixels.

The existing patch viewer still has scalar weight satellites. Sharing picking
does **not** turn those satellites into geometric arc controls. That authoring
work remains required for both triangular and quadrangular patch views.

The standalone arc page uses the shared declarative `fe-surface` live lifecycle.
No private startup script was added.

## Color packing and compiler evidence

The release arc build initially failed in the background fragment's helper chain.
A reduced shared-mb2 regression separates a literal color, float conversion,
checked arithmetic packing and bitwise packing. The first two compile; checked
packing retains six signed-overflow guard branches and fails because fullscreen
fragments have no trap channel. Riff-cat producer events identify `pack` as the
helper requiring an extra parameter; the small Sonatina snapshot identifies the
actual `smulo`/`saddo` guards. This is not an inlining or shader-size diagnosis.

`quilting_oklch::pack_rgba8_with_alpha` now expresses the RGBA8 representation with
four masked byte lanes, constant shifts and bitwise OR. Conversion, clamping and
rounding are unchanged. This is shared Fe color semantics, not a generated shader,
unchecked arithmetic substitution or hardcoded background. The compiler still
rejects the unproved checked-arithmetic version; its range inference and late
trap diagnostic have **not** been fixed by this change.

Executed evidence:

- Both arc and legacy patch ingots passed source checks after sharing picking.
- `shared_handle_selection_obeys_screen_radius_depth_and_identity`: passed in
  release Wasm, covering nonsquare viewport distances, invalid clip coordinates,
  depth and stable identity ordering.
- `display_rgba_alpha_preserves_rgb_and_all_alpha_bytes`: passed in release Wasm,
  including an independent 256-value sweep of every byte lane, alpha edge cases
  and signed high-bit preservation; 16.20 seconds including oracle setup.
- Shared-mb2 `fullscreen_color_helpers_preserve_packing_and_reject_unproved_overflow`:
  passed; bitwise example emits 1,726 bytes of WGSL. This is a small compilation
  regression, not GPU execution or a measurement of the complete arc shaders.
- The production arc build passes its background stage, then fails admission of
  the curve vertex stage. No successful new browser build is claimed.

Logs and captures are in `/laboratory/quilting/scratch/` with prefixes
`shared-handle`, `arc-color-helper`, `arc-bitwise-color` and `arc-raster` dated
`20260907`. In particular the small decisive IR is
`arc-color-helper-ir-20260907/0002-shade-post.sona`.

## Geometric authoring boundary

Local paper: Krasauskas and Zube, *Rational Bezier Formulas with Quaternion and
Clifford Algebra Weights*, `/home/micah/sync/PDFs/math/clifford patches.pdf`,
sections 2.5–2.6, formulae 13–15 and Lemmas 2–3.

The paper's low-degree constructions impose compatibility between boundary
circles and weights. Independently dragging four arbitrary circular edges does
not establish that one bilinear patch with those exact edges exists. The existing
scalar adjustments also do not constitute general geometric boundary authoring.

Next: finish the curve vertex-stage diagnosis and browser-test picking/dragging;
then expose a supported geometric construction through shared arc widgets, with
its constraints solved explicitly. Verify full boundary agreement before adding
the shared sampling maps. Do not silently fit incompatible arcs or claim tangent
continuity from endpoint agreement alone.
