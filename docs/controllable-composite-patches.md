# Controllable surfaces over the compositional atlas

Work in progress, September 7, 2026. This is a delivery slice of the
[consolidated goal](composition-and-recursive-atlas-goal.md), not its completion.

The existing `composite_atlas` Fe actor now has flat, curved-triangle and
curved-quad views. They share the worker-generated primary pool, permutations,
boundary maps, publication protocol and retained compatible draw snapshots.
There is no second atlas loader or JavaScript geometry implementation.

Curved views use `ProjectiveControl<EuclideanCliffordAlgebra>` and the generic
triangular/bilinear evaluators. Blue handles author corners; an orange handle
authors a point on the selected boundary arc. The resulting common pole couples
the other arcs. This is not independently configurable arbitrary circular edges,
nor a claim of tangent continuity between separately authored patches.

## Interaction and remaining limits

- Drag a handle in the camera plane; wheel moves a selected handle in depth.
- Drag empty space to orbit; wheel without a selection changes camera distance.
- Triangle mode uses bottom/right/left LoDs. The fourth corner and top LoD do
  not participate in its surface construction.
- Wire visibility, depth testing and four-sample antialiasing use shared Fe
  raster declarations and the WebGPU realization.
- `patch_valid` reports admission to the finite-common-pole construction.
  Invalid construction is rejected, not silently projected into another shape.
- Spacing currently measures the reference domain. Uniform curved-boundary
  arc length and interior stretch compensation remain unfinished.
- Vertex validity is not a conservative certificate for an entire triangle
  near a pole. Pathological chart coverage remains a separate acceptance gate.
- The flat coordinate fields at the UI boundary still need consolidation into
  a reusable typed authoring interface. Generic geometry is not flattened to
  work around the compiler.

## Evidence so far

The Wasm composition oracle passed in 44.51 seconds, including compilation and
the existing spacing/coherent-selection checks plus the new surface tests:
corner and authored midpoint interpolation, triangle exclusion of its hidden
fourth corner, and independent hypotenuse LoDs 0–8. This is not a frame-time or
browser startup measurement.

With shared Fe `c734eda1f`, the integrated release `fe web dev` build completed
in 87.888 seconds and served at `http://127.0.0.1:38331/`. Emission was 825,838
Wasm bytes and 201,298 WGSL bytes across four passes. Chrome page 65 rendered
all three modes without console warnings/errors. Captured triangle and quad
views had 1,074 and 4,296 draw vertices respectively at the tested settings.

An injected gesture through the shared event entry point selected handle 4 and
moved its point from approximately `(0,-0.7,0.6)` to
`(0.17146,-0.54169,0.53190)` while keeping valid state and a fixed camera. Empty
space dragging changed camera angles without changing that point. This checks
the Fe event path, not native pointer capture.

Browser testing also exposed a control-binding failure: an interleaved readout
shifted event ordinals relative to the input record. The old derive assumed
matching record order; changing the wire toggle instead changed `bottom_lod`.
A shared Fe reflection-based correction and Wasm regression are in progress.
Do not interpret the successful rendering checks as acceptance of the full
control panel until that correction is rebuilt and verified in Chrome.

The correction's Wasm execution regression subsequently passed (1.42 seconds
after compilation): interleaved readout, reversed input order, both valid input
indices and two unknown indices. The compile-time reflected-label comparison
regression passed in 1.16 seconds and its compiler change is on shared mb2 as
`dceb217ea`. The shared named-binding provider and migrated upstream examples
landed as `21abc4028`; other Quilting examples were migrated in `37ee60a`.

The rebuilt viewer then passed the previously failing browser checks:
all four quad arc choices updated `arc_edge`, `patch_valid` stayed 1, and
disabling wires set `wires=0` while preserving `bottom_lod=4`. Triangle mode,
its boundary selector and camera yaw also updated their intended fields.
An unequal triangle request `bottom=2, right=8, left=3` rendered 14,250 vertices,
reported 256 hypotenuse segments, and had `pending=0`, `status=0`. No console
warnings/errors appeared during these checks. At the tested 1268×1281 viewport,
document dimensions matched the viewport: there was no outer-page overflow.

Served receipt: `fe-render-df599506ed4c6d65.json`, Wasm
`fe-render-abe84f389b06938d.wasm`; the corrected build took 117.495 seconds
while other release tests were compiling. Counts remained 825,838 Wasm bytes
and 201,298 total WGSL bytes. These are build/artifact observations, not a
runtime performance comparison.

Upstream positive and missing-label-negative web binding regressions also
passed (2 tests, 2.64 seconds after compilation). The full reflection suite
finished **45 passed, 1 failed**: its forwarded-ground-parameter rejection
fixture now receives no diagnostics, contrary to its expectation. That fixture
does not exercise string comparisons; an unchanged-baseline run has not been
made, so this record does not claim the suite was previously green or that the
failure is resolved. Source checks passed for arc curves,
tessellation warp and paired triangle fans; this does not substitute for their
browser acceptance.

Build/test logs are under `/laboratory/quilting/scratch/`:
`composite-surface-oracle-20260907.log`,
`composite-patch-canonical-web-20260907.log`,
`composite-patch-named-bindings-web-20260907.log`,
`named-param-bindings-wasm-20260907.log`,
`named-param-binding-web-regressions-20260907.log`.

## Boundary samples stay on their own edge

September 8, 2026. Density placement composed two independent slice laws, one
per axis. That composition preserves any axis-parallel line and bends every
other one. The square's four sides are axis parallel and survived it. Its
diagonal is not, and the diagonal is the triangle patch's third edge, so with
automatic density on that edge visibly stopped being a circular arc while the
other two stayed correct.

Boundary samples now use the same segment law restricted to their own edge,
in the edge's canonical low-to-high direction so both adjacent uses feed
identical inputs. On the axis-parallel sides this is arithmetically the same
placement as before, so the published boundary tables are unchanged. On the
diagonal it is the correction. Interior samples still use both slice laws.

`diagonal_edge_deviation` in `composition_oracle` sweeps an edge and reports the
worst residual of the line through its two corners, and the Rust oracle asserts
all three edges of the triangle child stay under 1e-6 across three bulges.

## Free weights as an option

The authoring family derives every weight as the inverse of a corner's offset
from one pole. That shared form is exactly what makes all four boundary curves
circles through a common point, and it is the paper's point-space condition.
It is also a restriction: the wider Clifford-Bezier surfaces are not reachable
from it.

`free_weights` admits them. It adds an independent bivector to each corner
weight, in three distinct planes so no corner is left on the family by
accident, scaled by `weight_freedom`. Real per-corner scaling would not do:
positive real scales only reparameterize the patch, which the oracle already
records. The departure has to be multivector valued.

Two consequences are handled rather than hidden.

- Off the family the quotient carries a trivector part, so it is not purely a
  point. Rejecting it would draw nothing at all. The net now carries the
  largest residual its evaluator will still read as a point; the family keeps
  the strict tolerance and free weights widen it, so the surface is read as the
  vector part of a quotient that is no longer purely a point.
- Measured arc-length spacing is a claim about circular edges. The net carries
  `circular_edges`, and both the published boundary tables and the interior
  density placement read it, so they withdraw together the moment the patch
  leaves the family. Nothing reports an arc length for a curve that is not an
  arc.

`weight_freedom` at zero reproduces the constrained patch exactly, so the
toggle alone is never a silent geometry change. `free_weight_departure` in the
oracle asserts that, and asserts that any nonzero freedom actually leaves.

This is an option, not a new default. `pole_margin` is still re-derived after
the departure, so an unbounded free-weight patch still reports.
