# Shape distribution and sampled correspondence error

The shared mesh audit now optionally records more than minimum and mean shape.
The ordinary summary path retains its three vertex evaluations per triangle;
the detailed path adds one analytical evaluation at the triangle's UV centroid.
It decodes and evaluates the same packed atlas mesh, not a second triangulation.

## Shape

`quilting_patch::shape_histogram::ShapeHistogram` is a fixed twenty-bin Fe value.
Bins have width 0.05, are half open, and the final bin includes quality 1.
Invalid values are counted separately. Rank queries return the lower edge of a
bin, **not an exact percentile**. The oracle reports bins containing the fifth
and tenth percentiles using one-based ceiling ranks.

The histogram accepts finite rendered-triangle shape, not the local metric
prediction used to rank coarse layouts. Invalid surface vertices and undefined
triangle shapes are recorded as rejected observations, never silently omitted.

## Error

At each triangle's UV centroid, compare the exact analytical surface value with
the average of its three rendered vertices. Report maximum and RMS distance in
world units, invalid samples, and the UV location of the largest observation.

This is a **sampled correspondence error**. It is not nearest-point distance,
Hausdorff error, a normal-only displacement, or an error bound. Even a planar
surface can have nonzero correspondence error when parameterization is
nonlinear. One centroid can miss an edge or interior peak. These quantities
must not be presented as an exhaustive accuracy or coverage proof.

RMS weights triangles equally, not by area. Comparisons must keep the same
authored surface and disclose different triangle budgets. Nonfinite distances
or unrepresentable squared-error accumulations are counted as invalid.

## Verification

`curved_mesh_distribution_and_approximation_probe` checks histogram edge/rank
semantics and compares detailed counts against the existing summary for three
surfaces, four layouts, and two levels. It checks admission accounting, finite
error, RMS no greater than the sampled maximum, worst-location domain, and four
surface evaluations per emitted triangle. This is measurement-integrity
coverage, not a quality-improvement gate.

Raw log: `/laboratory/quilting/scratch/mesh-distribution-approximation-20260908.log`.
The aggregate diagnostics are not yet wired into the browser panel or worst-cell
overlay. The sampled-error color mode is implemented and browser-checked. Its
`error_scale` is an explicit world-unit ceiling for the
color ramp, kept fixed across comparisons; colors do not normalize themselves to
the current mesh. Zero error is cool; error at/above the scale is warm; invalid
measurements are magenta. Flat affine composition has zero correspondence error.

The optional error mode adds one surface evaluation per vertex invocation (the
three invocations of a face currently repeat its diagnostic). Ordinary shading
does not take that branch. This is a visualization, not a replacement for the
edit-time aggregate audit. No frame-performance improvement is claimed.

## September 8 result

The release Wasm test passed all 24 cases in 106.87 seconds, including Fe
compilation and audit work; this is not a browser planning-time benchmark.
All cases had zero invalid correspondence samples. The separate histogram test
also verifies that zero/out-of-range ranks and a rejected NaN sample trap.
This required the target-neutral assertion fix on shared mb2, `92a1812eb`.
After extracting the shared centroid-error helper used by both audit and viewer,
the 24-case test passed again in 117.66 seconds, including exact zero, scale and
winding checks. Log: `/laboratory/quilting/scratch/mesh-error-shared-helper-20260908.log`.

At 272 triangles on the strongly asymmetric fixture (bulge 2.4, dx -0.5,
dz 0.7), diagonal 0–2 and the midpoint square share the same fifth-percentile
bin [0.25, 0.30). Their sampled maximum errors are 0.22635 and 0.17244,
respectively; RMS errors are 0.04469 and 0.04003. The earlier minimum-shape
measurement favored the diagonal (0.24579 versus 0.11488). Thus the midpoint
tradeoff is more interesting than the worst triangle alone suggested: lower
sampled error, but a worse extreme triangle. Neither dominates all criteria.

On the second asymmetric fixture at 272 triangles, midpoint improves the
fifth-percentile bin to [0.45, 0.50), versus [0.35, 0.40) and [0.40, 0.45)
for the diagonals. At level 4 the methods have different counts and this
advantage does not simply persist. Retain explicit quality/cost comparisons;
do not turn these small-fixture results into a universal automatic selector.

## Served color mode

Chrome MCP verified the live canvas on port 38331, manifest
`assets/fe-render-ba1962679921ff2d.json`: 1,214,926 Wasm bytes and 630,875 WGSL
bytes across four shaders. On the same default curved quad, changing only the
error scale from 0.1 to 0.001 changes the predominantly cool mesh to a warm one
with visible per-triangle variation; geometry and resolved spoke counts remain
unchanged, pending is zero. Live-canvas captures were visually inspected:
`/laboratory/quilting/scratch/error-heatmap-scale01-20260908.png` and
`/laboratory/quilting/scratch/error-heatmap-scale0001-20260908.png`.
This checks rendered response, not pixel-perfect agreement with every oracle
sample. Full pathological capture/replay and worst-cell overlays remain pending.
