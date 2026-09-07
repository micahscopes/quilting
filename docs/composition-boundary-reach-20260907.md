# Boundary reach: one controlled comparison

The composition now exposes `boundary_reach`, a logarithmic control from1/16
to16 with the existing behavior at1. It changes how far boundary adjustments
reach into each child's interior, not edge maps, counts or geometric endpoints.
The default is deliberately unchanged; no universal optimum was established.

## Fe abstraction

`TriangleEdgeMaps::map_with_locality<L: IntervalSamplingMap>` uses a lawful
interval map to attenuate each edge displacement along slices parallel to that
edge. The existing method delegates to the identity locality law. This viewer
reuses `ProjectiveIntervalBias` as the locality law; no additional shader stage,
custom JavaScript transform or generated atlas is involved.

For span s along a slice and local edge coordinate t, the changed coordinate is
`s ((1 - L(s)) t + L(s) edge_map(t))`. The opposite barycentric coordinate stays
fixed. L(1)=1 preserves the complete edge map, while 0≤L(s)≤1 retains a convex
blend of identity and a monotone edge map along each slice. The sequential
composition has positive orientation in exact real arithmetic for strict laws;
the finite straight mesh can nevertheless fold under coarse sampling.

This comparison changes reach only. It does not test symmetric stage ordering,
relaxation, a new triangulation, or a different density objective.

## Bounded sweep

Six existing real-mesh cases × five reaches (1/16,1/4,1,4,16), identical seed42
primaries/counts and canonical edge maps. All results are retained in
`/laboratory/quilting/scratch/composition-locality-sweep-20260907.log`.

| Case | Reach | Mean metric shape ↑ | Density-mass CV ↓ | Flips | Shape below.1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uneven centered fan | 1 | .5484 | 1.2769 | 0 | 2 |
| Same | .25 | .5559 | 1.1349 | 0 | 2 |
| Same | .0625 | .5597 | 1.0582 | 0 | 2 |
| Uneven off-center fan | 1 | .5402 | 2.1338 | 0 | 9 |
| Same | .25 | .5468 | 1.9055 | 0 | 8 |
| Same | .0625 | .5517 | 1.7639 | 0 | 8 |
| Strong spoke skew | 1 | .6648 | .5463 | 0 | 0 |
| Same | .25 | .6853 | .4415 | 0 | 0 |
| Same | .0625 | .7103 | .3731 | 0 | 16 |

The strong optional concentration/twist case still has12 flips at every reach.
Broader reaches4/16 worsen the uneven cases. Narrowing reach improves averages
but can worsen the tail: the strong-skew minimum shape drops from.1631 at1 to
.01694 at1/16. None of this proves complete uniformization or safety for arbitrary
inputs. Even the narrowed uneven cases do not surpass the uncompensated baseline
in every metric; that baseline itself does not meet the required edge spacing.

## Gates and browser

- Three release Wasm tests pass in44.68s, including the30-case sweep, prior
  quality corpus, all6,561 density combinations and retention/cap/completion tests.
- Added exact boundary equality checks for257 samples on each triangle edge,
  five locality laws, and nontrivial independent edge biases.
- Reach-only edits change the visible policy without changing spacing revision;
  pending geometry retains the old reach along with the old scene.
- Chrome65, release `fe web dev`, manifest `fe-render-65465921425eeeeb.json`:
  the native log slider changes the Fe value1→.25 exactly. Uneven1/6/6/1 fan
  remains24,396 vertices, pending0, spacing revision2;424,174 changed pixels in
  the1018×1018 interior crop, zero red fold/exposed-background pixels. This is
  representative visual inspection, not exhaustive GPU geometry acceptance.
- Parent Wasm601,229 bytes; WGSL145,558 bytes /3passes. Dev rebuild42.100s,
  not rendering or generation timing. Final browser view is live at reach.25;
  page default remains1.

Final test log: `composition-locality-final-tests-20260907.log` in the same
scratch directory. Keep this as an explicit comparison control and reusable
extension law while carrying placement/edge contracts into controllable patches.
Do not replace the larger patch and gallery deliverables with more reach tuning.
