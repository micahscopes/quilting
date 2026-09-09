# Eight-triangle square: bounded experiment

The initial recipe splits a square into four corner triangles and four triangles
joining the midpoint diamond to the center. It is an experiment, not yet a
replacement for either diagonal or the four-triangle fan.

## Implemented contract

`quilting_atlas::midpoint_square::MidpointSquare` describes the topology using
exact integer coordinates on `[0,2]^2`. It admits uniform outer levels 1–8.
Level 0 is rejected: its boundary has no midpoint sample to reuse.

At outer level L, each child requests opposite-vertex levels `(L,L-1,L-1)`.
The diamond edge receives L; the two shorter edges receive L-1. All eight
children share one canonical atlas primary, with the existing permutation
mapping. This is a geometric dyadic count policy, not a measured curved-edge
or screen-space policy.

Outer half-edges reference the integer intervals of their original master edge.
Their endpoints are not independently approximated. Nonuniform spacing is
deliberately not admitted by this initial recipe: geometric midpoints need not
coincide with a nonuniform master's half-index samples.

## Evidence

The release Fe/Wasm oracle test
`midpoint_square_has_exact_coverage_and_protected_master_intervals` checks:

- all eight authored triangles have positive, equal integer area;
- sixteen distinct edges, eight outer edges, internal edges used twice with
  opposite orientation;
- exact master sample intervals at every admitted level and explicit rejection
  of unsupported levels and out-of-range indices;
- canonical levels and their return to the intended opposite-vertex order.

These checks establish the small coarse recipe, not the validity or quality of
the eventual warped fine mesh. In particular they do not establish curved
surface coverage, approximation accuracy, or a quality improvement.

## Next gate before adding a viewer mode

Generate actual packed primaries and evaluate them on asymmetric curved quads
through the existing surface evaluator. Compare both diagonals, the four-fan,
and this recipe with explicit triangle counts and quality-cost curves near the
256 and 1,024 triangle budgets. Keep boundary spacing identical. Report invalid
samples, folds, low-tail shape, area allocation and approximation error rather
than just average shape. Only a useful numerical result warrants extending the
viewer's four-child layout storage.

## First curved-mesh probe

`uniform_quad_layout_quality_cost_probe` uses the existing Fe surface evaluator,
packed atlas generator/decoder and finite-triangle measurement. All placements
in this ablation have uniform UV boundary spacing, no interior warp, seed 42.
This is not yet uniform physical arc spacing. The surface is a circular-edge
quad with either default corners, or an explicit displacement of active corner
p2. It includes every emitted triangle, including corner-near ones.

At outer level 3 the two diagonal layouts and eight-triangle layout each happen
to produce 272 triangles. Their minimum shape scores (higher is better) were:

| Surface: bulge, p2 x/z displacement | Diagonal 02 | Diagonal 13 | Eight triangles |
|---|---:|---:|---:|
| 0.6, 0 / 0 | 0.332143 | 0.332143 | 0.380628 |
| 2.4, -0.5 / 0.7 | 0.245792 | 0.104948 | 0.114884 |
| 1.2, 0.7 / -0.5 | 0.259354 | 0.275248 | 0.252678 |

The four-fan has 376 triangles at that level, so it is not an equal-cost winner
comparison. At level 4, counts differ: diagonals 1,212, fan 1,432, midpoint
1,088. Record these as quality-cost samples, not matched budgets.

Across the initial 24 observations, evaluated vertices were valid and no UV
triangles folded. This does not rule out curved-surface intersections or prove
approximation accuracy. Area variation remains substantial. In particular,
the strongly bent case at level 4 has two triangles below shape 0.1 in both the
diagonal-13 and midpoint layouts. The midpoint recipe is not a universal fix.

Decision: retain the small recipe and test surface-aware selection; do not
replace the viewer's existing layouts or claim successful uniformization. Add
low-tail and approximation measurements before accepting a selection policy.
The local logs in `/laboratory/quilting/scratch/midpoint-square-quality-20260908.log`
and its `-final-` counterpart retain the raw observations and verification.
Elapsed call times include primary generation and diagnostics (including
repeated primary builds per child), not a cached browser rendering benchmark.
