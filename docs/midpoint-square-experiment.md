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
