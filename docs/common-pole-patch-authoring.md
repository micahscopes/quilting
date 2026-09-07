# A verified patch family for geometric arc controls

This is a construction shared by triangular and quadrangular patches, not a
replacement for general compatible circular-boundary authoring. There is no new
viewer yet; the present result is the Fe constructor and executed geometry oracle.

For corners `p_i`, a geometric pole `q` and a shared reference `r`, construct

    w_i = inverse(p_i - q) * (r - q).

The existing triangular or bilinear interpolation then evaluates the usual
weighted-point numerator divided on the right by the weight denominator. The
common right factor `(r-q)` cancels. Moving `r` changes the representation, not
the surface. Moving `q` changes the geometric boundary arcs and the surface.
This pole is not the tessellation density's focal point.

In Euclidean Clifford algebra, set `S = sum B_i * inverse(p_i-q)`. Since the
basis sums to one, the result is exactly `q + inverse(S)` wherever defined.
Thus the point-space constraint follows from the construction; no scalar-weight
repair or projection of an invalid trivector onto a surface is needed.

`controls_from_pole<A,N>` is one Fe implementation for a fixed-size control net.
The opt-in `CommonPoleAlgebra` law is stronger than merely supporting arcs. Only
the Euclidean implementation opts in currently. Existing sparse GA/FCO products
do the model arithmetic. The helper rejects null/nonfinite endpoint or reference
displacements; interior poles still need quotient admission and chart handling.

Boundary consequences:

- Restricting a triangle or quad to an edge blends exactly its two controls.
  Sharing those controls therefore shares the complete parameterized curve.
- This does not by itself establish tangent-plane agreement across different
  surfaces, bit-identical independently evaluated floating-point paths, or a
  good interior triangulation. Canonical shared edge evaluation remains required.
- A bilinear quad is not generally two degree-one triangular patches. Exact
  subdivision must continue using the existing degree-aware restriction code.
- Arbitrary independent circles do not all belong to this common-pole family.

The triangle formula is grounded in Krasauskas and Zube, *Rational Bezier Formulas
with Quaternion and Clifford Algebra Weights*, section 2.5, formula 13, local
`/home/micah/sync/PDFs/math/clifford patches.pdf`. The quad vector-space statement
above follows from the partition-of-unity identity; it does not assert that an
arbitrary quad lies on one sphere.

## Executed evidence

Release Wasm test `common_pole_tri_and_quad_match_independent_inversion_and_shared_edges`
compares the production Fe numerator/denominator evaluator with an independent
f64 Euclidean inversion calculation. It tests both bases, four pole locations,
two reference choices, all 153 triangle and 289 quad points on a 16-step grid,
trivector residuals and 65 shared-edge samples per pole. The close-corner pole
is only 0.001 units from its corner. Rejected corner/null/nonfinite cases are
separate checks; these fixtures do not establish global absence of poles.

Observed maximum normalized error: `1.626367520657368e-6`; acceptance threshold
`1e-5`. Normalization is `abs(actual-expected)/(1+abs(expected))`, not screen error.
The oracle reuses no Fe-generated weights or Clifford multiplication code.
Log: `/laboratory/quilting/scratch/pole-patch-owner-copy-oracle-20260907.log`.

The generic array-of-controls test exposed an upstream Wasm lowering defect:
associated point/weight fields were admitted without their owning generic
instantiation. Materialization and copying must use the same owner-aware rules.
No separate triangle/quad constructor or JavaScript substitute was introduced.

## Next integration

The new `controls_from_edge<A,N>` bridge derives the pole from one authored
endpoint/middle-point/endpoint arc. For endpoint weights `wa`, `wb`, it uses

    q = (pb*wb - pa*wa) / (wb - wa).

This is the arc's parameter-infinity limit, not a finite-parameter denominator
singularity. It is computed directly with the model's checked quotient, not by
evaluating at an arbitrarily large parameter. The resulting common-pole net
preserves the entire edited parameterized boundary, not merely its midpoint.
Other boundary arcs change together with that edit. An affine edge has an
infinite limit and is explicitly rejected by this finite-pole constructor;
supporting that family remains unfinished, not a reason to invent a finite pole.

Release Wasm test `edge_authored_patch_preserves_midpoint_corners_and_reversal`
passes for every triangle/quad boundary, two spatial midpoint offsets, and both
edge orientations. It compares 17 points along each authored arc against the
independent dense Clifford oracle, checks all corners, and rejects invalid edge
identities and the affine limit. The surrounding `patch_` selection passed all
12 tests in 22.93 seconds. Receipt:
`/laboratory/quilting/scratch/edge-patch-planned-20260907.log`.
These are geometry/compilation checks, not yet browser-widget acceptance.

Render this bridge with the shared atlas, boundary spacing and camera widgets. Make any
coupling between edge edits explicit. Preserve the broader compatible-boundary
construction and higher-degree controls on the delivery checklist; this one
family is not completion of the patch-authoring goal.
