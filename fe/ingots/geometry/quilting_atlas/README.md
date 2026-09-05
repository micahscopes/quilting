# Triangle and quadrilateral atlas semantics

This ingot owns deterministic sampling and topology rules, independently of
where they execute. Storage, dispatch, and residency belong to providers.

The existing triangle atlas uses all 165 sorted edge-LoD triples through LoD 8.
Its equilateral metric is intentional: barycentric coordinate lanes are not
orthogonal Cartesian axes.

## Quad atlas in progress

`quad` defines four edge LoDs in counterclockwise order: bottom, right, top,
left. It canonicalizes under the square's eight rotations/reflections (D4),
not all 24 permutations. Adjacent dense edges are not interchangeable with
opposite dense edges. The 6,561 requests through LoD 8 form 1,035 classes.

Canonicalization returns a symmetry taking the requested square to the stored
square. A renderer uses its inverse on stored coordinates. Reflections also
reverse triangle winding. The stable base-9 key code is not a dense ordinal.

Boundary rings are counterclockwise, have no duplicate corners, and contain
exact Q14 dyadic subdivisions. Each edge owns its starting corner. These are
reference-domain samples, not yet uniform arc-length samples on a curved patch.

`planar` supplies shared exact orientation/incircle arithmetic, specialized by
`EquilateralMetric` or `SquareMetric`. Its two-word accumulator needs no native
64-bit GPU integers. Inputs are bounded Q14 points; custom integer metrics must
respect the documented lift bound. Triangle topology retains its original
entry points and metric.

Release Fe-to-Wasm gates in `quilting-fe-fixtures`, feature `fe-oracle`:

- `quad_atlas_wasm_canonicalizes_d4_and_preserves_exact_boundary_rings` checks
  every requested key, all 1,035 canonical rings (235,060 boundary samples),
  inverse symmetries, and invalid LoDs.
- `planar_metric_incircle_wasm_matches_independent_i128` compares both metrics
  with an independent wide-integer determinant, including degeneracy and a
  square whose cocircularity differs between the two metrics.

`quad_sampling` now defines a four-edge density policy, deterministic jittered
candidate slots, square-distance exclusion, and a reference boundary query.
Open edges inherit their own exact LoD exponent; corners use the larger
incident exponent. Interior exponents blend polynomial edge weights, with
bounded integer rounding, and commute exactly with D4. This policy is not an
optimality claim or an arc-length map on a curved surface.

Candidate capacities are derived from the largest edge LoD, with two trials
per cell on a grid twice that edge's resolution. The largest job has 524,288
candidate slots. This is conservative candidate work, not the final point
count or a claim that generating all 1,035 jobs simultaneously is affordable.
Providers must batch bounded jobs and reuse the common independent-set state
transitions. The reference boundary scan must be replaced by an equivalent
indexed query for large GPU jobs.

Additional release Wasm gates:

- `quad_sampling_wasm_preserves_edge_density_and_square_symmetry` checks all
  6,561 keys, 131,220 exact open-edge densities, corner and center rules, and D4
  covariance of the density field.
- `quad_sampling_wasm_candidates_match_counter_reference_and_boundary_exclusion`
  checks 526,344 candidate slots including every slot of an LoD-8 job against
  an independent counter-hash implementation, and selected boundary queries
  against wide-integer square distances.

Quad point-set selection, topology provider, generated atlas, and renderer
integration are not finished. The warp explorer's square grid is only a visualization
fixture, not a generated blue-noise/Delaunay quad atlas. Later surface warping
must share one canonical edge map and sample sequence between adjacent patches;
independent approximate arc-length inversions do not establish crack freedom.
