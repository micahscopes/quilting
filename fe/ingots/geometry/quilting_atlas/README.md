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

The quad sampler, topology provider, generated atlas, and renderer integration
are not finished. The warp explorer's square grid is only a visualization
fixture, not a generated blue-noise/Delaunay quad atlas. Later surface warping
must share one canonical edge map and sample sequence between adjacent patches;
independent approximate arc-length inversions do not establish crack freedom.
