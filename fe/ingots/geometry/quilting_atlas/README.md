# Triangle and quadrilateral atlas semantics

This ingot owns deterministic sampling and topology rules, independently of
where they execute. Storage, dispatch, and residency belong to providers.

## Wasm-worker direction

`active_sampling` is the CPU active-list sampler, sharing prescribed boundaries,
Q14 geometry and density laws across triangle and square domains. It generates
proposals around accepted samples instead of allocating a finest-grid candidate
population. Caller-owned scratch is reusable and capacity/budget failures are
explicit. Receipts count executed proposals and neighbor comparisons.

The initial exhaustive accepted-point query is a correctness reference for the
worker spatial index, not the intended full-atlas performance implementation.
Interior points obey symmetric larger-radius exclusion; prescribed boundary
points are exempt from mutual exclusion, not from excluding interior points.
Finite-attempt active-list exhaustion does not prove maximal coverage or a
continuous blue-noise spectrum. Sampling alone is not a completed triangulation
or atlas, and scalar test timings are not Wasm-worker benchmarks.

Known coverage counterexample: with seed 42, triangle `[0,0,4]` exhausts its
boundary-seeded active list with no interior samples even though the centroid
is admissible. The reference test explicitly demonstrates that gap.
`sample_with_exploration` adds configurable whole-domain darts after active
growth stalls, resuming growth when one is admitted. Both proposal sources use
the same exclusion rule. A regression verifies that this reaches the missed
interior. Finite rejected-dart budgets are statistical exploration, not
maximality certificates; domain proposals have their own work counter.
Large scratch arrays also exceed the EVM test target's stack: the
LoD-8 boundary-only contract test is not full LoD-8 Wasm sampling evidence.

The `active_sampling_oracle` validation ingot now executes the actual sampler
in Wasmtime at O0 and O2, without imports or a Rust sampling implementation.
Seed-42 mixed cases complete within a 4,096-point probe allocation:
triangle `[0,0,8]` produces 1,615 points (258 boundary), and square `[0,0,0,8]`
produces 619 points (259 boundary). Both optimization levels give the same
receipts, with zero exact-boundary or pairwise-exclusion violations. Triangle
runs also check deterministic replay. Exhaustive neighbor scans perform
10,788,031 and 3,101,542 comparisons respectively: this is a correctness
reference awaiting a sound variable-radius spatial index, not the fast atlas.
Fuel-instrumented sampling-plus-validation times are not generation benchmarks.

Run from `fe/tools/quilting-fe-fixtures`:
`cargo test --release --features fe-oracle active_sampling_wasm_ -- --nocapture --test-threads=1`.
This gate does not yet prove triangulation, browser worker behavior, all-key
coverage, maximality, or blue-noise spectral quality.

`sample_indexed` runs the same proposal/admission process with a `NeighborIndex`.
`Exhaustive` is the reference; `AcceptedGrid` retains linked point chains and
maximum radius bounds per cell, with a global-radius query window. Domains
must supply conservative coordinate-radius bounds; the exact domain distance
test still decides every admission. Grid dimensions affect performance only.

The Wasm `active_index_wasm_matches_exhaustive_streams` gate compares every point
coordinate and radius, proposal counts and completion on seven triangle/quad
cases, including mixed LoD-8 keys and multiple seeds. All streams match.
For triangle `[0,0,8]`, seed42, comparisons fall from10,788,031 to598,217 with
32,904 cell visits. The corresponding quad goes from3,101,542 to180,094
comparisons with1,227,836 cell visits.

One release O2 Wasmtime run measured five-call medians after warm-up, with fuel
disabled, including scratch initialization but excluding validation/IO:

| Case | Exhaustive | Indexed |
| --- | ---: | ---: |
| Triangle `[0,0,8]` | 141.59ms | 16.22ms |
| Square `[0,0,0,8]` | 31.02ms | 9.88ms |
| Triangle `[4,4,4]` | 2.85ms | 1.25ms |
| Square `[4,4,4,4]` | 4.30ms | 1.38ms |

These are local sampling comparisons, not Rust baselines, browser-worker timings,
triangulation costs or full-atlas performance claims. Small-radius windows
matter: before adding them, uniform square sampling regressed despite fewer
point comparisons. Keep both cell work and exact comparisons observable.

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

`topology` now shares point location, insertion, normalization, and flipped
triangle construction through `PlanarCoordinates`. Coordinates provide
incidence only; `interior_edge_flip_decision_with_metric` takes the distance
metric explicitly. `ConstrainedBoundary` distinguishes the triangle sampler's
AB/AC/BC storage order from the quad's direct circular order. Neither requires
pretending that a square point is a valid barycentric triple.

Release Fe-to-Wasm gates in `quilting-fe-fixtures`, feature `fe-oracle`:

- `quad_atlas_wasm_canonicalizes_d4_and_preserves_exact_boundary_rings` checks
  every requested key, all 1,035 canonical rings (235,060 boundary samples),
  inverse symmetries, and invalid LoDs.
- `planar_metric_incircle_wasm_matches_independent_i128` compares both metrics
  with an independent wide-integer determinant, including degeneracy and a
  square whose cocircularity differs between the two metrics.
- `planar_topology_wasm_preserves_incidence_orientation_and_area` checks
  location, normalization, edge/interior insertion, rejected off-segment
  points, and exact signed area.
- `planar_topology_wasm_flips_match_exact_metric_and_preserve_area` checks
  convexity, both metric decisions, deterministic cocircular tie-breaking,
  and the resulting triangle pair against independent i128 calculations.
- `atlas_topology_wasm_locks_triangle_and_quad_boundary_chains` checks every
  canonical triangle's boundary chain and probes every quad edge-LoD request.

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
transitions. The provider now replaces the reference boundary scan with an
exact nearest-sample query: at most six point tests for a triangle, eight for
a square. Each open edge has constant exclusion radius, so its nearest sample
suffices; corners are tested separately because their radii may differ.
Triangle projections use the equilateral metric. The unchanged full-ring
relation remains available as an oracle, not as the production query.

`neighborhood` maps conservative spatial windows back to stable candidate
slots without an auxiliary buffer. Uniform-density queries inspect at most
50 slots through LoD 8 in either domain. Mixed-density windows remain bounded
by the largest exclusion radius and may still be large. Deformed-surface
policies retain exhaustive queries unless they supply their own valid bound.

The release Fe-to-Wasm gate
`atlas_nearest_boundary_query_matches_exhaustive_relation_for_every_key`
matches 703,755 bounded/exhaustive queries across every canonical triangle key
and every square edge request through LoD 8, including corners, half-step ties,
interior points, and independently varied candidate radii. This proves those
comparisons, not full-atlas generation speed. GPU provider source checking also
passes; post-change browser validation is a separate outstanding gate.

Additional release Wasm gates:

- `quad_sampling_wasm_preserves_edge_density_and_square_symmetry` checks all
  6,561 keys, 131,220 exact open-edge densities, corner and center rules, and D4
  covariance of the density field.
- `quad_sampling_wasm_candidates_match_counter_reference_and_boundary_exclusion`
  checks 526,344 candidate slots including every slot of an LoD-8 job against
  an independent counter-hash implementation, and selected boundary queries
  against wide-integer square distances.

The shared GPU provider now realizes bounded quad point selection, convex
insertion, and square-metric Delaunay repair. The quad generator integration
gate records four independently checked Chromium jobs through LoD 3 in
`fe/web/quad-atlas`. A complete generated 0–8 atlas and scalable job scheduling
are **not finished**. Later surface warping
must share one canonical edge map and sample sequence between adjacent patches;
independent approximate arc-length inversions do not establish crack freedom.
