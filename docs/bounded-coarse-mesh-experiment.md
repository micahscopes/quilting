# Surface-aware coarse mesh filled by atlas triangles

This is an experiment in progress, not a replacement renderer or a claim of
Delaunay triangulation. It follows the failure of fixed connectivity plus
interior warping on the captured asymmetric quad.

## First gate: connectivity independent of placement

`quilting_atlas::coarse_mesh::CoarseMesh` holds at most sixteen parameter-domain
triangles. It starts from either square diagonal or one triangle. Strictly
interior insertion replaces one face by three; existing vertex IDs and all
outer edges remain unchanged. Proposals on an edge, outside a face, outside the
Q14 domain or beyond capacity are rejected without mutation. Counts and storage
are private so callers cannot construct an inconsistent active range.

`flipped(first, second)` replaces a shared interior diagonal only when the two
faces form a strictly convex quadrilateral. It preserves vertices and face
counts, rejects boundary/nonadjacent/invalid pairs and degenerate replacements,
and uses the same exact orientation predicate as insertion. It makes no metric
or quality decision; a caller must decide whether to accept the valid proposal.

The composition oracle exposes snapshots after bounded centroid insertions.
The independent Rust/Wasm check measures positive integer orientation, total
area, opposite interior-edge incidence, unchanged outer edges, retained vertex
coordinates, capacity and invalid-index behavior. These checks concern coarse
parameter connectivity. They do not prove fine-mesh coverage after spacing
correction, surface approximation, or absence of surface self-intersection.
The same test independently classifies every ordered face pair in the snapshots
for convexity and checks accepted flips' directed boundary, area and orientation.

Run from `fe/tools/quilting-fe-fixtures`:

```
cargo test --release --lib --features fe-oracle \
  coarse_connectivity_preserves_coverage_boundaries_and_vertex_identity -- --nocapture
```

## Next gates

1. Use surface measurements to propose relocation, local flips and insertion;
   exact parameter predicates remain the connectivity admission check. Reuse
   the existing pullback/triangle metrics, rather than assuming UV distances
   describe surface distances. Keep work explicitly bounded.
2. Resolve shared edge counts and spacing once per owner. Fill coarse faces
   using the existing atlas, including its winding and permutation handling.
3. Reject actual finite-mesh folds and missing coverage before comparing shape,
   area allocation and approximation error against the best fixed candidate at
   matched budgets. Include the captured triangle and asymmetric quad.
4. Only promote a useful, valid candidate to the viewer. Report unmet demand
   when sixteen coarse faces do not suffice; do not hide it with more LoD.

Centroid insertion is currently a connectivity test stimulus, **not** the
surface-aware sampling algorithm. No boundary is subdivided in this first
slice. Later boundary subdivision must reference existing shared master samples.

## Actual atlas assembly gate (September 9)

`coarse_atlas_assembly_reuses_surface_audit_and_preserves_affine_baseline`
passes in release Fe/Wasm. Arbitrary coarse plans now enter the existing packed
atlas decoder, exact surface evaluator, finite-triangle statistics and sampled
centroid-error path. No second geometry implementation or renderer is involved.

The baseline uses uniform parameter-space edge sampling, **not uniform surface
arc length**. Its legacy comparator disables automatic interior resolution and
freezes the diagonal to the same exponent: otherwise equal outer counts selected
68 versus 48 triangles in the initial LoD2 quad comparison. With this corrected,
the unchanged coarse plan reproduces the legacy count/quality/error statistics.

All sixteen measured assemblies (two captured geometries, LoDs 2/3, insertion
counts 0/1/3/7) have zero invalid surface samples and zero nonpositive UV
triangles. This is not a global intersection or coverage-multiplicity proof.
Selected LoD3 observations:

| Geometry | Insertions | Final triangles | Mean shape | Shape < 0.1 | Sampled max centroid error |
|---|---:|---:|---:|---:|---:|
| captured quad | 0 | 188 | 0.30548 | 26 | 2.5191 |
| captured quad | 7 | 1504 | 0.27261 | 484 | 2.9250 |
| captured triangle | 0 | 94 | 0.27704 | 9 | 11.3402 |
| captured triangle | 7 | 1410 | 0.23731 | 587 | 5.4199 |

Errors are in each fixture's world units and are not comparable across rows
belonging to different geometries. These are varying-budget observations, not
an efficiency win. The assembly gate passed; blind centroid insertion is not
accepted as a quality policy. Surface-guided location/connectivity/count choices
and production arc-length boundary maps remain to be integrated and tested.

The same run closed the radial-extension experiment: its exact atlas-boundary
assertions passed on both diagonals and the four-fan quad at LoDs 3/4, but every
case retained nonpositive UV triangles. The LoD4 fan has 27 such triangles out
of 1432, with 306 shapes below 0.1. Keep this as a rejected placement experiment,
not a viewer default or a substitute for the coarse-mesh work.

Log: `/laboratory/quilting/scratch/coarse-assembly-20260909-v2.log`.
The full test took 213.13 seconds including compilation; this is not a runtime
generation benchmark. The sampled centroid metric is not a global error bound.

## Surface-guided flip predictor (September 9)

`coarse_surface_guided_flips_compare_prediction_with_actual_atlas_mesh` passed
eight release Fe/Wasm cases: both captured geometries, three/seven insertions,
and finite-difference steps 0.001/0.0005. Each candidate scores the center and
three corner-near sites in each affected coarse face. Differences follow the
face's own two directions. A single bounded sweep accepts only strict geometric
flips whose minimum sampled tangent-triangle shape improves. This is not an
intrinsic Delaunay criterion or a production adaptation policy.

The actual atlas mesh keeps the same triangle budget before/after each sweep.
All eight runs have zero invalid samples, nonpositive UV triangles and invalid
centroid-error evaluations. Both difference steps choose the same flips here;
that is limited sensitivity evidence, not a general precision guarantee.

| Captured geometry | Insertions | Triangles | Accepted flips | Mean shape before → after | Minimum shape before → after |
|---|---:|---:|---:|---:|---:|
| quad | 3 | 752 | 2 | 0.25832 → 0.22529 | 0.001807 → 0.001807 |
| quad | 7 | 1504 | 5 | 0.27261 → 0.28565 | 0.001807 → 0.001807 |
| triangle | 3 | 658 | 0 | 0.22481 → 0.22481 | 0.019674 → 0.019674 |
| triangle | 7 | 1410 | 2 | 0.23731 → 0.24493 | 0.007297 → 0.010858 |

The sampled maximum centroid error remains 2.92497 on the quad and 5.41988 on
the triangle in every row. The predictor uses 352–1024 surface evaluations,
including the initial/final score, excluding authoring and fine-mesh auditing.
The run took 218.72 seconds including compilation, not planning time.

**Disposition:** reject this score alone as the automatic policy. It can accept
local changes while worsening aggregate fine-mesh shape, and does not resolve
the worst approximation error. Retain it as a measured proposal generator for
experiments. No new viewer default is justified. The next comparisons still
need the best fixed candidate, surface-aware count allocation, relocation and
shared arc-length boundary maps; these tests used frozen uniform UV spacing.

Log: `/laboratory/quilting/scratch/coarse-metric-20260909-v2.log`.
