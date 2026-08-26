# Conformal/QB optimizer prototype

This is an offline experiment, not a renderer path. Its first purpose is to
make candidate clustering and fitting measurable without pretending that an
ordinary Euclidean mesh error is invariant under all Möbius transformations.

## Proposed pipeline and API

1. **Ingest a primitive domain.** `ClusterInput` takes positions, indexed
   triangles, stable source-face IDs, optional face-domain IDs, and locked
   edges. A production domain ID must include material, UV/normal seams,
   skin/morph ownership, sharp edges, and any non-manifold policy. Attributes
   are not welded by position.
2. **Form small connected candidates.** `cluster_connected` grows across edge
   adjacency, scoring vertex reuse, normal-cone disagreement, and spatial
   extent while enforcing face/vertex budgets. Domain boundaries, locked
   edges, and non-manifold edges cannot become cluster interiors. `ClusterId`
   is a deterministic FNV-1a digest of sorted stable source-face IDs, so an
   unchanged membership has an unchanged identity after source-buffer
   reordering. Hash collisions must eventually be resolved by retaining the
   sorted membership as the authoritative identity.
3. **Optionally borrow meshoptimizer ordering.** With the
   `meshopt-prototype` feature, `cluster_meshopt_seeded` uses `meshopt` 0.6.2's
   `build_meshlets` result only as a seed order, then runs the same
   constraint-aware grower. The crate is MIT OR Apache-2.0 and vendors
   meshoptimizer 0.25. It is intentionally optional: current upstream
   meshoptimizer is newer, and its meshlet cost is not a QB fitting objective.
4. **Construct a shared coarse complex (not implemented yet).** Reduce cluster
   boundary loops to stable coarse vertices and triangulate the cluster
   adjacency graph. Boundary vertices and edges must have one owner in the
   coarse complex. Independent per-cluster corner selection is insufficient:
   it cannot guarantee matching patch boundaries.
5. **Fit existing first-order triangular QB patches.** Parameterize source
   samples onto coarse triangles, then feed `linear_global_fit_full`. Its one
   quaternion weight per coarse vertex is exactly the ownership rule required
   for C0 boundaries: neighboring patches share the two positions and weights
   defining their common rational edge. The output remains the current
   `QBTriPatch` (three positions plus three quaternion weights), directly
   consumable by canonical tessellation. Higher-order triangular rational
   patches are explicitly deferred.
6. **Score before accepting.** `score_patch_complex` reports source-space
   positional/normal error, a user-supplied envelope of representative Möbius
   transforms, shared-edge cracks, and rational-denominator conditioning. It
   rejects non-finite inputs, degenerate/inconsistently wound faces, and
   non-manifold coarse edges before measuring. `FitScore::scalar_objective` is
   a tunable candidate ranking, not a theorem.

## Objective

The source term reports Euclidean RMS/max position error and oriented normal
error in the persistent pre-Möbius material chart. It is a fitting diagnostic,
not the runtime walking/physics distance. Every conformal probe transforms both
the target samples and the QB patches, then reports raw active/output-chart
Euclidean error, transformed-bounds-relative error, transformed denominator
conditioning, peak local dilation, and pole-near sample count. The raw
output-chart values are the perceived geometry and the appropriate input to
runtime walking/physics tolerances. Useful probes include authored inversion
spheres, dilations spanning expected scene scales, and pole positions near (but
not on) the source surface.

The robustness envelope is deliberately empirical. Möbius maps preserve
angles and round geometry, but not Euclidean distance or nearest-neighbor
ordering; there is no universal conformally invariant Euclidean fitting error.
Near a pole, arbitrarily small source errors can be magnified arbitrarily. Such
cases must be rejected or handled conservatively, not hidden by normalization.
The configured pole threshold is a hard exclusion from this finite-Euclidean
objective, and is measured relative to the Möbius matrix/input scale so an
equivalent common real gauge does not change the verdict.

Boundary error samples every shared coarse edge from both incident patches,
both in the source chart and after every probe; exact shared ownership should
keep it zero, while output-chart sampling exposes numerical amplification near
a pole. Weight conditioning computes the exact minimum and maximum squared norm
of the affine quaternion denominator over the closed parameter triangle, then
divides by the largest corner-weight norm. The result is insensitive to a
common real weight scale and cannot miss a zero between grid samples. It is
evaluated both in the source material chart and after every probe transforms
the patch weights. A near-zero exact minimum is a hard rejection from the
scalar objective.

## Integration seam

The prototype's `ClusterSet` can feed a future boundary-complex builder. That
builder should emit:

```text
CoarsePatchComplex {
    stable_vertices: Vec<(StableVertexId, position, shared_weight)>,
    faces: Vec<[StableVertexId; 3]>,
    source_samples: Vec<(face, barycentric, position, normal)>,
    source_clusters: Vec<ClusterId>,
}
```

`linear_global_fit_full` already covers the shared-weight solve; canonical
tessellation already accepts the resulting `Vec<QBTriPatch>`. No renderer,
atlas, WASM, or shader change is needed for this experiment.

## Known limitations

- Cluster growth does not yet build the shared triangular coarse complex, so a
  good cluster score does not by itself produce a patch.
- The spatial/normal growth weights are heuristics. Meshoptimizer ordering may
  improve locality but is not assumed to improve QB fit quality.
- Probe normals use the exact Möbius differential on an oriented tangent frame;
  sphere reflection reverses that frame naturally. Flipped candidate normals
  therefore score as 180°, not 0°.
- The denominator range is exact for first-order QB patches. Position, normal,
  and crack error remain sampled diagnostics rather than Hausdorff bounds.
- Animated fitting needs pose-envelope samples and stable source identities;
  this prototype scores static sample sets.
- UVs, normals, skinning, morphs, and materials are represented only by domain
  barriers here. A production optimizer must propagate the complete attribute
  streams.

## Initial reproducible measurements

`cargo run -p quilting-remesh --release --example
conformal_optimizer_report --features meshopt-prototype` on the development
machine produced this first smoke-test snapshot (microseconds, not a stable
benchmark):

| Fixture/method | Source faces | Clusters/patches | RMS position | Max position | Boundary max | Time |
|---|---:|---:|---:|---:|---:|---:|
| sphere / connected growth | 320 | 14 | — | — | — | 1,270 µs |
| sphere / meshopt-seeded growth | 320 | 13 | — | — | — | 1,687 µs |
| cylinder / connected growth | 576 | 20 | — | — | — | 4,688 µs |
| cylinder / meshopt-seeded growth | 576 | 20 | — | — | — | 5,417 µs |
| exact sphere / shared linear QB fit | samples on 8 patches | 8 | < 1e-9 | < 1e-9 | 0 | 157 µs fit |
| exact sphere / flat baseline | samples on 8 patches | 8 | 0.028240 | 0.060696 | 0 | — |

The shared QB fit also stayed below `1e-9` positional error after a scale-by-8
probe; the flat baseline's output-chart RMS became `0.225921`. Under the
off-surface reflection probe, shared-QB output error stayed below print
precision while its sampled relative denominator minimum was `0.506785`.
These fixtures are intentionally favorable exact-QB ground truth. They prove
the API and diagnostics are wired correctly, not that arbitrary production
meshes will fit this well. Meshoptimizer seeding was slower at this tiny scale;
its value, if any, needs larger real meshes and fit-quality measurements.
