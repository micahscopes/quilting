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
3. **Optionally borrow meshoptimizer machinery.** With the
   `meshopt-prototype` feature, `cluster_meshopt_seeded` uses `meshopt` 0.6.2's
   `build_meshlets` result only as a seed order, then runs the same
   constraint-aware grower. `coarse_reduction` separately uses constrained
   simplification only inside Quilting-owned cut charts. The crate is MIT OR
   Apache-2.0 and vendors meshoptimizer 0.25. It is intentionally optional:
   current upstream meshoptimizer is newer, and its meshlet or Euclidean error
   is not itself a QB fitting objective.
4. **Construct a shared coarse complex.** `coarse_complex` now
   requires typed stable vertex/face IDs, validates finite nondegenerate
   consistently wound manifold input, derives cut edges from source borders,
   domains, and explicit locks, and splits face-corner fans across those edges.
   A deterministic whole-edge closure extends an open interior seam to a source
   boundary or closed cut network, recording every conservative extension as an
   induced cut. Emitted charts are revalidated for manifold vertex links,
   consistent winding, connectedness, and degree-two locked boundaries. This is
   backend-neutral ownership: no simplifier can half-join or silently weld a
   source edge. `coarse_reduction` now normalizes each chart before the f32
   backend call, locks every emitted boundary, fails on stable-ID collisions in
   backend position space, and revalidates exact boundary identity, winding,
   vertex links, connectivity, Euler characteristic, and boundary components.
   Requested and achieved triangle counts remain distinct because an owned
   boundary can impose a conservative floor. `coarse_patch_complex` welds the
   retained chart copies by stable source identity, revalidates the global
   oriented manifold and cut-edge incidence, and emits deterministic
   equal-area source quadrature with normalized surface weights and
   chart-restricted closest-point barycentrics. Sample and candidate-test
   budgets bound the temporary brute-force matcher. Independent per-cluster
   corner selection remains insufficient because it cannot guarantee matching
   patch boundaries.
5. **Fit existing first-order triangular QB patches.** Parameterize source
   samples onto coarse triangles, then feed
   `CoarsePatchComplex::fit_shared_qb`. Its one
   quaternion weight per coarse vertex is exactly the ownership rule required
   for C0 boundaries: neighboring patches share the two positions and weights
   defining their common rational edge. The fitter normalizes scene scale,
   column-equilibrates the augmented system, pins one stable gauge vertex per
   connected component, and uses sparse LSQR without forming condition-squared
   normal equations. An explicitly recomputed relative normal residual—not the
   iterative recurrence alone—is the convergence gate. Structural,
   non-convergence, non-finite, and exact closed-domain
   denominator-conditioning failures are explicit. The output
   remains the current `QBTriPatch` (three positions plus three quaternion
   weights), directly consumable by canonical tessellation.
   `linear_global_fit_weighted_full` scales every four-row sample block by the
   square root of its normalized surface measure; the legacy unweighted API
   remains a compatibility wrapper with its original summed objective. The
   weighted regularizer is defined against a unit-total data objective, so its
   production value must be selected by real-asset benchmarks rather than by
   source triangle count. Higher-order triangular rational patches are
   explicitly deferred.
6. **Score before accepting.** `score_patch_complex_weighted` reports source-space
   positional/normal error, a user-supplied envelope of representative Möbius
   transforms, shared-edge cracks, and rational-denominator conditioning. It
   rejects non-finite inputs, degenerate/inconsistently wound faces, and
   non-manifold coarse edges before measuring. Relative weighted errors use a
   separately memoizable `FitScoreContext` computed from authoritative source
   positions. The context owns the exact probes and their source-derived
   output extents together; neither mismatched probes, candidate controls, nor
   quadrature bounds can change the denominator.
   `FitScore::scalar_objective` is a tunable candidate ranking, not a theorem.

## Objective

The source term reports Euclidean RMS/max position error and oriented normal
error in the persistent pre-Möbius material chart. It is a fitting diagnostic,
not the runtime walking/physics distance. Every conformal probe transforms both
the target samples and the QB patches, then reports raw active/output-chart
Euclidean error, source-reference-extent-relative error, transformed
denominator conditioning, peak local dilation, and pole-near sample count. The
area measure remains fixed to the source/material surface under every probe;
this deliberately asks how the same authored material is represented after a
map. An output-area RMS would instead reweight by the square of local conformal
length scale. The raw output-chart values are the perceived geometry and the
appropriate input to runtime walking/physics tolerances. Useful probes include
authored inversion spheres, dilations spanning expected scene scales, and pole
positions near (but not on) the source surface.

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

`coarse_patch_complex::build_coarse_patch_complex` now provides the stable
positions, faces, provenance, cut records, and area-aware correspondences for
the weighted fitter. The remaining seam is intentionally narrow:

```text
CoarsePatchComplex {
    vertices: Vec<CoarseVertex>,
    faces: Vec<CoarseFace>,
    correspondence: Vec<CorrespondenceSample>,
    ...
}
```

`CoarsePatchComplex::fit_shared_qb` converts these correspondences directly to
the weighted shared solve; `weighted_score_samples` supplies the same measure
and oriented source normals to conformal scoring, while `fit_score_context`
derives candidate-independent normalization from retained stable source
positions. Canonical tessellation already accepts the resulting
`Vec<QBTriPatch>`. No renderer, atlas, WASM, or shader change is needed for this
experiment.

## Known limitations

- Area measure now reaches fitting and scoring, with sample-splitting
  invariance tests. Its default regularization and acceptance thresholds still
  need real-asset calibration. Boundary locks deliberately retain the complete
  authored cut polyline; shared boundary coarsening is a later optimization.
- Correspondence is chart-restricted, orientation-filtered, and deterministic,
  but still brute-force and geometrically nearest. Production-scale meshes need
  a chart BVH, and near-coincident folded sheets need a topology-aware policy.
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
- Sparse LSQR preserves the at-most-twelve nonzeros in each sample equation and
  one nonzero in each regularization row. It is iterative: difficult production
  complexes still need convergence/workload telemetry and may eventually
  justify a mature sparse QR or LSMR backend. Returning to dense normal
  equations is not an acceptable speedup.

## Initial reproducible measurements

`cargo run -p quilting-remesh --release --example
conformal_optimizer_report --features meshopt-prototype` on the development
machine produced this first smoke-test snapshot (microseconds, not a stable
benchmark):

| Fixture/method | Source faces | Clusters/patches | RMS position | Max position | Boundary max | Time |
|---|---:|---:|---:|---:|---:|---:|
| sphere / connected growth | 320 | 14 | — | — | — | 1,723 µs |
| sphere / meshopt-seeded growth | 320 | 13 | — | — | — | 3,276 µs |
| cylinder / connected growth | 576 | 20 | — | — | — | 7,347 µs |
| cylinder / meshopt-seeded growth | 576 | 20 | — | — | — | 10,813 µs |
| exact sphere / shared sparse-LSQR QB fit | samples on 8 patches | 8 | < 1e-9 | < 1e-9 | 0 | 1,960 µs fit |
| exact sphere / flat baseline | samples on 8 patches | 8 | 0.028240 | 0.060696 | 0 | — |

The shared QB fit also stayed below `1e-9` positional error after a scale-by-8
probe; the flat baseline's output-chart RMS became `0.225921`. Under the
off-surface reflection probe, shared-QB output error stayed below print
precision while its sampled relative denominator minimum was `0.506785`.
These fixtures are intentionally favorable exact-QB ground truth. They prove
the API and diagnostics are wired correctly, not that arbitrary production
meshes will fit this well. Meshoptimizer seeding was slower at this tiny scale;
its value, if any, needs larger real meshes and fit-quality measurements.
The earlier 157 µs fit used dense normal equations. A subsequent
column-pivoted dense QR reference was numerically sound but did not finish the
full `curved_vs_flat` run in 60 seconds: it discarded the matrix's sparse
structure and swept every trailing dense column. The sparse-LSQR replacement
completed that entire 12-fit example—including 78-, 90-, and 140-patch coarse
fixtures plus their input construction—in 0.165–0.67 seconds across repeated
cached-binary runs on the same development machine. This is encouraging
engineering evidence, not a stable benchmark; the upstream QEM fixture builder
also resolves some symmetric ties nondeterministically.

`curved_vs_flat` labels its denominator diagnostic `sample|bot|`: its legacy
correspondence margin can produce affine barycentrics just outside a coarse
triangle. That sampled extrapolated value may therefore fall below the fitter's
exact in-domain relative-denominator guard without contradiction.
