# Four-Patch Center

Standalone page: `fe/web/exact-patch-subdivision/index.html`.
Run with the release compiler:

```sh
.toolchains/fe/target/release/fe web dev fe/web/exact-patch-subdivision/index.html --port 8774
```

The center is a parameter on the original surface. Four triangles partition
the square parameter domain around that center. Restricting the original
bilinear numerator and weight fields to these triangles gives quadratic
triangular fields. Their quotient evaluates the same original patch. The
center changes the partition and display mesh, not the underlying surface.

**Fan concentration** now redistributes samples toward that shared center
(values above one) or toward the outer boundary (below one). It is independent
of the shape-weight controls and uses the same value on all four children.
Outer samples remain fixed and shared radial samples remain paired. It only
applies when four pieces are displayed; the no-subdivision comparison ignores
it. The center search currently scores the unwarped child regions, not this
new sampling distribution.

The coordinate map can also be absorbed exactly into the quadratic numerator and
denominator controls: each corner/edge coefficient is multiplied by the
corresponding product of two barycentric weights. Both fields gain the same
scalar denominator-squared factor, which cancels in their quotient. The shared
Fe operation is generic over `ProjectiveAlgebra` and `LinearElement`; it does
not hardcode a GA metric. The display now maps atlas coordinates into the
original chart and evaluates the original patch directly. This is algebraically
equivalent but avoids reconstructing tiny child nets, whose f32 derivatives can
lose precision. Exact restriction/pullback remains a separately tested operation.

This viewer uses the same resident blue-noise atlas as Radial Atlas Fan:
165 canonical edge triples spanning LoD 0–8, with permutations restored before
evaluation. `mesh_lod` selects the outer edges and `radial_lod` the edges
incident to the center. A GPU compute pass writes the selected tile's indirect
draw count; the render pass instances it across the exact children. Interaction
does not read geometry back to the CPU or rebuild/upload a display mesh.
The resident artifact is 24,871,552 bytes. Both this explorer and Two Triangles,
Two Fans use `quilting_patch_view` for atlas lookup, original-surface evaluation,
analytic normals, depth shading, and pixel-width wires. Leaf transforms and
atlas lookups are still repeated per vertex; caching per-leaf plans remains an
optimization opportunity.
A fixed pair of LoDs retains the same triangle count as concentration changes.
Valid parameter topology does not guarantee an accurate or well-shaped display
mesh on the surface.

## Meaning and limits

| Part | Contract |
| --- | --- |
| Object | The authored isotropic Clifford patch, or its explicitly marked grade-one projection when the weights leave the vector-valued family. |
| Domain | Unit square, partitioned into four oriented triangular regions with an interior center. |
| Metric | Euclidean lengths and surface area in the admitted 3D chart. This explorer does not assert a metric shared by all GA models. |
| Exact operation | Restriction of numerator and weight fields; matching shared curves and tangent planes wherever the original surface is regular. Floating-point evaluation remains approximate. |
| Approximate operation | Triangle rendering, numerical area integration, center search, and the optional split decision. |
| Search | Seven shrinking 3-by-3 neighborhoods starting at the square center; a local search, with no claim of global optimality. |
| Search score | Worst sampled tangent-triangle quality and/or balance of estimated child areas. Four equal-area subtriangle centers sample each child. |
| Cost | At most 64 center evaluations, each with 16 child differential evaluations. Search runs on relevant parameter changes in Fe/Wasm. Rendering uses the selected center. |
| Split decision | A heuristic from five root differential samples. Its threshold is dimensionless; it is not a screen-error tolerance or a certificate. |
| Failure | Samples can miss narrow features; local optimization can miss a better center; invalid weights, singularities, and f32 cancellation require distinct diagnosis. |
| Falsifier | Compare the chosen partition against dense independent sampling at equal triangle counts. A higher score need not imply lower display error. |
| Role | An exact restriction and center-placement experiment toward the virtual patch hierarchy. Recursive leaves, canonical shared-edge storage, per-leaf atlas demand, and certified stopping remain to be integrated. |

The shape score is `4 sqrt(3) area / sum(edge_length_squared)` on each tangent
triangle. It approaches zero as a triangle flattens, including when none of
its three edges is short. Its normalization preserves quality across uniform
scales. Area estimates integrate the magnitude of the surface Jacobian and
are kept separate from quality.

All four children share the same radial LoD and concentration, pairing their
radial boundary samples. Their outer samples are unaffected by concentration.
Independent per-edge demand in a larger hierarchy still requires canonical
shared-edge ownership and reconciliation; matching analytic curves alone is
insufficient.

The no-subdivision comparison displays the square through its fixed diagonal;
it uses two triangular restrictions internally with uniform `mesh_lod` on
every edge, ignoring `radial_lod` and concentration. When both LoD controls
agree, the four-region path has twice as many triangles as this comparison.
This is not an equal-budget performance or approximation-quality comparison.

For this authored example, the four scale controls preserve the vector-valued
surface when `w00*w11 = w10*w01`. The tuple `0.02, 0.4, 0.4, 8` produces strong
parameter distortion while preserving that relation. **Resolve weight constraint**
is checked by default. Each weight edit projects all four positive log-weights
onto `log(w00) - log(w10) - log(w01) + log(w11) = 0`, minimizing their summed
squared changes. The implementation uses two square roots, not logarithms.
A common scale fits the corrected values into the slider range without changing
the relative weights. The UI displays those actual corrected values. Correction
runs once per weight edit or when enabling the checkbox, not per vertex/frame.
Unchecking leaves the current surface unchanged and permits independent edits;
rechecking corrects the current weights (it does not restore previous values).
This condition is specific to the fixed authored control points; arbitrary
control-point edits and other patch models need their own admission rules.

Independent scales such
as `0.02, 8, 8, 8` generally leave that family; the red diagnostic identifies
sampled non-vector residuals rather than tessellation error. Near its tolerance,
rounding can also affect admission. Red does not mean that the displayed xyz
projection is undefined, nor that the triangles themselves are invalid.

## Alternative atlas experiment: independent interior density

Add an adaptive blue-noise atlas key `(A, B, C; M)`: three edge densities and
one interior density. `M` controls the interior sampling field, not the location
of this explorer's movable split center. Require exact canonical boundary
samples determined solely by each edge's density; varying `M` must leave them
unchanged. Blend the interior target toward those boundary conditions.

Permute only `A, B, C`, together with barycentric coordinates; `M` is invariant.
For nine levels (0–8) this means 165 canonical edge triples times nine interior
levels: 1,485 keys before feasibility/storage policy, not a handpicked ladder.
Start with a focused standalone explorer sharing atlas topology validation and
permutation tests. Measure quality, boundary agreement, generation time, memory,
and allocation/overflow before choosing full residency versus on-demand caching.
High boundary demand still imposes a minimum amount of interior triangulation;
the center control cannot promise an arbitrary independent triangle count.

This is an alternative, not implemented. The current radial fan reuses the
ordinary atlas instead of multiplying residency by nine. A richer atlas complements exact child
subdivision; it does not by itself resolve a patch singularity or uneven
screen-space stretch within a large child.

## Evidence

`projective_control_pullback_wasm_matches_exact_surface_and_normals` checks
525 interior samples across five concentrations from 1/16 to 16. The production
Fe reweighted-control evaluator is compared against direct independent f64
surface evaluation at explicitly warped coordinates; analytic normals are
compared with independent f64 finite differences. Numerical conditioning can
still change under homogeneous scaling, especially near singularities.

Browser check (2026-09-04): `fan_concentration=4`, `mesh_lod=2`, and
`radial_lod=5` rendered the curved four-piece surface with the resident atlas
in Chrome MCP. The corner diagram retained the same center and partition.
This build reports 49,076 bytes of Wasm and 126,472 bytes of WGSL across five shaders. This is an
interaction/visual check, not evidence that automatic meshing is solved.

`radial_atlas_concentration_is_a_budget_neutral_tradeoff` uses the actual
resident equal-edge LoD-3 tile: 400 triangles across four children, at every
tested concentration. Fifteen cases check positive parameter-triangle area,
complete parameter-domain coverage, and finite sampled chord error. At the
default centered paper patch, sampled maximum error is 0.290 at concentration
1 versus 0.914 at 4. On the compatible squeezed patch it is 2.327 at 1 versus
1.909 at 0.25: redistribution helps that sampled maximum but does not solve
the large error. These are four probes per rendered triangle, not certified
bounds. The reported RMS weights parameter area, not surface area; it can
underrepresent a stretched surface region.

`separable_corner_weights_wasm_resolve_symmetrically` executes the shared Fe
correction in Wasm over 1,296 weight combinations, comparing against an
independent f64 logarithmic projection. It checks the constraint, idempotence,
slider-fit range, and retention of the compatible extreme tuple above.

Browser check (2026-09-04, release `fe web dev`, Chrome MCP): the checkbox
started enabled. Editing w00 toward its minimum corrected all four values;
their diagonal-product ratio was `1.000000034`. Disabling correction and editing
w10 left an independent ratio `0.048437066`; re-enabling restored it to
`1.000000092`. Automatic center search updated the corner marker and rendered
the surface. No browser console warnings/errors were reported. This verifies
the interaction, not the quality of the resulting display mesh.

The dense f64 Clifford oracle checks the restriction identity and shared edge
positions, including independent extreme weights. For the regular authored
surface it also checks the shared tangent planes. These finite tests support
the algebraic restriction law; they do not prove display-mesh accuracy.

`tangent_triangle_wasm_detects_flattening_and_preserves_scale` runs the shared
Fe quality calculation as Wasm. It checks nearly collinear triangles, rigid
axis permutation, uniform scaling from `1e-12` to `1e12`, rank loss, nonfinite
inputs, and nonrepresentable area. The Rust code is test infrastructure;
the demo's scoring, search, and rendering are authored in Fe.

An independent f64 experiment also shrinks exact triangle regions around a
regular interior anchor and compares their flat display triangles with sampled
surface points and normals. In six halvings, maximum sampled local chord error
falls from 2.550 to 0.002597 on the authored patch and from 2.559 to 0.0000591 on
the valid squeezed patch above. A bilinear saddle has the expected exact
fourfold error decrease per halving. These results motivate local subdivision;
they do not establish a full covering mesh, an efficient stopping rule, or
improvement at equal total triangle budgets.

```sh
cargo test --release --manifest-path fe/tools/quilting-fe-fixtures/Cargo.toml --lib clifford_oracle
cargo test --release --manifest-path fe/tools/quilting-fe-fixtures/Cargo.toml --features fe-oracle --lib tangent_triangle_wasm_detects_flattening_and_preserves_scale
```
