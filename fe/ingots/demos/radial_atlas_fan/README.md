# Radial Atlas Fan

Run the standalone explorer with the release compiler:

```sh
.toolchains/fe/target/release/fe web dev fe/web/radial-atlas-fan/index.html --port 8775
```

Drag the yellow interior point. **Concentration** redistributes existing
samples toward or away from it. **Interior LoD** changes the number of samples
along its three incident edges. **Outer A/B/C** independently select boundary
edge levels. All levels run from 0 to 8 and use the existing complete atlas,
including canonicalization and barycentric permutations.

**Endpoint A/B/C** redistribute samples along outer edges. Equal endpoint
weights leave those samples unchanged. Adjacent patches must share each
edge's endpoint weights and LoD, accounting for orientation. Different weights
at the vertex opposite a shared edge do not affect that edge's sampling.

## Meaning

Split one reference triangle around an interior point H into children
`(H,B,C)`, `(H,C,A)`, `(H,A,B)`. For a child, transform its atlas barycentrics by
`(a,b,c) -> (wa*a,wb*b,wc*c)/(wa*a+wb*b+wc*c)`, then map them into the root
triangle. A child uses H's concentration weight and the weights of its two
outer vertices. This is sampling-coordinate transformation, not modification
of an analytic surface's control points.

| Part | Contract |
| --- | --- |
| Object/domain | A parameter-space triangle covered by three ordinary atlas triangulations. |
| Metric | Reference-triangle Euclidean coordinates; not yet a surface-area or screen-demand solver. |
| Guarantee | Positive finite weights and a strictly interior H give piecewise-projective homeomorphisms in exact arithmetic. Lines remain lines; triangles do not fold. |
| Boundary | Shared endpoint weights and sample parameters determine each edge independently of the opposite vertex. Canonical values are required for bitwise f32 agreement. |
| Smoothness | Each child map is smooth. Across fan seams the maps agree in position, but transverse derivatives need not agree. The eventual evaluated analytic surface is unchanged. |
| Density | Concentration redistributes a fixed budget. LoD changes the budget. Neither guarantees uniform surface sampling. |
| Cost | Existing immutable 24.9 MB atlas; one three-entry count pass writes an indirect draw. No interaction-time sampling, triangulation, mesh-buffer upload, or GPU readback; small control uniforms still cross to the GPU. |
| Failure | Extreme ratios or near-boundary H can create skinny triangles and precision loss. Projective warping preserves topology, not Delaunay optimality or blue-noise spacing. |
| Falsifier | Signed-area changes, missing coverage, inconsistent shared samples, or unacceptable approximation error at equal triangle budgets. |
| Role | Shared coordinate-map primitives for subsequent exact-patch sampling and recursive virtual patchworks. |

The three endpoint weights provide compatible projective edge warps, not three
arbitrary independent edge functions. Their directed endpoint ratios around a
triangle multiply to one. More general edge functions need a different interior
extension. A shared-edge owner in a future PatchGraph should store canonical
endpoint weights/sample parameters, rather than have neighbors infer them.

## Evidence and remaining work

`radial_atlas_fan_wasm_preserves_seams_and_triangle_orientation` executes the
production Fe maps as Wasm. It checks 257 dyadic positions on each radial seam
with two interior points and five concentration strengths, requiring bitwise
agreement. It checks reversed outer edges with nine endpoint-weight pairs,
independent opposite weights, and the same 257 samples. It also checks all six
permutations of the frozen matrix atlas, inverse maps, triangle orientation,
simplex containment, and total area coverage. This is not exhaustive across the
full LoD-8 resident atlas or all floating-point inputs.

Browser receipt (2026-09-04): release `fe web dev` served three compiled passes
(draw preparation, background, fan). The default fan was visually inspected.
Control edits reached Fe state for concentration 4, endpoint A weight 2, and a
moved interior point; the renderer continued reporting WebGPU with no console
warnings/errors. Interior LoD 8 with outer A at 0 was also admitted without a
reported error, then restored to LoD 4/3. Screenshot capture was intermittent;
these state/error checks are not a claim of exhaustive visual validation at
the extreme settings. The delivered build has 12,486 bytes of Wasm and 44,903
bytes of browser-reported WGSL across three shaders, excluding its atlas data.

```sh
cargo test --release --manifest-path fe/tools/quilting-fe-fixtures/Cargo.toml --features fe-oracle --lib radial_atlas_fan_wasm -- --nocapture
```

Still required: integrate this map with exact surface evaluation and the child
hierarchy; test extreme patch fixtures at equal budgets; select concentration
from measured surface distortion; implement canonical graph-wide edge state,
deep-linked controls, and the gallery's compiler-derived source/provenance viewer.
