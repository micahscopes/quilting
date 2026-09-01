# Classic Quilting demo ingots in Fe

Status: implementation handoff specification, 2026-09-01. This document is
deliberately self-contained. It records what was inspected, what can be reused,
what must not be inherited, and a stop-gated build order for another agent.

The evidence pins are:

- classic TypeScript Quilting: `a565a4c85c1186142093361c4068537c36aab934`
  (`todo: faster tessellation atlas generation`, 2022-06-24), with earlier
  milestones named in the source map;
- current Rust Quilting: `cd9f810` (`Define HHHS worker close request`) on the
  `rust` branch when inspected; unrelated Hyperscape edits appeared later and
  are not inputs to this design;
- Fe source worktree: `60834b9af8269681b2444199913ad9bfa4104ab4`
  on the `mb2` branch when inspected; that worktree already had unrelated dirty
  work and must not be cleaned or folded into this project; and
- MB2 architecture-document repository: `7302ce1ac3c059b05a3af39ce77a2a4601bb8292`
  when inspected. It also had unrelated dirty state.

This document authorizes no dependency cutover by itself. It specifies a demo
family and the evidence needed to promote each implementation rung.

## 1. Outcome and decision

Build a small family of composable Fe ingots that teaches the classic
Quilting pipeline one concept at a time, ending in one interactive patch
laboratory:

1. deterministic blue-noise samples in a triangular parameter domain;
2. the constrained Delaunay triangulation of those samples;
3. one quaternionic-Bézier triangular surface evaluated over that topology;
4. two or more patches sharing exact boundary samples;
5. requested versus resident edge LOD, canonical S3 permutations, parity, and
   seam reconciliation;
6. draggable patch handles and weights;
7. wire, density/LOD, normal, and simple material presentations; and
8. canonical URL state plus a deterministic replay log.

The culminating demo should feel like the compact, playful old Quilting demos,
but its architecture must follow the current stack:

- Fe owns the demo's semantic state, pure transition logic, typed surface
  declaration, and the small shader-specialized arithmetic it can honestly
  compile today.
- Rust Quilting owns sampling, constrained triangulation, atlas construction,
  permutation semantics, topology validation, serialization, and scalable
  WebGL2/WebGPU renderer integration.
- A compiler-generated or narrowly specified browser adapter owns browser
  events, workers, canvas/device creation, transfer, IndexedDB, and device or
  context recovery. It does not own a second scene model.
- Static atlas construction is not frame work. A cached or build-time atlas is
  the default. Runtime direct blue-noise regeneration is an explicit
  educational/background mode.
- WebGPU compute is used only where measurement and data residency justify it.
  Branchy robust triangulation is not moved to a shader merely to say that it
  is on the GPU.

Do **not** port the TypeScript renderer, its global mutable objects, its render
loop, its package graph, or its string-keyed mesh bookkeeping into Fe. The old
code is an idea mine and a visual reference, not an architecture template.

### 1.1 The crucial QB naming correction

This must be decided before implementation.

In current Quilting, **QB means quaternionic-Bézier**, not “quadratic
Bézier.” `quilting_core::patch::QBTriPatch` has three position quaternions and
three quaternion weights. Its homogeneous numerator and denominator are
degree one in barycentric coordinates. Quaternion division makes the resulting
surface curved and Möbius-compatible even though the barycentric basis is
linear.

A conventional **quadratic triangular Bézier** patch has six control points
and degree-two Bernstein basis functions. It is a useful educational extension,
but it is not the current Quilting patch ABI and cannot silently replace it.

The recommended demo order is:

- required: the actual current three-control quaternionic-Bézier patch;
- optional comparison ingot: a six-control polynomial quadratic triangle; and
- deferred: rational/quaternionic higher-order fitting, which needs its own
  type, restriction rule, seam contract, shader specialization, and fixtures.

If the product requirement really means the six-control quadratic model, stop
at milestone Q0 and name that choice explicitly. Do not call one model the
other in code, UI, fixtures, or documentation.

## 2. What the classic TypeScript implementation actually did

The best final classic snapshot is `a565a4c`. It is explicitly marked
experimental and AGPL-3.0-or-later; its examples were marked CC BY-NC-SA 4.0.
Those terms matter if any source or example art is copied. Reimplementing the
mathematics from the current dual-licensed Rust crates is preferable.

### 2.1 Gems worth preserving

`src/tessellation.ts` contains the central old pipeline:

- `triEdgeWeightInterpolator(A, B, C)` interpolates an edge-derived density
  over the barycentric domain.
- `triPatch(resA, resB, resC, opts)` inserts exact boundary samples first:
  `resC + 1` points on AB, `resB + 1` on AC, and `resA + 1` on BC, with corners
  inserted once. It then fills the interior with a variable-radius Poisson
  sampler.
- `sampleTriangle` uses the square-root transform of two uniform random values
  to sample a triangle without center bias.
- `tessellation` sends the complete sample set to Delaunator.
- `prepareMesh` converts the flat triangulation into positions, triangle cells,
  normals, and expanded per-cell data.

The valuable conceptual move is “pin the stitch boundary, then distribute
interior samples, then triangulate.” The boundary is the interface between
patches; interior point placement is a quality policy.

`src/load-tessellation-atlas.ts` adds the second major idea:

- enumerate edge-LOD triples with replacement;
- sort each triple and generate only one canonical member of each S3 orbit;
- generate canonical patches in workers;
- combine them into one atlas; and
- build a lookup that carries base index, count, edge permutation, and the
  vertex permutation needed to stamp the canonical patch in all six
  orientations.

`src/permutator.ts` contains the six S3 permutations and the old explicit
edge-permutation to vertex-permutation table. Its implementation is crude, but
the symmetry reduction is fundamental.

`src/tessellation-worker.ts` is only a `threads/worker` exposure of
`tessellationMesh`. Commit
`f32407065ef128ac225ca25ce35153a3cd53d0ba` introduced worker-based atlas
generation. The good idea is independent patch jobs off the canvas thread,
not the particular worker library.

`examples/tessellation.ts` is the clearest interactive affordance:

- mouse proximity assigns low, medium, and high face LODs;
- each shared edge takes the maximum of the incident face requests;
- the resulting three edge values select an atlas entry;
- an instanced permutation stream maps canonical barycentrics to each face;
- adjacent equal atlas ranges are coalesced; and
- triangle and patch counts remain visible.

`examples/terrain/terrain.ts` is the strongest visual composition:

- a moving scalar field chooses quantized LOD;
- face adjacency reconciles edge LODs;
- per-corner LOD is interpolated for color;
- the canonical atlas is instanced across a simple grid; and
- the draw loop is independent of the slower 100 ms LOD update stream.

That separation between a continuously drawing resident mesh and a slower
topology-selection stream is still useful, provided publication is atomic and
stale results are rejected.

The older `6e461da427817f3c3548947a42660eaa5920fac0` snapshot contains the
quaternion/CGA patch experiment in `gpu/gl/snippets/patches.glsl`, Bernstein
helpers in `src/bernstein.ts`, and variable-density boundary sampling in
`src/lod-patches.ts`. It demonstrates the visual direction, but computes
normals by evaluating nearby parameter points. Current Rust's analytic
quotient-rule normal is the implementation to retain.

### 2.2 Classic mechanisms not to inherit

The following are historical liabilities, not shortcuts:

- `Math.random()` makes atlas topology non-replayable.
- boundary classification and density interpolation depend on loose floating
  conventions and third-party packages without a frozen artifact ABI.
- extensive `any`, `@ts-ignore`, mutable face fields, `window.*` debugging
  globals, and JSON-string keys erase ownership and invariants.
- `Promise.all` over every atlas key has no generation identity, cancellation,
  admission budget, backpressure, or partial-failure policy.
- worker lifecycle and pool shutdown are not part of the model.
- every worker returns nested JavaScript arrays which are flattened, copied,
  converted, and copied again before upload.
- MDA, lodash, Delaunator, thi.ng, PicoGL, Most, RBush, and rendering glue are
  entangled in examples rather than separated behind typed contracts.
- pointer LOD uses repeated `Array.includes` inside face loops and rebuilds
  mutable face annotations.
- draw-range consolidation is rediscovered every frame in JavaScript.
- finite-difference normals use a fixed epsilon unrelated to patch scale.
- failures commonly become missing values or console output rather than a
  retained last-good atlas plus a typed error.
- the classic README itself lists draggable controls, faster atlas generation,
  general Möbius authoring, and modularization as unfinished.

The visual immediacy is worth recovering. The state and resource architecture
is not.

## 3. What current Rust Quilting provides

The Rust workspace is not one monolith. Reuse its narrow authorities.

### 3.1 Ready to reuse inside this repository

These components have concrete tests and are suitable as the initial
authority for the demo:

| Capability | Current source | What is ready |
| --- | --- | --- |
| Reference triangle | `quilting-core/src/triangle.rs` | equilateral vertices, containment, Cartesian/barycentric conversion, edge interpolation |
| Deterministic samples | `distressed-blue-noise/src/lib.rs` | PCG-seeded variable-density Bridson sampler, fixed seed preservation, rectangular and equilateral domains, constant-spacing jittered path |
| Patch boundary sampling | `quilting-core/src/sampling.rs` | exact edge seed counts/order, deterministic seed, barycentric snapping and renormalization |
| Robust constrained triangulation | `quilting-core/src/delaunay.rs` via `cdt 0.1.0` | triangular contour construction and constrained Delaunay; `cdt` uses exact `geometry-predicates` orientation and incircle tests |
| Symmetry | `quilting-core/src/permutation.rs` | all six S3 permutations, canonical sorted key, geometric remap, parity and winding tests |
| Atlas | `quilting-core/src/atlas.rs` | direct and hierarchical modes, reachable-key subsets, exact boundary tests, lookup, serialization, deterministic merge |
| Patch math | `quilting-core/src/patch.rs` | three-control QB evaluation, exact quotient-rule differential, normal, exact rational restriction, Möbius transform |
| Seam topology | `quilting-mesh/src/lib.rs` | half-edge adjacency and canonical shared-edge identity, including exact geometric welding across duplicated render vertices |
| Educational model | `quilting-core/src/educational.rs` | triangle/plane/cube patch-lab shapes, deterministic LOD fields, requested/resident accounting, 2:1 or 4:1 grading |
| Backend-neutral render values | `quilting-core/src/render*.rs` | immutable scene/frame/command and pipeline-description boundaries |
| Authoritative browser renderer | `quilting-renderer` | current WebGL2/glow renderer, Naga-lowered shaders, retained resources, preparation and visibility paths |
| Shared shader math | `quilting-shaders/shaders/surface/qb_eval.wgsl` | exact QB evaluation and analytic normal; permutation parity is explicit |
| WASM bridge | `quilting-wasm/src/lib.rs` | atlas construction/export/import, packed patch export, LOD and demo bindings |

“Ready” here means appropriate for reuse under the repository's current test
and release discipline. It does not mean the crates have a stable public 1.0
API.

The direct and hierarchical atlas policies are deliberately different:

- `Direct` independently blue-noise samples and constrained-Delaunay
  triangulates every requested canonical key. It has better fresh interior
  triangle quality but costs more time and topology.
- `Hierarchical` directly samples only irreducible roots, then produces
  power-of-two descendants by exact midpoint subdivision. It is the current
  runtime default.

Measured LOD-64 native release results already archived in
`docs/benchmarks/2026-08-22-hacker-night-baseline.md` are useful calibration,
not browser promises:

| Mode and grading | Keys | Triangles | Median build |
| --- | ---: | ---: | ---: |
| hierarchical 2:1 | 19 | 12,286 | 1.484 ms |
| hierarchical 4:1 | 34 | 19,106 | 2.658 ms |
| direct blue-noise 2:1 | 19 | 18,234 | 132.862 ms |
| direct blue-noise 4:1 | 34 | 26,742 | 197.729 ms |

At exponent 7, current browser startup evidence reported about 29 ms atlas
generation and 40.7 ms packing inside a 129.7 ms atlas phase. The environment
pipeline was much more expensive in that run. Measure this new demo rather
than assuming the same ratios.

### 3.2 Experimental or incomplete surfaces

Do not present these as solved:

- `quilting-webgpu` calls itself a shadow/conformance backend. It is now
  extensive, including retained pipelines, device-loss handling, LOD compute,
  visibility compaction, indirect tables, picking, PBR, and focus passes, but
  live browser parity is still gated and WebGL2 remains the authority path.
- `distressed-blue-noise` declares a `parallel` Rayon feature but its sampler
  contains no Rayon path. Parallelism currently exists across atlas keys in
  `quilting-core`, not within one Bridson sample.
- `TessellationAtlas::build_subset` is reasonable for independent direct keys.
  In hierarchical mode it currently builds the full hierarchy and then
  extracts the assigned keys. Spawning N such helpers repeats work N times.
  Current Hyperscope therefore intentionally uses one retained atlas worker.
- `quilting-wasm` exposes subset build, merge, export, and import, but that ABI
  predates the stricter generation/cancellation/backpressure contract below.
- current triangulation functions panic on missing corners or CDT failure.
  A browser-facing demo wrapper must return a typed `Result` and retain the
  previous generation.
- the core LOD module still contains some thread-local UI settings. The demo
  should pass immutable config values through its own request contract.
- current Fe authored raster supports fixed non-indexed
  `TriangleList<const VERTICES>`. It does not yet provide the dynamic indexed
  vertex/index-buffer and external-resource surface needed to make Fe own a
  large runtime atlas draw.
- an Fe-to-Quilting shader importer is only a bounded recommendation in
  `quilting/docs/fe-webgl-ga-prototype.md`, not an implemented production seam.

## 4. Mathematical and topological contract

An implementation agent must understand this section before touching browser
or GPU code.

### 4.1 Reference triangle and barycentric coordinates

Use the current equilateral reference triangle, not the old right triangle:

```text
A = (0, 1)
B = (-sqrt(3)/2, -1/2)
C = ( sqrt(3)/2, -1/2)
```

A parameter point is represented by barycentric coordinates
`b = (a, b, c)` with `a + b + c = 1`. It maps to Cartesian parameter space as
`a*A + b*B + c*C`. It lies inside or on the triangle when all three components
are nonnegative, within a stated numerical tolerance.

The edge convention is normative:

- edge 0, called A in an edge-LOD triple, is BC and has barycentric `a = 0`;
- edge 1 is CA and has `b = 0`;
- edge 2 is AB and has `c = 0`.

Thus `[res_a, res_b, res_c]` means subdivisions opposite control vertices
`[p0, p1, p2]`. Edge `e` has exactly `res_e + 1` boundary vertices at
parameters `i / res_e`, including both corners. Corners are inserted once.

Near-zero barycentric components produced by Cartesian conversion are snapped
to exactly zero and the triple is renormalized. This is not cosmetic: epsilon
times a control near a Möbius pole can become a visible inter-patch crack.

### 4.2 The actual Quilting QB triangle

Let each Euclidean control point be a pure imaginary quaternion `p_i`, each
weight be a general quaternion `w_i`, and let barycentrics be
`lambda = (lambda_0, lambda_1, lambda_2)`. Current Quilting evaluates

```text
N(lambda) = sum_i lambda_i (p_i w_i)
D(lambda) = sum_i lambda_i w_i
X(lambda) = N(lambda) D(lambda)^-1
position  = imaginary_part(X)
```

Control and weight order is `[0, 1, 2] = [A, B, C]`. Quaternions use
`(w, x, y, z)` in Rust and WGSL. Do not confuse the scalar component with the
first lane of current renderer position records, which stores a source vertex
index or adaptive-leaf tag.

Use analytic derivatives. With parameter `u` moving control 0 toward control 1
and `v` moving control 0 toward control 2:

```text
dN/du = p1*w1 - p0*w0       dD/du = w1 - w0
dN/dv = p2*w2 - p0*w0       dD/dv = w2 - w0
dX/du = (dN/du - X*dD/du) D^-1
dX/dv = (dN/dv - X*dD/dv) D^-1
normal = normalize(cross(imag(dX/du), imag(dX/dv)))
```

The shader may right-multiply both tangents by `conjugate(D)` instead of
`D^-1` before normalization. The omitted positive scalar is common to both
tangents and the form stays better bounded near a small denominator. Odd S3
permutations reverse orientation, so multiply the normal by permutation
parity.

Identity weights make a flat triangle. Non-identity quaternion weights curve
the surface. A denominator approaching zero indicates the Möbius pole region;
the demo must expose a bounded fade/error visualization rather than allowing
NaNs to poison the frame.

### 4.3 Optional conventional quadratic triangle

If milestone Q0 explicitly admits a polynomial quadratic comparison, name six
controls by exponents whose sum is two:

```text
P200, P110, P020, P101, P011, P002
```

For barycentrics `(a,b,c)`:

```text
P(a,b,c) =
    a^2 P200 + 2ab P110 + b^2 P020
  + 2ac P101 + 2bc P011 + c^2 P002
```

The three corners are `P200`, `P020`, and `P002`; the other controls bend the
three edges. Derivatives are obtained by differentiating this basis after
substituting `a = 1-u-v`, `b = u`, `c = v`, then crossing the two tangent
vectors. Shared polynomial edges require matching the three controls on that
edge, with reversed order on the oppositely oriented neighbor.

This model needs six draggable handles and a different GPU record. It cannot
reuse `QBTriPatch`, its three-weight ABI, or its exact rational restriction.

### 4.4 Blue noise and boundary treatment

“Blue noise” in this demo means a deterministic variable-density Poisson-disk
sample, not arbitrary jitter and not a promise about an ideal power spectrum.

The direct path is:

1. Create the exact ordered boundary seeds for the requested edge counts.
2. Give every candidate a positive finite local exclusion radius derived from
   the edge-density interpolation.
3. Use a seeded PCG generator and Bridson active list. Generate up to `k`
   annulus candidates around a selected active point.
4. Reject candidates outside the equilateral triangle.
5. Use a background grid sized from the estimated minimum radius and reject a
   candidate if it violates the maximum of its radius and a neighbor's radius.
6. Retire an active point after `k` failed candidates.
7. Preserve boundary seeds exactly. Never jitter them.

The current default is `seed = 42`, `k_candidates = 30`. A demo control may
change the seed or quality, but the values must enter canonical URL and replay
state. A random button chooses and displays a concrete seed; it never installs
an unrecorded ambient RNG.

For constant edge density, the jittered hex sampler may be shown as a separate
comparison. It is not called “the same result” as direct Bridson sampling.

### 4.5 Constrained Delaunay and robust predicates

Feed the complete boundary-plus-interior point set to a constrained Delaunay
triangulation whose contour is ordered A to B to C to A. Boundary segments
must be constraints; unconstrained Delaunay may create exterior fins around a
non-convex contour and is not the general interface.

Orientation and incircle decisions must use robust/exact predicates. Do not
rewrite them as naive `f32` determinants in Fe or WGSL. Current Rust uses
`cdt 0.1.0`, which delegates these decisions to `geometry-predicates` exact
predicates. If that dependency changes, adversarial collinear, cocircular,
near-degenerate, and duplicate-point fixtures are a release gate.

Canonicalize the final artifact independently of hash-map or worker completion
order:

- canonical keys sorted lexicographically;
- one stated point order, preferably boundary order followed by sampler order;
- counter-clockwise triangles;
- rotate each triangle's index triple so the smallest index is first;
- lexicographically sort triangles only if doing so does not invalidate a
  separately documented locality order;
- include algorithm, sampler, predicate, and artifact schema versions in the
  cache key; and
- hash the canonical bytes, not nested language objects.

If the current sampler order is retained for stable golden fixtures, say so
and do not add a post-sort casually: triangulation tie-breaking can change.

### 4.6 Atlas, LOD, permutations, and seams

LOD is an edge subdivision count, always a positive power of two in the
resident atlas. LOD 1 is the source triangle itself. `[2, 8, 8]` requests two
segments on BC and eight on the other edges; it does not mean “LOD index 2.”

There are three distinct values which the UI must not conflate:

- **requested face/edge LOD:** what the scalar field or screen metric asked
  for;
- **reconciled shared-edge LOD:** the common value both incident faces must
  use on their shared edge; and
- **resident face LOD:** the atlas key after within-face grading promotes low
  edges so `max/min` is at most the selected 2:1 or 4:1 policy.

Shared-edge equality prevents cracks. The within-face 2:1 or 4:1 rule is a
quality/residency policy and is not mathematically required for a seam. The
demo should let users see requests, promotions, and resident values separately.

Sort an edge triple to obtain its canonical atlas key. One of the six S3
permutations maps canonical barycentrics back to the face-local order. Even
permutations are rotations; odd permutations are reflections and reverse
winding. The atlas needs one geometric entry per canonical key, not six copied
meshes.

The direct atlas is both a real topology source and educational content. The
hierarchical atlas uses blue noise only for irreducible roots and midpoint
subdivision for descendants. Therefore the honest answer to “is blue noise
plus Delaunay visual content, atlas generation, or both?” is:

- always visual content in the blue-noise and Delaunay ingots;
- the direct-atlas generation algorithm in the quality/background mode; and
- only root generation in the production hierarchical runtime path.

## 5. Proposed Fe ingot set

Keep the packages small enough to test independently. Names are proposed and
may change once checked against the workspace namespace, but their ownership
must not blur.

### 5.1 Pure foundation ingots

`ingots/quilting_domain`

- `Vec2F`, `Vec3F`, and `Bary3F` records;
- equilateral constants and Cartesian/barycentric conversion;
- `EdgeId::{BC, CA, AB}` and the opposite-corner convention;
- containment, normalization, and edge-parameter helpers; and
- no browser, worker, GPU, scene, or allocator effect.

`ingots/quilting_quaternion`

- fixed four-scalar quaternion add, subtract, multiply, conjugate, norm, and
  inverse;
- fail-closed or diagnostic behavior for a denominator below the declared
  epsilon;
- straight-line scalar arithmetic suitable for Wasm and shader lowering; and
- no general runtime GA blade machinery. The QB formula is small enough that
  a dynamic algebra representation would be overhead, not elegance.

`ingots/quilting_qb`

- `QbTriPatch`, `QbSample`, and `QbDifferential`;
- pure evaluation and analytic derivatives from section 4.2;
- explicit `(w,x,y,z)` layout;
- S3 barycentric remap and normal parity; and
- optional `ingots/quilting_quadratic_tri` only after the Q0 decision.

`ingots/quilting_atlas_contract`

- semantic `AtlasKey`, `AtlasPatchRange`, `AtlasManifest`, `AtlasVersion`, and
  `Permutation3` values;
- validators for positive powers of two, sorted canonical keys, in-range
  offsets, exact boundary counts, and complete permutation metadata;
- no triangulation implementation; and
- reflected byte layout shared by generated fixtures, Wasm workers, and
  WebGPU bindings.

`ingots/quilting_demo_state`

- closed demo mode, visual mode, grading policy, patch model, camera, selected
  handle, deterministic seed, and requested LOD records;
- a pure reducer from typed demo events to state plus explicit effects;
- canonical URL encode/decode and schema version;
- replay event records with logical step/generation rather than wall-clock
  dependence; and
- state admission/clamping in one place.

### 5.2 Visual ingots

`demos/sketches/classic_quilting_blue_noise`

- draw boundary samples, interior samples, and optional exclusion circles;
- color corners, edge samples, and interior samples distinctly;
- show seed, `k`, point count, minimum measured separation, and artifact hash;
- let the user switch direct Bridson versus uniform jittered comparison; and
- consume a fixed generated fixture first, then a worker result later.

`demos/sketches/classic_quilting_delaunay`

- reuse the exact point artifact from the previous ingot;
- reveal triangles progressively, with boundary constraints emphasized;
- optionally show selected circumcircles and triangle compactness;
- include a deliberately adversarial near-collinear fixture; and
- never triangulate independently in the shader.

`demos/sketches/classic_quilting_patch`

- evaluate one actual QB triangle over the Delaunay topology;
- show the parameter domain beside or overlaid on the 3D patch;
- provide three position and three quaternion-weight handles/controls;
- show analytic normals and a pole-conditioning/fade diagnostic;
- support wire, normal, barycentric, stretch, and simple material views; and
- optionally add the separately named six-control quadratic comparison.

`demos/sketches/classic_quilting_stitches`

- at least two source triangles with opposite orientation on one edge;
- exact shared boundary positions and independently different interior points;
- a seam mode which intentionally shows an unreconciled request, followed by
  the reconciled resident result;
- edge labels in face-local and canonical order; and
- zero shared-edge mismatches as a machine-visible assertion in normal mode.

`demos/sketches/classic_quilting_lod`

- one patch plane or small cube with a moving wave/radial/sweep field;
- requested, shared-edge reconciled, and resident LOD overlays;
- all six S3 permutations and odd/even parity visualization;
- 2:1 and 4:1 grading with promotion count and rendered triangle count; and
- an explicit direct-versus-hierarchical topology comparison.

`demos/capstones/classic_quilting_lab`

- composes the previous modes as named views over one semantic state;
- can step through them like slides without reloading unrelated resources;
- supports draggable controls in patch and stitched modes;
- exposes all relevant state in the URL and replay packet;
- loads a last-good cached atlas immediately and regenerates in the
  background when requested; and
- is the only package that knows the complete navigation between views.

### 5.3 Proposed Fe records

The syntax below is a design sketch to be adapted to the exact Fe compiler
surface. It is not permission to bypass type checking with generated text.

```fe
pub struct Bary3F { pub a: f32, pub b: f32, pub c: f32 }
pub struct Vec3F { pub x: f32, pub y: f32, pub z: f32 }
pub struct QuatF { pub w: f32, pub x: f32, pub y: f32, pub z: f32 }

pub struct QbControl { pub point: QuatF, pub weight: QuatF }
pub struct QbTriPatch {
    pub a: QbControl,
    pub b: QbControl,
    pub c: QbControl,
}

pub struct AtlasKey { pub a: u32, pub b: u32, pub c: u32 }
pub struct AtlasPatchRange {
    pub key: AtlasKey,
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_triangle: u32,
    pub triangle_count: u32,
}

pub enum ClassicView {
    BlueNoise, Delaunay, SinglePatch, StitchedPatches, LodPermutations,
}
pub enum ClassicVisual {
    Points, Wire, Material, Normals, Barycentric, Lod, Permutation, Seam,
}
pub enum Grading { TwoToOne, FourToOne }

pub struct ClassicState {
    pub view: ClassicView,
    pub visual: ClassicVisual,
    pub seed: u32,
    pub requested: AtlasKey,
    pub grading: Grading,
    pub selected_handle: i32,
    pub generation: u32,
}
```

Use const generics to specialize fixed shader/data shapes when the current Fe
compiler can prove them: patch order, control count, fixed small fixture point
count, fixed non-indexed expanded triangle count, and workgroup size. Runtime
controls choose among already admitted variants or request a new background
artifact; they do not turn every array into an unbounded dynamic value.

### 5.4 Current Fe raster constraint and the honest first implementation

At Fe commit `60834b9`, authored raster provides
`VertexStage<V, TriangleList<N>>`, `FragmentStage<V>`, `RasterVertex<V>`, and
source-ordered pass composition. It can already render fixed generated
topologies and small handle overlays. The QCGA pencil demo's 54-vertex marker
pass is the model for fixed control-handle quads.

It cannot yet own a runtime-sized indexed tessellation buffer. Therefore:

- M1 expands a small golden index buffer into a fixed non-indexed
  `TriangleList<N>` at generation time. This proves the Fe patch evaluator,
  varyings, wire/material modes, and controls.
- M2 may compile several fixed LOD variants. Inactive variants must not all
  consume full raster work; prefer separate admitted pass variants or host
  selection. Degenerate-vertex masking is acceptable only for tiny teaching
  fixtures and must be measured/labeled.
- M4 introduces the current Rust renderer or a new upstream Fe indexed-buffer
  draw policy for scalable runtime atlas rendering.

An upstream Fe request should ask for, without dictating internal design:

1. an indexed triangle draw policy with compiler-reflected vertex and index
   resources;
2. read-only external vertex/storage resources available to both vertex and
   fragment stages;
3. runtime draw ranges or a compiler-typed indirect draw description;
4. bundle reflection that preserves exact resource offsets, alignment, and
   entry-point visibility; and
5. browser actor support for those resources with device-loss reconstruction.

Until that exists, say “Fe owns the patch arithmetic and demo state; Rust owns
the scalable topology draw.” Do not claim a fully Fe-owned renderer.

## 6. Browser and effect boundary

The recommended ownership map is:

| Resource or decision | Sole owner |
| --- | --- |
| admitted demo state and reducer | resident Fe/Wasm actor |
| URL and replay semantics | Fe state ingot; browser adapter transports history operations |
| reference triangle, sampling, CDT, atlas validation | Rust Quilting worker code |
| worker pool, MessagePorts, transfer, restart | compiler-generated browser runtime |
| canvas and pointer capture | browser adapter under typed Fe surface intents |
| WebGPU adapter/device/queue/surface | one main/canvas-thread broker per device epoch |
| WebGL2 context and retained GL objects | Quilting WebGL backend per context epoch |
| immutable atlas CPU artifact | worker result/cache value, generation-tagged |
| GPU atlas buffers and pipelines | renderer backend per device/context epoch |
| current draggable controls | Fe state; GPU receives an immutable frame packet |
| diagnostics and benchmark samples | bounded observer, never semantic state |

Follow the MB2 actor precedent from
`demos/webgpu-qcga3d-quadric`: generated intent selects worker versus
main-thread effects, the domain wrapper does not restate request IDs or lane
sets, in-flight requests are rejected across restart epochs, and owned byte
buffers transfer rather than alias.

Do not replicate GPU handles, worker IDs, pipeline caches, or frame counters
into URL/replay state.

### 6.1 Interaction and picking

For three or six control handles, CPU/Wasm projected-handle picking is simpler
and lower latency than an ID-buffer readback:

1. project each handle with the current camera;
2. choose the nearest within a device-pixel-radius threshold;
3. capture the pointer;
4. map drag rays onto an explicit manipulation plane;
5. use a modifier or wheel for the plane normal/depth axis; and
6. dispatch one coalesced semantic update per animation frame.

The selected handle is a stable semantic `(patch, control, channel)` identity,
not a GPU vertex index. Draw its marker as a fixed authored raster overlay,
like `qcga_pencil_de::marker_vertices`.

If later selecting tessellated surface triangles, use a retained pick pass
with stable patch/source IDs and asynchronous readback. Tag the request with
the state and device generation; discard stale results. Never block the
interaction frame waiting for GPU readback.

Dragging a shared geometric corner updates every patch reference to that
corner. Dragging a face-local quaternion weight does not mutate neighboring
weights unless an explicit higher-order seam policy says it should.

### 6.2 URL state and replay

Every visible state needs a canonical field or an explicit reason for being
ephemeral. A suggested route is:

```text
?demo=quilting&view=stitched&visual=seam&seed=42
&lod=2,8,8&ratio=4&patch=qb&handle=1
&camera=...&weights=...
```

Requirements:

- parse through one versioned Rust/Fe admission schema, not scattered DOM
  reads;
- omit default values when canonicalizing;
- preserve unknown future fields only if the router contract explicitly does
  so;
- use decimal text with a stated float canonicalization policy;
- replay normalized semantic events, not raw pointermove frequency;
- record seed, algorithm version, fixture hash, and atlas generation intent;
- make pause/step/reset deterministic; and
- expose a machine-readable `window.__classicQuiltingAcceptance` only as
  bounded diagnostics, never as state authority.

## 7. Artifact and buffer ABI

Use an explicit little-endian, versioned binary schema. Avoid JavaScript object
graphs and nested arrays. All GPU storage records should be multiples of 16
bytes unless a reflected backend layout proves another portable stride.

Suggested narrow demo records:

| Record | Words / bytes | Fields |
| --- | ---: | --- |
| `AtlasVertex` | 4 / 16 | barycentric `a,b,c` as f32 bits; flags/source class u32 |
| `AtlasTriangle` | 4 / 16 | three u32 vertex indices; padding/flags |
| `AtlasPatch` | 8 / 32 | canonical key 3; first/count vertices 2; first/count triangles 2; flags/version 1 |
| `QbControl` | 8 / 32 | position quaternion vec4; weight quaternion vec4 |
| `QbPatch` | 32 / 128 | three controls = 96 bytes; patch ID, permutation, visual flags, padding = 32 bytes |
| `DrawInstance` | 8 / 32 | patch index, atlas patch index, permutation/parity, material/selection fields |

If using the current Quilting renderer instead of the narrow teaching path,
reuse `quilting_core::instance_layout` exactly: 52 f32 / 208 bytes per prepared
face record and a separate 4-byte visibility stream. Never maintain a second
copy of those offsets in Fe.

The artifact header must carry at least:

```text
magic, schema_version, algorithm_version, endianness,
master_seed, key_count, vertex_count, triangle_count,
patch_table_offset, vertex_offset, triangle_offset,
payload_hash
```

Validate all additions and multiplications for overflow before slicing. Reject
NaN/Inf barycentrics, zero LODs, unsorted canonical keys, out-of-range indices,
overlapping ranges, and nonzero reserved words. Publish only after the complete
artifact validates.

For uniforms, obey the adapter's reported minimum uniform-buffer offset
alignment. A 64-byte logical frame packet may require a 256-byte dynamic
stride. Storage-buffer struct alignment and vertex-buffer stride are separate
contracts; reflection must state each.

## 8. Construction DAG and worker parallelism

### 8.1 Static versus dynamic work

The default demo bundle should ship small golden fixtures for the teaching
views and either a build-time atlas or a versioned IndexedDB cache for the
capstone. The frame loop never performs sampling or Delaunay triangulation.

The complete construction DAG is:

```text
admit request
    -> enumerate and sort canonical keys
    -> derive irreducible roots and per-key deterministic seeds
    -> [sample root/key] x independent jobs
    -> [constrained triangulate root/key] x independent jobs
    -> validate each root boundary/topology
    -> hierarchical descendants, level by level where selected
    -> canonicalize records
    -> deterministic key-ordered merge
    -> validate complete atlas and compute hash
    -> pack transfer buffers
    -> cache immutable artifact
    -> upload into a candidate GPU residency
    -> atomically publish candidate if generation is still current
```

The renderer retains the previous valid artifact through every failed or stale
candidate.

### 8.2 Worker pool contract

Use a bounded pool, not one worker per patch and not an unbounded message fanout.

- Start with `min(max(hardwareConcurrency - 1, 1), 4)` workers as a policy
  ceiling, then tune from measurements. Low-core/mobile devices may use one.
- Compile/fetch the Wasm module once. Structured-clone the compiled
  `WebAssembly.Module` where supported; each worker owns its own instance and
  linear memory.
- One pool coordinator owns job IDs, generation IDs, queue admission, restart,
  deterministic merge, and publication.
- At most one generation is active and one latest generation is pending.
  Intermediate slider generations are coalesced.
- A request includes `generation`, algorithm/schema versions, ordered keys,
  build mode, grading ratio, master seed, and hard byte/work budgets.
- Worker results are immutable transferable `ArrayBuffer`s. Transferred buffers
  become detached at the sender; no code may retain a supposedly usable view.
- Results include their key and hash. The coordinator sorts by key rather than
  merge arrival order.
- Cancellation is cooperative at safe stage boundaries. A shared atomic cancel
  word is optional under `crossOriginIsolated`; generation rejection is still
  mandatory because cancellation can race completion.
- Worker death fails its assigned jobs with a typed epoch error. Restarting a
  worker increments its epoch and never reuses pending request identities.
- Backpressure counters include queued jobs, active jobs, coalesced requests,
  canceled jobs, stale results, transferred bytes, and peak retained bytes.

`SharedArrayBuffer` is not a v1 requirement. It needs COOP/COEP deployment,
atomic publication, and explicit ownership of partially written regions. For
these moderate immutable artifacts, transfer is simpler and often fast enough.
Use shared memory only after transfer and peak-memory traces show a real
problem.

### 8.3 What to parallelize

Good axes:

- independent direct canonical keys;
- independent irreducible hierarchical roots;
- constrained triangulation after each root's sample is complete;
- independent validation/hash tasks for large artifacts; and
- independent build-time fixture generation.

Bad axes:

- the active-list iterations inside one Bridson sample, if bit-replayable output
  matters;
- one small Delaunay triangulation split across workers;
- child hierarchy levels before their parent exists;
- N workers each calling current hierarchical `build_subset`, because it
  rebuilds the full hierarchy per worker; and
- any work whose transfer/merge cost exceeds its computation.

Refactor Rust atlas construction, if measurement justifies it, around an
explicit `plan -> root jobs -> derive descendants -> merge` API. Do not hide
browser coordination inside `TessellationAtlas`.

### 8.4 Deterministic seeds and reductions

Derive each independent seed from a stable integer hash of:

```text
master seed, canonical key, build mode, root depth, sampler version
```

Do not assign seeds by worker index or completion order. Use an explicitly
specified 64-bit mixing function and golden vectors.

Every parallel reduction must define its order. Merge keys lexicographically;
append point and triangle buffers in that order; rebase indices with checked
integer arithmetic; and hash after final packing. Two pool sizes must produce
identical artifact bytes for the same request.

## 9. WebGPU execution design

WebGPU is a scalable backend and a teaching opportunity, not a mandate to move
every algorithm into a workgroup.

### 9.1 Warm-frame mapping

For the small patch demos, evaluate the QB surface directly in the vertex
shader. Each submitted atlas vertex reads barycentrics and one patch instance,
evaluates position plus analytic normal, and emits varyings. A separate compute
pass would add buffer traffic and synchronization without reuse.

For a larger stitched scene, the current Quilting mapping is the model:

- LOD pass 1: one invocation per source face, workgroup size 64;
- LOD reconciliation/canonicalization: one invocation per face, reading three
  neighbors and the atlas lookup;
- preparation: one invocation per resident patch/leaf when a reusable prepared
  record is justified;
- eligibility/count: one invocation per source instance;
- prefix/compaction: stable scan and scatter into resident ranges;
- indirect arguments: one fixed 20-byte indexed record per draw batch; and
- render: vertex pulling from the canonical atlas plus resident patch records.

Workgroup size 64 is a current validated baseline, not a universal optimum.
Compile-time-specialize 32/64/128 variants only if adapter measurements warrant
them. Include workgroup size in shader/pipeline identity.

Compute dispatch must use `ceil_div(item_count, workgroup_size)`, check
`maxComputeWorkgroupsPerDimension`, and chunk oversized dimensions. Shader
bounds checks remain required.

### 9.2 Scan and compaction

A stable compaction pipeline normally needs:

1. per-item eligibility;
2. per-workgroup local scan/count;
3. scan of workgroup totals;
4. stable scatter using group offsets; and
5. indirect/range record construction.

The current shaders deliberately use a serial `@workgroup_size(1)` scan for a
small batch table and 64-lane scans inside each batch. That is a sensible
measured simplification while batch count is small. Do not replace it with a
complex global scan until batch-count traces justify the work.

Keep `first_instance = 0` in portable indirect records unless the optional
indirect-first-instance capability is explicitly admitted. The vertex stage
can combine the compacted range prefix with local instance index, as current
Quilting does.

### 9.3 GPU atlas-generation experiment

This is a later experiment, not the first implementation.

Bridson sampling has a sequential active list and conflict-dependent RNG
consumption. A workgroup implementation would be a different algorithm unless
it preserves those semantics. A plausible GPU-native blue-noise experiment is
round-based deterministic candidate generation:

- one workgroup per canonical root or spatial tile;
- integer/fixed-point counter-based random candidates keyed by
  `(seed, round, candidate)`;
- spatial-bin counts and prefix offsets;
- conflict marking against neighboring bins;
- deterministic tie-break by candidate ID;
- parallel scan/compaction of accepted candidates; and
- bounded rounds with an explicit completion/quality report.

This can be statistically and visually excellent, but it should receive a new
algorithm version and its own golden invariants. Floating GPU tie-breaking is
not assumed bit-identical across adapters.

Keep robust constrained Delaunay on CPU/Wasm initially. A GPU Delaunay
implementation is branchy, mutation-heavy, and predicate-sensitive. Moving
sample points GPU -> CPU -> GPU may cost more than the saved sampling time.
Measure candidate-generation time, readback bytes/time, CDT time, and reupload
before proceeding.

### 9.4 WebGL2 fallback

The fallback is the current Quilting model:

- atlas generated/cached off-thread;
- packed barycentrics and indices uploaded once;
- canonical S3 entry instanced over faces;
- transform feedback or CPU/Wasm for preparation/LOD where required;
- optional multi-draw extension, otherwise deterministic grouped draw ranges;
- vertex-stage conservative rejection saves raster/fragment work but not
  vertex invocation; and
- no attempt to emulate general WebGPU compute in WebGL2.

The same immutable demo state and atlas artifact feed both backends. Backend
resources never cross context/device epochs.

### 9.5 Device and context loss

On WebGPU `device.lost` or WebGL context loss:

- stop new submissions for that epoch;
- reject or mark all pending frame/pick completions with the old epoch;
- retain semantic state and immutable CPU/cache artifacts;
- destroy/release what the API permits;
- request a new adapter/device/context under the typed recovery policy;
- rebuild pipelines and GPU buffers from the last validated artifact; and
- publish the new epoch only after a complete drawable residency exists.

Never continue using a pipeline, buffer, bind group, VAO, or query from an old
epoch. The fallback badge must distinguish WebGPU unavailable, device lost and
recovering, WebGL fallback active, and CPU-only educational mode.

## 10. Per-frame and asynchronous ownership

The warm frame should contain only:

- apply at most one coalesced Fe state transition;
- advance an optional deterministic demo clock;
- update the small camera/control/frame packet only if changed;
- select an already resident atlas range/LOD snapshot;
- encode visibility/LOD compute only for invalidated epochs;
- encode draw and optional pick/overlay passes; and
- submit/present.

The warm frame must not contain:

- blue-noise sampling;
- Delaunay triangulation;
- atlas serialization or IndexedDB access;
- shader compilation after residency;
- synchronous GPU readback;
- nested JavaScript array flattening;
- URL parsing; or
- unbounded diagnostic allocation.

Asynchronous tasks include atlas generation, cache read/write, device recovery,
golden verification, and optional delayed telemetry. Each publishes through a
generation/epoch fence.

## 11. Error model and last-good publication

Use a structured error with at least:

```text
domain: Request | Sampling | Triangulation | Validation | Packing |
        Worker | Cache | Upload | Shader | Device | Context
code: stable machine-readable reason
generation: request generation
key: optional canonical atlas key
retryable: boolean
public_message: bounded non-sensitive text
diagnostic: bounded developer detail
```

Examples include invalid LOD, work budget exceeded, nonfinite radius, missing
boundary corner, CDT failure, index overflow, malformed transfer, stale
generation, worker restart, cache version mismatch, shader validation failure,
and device epoch loss.

Construction is transactional:

1. build candidate CPU artifact;
2. validate candidate completely;
3. create candidate GPU resources;
4. validate required pipelines/bindings;
5. check generation and device epoch again; and
6. atomically replace the resident artifact.

Any failure leaves the old resident demo drawable. UI status reports that a
new request failed; it does not blank the canvas or install a partial atlas.

## 12. Performance budgets and instrumentation

These are initial targets and stop gates, not claims. Record hardware, browser,
backend, build profile, atlas request, cache state, and adapter limits with each
result.

### 12.1 User-visible targets

- pointer-to-handle feedback: p95 at or below one 60 Hz frame;
- Fe reducer plus frame packet construction: p95 below 1 ms for the teaching
  scene;
- unchanged frame: zero atlas/control uploads and zero CPU/GPU readback;
- cached first teaching view: useful paint target below 250 ms on the reference
  presentation machine;
- cold direct atlas request: stays interactive, reports progress, and never
  blocks the main thread longer than 8 ms;
- stale slider generations: never become resident;
- device/context recovery: restores the last-good artifact without semantic
  state loss.

### 12.2 Measurements to expose

For atlas generation:

- key/root count;
- sample, triangulate, validate, derive, pack, transfer, merge, hash, cache,
  and upload milliseconds;
- per-worker active/queue time;
- points, triangles, bytes, and peak retained bytes;
- canceled/coalesced/stale job counts;
- minimum and p1 triangle compactness;
- boundary mismatch count; and
- complete artifact hash.

For frames:

- reducer time, CPU encode time, queue-write bytes, number of dispatches, draw
  calls, submitted vertices/instances/triangles, and pipeline memo hits/misses;
- GPU timestamps when supported, with unsupported clearly distinguished;
- pick request/readback latency only when a pick occurs; and
- context/device epoch and recovery count.

Use `performance.mark/measure` or compiler-generated equivalent for browser
stages and bounded rolling histograms for long sessions. Avoid console logging
inside hot loops. Never wait on a GPU fence merely to populate ordinary UI
stats.

### 12.3 “GPU is faster” acceptance rule

No path is promoted because its isolated kernel is faster. Compare end to end:

```text
CPU/Wasm compute + transfer + upload
versus
GPU dispatch + intermediate buffers + synchronization + any readback
```

For small fixtures, CPU or vertex-shader evaluation may win. For large face
classification and resident compaction, GPU residency can win by eliminating
round trips. Measure both.

## 13. Tests and golden fixtures

### 13.1 Pure math tests

- Cartesian/barycentric round trip at corners, edges, centroid, random
  interiors, and near-boundary values;
- barycentrics sum to one after snapping/renormalization;
- identity-weight QB patch is planar;
- QB CPU and Fe evaluators agree on a fixed grid within a declared f32/f64
  tolerance;
- analytic tangents agree with central differences away from the pole;
- normal orientation flips exactly for odd S3 permutations;
- denominator-near-zero values produce finite bounded diagnostic output; and
- optional quadratic patch reproduces corners/edge curves and analytic
  derivatives.

### 13.2 Sampling and triangulation tests

- seed 42 produces byte-identical samples for repeated runs;
- pool sizes 1, 2, and 4 produce identical final artifact bytes;
- every point lies in the equilateral domain;
- boundary points are present exactly once with exact expected parameters;
- no pair violates the specified variable-radius conflict rule;
- all triangle indices are in range and all retained triangles have positive
  orientation/nonzero area;
- no constrained boundary segment is crossed;
- no triangle centroid lies outside the domain;
- near-collinear, cocircular, duplicate, signed-zero, and very different edge
  resolution fixtures return a deterministic result or a typed rejection; and
- direct and hierarchical modes state different expected interior topology but
  identical boundary counts.

### 13.3 Atlas and seam tests

- canonical round trip for every permutation of `[2,4,8]`;
- parity matches geometric winding for all six S3 elements;
- exact boundary count for `[1,1,1]`, `[1,1,2]`, `[1,2,2]`, `[2,4,8]`, and
  `[4,16,16]`;
- two oppositely oriented patches emit identical shared-edge parameter points;
- shared-edge reconciliation produces zero mismatches;
- 2:1 and 4:1 promotion counts are frozen on a small teaching mesh;
- atlas serialize/deserialize is byte-stable under the schema version; and
- malformed offsets, counts, reserved words, and hashes fail closed.

### 13.4 Cross-backend tests

- Rust f64 oracle, Fe/Wasm f32, WGSL, and WebGL-lowered GLSL compare sample
  positions/normals under an explicit tolerance and association policy;
- fixed teaching images compare canonical RGBA origin/channel order with a
  documented pixel tolerance;
- wire and material modes render nonempty and distinct images;
- no warm-frame readback occurs;
- WebGL and WebGPU consume the same artifact hash and command ordering;
- shader/program memo hit, miss, and epoch invalidation counters are tested;
- WebGPU absence activates the honest fallback without a red error;
- device/context loss reconstructs from the last-good CPU artifact; and
- stale worker, pick, and GPU completions cannot overwrite newer state.

### 13.5 Browser and replay tests

- every named view reloads from its canonical URL;
- a handle drag, weight edit, LOD policy change, pause/step, and reset replay to
  the same state/artifact hash;
- rapid controls produce one active plus one latest pending atlas generation;
- a worker restart rejects its in-flight request and a later request succeeds;
- `pagehide` closes worker/device resources;
- pointer capture release cannot leave a handle stuck selected; and
- keyboard-only selection and adjustment have a documented path.

Golden artifacts should live under a versioned fixture directory with a
manifest naming source commit, generator version, seed, keys, hashes, point and
triangle counts, and license. Do not check in an opaque binary without its
generator command and independent validation summary.

## 14. Milestones and stop gates

### Q0: settle the patch model

Deliverable: one written decision: actual three-control quaternionic-Bézier is
required; conventional six-control quadratic is optional or separately
required.

Stop if terminology remains ambiguous.

### M0: freeze artifacts and ABI

- wrap current Rust sampler/CDT/atlas in a `Result`-returning fixture exporter;
- emit the small keys and adversarial fixtures from section 13;
- specify and validate the binary ABI;
- record hashes and benchmark stage times; and
- make pool-size determinism green.

Stop if exact boundary invariants or deterministic bytes cannot be reproduced.

### M1: fixed Fe teaching views

- land `quilting_domain`, `quilting_quaternion`, and `quilting_qb`;
- import one generated topology as an expanded fixed `TriangleList<N>`;
- render blue-noise, Delaunay, and single-patch views;
- compare Fe/Wasm, WGSL, and Rust vectors; and
- add wire/material/normal modes.

Stop if current Fe authored raster cannot express the fixed topology without a
compiler hack or unreflected resource.

### M2: Fe-native interaction and replay

- add typed surface state, parameter bindings, reducer, URL codec, and replay;
- add projected handle picking and fixed marker raster pass;
- drag positions and adjust quaternion weights; and
- prove no main-thread semantic JavaScript state.

Stop if pointer coalescing or state ownership requires a second JS reducer.

### M3: stitched patches and LOD explanation

- add the shared-edge fixture;
- visualize requested/reconciled/resident values;
- show all S3 permutations and parity;
- support 2:1/4:1 policies; and
- make seam and promotion assertions machine-visible.

Stop if visible geometry and the diagnostic LOD colors are not driven from the
same resident record.

### M4: bounded worker generation

- add compiler-derived worker intents and the generation-tagged ABI;
- direct-generate one root/key in the background;
- add coalescing, cancellation, restart, deterministic merge, and last-good
  publication;
- cache the complete artifact by schema/algorithm/request key; and
- measure one versus multiple workers before choosing a default.

Stop if hierarchical pool jobs repeat the full build or peak memory regresses
relative to one worker.

### M5: scalable renderer integration

Choose one honestly:

- integrate the current Rust Quilting WebGL2/WebGPU render contract and treat
  Fe as state/shader specialization; or
- consume an upstream Fe indexed/external-buffer raster surface after its
  independent tests exist.

Do not fork a second TypeScript renderer.

### M6: GPU parallel experiments

- move face LOD/visibility/compaction to the retained WebGPU path;
- retain zero-readback warm frames;
- benchmark workgroup variants and scans; and
- only then prototype a separately versioned GPU blue-noise algorithm.

Stop if transfer/readback makes the end-to-end GPU path slower or if robust
predicate semantics are weakened.

### M7: release-quality gallery

- browser/device matrix, fallback, recovery, accessibility, attribution,
  canonical routes, replay fixtures, benchmark record, and artifact preflight;
- no dirty generated assets; and
- explicit label of which pieces are Fe-authored, Rust-authored, generated,
  WebGL2, WebGPU, or fallback.

## 15. First-agent task packet

The next agent should execute these steps in order and stop at each named gate.

1. **Preserve worktrees.** Record `git status` in both repositories. Do not
   clean, reset, or absorb the known unrelated MB2 dirt. Create a dedicated
   branch/worktree if authorized. Commit only coherent slices.
2. **Re-read the pinned sources.** Use the source map below and confirm no
   upstream Fe or Quilting change has superseded a stated limitation.
3. **Close Q0.** Add a short decision note to the implementation PR/commit:
   current QB first, six-control quadratic optional and separately named.
4. **Define the artifact ABI in tests first.** Add Rust structs/checked codec in
   the narrowest appropriate crate or a fixture-export tool. Do not expose
   `HashMap` iteration order or native `usize` in the bytes.
5. **Generate four tiny fixtures.** Seed 42 keys `[1,1,1]`, `[1,1,2]`,
   `[1,2,2]`, and `[2,4,8]`, plus one near-degenerate rejection fixture.
   Record counts, hashes, boundary parameters, and the exact command.
6. **Create `quilting_domain` and `quilting_quaternion`.** Make pure Fe/Wasm
   oracle tests green before adding a shader.
7. **Create `quilting_qb`.** Implement the actual three-control formula and
   analytic derivatives. Compare a barycentric grid against
   `QBTriPatch::eval_differential`.
8. **Prove one fixed Fe raster.** Expand the smallest fixture to a generated
   non-indexed `TriangleList<N>`. Render position/normal colors through current
   authored raster. Run the existing authored-raster Naga/lavapipe/wasmtime
   style gates.
9. **Add blue-noise and Delaunay views.** These consume the same fixture; they
   do not each regenerate it. Add boundary/interior classification and one
   selected-triangle diagnostic.
10. **Add Fe state and interaction.** Follow current typed `Surface`, `Param`,
    `SurfaceTransition`, `LatestPerFrame`, and fixed raster marker patterns.
    Keep the domain wrapper free of request IDs, MessageChannels, and placement
    restatement.
11. **Add stitched and LOD fixtures.** Drive visible tessellation and colors
    from one resident key/permutation record. Freeze seam/promotion assertions.
12. **Instrument before workers.** Establish fixed-fixture frame and cold Rust
    generation baselines.
13. **Add one worker lane.** Reuse compiler-derived structured worker runtime;
    add generation, epoch, bounded queue, transfer, typed error, and last-good
    publication. Test cancellation/restart before adding a pool.
14. **Refactor root-job planning if needed.** Never use current hierarchical
    `build_subset` in multiple workers. Add pool width only after independent
    roots exist and one-worker evidence is recorded.
15. **Make the scalable renderer choice at M5.** If upstream Fe indexed raster
    is still absent, integrate Quilting rather than inventing JS buffer
    semantics.
16. **Defer GPU sampling.** First establish resident WebGPU LOD/compaction and
    no-readback rendering. GPU blue noise is a separately versioned experiment.
17. **Commit each green rung.** A fixture/ABI commit, Fe math commit, fixed
    raster commit, interaction commit, stitched/LOD commit, and worker commit
    should remain individually reviewable and revertible.

## 16. Dependency decisions

| Need | Decision | Reason |
| --- | --- | --- |
| Variable-density sampling | reuse `distressed-blue-noise` | deterministic PCG, domain and seed tests already exist |
| Constrained Delaunay | reuse current Rust `cdt` path | exact orientation/incircle predicates; do not port Delaunator or write WGSL predicates |
| Atlas/S3 | reuse `quilting-core` | canonicalization, direct/hierarchical modes, boundary and parity tests |
| Seam topology | reuse `quilting-mesh` | canonical half-edge identity and exact seam welding |
| QB CPU oracle | reuse `quilting-core::patch` | current semantic authority and analytic derivatives |
| QB Fe shader arithmetic | small dedicated Fe ingot | fixed straight-line quaternion math; no runtime GA machinery needed |
| General GA shader specialization | optional later use of `ga_expr` | useful for conformal experiments, not required to evaluate this QB formula |
| WebGL2 render | reuse `quilting-renderer` for scalable rung | current authority; retained resources and WGSL-to-GLSL path |
| WebGPU render/compute | reuse and gate `quilting-webgpu` | deep conformance work exists, but browser promotion remains evidence-gated |
| Browser workers | reuse compiler-generated MB2 actor runtime | typed placement, restart epochs, bounded/canonical interface precedent |
| Classic npm libraries | do not adopt | duplicated authority, nondeterminism, copies, global state, licensing surface |
| Precomputed atlas | ship small fixtures; cache larger artifacts | fastest reliable first view; generator/version remains available |

## 17. Risks and non-goals

Highest risks:

- building a six-control quadratic demo while believing it is current QB;
- overstating current Fe dynamic raster/resource support;
- duplicating hierarchical atlas work across workers;
- nondeterministic merge or GPU floating tie-breaking breaking replay;
- copying an AGPL/CC-NC example instead of reimplementing from current Rust;
- making direct high-LOD atlas generation part of every load;
- creating another JS scene/state owner to work around Fe boundaries;
- moving CDT to GPU and losing exact predicates; and
- publishing partial worker or device results over a last-good atlas.

Non-goals for the first capstone:

- arbitrary glTF scene loading;
- mesh simplification or QB patch fitting;
- general higher-order rational patch fitting;
- full Hyperscape navigation, focus, fuzzy vision, or HHHS sync;
- a new TypeScript renderer;
- a universal GPU Delaunay implementation;
- mobile performance claims before a measured device matrix; and
- making WebGPU the default backend before existing parity gates pass.

## 18. Exact source map

### 18.1 Classic TypeScript Quilting

Inspect with `git show <commit>:<path>` from `/laboratory/quilting`.

| Commit/path | Symbols or evidence | Use |
| --- | --- | --- |
| `a565a4c:src/tessellation.ts` | `triEdgeWeightInterpolator`, `triPatch`, `tessellation`, `prepareMesh`, `tessellationMesh` | boundary-first variable-density sampling and Delaunay pipeline |
| `a565a4c:src/load-tessellation-atlas.ts` | `loadTessellationAtlas` | worker fanout, S3-reduced canonical atlas, combined ranges |
| `a565a4c:src/permutator.ts` | `permutationIndices3`, `vertPermFromEdgePerm` | six orientations and old edge/vertex mapping |
| `a565a4c:src/tessellation-worker.ts` | exposed `tessellationMesh` | minimal historical worker boundary |
| `a565a4c:scripts/generate-tessellations.ts` | `generateTessellations` | offline pack experiment, 12-bit coordinate quantization, permutation metadata |
| `a565a4c:examples/tessellation.ts` | pointer-neighborhood LOD stream and range grouping | classic interactive LOD affordance |
| `a565a4c:examples/terrain/terrain.ts` | periodic moving LOD field, shared-edge max, instanced atlas | culminating visual reference |
| `a565a4c:src/transformer.ts` | `cellsTransformer` | historical transform-feedback positions/weights/LOD experiment; do not reuse readback getters |
| `6e461da:gpu/gl/snippets/patches.glsl` | `bilinearTri`, `bilinearQuad` | early quaternion/CGA rational patch visual experiment |
| `6e461da:src/bernstein.ts` | `B`, `Btri`, `BxB` | historical Bernstein helpers and evidence that degree/order must be named |
| `6e461da:src/lod-patches.ts` | `triPatch`, `quadPatch` | early boundary-seeded variable-density ideas |
| `f324070` | `use workers to compute tessellations` | worker milestone |
| `5ef85c1` | `tessellation achieved!` | topology milestone |
| `33b6779` | `tessellation atlas example` | atlas-demo milestone |

### 18.2 Current Rust Quilting

All paths below were inspected at `cd9f810` unless a later implementation
commit deliberately supersedes them.

| Path | Symbols | Use or caution |
| --- | --- | --- |
| `crates/quilting-core/src/triangle.rs` | `VERTICES`, `cartesian_to_bary`, `bary_to_cartesian` | normative parameter domain |
| `crates/distressed-blue-noise/src/lib.rs` | `Domain`, `SamplerConfig`, `PoissonSampler::sample`, `sample_jittered` | deterministic sampling; declared parallel feature has no sampler parallelism |
| `crates/quilting-core/src/sampling.rs` | `PatchConfig`, `tri_patch`, `tri_patch_jittered` | exact boundary seeding and barycentric snap |
| `crates/quilting-core/src/delaunay.rs` | `triangulate_2d_constrained` | contour/CDT wrapper; browser facade must replace panics with errors |
| `crates/quilting-core/src/permutation.rs` | `S3_PERMUTATIONS`, `canonical_form`, `perm_sign`, `remap_position` | exact symmetry/parity authority |
| `crates/quilting-core/src/atlas.rs` | `BuildMode`, `build_for_keys`, `build_hierarchical_for_keys`, `build_subset`, `merge_from` | atlas authority; do not multi-worker current hierarchical subset |
| `crates/quilting-core/src/patch.rs` | `QBTriPatch`, `eval`, `eval_differential`, `restrict`, `transform` | actual three-control QB definition |
| `crates/quilting-core/src/instance_layout.rs` | `STRIDE_BYTES`, `ATTR_MAP`, offsets | normative scalable renderer ABI |
| `crates/quilting-core/src/educational.rs` | `PatchLabMesh`, `PatchLabField`, `PatchLabLodResult` | reusable teaching topology and accounting |
| `crates/quilting-mesh/src/lib.rs` | `HalfEdgeMesh::canonical_edge`, `from_triangles_welded_exact` | seam topology |
| `crates/quilting-shaders/shaders/surface/qb_eval.wgsl` | `eval_qb`, `eval_qb_with_normal` | shader formula and parity |
| `crates/quilting-shaders/shaders/compute/lod_pass1.wgsl` | `classify_lod_pass1`, workgroup 64 | current GPU face classifier model |
| `crates/quilting-shaders/shaders/compute/lod_pass2.wgsl` | `classify_lod_pass2` | neighbor reconciliation and canonical atlas lookup |
| `crates/quilting-shaders/shaders/compute/visibility_scan.wgsl` | `scan_visible_batches` | intentionally serial small-batch prefix |
| `crates/quilting-shaders/shaders/compute/visibility_scatter.wgsl` | `scatter_visible_instances` | 64-lane stable batch scatter |
| `crates/quilting-wasm/src/lib.rs` | `build_required_atlas`, `export_all_patches`, `build_atlas_subset`, `merge_atlas_bytes` | current bridge to wrap/refine |
| `hyperscope_worker.js` | atlas messages and animated LOD generations | incumbent protocol evidence, not Fe architecture |
| `hyperscope.html` | one-worker atlas policy, debounced live replacement, cache | evidence that repeated helper initialization outweighed current small-root parallelism |
| `docs/runtime-render-pipeline.md` | “Work outside the frame loop,” atlas policy, WebGPU sequence | current directional architecture |
| `crates/quilting-core/examples/bench_runtime_atlas.rs` | policy/topology benchmark | repeatable baseline tool |

### 18.3 Fe/MB2

The source worktree is `/laboratory/fe-stuff/fe-worktrees/mb2` at `60834b9`;
the architecture docs are under `/laboratory/fe-stuff/mb2/docs` at `7302ce1`.

| Path | Evidence | Consequence |
| --- | --- | --- |
| `ingots/std/src/webgpu.fe` | `GpuProgram`, `FragmentSurface`, `VertexStage`, `FragmentStage`, `TriangleList`, `Dispatch`, `WorkerGpu` | fixed authored raster and typed GPU/worker effect vocabulary exist; dynamic indexed raster does not |
| `demos/sketches/qcga_pencil_de/src/lib.fe` | `marker_vertex`, `marker_vertices`, `navigate` | model for Fe-owned transitions plus fixed handle overlay |
| `crates/codegen/tests/fixtures/actor_raster_typed/src/lib.fe` | typed varying and `TriangleList<3>` | smallest fixed patch render precedent |
| `crates/codegen/tests/fixtures/actor_layered_raster/src/lib.fe` | source-ordered fullscreen plus overlay | composition precedent |
| `crates/codegen/tests/authored_raster_e2e.rs` | Naga, wgpu/lavapipe, Wasmtime checks | verification style to extend |
| `demos/webgpu-qcga3d-quadric` | generated canonical actor, restart and exact worker/GPU evidence | worker ownership/restart precedent |
| `docs/mb2-renderer-in-fe-spec.md` | fixed Fe render-stage design and byte-exact evidence discipline | do not skip backend-independent oracle |
| `docs/mb2-fewasm-webgpu-architecture.md` | imports-as-capability op set and honest JS linkage | browser adapter must be mechanical, not semantic |
| `docs/mb2-wasm-worker-webgpu-interop.md` | one device broker, worker placement, transfer and suspension rulings | concurrency boundary |
| `docs/mb2-webgpu-browser-design.md` | WebGPU backend distinctions, build ladder, honesty split | do not overclaim compiler or browser support |
| `/laboratory/quilting/docs/fe-webgl-ga-prototype.md` | Fe as optional shader-specialization frontend | Fe must not become a second scene/renderer authority |

## 19. Final architectural test

The implementation is on course if the following sentence remains true:

> One deterministic semantic demo state selects one validated immutable atlas
> artifact and one backend-neutral frame; Fe explains and specializes the patch
> math, Rust owns topology and scalable rendering, workers build only immutable
> background candidates, and WebGL2 or WebGPU lower the same intent under an
> explicit resource epoch.

If an implementation needs a second JavaScript scene graph, a second
triangulator, six copied permutation meshes, per-frame atlas generation,
synchronous readback, or a false “quadratic QB” equivalence, stop and return to
the contracts above.
