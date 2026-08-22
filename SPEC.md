# Quilting: Posthoc Specification

## 1. Vision

Quilting renders triangle meshes under conformal (Mobius) transformations in real time on the web. It is, in short, a way to take any 3D model and warp it through sphere inversions, reflections, and other angle-preserving maps of 3-space — live, interactively, with full PBR materials.

The name comes from the technique: each triangle face of the input mesh gets its own pre-tessellated "quilt patch," a small triangle sub-mesh whose density adapts to how much the conformal transform curves that region of space. These patches are stitched together at their edges (matching LODs guarantee no T-junction cracks) and rendered via instanced draw calls. The GPU evaluates quaternionic Bezier (QB) surfaces per-vertex — the rational Bezier weights encode the Mobius transform — so the CPU only needs to update a few uniform quaternions per frame.

The project exists because WebGL2 has no geometry shaders and no tessellation shaders. You can't generate sub-triangle detail on the fly. Quilting works around this by pre-computing a tessellation atlas (a dictionary of patch geometries keyed by edge LOD triples) and stamping those patches onto each face at render time. The conformal math ensures the patches curve correctly through the Mobius warp.

The long-term vision: **Hyperscope**, "an environment for cartoons" — a full glTF viewer where arbitrary 3D models can be loaded, animated via spacetime slicing (4D hyperplane intersection of animated meshes), and deformed through conformal transformations with production-quality PBR rendering.

### History

The original implementation was TypeScript/GLSL on the `main` branch. A cleaner rewrite was lost when Micah's laptop was left on BART in 2022. The current codebase is a Rust/WASM rewrite on the `rust` branch, using WGSL shaders compiled to GLSL ES 300 via naga.

## 2. Core Math

### Quaternions as 3D Points

Following Krasauskas & Zube, R^3 is identified with the imaginary quaternions Im(H). A 3D point (x, y, z) is the pure quaternion `0 + xi + yj + zk`. The `Quat` type stores `(w, x, y, z)` with `w` as the scalar (real) part.

Quaternion layout convention (critical — must be consistent everywhere):

- Rust: `Quat { w, x, y, z }` where `w` is real, `(x, y, z)` is imaginary
- WGSL/GLSL: `vec4(w, x, y, z)` — scalar in `.x`, vector in `.yzw`
- Instance data packing: `[w, x, y, z]` per quaternion — 4 floats. The one
  exception is the three position slots, whose first component carries a
  vertex index for GPU skinning instead of a scalar part (positions are pure
  imaginary, so the slot was free). See `quilting-core::instance_layout`.

The `q_to_point` function extracts `.yzw` (the imaginary part) to get the 3D position.

### Mobius Transforms

A Mobius transformation in R^3 is:

```
F(x) = (a*x + b) * (c*x + d)^{-1}
```

where `a, b, c, d` are quaternions, `x` is a pure imaginary quaternion (3D point), and multiplication/inversion are quaternion operations.

Represented as a 2x2 quaternion matrix `[[a,b],[c,d]]`. Composition is matrix multiplication. Key transforms:

- **Identity**: `a=1, b=0, c=0, d=1`
- **Translation by t**: `a=1, b=t, c=0, d=1`
- **Rotation qxq-bar**: `a=q, b=0, c=0, d=q` (for unit q)
- **Sphere reflection** (center c, radius r): `a=c, b=-(c^2+r^2), c=1, d=-c`
- **Sphere inversion**: compose two sphere reflections (orientation-preserving)

The `c` coefficient determines whether the transform is affine (`c=0`, no curvature) or truly conformal (`c!=0`, introduces curvature requiring tessellation). The `is_affine()` check tests `|c|^2 < 1e-20`.

### Conformal Weight Transform

Under a Mobius transformation, a QB weight at position `x` transforms as:

```
w' = (c*x + d) * w
```

This is equation 5 from Krasauskas & Zube 2014. The conformal weight encodes how much the transform stretches space at that point. Near the Mobius pole (where `c*x + d -> 0`), the weight goes to zero and space stretches to infinity.

### Quaternionic Bezier (QB) Surface Evaluation

A QB triangle surface evaluates a point at barycentric coordinates `(lambda_0, lambda_1, lambda_2)`:

```
X(bary) = (sum_i lambda_i * p_i * w_i) / (sum_i lambda_i * w_i)
```

where `p_i` are position quaternions and `w_i` are weight quaternions. This is a rational quaternion combination. When all weights are identity (`w = 1`), this reduces to linear interpolation of the positions. Under Mobius transforms, the weights encode the conformal curvature.

Normals are computed analytically via the quotient rule:

```
dX/du = (dtop_u - X * dbot_u) * bot^{-1}
dX/dv = (dtop_v - X * dbot_v) * bot^{-1}
normal = cross(dX/du, dX/dv)
```

where `u` and `v` are the barycentric tangent directions (bary.x->bary.y and bary.x->bary.z).

### Key Papers

- Zube 2013: Quaternionic Bezier curves, surfaces and volume (foundational)
- Krasauskas & Zube 2014: Rational Bezier Formulas with Quaternion and Clifford Algebra Weights (the core technique)
- Krasauskas & Zube 2015: Representation of Dupin cyclides using quaternions
- Zube 2018: Interpolation method for quaternionic-Bezier curves
- Krasauskas & Zube 2024: Quaternionic Bezier parameterizations of bidegree (2,1)

## 3. Architecture

### Crate Structure

```
quilting/
  crates/
    quilting-core/          # Math, tessellation atlas, LOD, evaluation, instance layout
    quilting-mesh/          # Half-edge mesh data structure
    quilting-shaders/       # WGSL shader modules, naga compilation to GLSL
    quilting-renderer/      # glow-based WebGL2 renderer (UBOs, VAO cache, draw calls)
    quilting-gltf/          # glTF/GLB loader (meshes, materials, animations, skins)
    quilting-remesh/        # VSA clustering + QB patch fitting (dense mesh -> curved patches)
    quilting-wasm/          # WASM entry point (wasm-pack, JS interop)
    distressed-blue-noise/  # Variable-density Poisson disk sampler (Bridson)
    fuzzy-vision/           # JFA-based variable per-pixel blur post-process
    trunk-stub/             # Placeholder cdylib so Trunk has something to build
  hyperscope.html           # Primary frontend (Hyperscope app)
  hyperscope_worker.js      # Web worker for WASM computation
```

`quilting-wasm` and `trunk-stub` are excluded from the workspace's
`default-members`: the former only compiles for `wasm32` (it calls
`glow::Context::from_webgl2_context`), so a bare `cargo test` would fail on a
fresh clone. Build it with `cargo check --target wasm32-unknown-unknown -p
quilting-wasm`.

There is no longer a `quilting-spacetime` crate. The 4D slicing pipeline it
implemented was removed wholesale (see §6); the sections below describe it as
historical design, not as shipping code.

### Pipeline

```
glTF file
  |
  v
quilting-gltf: parse meshes, materials, textures, animations, skins
  |
  v
renderer-owned source textures + immutable canonical atlas
  |                                      |
  | each render frame                    | asynchronous LOD job
  v                                      v
GPU patch preparation                 worker GPU classification + edge coherence
  - current animation pose              - fence, then 6-float/face readback
  - ordinary + conformal state           - retain/reconcile/grade topology
  - conservative frustum result          - upload only changed 8-float records
  |
  v
retained prepared patch buffers
  |
  v
GPU vertex/fragment draws: fused Mobius-QB evaluation + PBR/debug shading
```

The exact per-frame and asynchronous responsibilities, including the WebGPU
backend boundary, are maintained in `docs/runtime-render-pipeline.md`.

### Shader Compilation

Shaders are authored in WGSL as modular files with import paths (`#import quilting::math::quaternion`). The `quilting-shaders` crate uses naga to compile WGSL to GLSL ES 300 for WebGL2. Naga renames samplers to `_group_0_binding_N_fs` — the renderer matches these by binding number regex.

## 4. Tessellation Atlas

### Concept

The tessellation atlas is a dictionary of pre-computed triangle sub-meshes, keyed by edge LOD triples `(p, q, r)` where each value is a power of 2. A face with edge LODs `(4, 8, 16)` looks up its tessellation pattern in the atlas, which provides the barycentric coordinates and triangle indices for a sub-mesh that has 4 subdivisions along one edge, 8 along another, and 16 along the third.

### Power-of-2 LODs

Edge LODs are always powers of 2:

- Density/curvature demand is computed from the largest Mobius-deformed face
  median plus the exact interior conformal scale, divided by a target edge size
  (mesh radius / density), then snapped to the nearest power of two
- Optional screen attenuation computes a power-of-two capacity rounded down so
  a subdivision never falls below `min_px_per_sub` screen pixels, then caps the
  density/curvature demand; it never adds tessellation
- A true pole intersection explicitly saturates to the atlas ceiling
- Results are clamped to [1, 512] (`evaluate::MAX_LOD`)

The practical ceiling is usually lower than 512: `quilting-wasm` clamps to the
largest LOD actually present in the built atlas, and the atlas is normally
built with `max_lod_exp` of 8 or 9.

This keeps the atlas size manageable (only need patches for power-of-2 triples) and provides natural 2x jumps that align well with hierarchical subdivision.

### S3 Permutation Reuse

A triple `(2, 4, 8)` needs the same tessellation pattern as `(4, 8, 2)` or `(8, 2, 4)` — just with the vertices permuted. The atlas stores only **canonical (sorted) triples** where `p <= q <= r`, and maps arbitrary triples to canonical form via S3 (symmetric group on 3 elements, 6 permutations).

The 6 S3 permutations correspond to rotations and reflections of the equilateral reference triangle:
- 3 even permutations (identity, 120 deg, 240 deg rotation)
- 3 odd permutations (reflections across each altitude)

`canonical_form([4, 8, 2])` returns `{ res: [2, 4, 8], perm_index: 4 }`. The permutation index tells the system how to remap barycentric coordinates when looking up the patch.

For odd permutations (reflections), `perm_parity = -1` and the normal is flipped. The vertex shader's `perm_bary()` function applies the permutation.

In practice, UV/normal permutation was found to cause visible orientation jumps when LOD redistribution changes the perm_index. The current approach does NOT permute UVs or normals — only the tessellation bary coords are remapped.

### Build Modes

- **Direct**: each canonical triple is independently generated via Bridson (Poisson disk) sampling + Delaunay triangulation. Produces high-quality patches but slow for large LOD ranges.
- **Hierarchical** (current default): base patches at minimum LOD are generated directly. Higher LODs are derived by midpoint subdivision: `(2p, 2q, 2r)` from `(p, q, r)` by splitting every triangle into 4. This preserves boundary vertex positions exactly and is much faster.

### Edge Stitching Guarantee

Adjacent faces sharing an edge must have the same LOD on that edge. This is guaranteed by the half-edge mesh structure: both half-edges of a shared edge map to the same canonical edge index, so they always get the same LOD value. No HashMap needed — a flat Vec indexed by half-edge index suffices.

Within a patch, boundary vertices (on the triangle edges) are placed at exactly `1/N` spacing along each edge, where N is the edge's LOD. Since both adjacent faces use the same N for the shared edge, their boundary vertices line up perfectly. No T-junction cracks.

## 5. GPU Pipeline

### Current Architecture: Fused Mobius-QB Evaluation

The vertex shader performs a fused Mobius + QB evaluation that avoids separate transform and evaluation passes:

```
// Numerator: (a*p_i + b) * w_i
pw_i = qmul(qmul(mob_a, p_i) + mob_b, w_i)

// Denominator: (c*p_i + b) * w_i
bw_i = qmul(qmul(mob_c, p_i) + mob_d, w_i)

// Rational combination
top = sum(lambda_i * pw_i)
bot = sum(lambda_i * bw_i)
X = top * bot^{-1}
```

This folds the Mobius transform directly into the rational Bezier form. Only one quaternion inverse is needed total (on `bot`), rather than one per control point. The math works because the per-vertex inverses `(c*p_i + d)^{-1}` cancel algebraically when the Mobius-transformed positions and weights are combined in the rational quotient.

### Per-Frame Uniforms

The vertex UBO (binding 0) contains:
- `mvp`: 4x4 model-view-projection matrix
- `mv`: 4x4 model-view matrix (for view-space normals/positions)
- `use_qb`: 1 for QB evaluation, 0 for flat (linear) interpolation
- `mob_a, mob_b, mob_c, mob_d`: the four Mobius quaternions as vec4
- camera position plus the ordinary model and inverse-normal matrices

Permutation index is per instance and parity selects WebGL raster winding per
batch; neither is a frame UBO field. A Möbius or camera change immediately
updates uniforms and current-pose GPU preparation, and asynchronously requests
a new topology classification. Unchanged classifications do not rebuild or
re-upload batches.

### Instance Data Layout

**Normative source: `quilting-core::instance_layout`.** That module owns the
stride, the field offsets, the attribute map, and an `InstanceWriter` that
packs a record without anyone rediscovering offsets. Read it rather than
trusting a table; this section only sketches the shape so the pipeline
description reads coherently.

One face instance is **52 floats / 208 bytes / 13 instanced vec4 attributes**,
all with a divisor of 1: three positions, three rational QB weights, edge
LODs, vertex LODs, three UV pairs packed into two vec4s, and three smooth
normals.

Two things about this layout are not guessable:

- The `.x` component of each position vec4 is a **vertex index** used by GPU
  skinning, not the quaternion scalar part. Everywhere else in the codebase a
  quaternion is `(w, x, y, z)`; this is the one deliberate exception.
- QB weight quaternions (`@location` 4, 5, 6) are explicit source-patch data.
  Ordinary triangle meshes store identity weights. Fitted/remeshed rational
  patches retain their non-identity weights, which the fused Mobius-QB shader
  combines with `(c*p + d)` during evaluation. Replacing these fields with
  constant identity weights silently flattens every fitted rational patch.

The briefly used 40-float / 160-byte compact layout omitted source weights. It
was valid for ordinary linear triangles but not as the canonical production
record because it destroyed fitted/remeshed QB geometry.

### LOD Computation

The numerical classifier runs in two worker-GPU passes; CPU WASM derives pole
parameters, polls the fence, and applies the retained-topology invariants after
the compact readback:

1. Compute a target edge size: mesh bounding-sphere radius / tessellation density
2. Per face, push the three edge midpoints through the Mobius transform and
   measure the three deformed medians (vertex to deformed opposite midpoint)
3. Use the largest median for a uniform face demand, then snap to the nearest
   power of two
4. Reconcile the demand across each canonical shared edge in the second GPU
   pass (this is what makes invariant 1 hold)
5. Include the exact interior conformal-dilation demand in the density-driven
   world LOD; a true pole intersection remains an explicit maximum-LOD safety
   case
6. If screen attenuation is on, derive a power-of-two screen capacity from the
   deformed rim plus interior extent, then take `min(world_demand, capacity)` so
   every added subdivision spans at least `min_px_per_sub` pixels; attenuation
   never raises LOD or replaces the tessellation-density control
7. Clamp to [1, `MAX_LOD` = 512], where LOD 1 is the source triangle itself

LOD is computed even for the identity Mobius: the atlas still supplies the
sub-triangle sampling that QB evaluation needs.

The completed payload is six floats per source face. Main-thread WASM retains
off-camera topology, reconciles exact duplicated seam vertices, enforces a 2:1
within-face ratio, and updates only changed draw buckets.

### Conformal Fade

Near the Mobius pole, `|bot|^2 -> 0` and the surface stretches to infinity. A `smoothstep(0.0001, 0.001, dot(bot, bot))` fade factor allows the fragment shader to attenuate these regions.

This is a *visual* fade and is deliberately much wider than the *numeric* pole guard in `qinv` (`|q|^2 < 1e-20`, mirrored by `SINGULARITY_NORM_SQ` in `quilting-core`). Geometry is already faded out sixteen orders of magnitude before the inverse has to fall back to a sentinel, so the sentinel should never be visible — it exists to keep NaNs from propagating, not to be looked at. The CPU and GPU guards must use the same constants, because the CPU computes the LODs and smooth normals for exactly the geometry the shader evaluates.

### What Becomes Static (Updated Only When Geometry Changes)

- Source-patch resource with original positions and rational QB weights
- Tessellation atlas VBO
- Texture uploads

### What Updates Per Frame

- Mobius a,b,c,d uniforms (64 bytes)
- MVP/MV matrices
- Camera info

## 6. Spacetime

**This section is historical.** The `quilting-spacetime` crate was deleted in
commit `b077ca5` ("Remove spacetime/prebake pipeline, unify on QB + async
LOD"); none of the code described below is in the workspace. It is kept as a
record of the design, since the roadmap in §10 still intends to revive it.

### 4D Slicing

Quilting treats animated meshes as 4D objects: a triangle mesh whose vertices trace trajectories through (x, y, z, t) spacetime. Each face sweeps out a triangular prism between consecutive keyframes.

A hyperplane in R^4, defined by `dot(normal, x) = offset`, intersects this prism complex to produce a 3D triangle mesh — the "slice." A pure time slice (`normal = (0,0,0,1)`) gives the mesh at a specific time. A tilted slice mixes spatial and temporal directions, creating effects like motion blur or relativistic distortion.

### HyperMesh

Built from glTF animations (skinned meshes and morph targets) baked into per-vertex Hermite spline trajectories. Each `VertexTrajectory` has a list of `HermiteSegment`s with cubic Hermite interpolation between keyframes. UVs and smooth normals are carried through unchanged (they're properties of the mesh topology, not the animation).

### Marching Prism Slicer

The slicer computes hyperplane intersections along each vertex trajectory (cubic root solving), matches them per-face into coherent triangles (temporal proximity), and groups connected triangles into layers. Faces spanning more than half the animation period are discarded as temporally incoherent.

### Time Embeddings

- **Linear**: time is the real part of the quaternion, `q = (t, x, y, z)`. Simple but produces boundary artifacts at the animation endpoints.
- **Toroidal**: time wraps around a circle in the (w, z) plane: `q = (R*cos(2*pi*t/period), x, y, R*sin(2*pi*t/period))`. Produces closed surfaces — no boundary artifacts. Radius R controls the "thickness" of the torus. A radar-sweep half-plane filter selects one period.

### Spacetime Modes (in Hyperscope)

- **Classic**: 3D Mobius applied post-slice, linear time, auto-animate
- **Toroidal**: time wraps as hypertorus, radar sweep, half-plane filter

### Status

Removed. The pipeline worked, but the 4D Mobius pre-slice (applying a Mobius transform in R^4 before slicing) was never made to behave — the hyperplane doesn't track the deformed torus correctly — and carrying the crate through the QB/async-LOD unification was not worth it. Recovering it means reading `crates/quilting-spacetime` out of history at `b077ca5^`.

## 7. glTF/PBR

### glTF Loading

The `quilting-gltf` crate parses glTF 2.0 and GLB files using the `gltf` crate. It extracts:

- **Meshes**: positions, normals, UVs (TEXCOORD_0), indices, triangulated
- **Materials**: full PBR metallic-roughness model
- **Animations**: keyframe channels (translation, rotation, scale, morph weights)
- **Skins**: joint hierarchies and inverse bind matrices
- **Images**: decoded to RGBA8 from any source format
- **Scene graph**: node hierarchy with transforms

Lenient loading: files with unsupported required extensions (KHR_materials_unlit, KHR_texture_basisu) are loaded via validation-free parsing. Missing textures fall back to average color.

### Animation Baking

Skeletal (skinned) and morph target animations are baked into per-vertex position trajectories. This eliminates the need for runtime skinning — the slicer works directly with position trajectories.

### PBR Material Model

Cook-Torrance GGX microfacet BRDF with metallic-roughness workflow:

- **Diffuse**: Lambertian, modulated by `(1 - F) * (1 - metallic)`
- **Specular**: GGX normal distribution, Smith geometry, Schlick Fresnel
- **F0**: `mix(0.04, base_color, metallic)` — dielectric vs metal reflectance

### Texture Maps

5 PBR texture units, all optional:

| Binding | Map | Format | Notes |
|---------|-----|--------|-------|
| 0 | Base color | sRGB + alpha | Converted to linear in shader via `pow(rgb, 2.2)` |
| 1 | Metallic-roughness | Linear | B=metallic, G=roughness (glTF convention) |
| 2 | Normal | Linear | Tangent-space, scaled by `normal_scale` |
| 3 | Emissive | sRGB | Multiplied by `emissive_factor` |
| 4 | Occlusion | Linear (R) | Applied to ambient term only |

### Alpha Modes

- **OPAQUE**: alpha forced to 1.0
- **MASK**: discard fragments below `alpha_cutoff`
- **BLEND**: sorted back-to-front for correct transparency

### IBL (Image-Based Lighting)

No cubemap or SH probes from the scene. Instead, analytical approximations:

- **Diffuse**: L0+L1 spherical harmonics simulating outdoor sky/ground gradient
- **Specular**: analytical environment reflection (sky/horizon/ground gradient, roughness-based blur)
- **DFG**: Narkowicz analytical approximation (replaces BRDF LUT texture)

### Tangent Frame

Computed in the vertex shader from UV edge vectors and original (pre-Mobius) vertex positions. Gram-Schmidt orthonormalized against the interpolated normal. NaN guards protect against degenerate tangents from stretched Mobius faces.

### Unlit Mode

`KHR_materials_unlit`: outputs base color directly with gamma correction, no lighting computation. Per-material face culling respects the `double_sided` flag.

## 8. Key Invariants

These must always hold. Violating any of them produces visible artifacts.

1. **Edge LOD matching**: adjacent faces sharing an edge must have the same LOD on that edge. Enforced by canonical edge indexing via the half-edge mesh. Violation causes T-junction cracks.

2. **Quaternion layout (w,x,y,z)**: the scalar part `w` is always first, in Rust, in WGSL, and in the instance data packing. Swapping components silently produces wrong geometry. The documented exception is the instance buffer's position slots, which reuse the (always-zero) scalar component to carry a skinning vertex index.

3. **Power-of-2 LODs**: all edge LODs must be powers of 2. The tessellation atlas only contains patches for power-of-2 triples. A non-power-of-2 LOD will miss the atlas lookup.

4. **Perm parity tracks normal orientation**: odd S3 permutations (reflections) flip the winding order. `perm_parity` must be -1 for these to maintain correct normals.

5. **Mobius weight formula**: `w' = (c*x + d) * w`, NOT `w' = (c*x + d)^{-1} * w`. The inverse form is wrong and produces inside-out geometry.

6. **Instance data alignment**: the packer, the renderer's VAO setup, and the WGSL vertex shader must all agree on the instance stride and field offsets, and every instanced attribute must have a divisor of 1. Misalignment silently shifts every subsequent field, so it produces garbled geometry rather than an error. `quilting-core::instance_layout` is the normative definition (currently 52 floats / 208 bytes / 13 attributes) and carries tests that the attributes tile the stride exactly. Anything that hardcodes a stride or offset instead of using those constants is a silent-corruption hazard.

7. **UVs and normals are NOT permuted**: only tessellation bary coords go through S3 permutation. Permuting UVs/normals causes visible orientation jumps when LOD redistribution changes the perm_index.

8. **Smooth normals zeroed under conformal transforms**: the vertex shader checks `dot(sn0,sn0) + dot(sn1,sn1) + dot(sn2,sn2) > 0.01` to decide whether to use smooth normals or QB analytical normals. Under Mobius, the CPU zeroes out smooth normals so the shader falls back to QB normals (flat per-face but geometrically correct).

9. **UBO std140 layout**: vertex uniforms at binding 0, PBR uniforms at binding 1. The PBR UBO is 80 bytes. std140 packing rules apply (vec4 alignment).

10. **WASM built in release mode**: `wasm-pack build` without `--dev`. Debug builds are unacceptably slow for LOD computation and instance packing.

## 9. Known Limitations

### Smooth Normals Under Mobius

Smooth normals from glTF don't transform correctly through conformal maps. Multiple approaches were tried and reverted:

- Conformal rotation of normals via quaternion `(c*p+d)` conjugation
- Post-transform smooth normals from Mobius-deformed positions (equal-weight averaging)
- Gram-Schmidt orthonormalized TBN

All produced artifacts: desaturation on metallic models, orientation issues, NaN cascades. Current solution: zero out smooth normals under conformal transforms, fall back to QB analytical normals. These are flat per-face but geometrically correct for the deformed surface.

### Normal Map Artifacts on Coarse Tessellation

The tangent frame is approximated from UV edge vectors of the original (pre-Mobius) mesh vertices. On coarsely tessellated faces, the screen-space derivative TBN can produce artifacts. Using glTF tangent attributes would improve this.

### WebGL Topology Readback

Current-pose visibility and patch preparation stay on the render GPU, but
WebGL2 cannot compact instances and emit indirect draw commands. Each completed
adaptive classification therefore reads back 24 bytes per source face. The
readback is fenced and asynchronous, yet it remains the principal scaling
boundary for very large animated source meshes.

### RefCell Panic Cascade

After a WASM panic (e.g., from an unexpected glTF format), RefCell borrows in the JS-WASM interop layer can cascade, requiring a page reload.

### LOD Flickering During Interaction

LOD changes cause tessellation pattern changes, which can produce visual flickering. Mitigated by:
- Power-of-2 snapping plus fenced asynchronous results
- One in-flight job plus one coalesced newest-state follow-up
- Retaining the previous valid topology for temporarily invisible faces
- Exact seam reconciliation and 2:1 within-face grading before batching
- Clamping to the largest LOD the atlas was actually built with

### UV Permutation

UV/normal permutation was removed because it causes visible orientation jumps when perm_index changes. This means the S3 permutation optimization only applies to the tessellation geometry, not to material coordinates.

### Missing glTF Features

- KHR_lights_punctual (point/spot/directional lights from glTF)
- Draco mesh compression

## 10. Future Direction

### GPU-resident topology

The remaining geometry bottleneck is not LOD arithmetic—it is returning the
worker-GPU result to the CPU so WebGL can form draw buckets. The next backend
should keep reconciliation, resident topology, visibility compaction, and
indirect draw generation in GPU storage buffers. A universal maximum-density
mesh with degenerate vertices would waste vertex work and is not the target.

### Spacetime FX Roadmap

Parked but planned:
- Proper 4D Mobius with hyperplane tracking on deformed torus
- GPU-side marching (vertex shader prism intersection)
- Multi-layer rendering (OIT/depth peeling)
- Lorentz boost as Mobius in the (x,t) plane
- S^3 slicing (curved hyperplane = conformal bubble through spacetime)

### WebGPU Migration

Surface and material shaders are already WGSL and compile through naga for
WebGL2. The LOD passes, blur, and a few utility programs remain handwritten
GLSL and must be ported. WebGPU compute should emit the same topology and
prepared-patch records already defined in `quilting-core`, then generate
indirect draws without a steady-state CPU readback. See
`docs/runtime-render-pipeline.md` for the staged migration contract.

### Production Rendering Gaps

1. Scene-authored IBL/light probes and `KHR_lights_punctual`
2. glTF tangent attributes for better normal mapping
3. Draco mesh compression support
