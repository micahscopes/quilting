# Fe geometric-algebra shaders on Quilting WebGL2

Status: bounded research recommendation, 2026-09-01. This records the inspected
`mb2` worktree at Fe commit `60834b9af` and the corresponding Quilting `rust`
architecture. It authorizes no dependency or renderer cutover by itself.

## Decision

A Fe-to-WebGL2 path is viable and interesting for small fixed geometric-
algebra calculations. Fe should be an optional **shader-specialization
frontend**, not a second Hyperscape scene model, FRP runtime, renderer, or
resource authority.

The intended ownership remains:

- Hyperscape owns semantic scene state, conformal frames, navigation, and
  backend-neutral extraction.
- Quilting owns tessellation, LOD topology, render descriptors, commands, and
  backend resource lifetimes.
- Fe may compile a small fixed per-vertex or per-fragment algebraic function to
  ordinary shader arithmetic.
- Quilting's WebGPU and WebGL2 implementations interpret the same immutable
  shader/pipeline descriptors under independent device epochs.
- JavaScript supplies browser contexts and events only.

This direction applies Conal Elliott's useful separation—state a denotation,
derive realizations, and isolate effects—without introducing another owner of
scene or renderer state.

## Inspected evidence

The relevant Fe worktree is `/laboratory/fe-stuff/fe-worktrees/mb2`.

- `demos/sketches/cga3d/src/lib.fe` declares a named reflected CGA basis and
  derives `CompileNamedVectorDotF32` for point/sphere incidence.
- `demos/sketches/qcga/src/lib.fe` applies the same mechanism to a sparse
  15-dimensional QCGA point/quadric incidence calculation.
- `ingots/ga_expr/src/lib.fe` supplies the compile-time expression machinery.
  The generated shader contains scalar SSA arithmetic; it does not retain a
  runtime blade loop, GA expression tree, or dynamic sign dispatch.
- `demos/sketches/qcga_pencil_de/src/lib.fe` demonstrates the richer future
  target: fragment distance estimation and an authored raster overlay.

A cached current `qcga` bundle makes both the opportunity and the incompatibility
concrete. Its manifest describes four contiguous `f32` inputs (`lambda`,
`theta`, `zoom`, and `res`) with a 16-byte span. The generated WGSL begins with:

```wgsl
struct Input {
    p2_: f32,
    p3_: f32,
    p4_: f32,
    p5_: f32,
}

@group(0) @binding(1)
var<storage> input: Input;
```

The following fragment body is straight-line scalar arithmetic. This is the
right form for shader execution, but WebGL2 has no shader-storage buffers. The
same four broadcast values fit naturally in one `std140` uniform block.

Fe's current raster surface is deliberately narrower than Quilting:

- `std::webgpu::TriangleList<N>` is a fixed non-indexed draw policy; its source
  explicitly leaves vertex/index-buffer variants for later.
- `web_bundle.rs` rejects authored raster programs with external resources
  because those resources are not yet wired through both stages.
- compute pass graphs have no WebGL2 realization.

Therefore Fe is ready to specialize fiber algebra, not to own GLB meshes,
resident tessellation, mesh combinatorics, materials, culling, compaction,
indirect arguments, JFA, or context recovery.

## Existing Quilting seam

Quilting already has the functional/effectful rendering boundary this work
needs:

1. `quilting-core::render_pipeline` defines immutable WGSL/GLSL shader module,
   bind-group, vertex-layout, and pipeline descriptors. Exact source,
   compiler-catalog revision, stage, entry point, target, and definitions form
   comparable memo keys.
2. `quilting-core::render_memo::DeviceMemo` owns epoch-scoped effect results and
   returns old resources for explicit destruction when the device/context
   epoch changes.
3. `quilting-renderer::shader::WebGlProgramMemo` memoizes compiled shaders and
   linked programs by those complete descriptors.
4. `quilting-shaders` validates WGSL through Naga and emits GLSL ES 300 with
   `is_webgl: true`.
5. The WebGL lowering rejects compute stages explicitly instead of pretending
   to emulate them.

The missing seam is consequently small and testable: validate an Fe shader
bundle, lower a deliberately restricted broadcast input record to a uniform
layout accepted by both backends, and construct Quilting descriptors. It does
not require a TypeScript renderer or a new resource cache.

## Bounded prototype

Keep the prototype outside default release dependencies. Prefer a small
build-time importer/tool or generated fixture crate; do not make the browser
download the Fe compiler.

### Gate 1: no-state fragment

- Compile a constant/no-state Fe fragment to WGSL.
- Admit it through `ShaderModuleDescriptor` with the exact Fe compiler/catalog
  revision.
- Use Quilting's existing Naga path to emit GLSL ES 300.
- Compare Fe Wasm, WGSL, and GLSL color vectors without opening a browser.

This proves the compiler interchange before introducing a layout rewrite.

### Gate 2: broadcast uniform subset

Add one fail-closed importer rule for a closed scalar/record input:

- recursively nested records of finite-width `f32`, `i32`, or `u32` scalars;
- one read-only broadcast binding;
- exact reflected offsets, alignment, padding, and minimum block size;
- no arrays, runtime-sized values, writable storage, atomics, textures,
  samplers, external resources, or compute stages;
- distinct vertex and fragment entry-point descriptors; and
- a manifest fingerprint included in compilation identity.

The importer may transform the Fe input binding into uniform-compatible WGSL,
or Fe may eventually emit that target directly. It must not rely on Naga to
silently reinterpret a storage buffer as a uniform block.

Run `cga3d` and `qcga` through this subset. Compare point/sphere and
point/quadric incidence against independent Rust/Fe Wasm vectors, including
zeros, near-zero incidence, large/small coordinates, and sphere-at-infinity or
degenerate pencil cases.

### Gate 3: authored raster overlay

Only after the fragment path is proven, try the fixed
`qcga_pencil_de` vertex/fragment overlay. Continue to reject:

- indexed or runtime-sized geometry;
- external buffers/textures;
- transform feedback;
- compute or indirect dispatch; and
- any request for scene, LOD, or resource authority.

Those are separate Fe features, not preconditions for useful algebra shaders.

## Required evidence

The experiment is successful only with all of the following:

- byte-exact reflected input offsets and minimum UBO sizes;
- independent CPU/Fe-Wasm/WGSL/GLSL algebra vectors within an explicit float
  tolerance and association policy;
- WebGL compile/link success for every admitted stage and fail-closed rejection
  tests for every unsupported feature;
- WebGPU/WebGL image hashes under a documented tolerance once shared GPU
  testing is safe;
- identical backend-neutral command ordering;
- shader/program memo hit, miss, and failed-creation counts;
- bytes uploaded per frame, with unchanged state causing zero uploads;
- context/device epoch invalidation and resource recreation evidence; and
- no new JavaScript semantic state, scene graph, or renderer lifecycle.

## Stop conditions

Stop or redesign the experiment if it requires any of the following:

- copying Hyperscape scene state into Fe or TypeScript;
- bypassing `ShaderModuleDescriptor`, `RenderCommand`, or device-epoch memos;
- a permissive shader rewrite whose accepted resource/layout subset cannot be
  stated and tested exactly;
- emulating WebGPU compute in the WebGL2 compatibility lane; or
- replacing Quilting's resident atlas/vertex-pulling path with Fe's current
  fixed `TriangleList<N>` surface.

The expected valuable result is a small algebraic shader frontend shared by
both backends. A full Fe Hyperscope renderer would be a separate substantial
project and is not recommended by the current evidence.
