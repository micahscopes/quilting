# Tessellation Warp

One flat-domain explorer, with the same edge skew, interior twist, draggable
focus, concentration, wire, and responsive MSAA rendering for both domains.
It neither fits a surface nor changes the topology in response to warping.
Red triangles indicate that the chosen deformation folded the displayed mesh.

```sh
.toolchains/fe/target/release/fe web dev fe/web/tessellation-warp/index.html --port 8777
```

Triangle uses the existing complete precomputed 0–8 atlas (`lod`). Quad now
uses an actual generated square tile instead of a regular grid (`quad_lod`).
Its temporary 0–3 bound is an **unfinished implementation**, not an acceptable
replacement for the requested complete quad atlas. Shared candidate-neighbor
search, bounded scalable topology construction, and complete 0–8 residency
remain required. The two controls make the temporary asymmetry visible instead
of silently clamping a requested 8 to 3.

The quad job keeps points and indices in GPU storage. A key certificate decides
whether generation is needed; warp parameters do not participate in that key.
Triangle mode disables the quad preparation passes via Fe `PassActivation`.
GPU resource recreation starts with an empty certificate and regenerates.
Cached quad frames still submit guarded dispatches; avoiding their submission
cost is separate from avoiding the sampling/triangulation work itself.

## Integration evidence (2026-09-04)

Release compiler `8b7ff12ce`, protocol 13, 11 passes; cold publication 80.7 s,
13,436 control-Wasm bytes, 313,936 compiler-reported WGSL bytes. These are build
measurements, not GPU generation timings.

Chromium showed both modes. Neutral quad level 3 has 58 points, 82 triangles,
and valid sampling/insertion/Delaunay receipts. Changing edge skews to 0.65/1.4,
twist to 1.1, focus x to 1.6, and concentration to 1.2 visibly changed the warp
while leaving the complete source point/index buffers unchanged and generation
counter at 1. Changing quad LoD to 0 produced four corners/two triangles and
counter 2. Triangle LoD 8 rendered after switching domains. No console errors
or warnings were reported during this integration check. Parameter transitions
were exercised through the public Fe surface parameter interface; this is not
a claim of exhaustive physical pointer, keyboard, resize, or recovery testing.
