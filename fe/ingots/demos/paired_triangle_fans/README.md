# Two Triangles, Two Fans

```sh
.toolchains/fe/target/release/fe web dev fe/web/paired-triangle-fans/index.html --port 8776
```

The original quad is covered by two triangles sharing its diagonal. Each
triangle has an independently movable interior point and three atlas tiles
around it: six pieces altogether. The lower-left diagram shows their parameter
regions with the same colors as the 3D view.

- `first_b/c` and `second_b/c` are positive barycentric ratios to the first
  corner of each parent triangle. Equal ratios place its point at the center.
- Each half has its own concentration and radial LoD. Concentration above one
  moves samples toward its interior point without changing the triangle count.
- `outer_lod` controls the square perimeter. `diagonal_lod` is one shared value
  used by both halves, irrespective of their independent radial settings.
- Shape-weight correction starts enabled. As in Four-Patch Center, it resolves
  the fixed authored patch's vector-valued constraint; disable it to explore
  the xyz projection of independently weighted Clifford values.
- Drag to orbit; scroll to approach/retreat; toggle wires independently.

## What is exact, and what is approximate?

`TriangleFan3::domain` composes each triangular child with its parent's affine
domain. A positive projective map concentrates atlas samples within that child.
The original surface is evaluated at the resulting parameter coordinate;
neither surface fitting nor a separate child control net is needed to draw it.
It is an exact parameter restriction in real arithmetic, implemented in f32.
The straight triangles connecting evaluated points are only an approximation.

The existing atlas supplies every canonical edge triple through LoD 8. Tile
permutations are restored before applying the sampling map. One Fe compute pass
writes the total vertex count, and one indirect draw covers all six pieces.
There is no per-interaction GPU-to-CPU geometry readback or mesh upload. The
24.9 MB fixture is uploaded when the page initializes. The renderer currently
locates a leaf with a bounded six-tile prefix scan per vertex; there is no claim
of optimal batching or measured runtime performance yet.

Outer child edges have equal endpoint weights, so concentration leaves them
fixed. The diagonal is traversed in opposite directions by its two neighbors.
Within each fan, adjacent children share concentration and radial LoD. These
rules preserve matching boundary samples, even with independent fan centers.
This focused ownership rule is not yet a general PatchGraph edge registry.

The subdivision coordinates and projective map are shared geometry operations
in `quilting_patch`. `quilting_patch_view` shares the rendering path with the
four-piece explorer: resident atlas lookup, analytic evaluation, normals,
pastel shading, depth, and antialiased wires. The GPU actor owns draw preparation
and rendering effects; input state and correction run through Fe transitions.

## Evidence and limits

`nested_triangle_fans_wasm_preserve_domain_surface_and_diagonal` executes the
production Fe fan-domain and projective maps in release Wasm. Four independent
parameter settings check positive orientation and complete coverage by all six
children; 72 interior points compare exact restricted controls against an
independent dense f64 surface oracle. Along 257 diagonal samples, radically
different centers and concentrations produce bit-identical parameter positions,
original-patch positions, and original-patch normals in Wasm.

The same test deliberately retains a numerical counterexample: reconstructing
very thin child control nets produces a maximum neighboring normal-component
disagreement of about 0.1198 in f32. The direct original-chart evaluation used
for display avoids that additional source of cancellation. It does not resolve
singularities or conditioning already present in the original patch.

This does not certify good triangle shapes, adequate surface sampling, GPU
bitwise agreement, or approximation error. Independent fan controls can produce
terrible distributions. Six tiles are not automatically cheaper than four.
Use this as a focused topology comparison, not as a declaration that meshing is
solved. Geometry-driven edge counts and spacing are still pending.

Browser check (2026-09-04, release `fe web dev`, Chrome MCP): the default six
regions and parameter overlay render. Moving the first center, setting the
concentrations to 4 and 0.25 and the radial LoDs to 5 and 2 updates both views
without console errors. This deliberately produces visibly uneven triangles;
the controls work, but automatic placement is not implemented. The browser
reports 17,284 bytes of Wasm and 98,469 bytes of WGSL across four shaders.

## Next comparison: from tessellation upward

The fan need only define a sampling map; it need not create separate geometric
patches. Compare it with a direct quad atlas using the same principles:

1. One canonical boundary curve and monotone endpoint-preserving sampling map
   per shared edge, with LoD driven by evaluated length and approximation error.
2. An interior extension of those boundary maps, plus independent concentration
   and focal controls whose additional displacement vanishes at the boundary.
3. Explicit checks for foldovers and inadequately sampled regions. Valid edge
   maps do not alone guarantee that their interior interpolation is injective.

Triangle/quad topology, edge demand, and sampling placement should remain
separate decisions. Subpatching is an available fallback, not a prerequisite
for adjusting sample density. No direct quad atlas or independent edge-warp
extension is implemented by this explorer yet.

```sh
cargo test --release --manifest-path fe/tools/quilting-fe-fixtures/Cargo.toml --features fe-oracle --lib nested_triangle_fans_wasm_preserve_domain_surface_and_diagonal -- --nocapture
```
