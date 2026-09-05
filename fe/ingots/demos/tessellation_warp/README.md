# Tessellation Warp

```sh
.toolchains/fe/target/release/fe web dev fe/web/tessellation-warp/index.html --port 8777
```

A flat, focused explorer of boundary skew, interior twist, movable focus, and
concentration. It evaluates no curved surface and constructs no child patches.
The same connectivity remains resident as the coordinates move.

The triangle uses an existing blue-noise/Delaunay atlas tile, selected at a
uniform edge LoD from 0–8. The square currently uses a regular fixed-diagonal
grid at the same dyadic resolution. **That grid is a visualization fixture,
not the requested blue-noise quad atlas.** WebGPU writes an indirect draw count
and renders the mapped points; changing the warp does not read geometry back
to the CPU or upload a replacement mesh. The existing full triangle artifact
is uploaded at initialization; generation of that artifact is not timed here.

## Controls

- Domain switches between the triangle atlas and square-grid fixture.
- Each edge skew is positive; one is neutral. Triangle a/b/c traverse B→C,
  C→A, A→B (d is unused). Square a/b/c/d are bottom/right/top/left; horizontal
  edges run in increasing x and vertical edges in increasing y.
- Twist rotates nested polygon shells about the focus; it fades to zero at
  the boundary. Zero is neutral.
- Click or drag to place the yellow focus. The positive x/y ratio controls
  also move it while keeping it strictly inside the domain.
- Concentration above one draws samples toward that focus; below one toward
  the boundary; one is neutral. It does not change connectivity or count.
- LoD changes resolution. Wires can be hidden or made thinner.

Red marks a rendered triangle with reversed or zero signed area. This is a
discrete mesh failure, not a surface-family diagnostic. It must not be hidden
by removing the offending triangles when adapting this method to surfaces.

## Shared Fe maps and their contracts

`quilting_patch::sampling_warp` provides:

- `IntervalSamplingMap`: a law-bearing interface for a finite, strictly
  increasing map of [0,1] fixing both endpoints. The trait does not prove those
  laws; implementations require evidence. `ProjectiveIntervalBias` is the
  current authored example, not a general uniformizer.
- `TriangleEdgeMaps` and `QuadEdgeMaps`: independent interval laws, extended
  by composition of monotone slice maps. Triangle extensions fix the other
  two edges; quad horizontal and vertical extensions preserve their assigned
  boundaries. Composition order affects the interior, not the edge rules.
- `InteriorTwist`: rotation in normalized polygon-radius coordinates. In
  those coordinates the map is `(r, angle) -> (r, angle + twist*(1-r)^2)`.
- `InteriorRadialMap`: a replaceable monotone radius law preserving the
  focus and every boundary point. `InteriorConcentration` supplies the current
  projective radius law `r/(k+(1-k)*r)`.

The continuous maps are orientation-preserving/injective in real arithmetic
under their stated domain/law assumptions. Polygon sectors can have derivative
discontinuities. **Mapping only the vertices and connecting them with straight
segments does not inherit a continuous-map injectivity guarantee.** Finite
resolution, floating-point conditioning, and extreme controls require separate
mesh evidence. This experiment exposes that distinction directly.

## Tests and browser evidence

`sampling_warps_wasm_preserve_independent_boundaries_and_interior_domain`
executes the production maps in release Wasm. It checks neutral identity, all
seven triangle/square edges at 257 positions and five skew strengths, and
independence from extreme twist/focus/concentration changes. It also checks
finite in-domain results on sampled interiors and preservation of the focus.
It records a counterexample: an 8×8 square grid with twist 6 and concentration
4 has 20 inverted/collapsed straight triangles out of 128, despite the
continuous map's injectivity. This is not a certified mesher.

`projective_interval_inverse_uniformizes_a_rational_line` checks a simple
analytic case: inverse projective bias applied to a rationally parameterized
straight edge gives uniform geometric positions, within an absolute 3e-6
f32 tolerance. It does not establish a closed-form inverse for general curves.

Browser check (2026-09-04, Chrome MCP, release `fe web dev`): the triangle
baseline renders. Switching to square, changing the four edge skews to
4/0.25/2/0.5, twist to about 1, and concentration to 2 updates the flat mesh
without console errors. The build reports 13,078 bytes of Wasm and 52,568 bytes
of WGSL across three shaders. This is an interaction receipt, not a performance
benchmark or a proof that the displayed mesh never overlaps.

## Next: actual atlas and surface uniformization

The intended pipeline, rather than more fan/subpatch experiments by default:

1. Generate blue-noise/constrained-Delaunay atlases directly for triangular and
   quadrilateral domains in Fe, reusing deterministic sampling and topology
   machinery across workers and GPU providers. Preserve canonical dyadic edge
   samples. Triangle symmetry is S3; a square admits D4 rotations/reflections,
   not arbitrary sorting of four edge values. Opposite/adjacent edges must not
   be conflated. The current exact incircle predicate uses the equilateral
   metric; square topology needs its own correctly specialized metric.
2. Measure actual patch boundary curves in an explicitly chosen geometric
   metric. Choose shared edge LoDs and map atlas edge coordinates directly to
   uniform arc-length positions. For curve C(t), use the inverse of normalized
   cumulative speed. Prefer a family-provided analytic inverse where valid;
   otherwise use a numerically controlled representation with an explicit error
   receipt. A hand-authored scalar skew does not meet this requirement.
3. Own the oriented curve, spacing map, and dyadic sample sequence once per
   shared edge. Neighbors consume those same samples, reversed as appropriate.
   Uniformity and seam equality are separate gates; independent approximate
   inversions must not become competing authorities for one seam. The target
   is uniform spacing along the actual curve, not equal UV increments or equal
   endpoint chords across a curved segment.
4. Extend the geometry-derived boundary adjustment inward while preserving it
   exactly, then assess density, twist, triangle quality, normal/display error,
   and conditioning against the actual patch interior. Determine how much the
   boundary-driven map already solves before adding stronger interior methods.
   Keep additional focal/concentration controls boundary-fixed. Subdivision is
   a fallback option, not a prerequisite of this construction.

For LoDs 0–8, triangle keys have 165 symmetry classes; square edge keys have
1,035 under D4. These are key counts, not an assertion that the complete quad
atlas exists or fits any particular memory/runtime budget. Neither direct quad
atlas generation nor actual curved-patch uniformization is completed here.

```sh
cargo test --release --manifest-path fe/tools/quilting-fe-fixtures/Cargo.toml --features fe-oracle --lib sampling_warps_wasm_preserve_independent_boundaries_and_interior_domain -- --nocapture
cargo test --release --manifest-path fe/tools/quilting-fe-fixtures/Cargo.toml --features fe-oracle --lib projective_interval_inverse_uniformizes_a_rational_line
```
