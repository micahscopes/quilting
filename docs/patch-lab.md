# Hyperscope Patch Lab

The Patch Lab is a small interactive scene inside the ordinary Hyperscope
renderer. It is meant to make adaptive Quilting topology legible before a
large glTF, animation, screen attenuation, or conformal pole complicates the
picture.

Open it from the **Patch Lab** section of the sidebar and choose one of:

- **Tri patch** — one rational quaternionic-Bézier triangle. The bend slider
  changes one quaternion weight, while the three edge controls request the
  `BC`, `CA`, and `AB` subdivisions independently.
- **Plane** — an indexed checkerboard-diagonal grid. Wave, radial, and sweep
  functions are sampled at edge midpoints, so both incident faces request the
  same value before reconciliation.
- **Cube** — eight source vertices and twelve source triangles, useful for
  seeing the same field turn corners without hiding the source topology in a
  dense model.

**Surface + wire** shows the resulting tessellation directly. **LOD colors**
shows its resident topology classes. The status readout distinguishes the
requested field from the renderable result, including promoted faces and
edges, the resident LOD histogram, rendered triangle count, and shared-edge
mismatch count.

## Invariant being demonstrated

The field first quantizes each requested edge level to a power of two. Rust
then calls the production resident-LOD reconciler, which monotonically
promotes values until:

1. both faces incident to a shared edge use exactly the same subdivision; and
2. the largest and smallest edge resolutions in one source face differ by at
   most 2:1.

The first rule is the crack-free stitching invariant. The second is a current
grading policy, not a mathematical requirement of the atlas or shared edges.
It bounds anisotropic triangle fans and makes a detail peak decay by one
power-of-two level per neighboring source face, but its monotone promotions
can also create a conservative halo of extra resident triangles. The runtime
atlas is restricted to keys reachable under that policy only after grading;
the atlas is not what causes the promotion.

For example, requesting `1 / 8 / 128` on the single triangle produces a
renderable `64 / 64 / 128`. The UI reports that as two promotions rather than
silently presenting the reconciled values as if they had been requested.

The deterministic geometry and field implementation lives in
`quilting_core::educational`. Browser code only selects parameters, transfers
the compact six-float-per-face LOD result, and displays diagnostics. This is
intentional: a native or future WebGPU frontend can reuse the same lesson and
the same invariants without porting JavaScript behavior.

## Runtime isolation

Entering the lab pauses model animation, clears animation textures, and stops
authored presentation updates from overwriting its source geometry. Camera,
SpaceMouse, render-mode, and Möbius controls remain active. **Exit lab** reloads
the exact URL that was open before entry, restoring the model or presentation
without maintaining a second copy of its runtime state.
