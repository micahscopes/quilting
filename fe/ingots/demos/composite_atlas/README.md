# Composed tessellations and controllable patches

This explorer places runtime-generated primary triangle tessellations into a
square using either diagonal or a four-triangle fan with a movable interior
point. Curved triangle and quad modes reuse this atlas and the shared patch
evaluator; a curved triangle uses one primary. This is not a direct quad atlas.

- All165 canonical triangle keys for exponents0–8 are generated in Fe/Wasm by
  four typed workers, with requested keys prioritized. No atlas file is downloaded.
- Primary geometry is retained in a shared GPU buffer. Changing the split
  or center does not regenerate or duplicate the primary geometry.
- Child lookup and directed edge identity come from
  `quilting_atlas::composition`. Boundary interpolation uses the canonical
  endpoint order, including the diagonal or center spokes.
- WebGPU computes indirect draw arguments and renders; sampling and
  triangulation are Wasm work. The first selected tile can render as soon as it
  arrives. Completion messages request frames; there is no polling scheduler.
- Outer LoDs are independent. Automatic interior LoDs use reference-domain
  demand; manual diagonal/spoke requests are retained when switching modes.
  This is not yet automatic screen-space or surface-length LoD selection.
- Endpoints and an on-arc midpoint shape the supported circular-edge family.
  Exploratory free weights leave that family. Surface admission is not a
  certificate that its tessellation has good shape or no folds.
- Boundary spacing, interior area correction and optional warps can be compared
  on the same geometry. Strong cases still fold; neither global blue noise nor
  global Delaunay legality is claimed for the composed square.

Use `color_view` for fixed-scale shape or sampled surface-error diagnostics.
`highlight_folds` marks nonpositive UV triangles magenta, including curved views.
`rendered_triangles` reports the retained submitted mesh, excluding handles but
including clipped, hidden and folded triangles. Pending work retains that count.
With measured density and compensation enabled, toggle `interior_area_correction`
to compare the two placements without changing their boundary maps or counts.

Pathological fixtures and measured limitations are documented in
`docs/captured-triangle-boundary-failure.md`,
`docs/captured-quad-placement-failure.md`, and
`docs/bounded-coarse-mesh-experiment.md` at the repository root. The coarse mesh
and count experiments there are not automatically promoted into this viewer.

Serve with the shared release Fe CLI:

```sh
/laboratory/fe-stuff/fe-worktrees/mb2/target/release/fe web dev \
  fe/web/composite-atlas/index.html --port 0
```

Browser verification and timings must be recorded separately; source presence
and successful compilation alone do not establish a working explorer.
