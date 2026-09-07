# Composed tessellations: affine control

This explorer places runtime-generated primary triangle tessellations into a
square using either diagonal or a four-triangle fan with a movable interior
point. It is the first control for a family of compositional methods, not a
replacement for the complete mixed-edge atlas.

- Nine uniform primary keys `(l,l,l)`, `l = 0..8`, are generated in Fe/Wasm by
  four typed workers. No atlas file is downloaded.
- Primary geometry is retained and shared by GPU instances. Changing the split
  or center does not regenerate or duplicate the primary geometry.
- Child lookup and directed edge identity come from
  `quilting_atlas::composition`. Boundary interpolation uses the canonical
  endpoint order, including the diagonal or center spokes.
- WebGPU computes indirect draw arguments and renders; sampling and
  triangulation are Wasm work. The first selected tile can render as soon as it
  arrives. Completion messages request frames; there is no polling scheduler.
- LoD selects a uniform primary, not independent outer-edge densities. Affine
  placement changes the point-spacing metric. Neither global blue noise nor
  global Delaunay legality is claimed for the composed square.

The next comparison adds mixed-edge requests with shared diagonal density,
explicit primary permutations, and boundary-preserving warps. Separate that
from the question of choosing a better primary sampling metric.

Serve with the shared release Fe CLI:

```sh
/laboratory/fe-stuff/fe-worktrees/mb2/target/release/fe web dev \
  fe/web/composite-atlas/index.html --port 0
```

Browser verification and timings must be recorded separately; source presence
and successful compilation alone do not establish a working explorer.
