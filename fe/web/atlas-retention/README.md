# GPU atlas retention acceptance

This fixture exercises storage for completed atlas tiles. It is **not** the
full atlas generator and its authored points are not sampling evidence.

The Fe actor runs initialization, reservation, parallel copying, scratch
overwrite, invalid copying, and publication as ordered compute passes. One
valid tile must remain intact after its scratch storage is overwritten. An
uncertified tile, an oversized reservation, an incomplete copy, and a face
with an invalid local index must never become ready. Wrong-epoch and repeated
publication must not affect counts.

`verify.mjs` checks independently specified expected results. Its Node tests
test the oracle with synthetic data, not GPU execution. Browser readback is
diagnostic only; the production retention path does not return geometry to
the CPU.

`browser-evidence.json` records actual execution of all six Fe-generated
passes in Chromium using the portable eight-storage-binding device limit.
The independent oracle passes: one ready tile, four rejected tiles, and the
ready tile unchanged after scratch overwrite. The recorded compiler is mb2
`ce2ff6862`; its same-named-stage/helper regression and all 46 actor tests pass.
This fixture compiled in 32.9 seconds, including dependency diagnostics;
that is **compiler time**, not full-atlas generation time.

From the repository root:

```sh
.toolchains/fe/target/release/fe test fe/ingots/geometry/quilting_atlas -O2
node --test fe/web/atlas-retention/verify.test.mjs
.toolchains/fe/target/release/fe web dev fe/web/atlas-retention/index.html --port 8783 --no-watch
```

## Contracts

- Reservation and publication are single-writer operations.
- Each copy lane executes exactly once per reservation; publication is a
  separate dispatch after the copy. Completion counts are not a general
  substitute for this scheduling contract.
- Initialization starts a new generation only after old commands have drained.
- Arena ranges remain consumed when copying fails; partial data stays hidden.
- Rendering must select only ready records from the current generation.
- The caller supplies mesh certification; retention checks copying and bounds,
  not Delaunay validity, coverage, or sampling quality.

## Remaining full-provider integration

1. Feed the canonical triangle/quad job cursor into the real sampling and
   triangulation passes, with reusable scratch and bounded resident arenas.
2. Compose job iteration with the existing sampling/insertion/repair cycles.
   Fe mb2 `6717e746f` now supports nested `CycledDispatch` policies and compact
   shared schedules. Its six-stage browser fixture records the exact nested
   execution order. Wire this facility into the generator, including bounded
   queue pacing; the full atlas job loop is not connected yet.
3. Connect mesh certificates and compacted counts to reservation, copying,
   publication, and ready-only atlas rendering.
4. Scale scratch allocation and convergence policy through LoD 8; surface
   exhaustion or unconverged work rather than silently dropping tiles.
5. Validate all canonical keys, boundaries, symmetry transforms, and retained
   meshes; then measure full generation, shader compilation, queue time, and
   peak/resident memory separately. No full LoD 0–8 timing exists yet.
