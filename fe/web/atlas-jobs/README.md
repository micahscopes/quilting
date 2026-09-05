# Full quad job coverage

This is a GPU job-iteration acceptance fixture, **not** full tessellation
generation. It declares all 1,035 canonical quad jobs through LoD 8. Each job
advances a typed GPU record and writes its key into a diagnostic history. No
geometry is downloaded, uploaded, or replaced by that history.

The production cursor and the portable iterator share `next_canonical_key` in
`quilting_atlas::quad`. GPU storage layout is derived from the local
`QuadJobState` type. The record retains separate scan position and dense ordinal.

`verify.mjs` independently enumerates square rotations/reflections, sorts the
canonical representatives, and requires exact ordered coverage. It also checks
that a stale-epoch advance leaves the last job unchanged and that exhaustion
invalidates it. This tests D4 symmetry, not arbitrary permutations of four edges.

```sh
node --test fe/web/atlas-jobs/verify.test.mjs
/laboratory/fe-stuff/fe-worktrees/mb2/target/release/fe web dev fe/web/atlas-jobs/index.html --port 8783 --no-watch
```

Browser evidence (2026-09-05): shared Fe `03b477058`, Sonatina `2e3cc1f5`,
plus the existing provider-identity overlay, compiles all four passes.
`gpu.verify.mjs` executes those exact pipelines into separate diagnostic
buffers with readback permission. It uses the runtime's lazy pipeline
realization and does not change production resource permissions. The resulting
1,035-entry GPU history passes `verifyQuadJobCoverage` in exact order, with
receipt `[1035,1,6560,0]`. Ten independent oracle tests also pass.

Saved evidence: `/laboratory/quilting/scratch/atlas-job-state-20260905/`,
including `browser-job-history.json`, `fixed-next-job.capture.json` and the
compiler-emitted artifacts in `fixed-capture/`. The test establishes job
iteration correctness, not full atlas geometry or generation timing.

Next integration: run the real sampling/triangulation/retention stages between
job advancement and the next iteration. Then validate retained topology,
boundaries, memory bounds and queue pacing through LoD 8. Passing key coverage
alone does not establish any of those geometry or performance properties.
