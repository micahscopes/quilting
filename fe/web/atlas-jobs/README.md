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
.toolchains/fe/target/release/fe web dev fe/web/atlas-jobs/index.html --port 8783 --no-watch
```

Current evidence: the Node tests use synthetic inputs to validate the oracle;
they are not GPU execution evidence. The GPU build is pending resolution of
the storage-layout derive-provider ambiguity documented in upstream mailbox
`FE-QF-COORD-20260905`. Compiler edits are paused for ownership coordination.
Do not claim a browser pass or generation timing until this fixture is actually
built and the real buffer history passes the independent oracle.

Next integration: run the real sampling/triangulation/retention stages between
job advancement and the next iteration. Then validate retained topology,
boundaries, memory bounds and queue pacing through LoD 8. Passing key coverage
alone does not establish any of those geometry or performance properties.
