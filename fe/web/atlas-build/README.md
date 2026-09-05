# Full GPU atlas construction

`index.html` selects all 1,035 D4 quad keys; `triangles.html` selects all 165
S3 triangle keys. Both include every exponent from 0 through 8. There is no
edge-ratio restriction, density ladder, or downloaded geometry artifact.

The Fe actors place existing shared sampling, insertion, adjacency, repair and
certification kernels inside an outer canonical-job cycle. One scratch arena
is reused. Each certified tile reserves checked ranges, copies its geometry,
and publishes its directory record only after the copy completes. A queue
completion boundary after each job limits queued scratch work; no geometry or
count readback controls generation. Resident points and triangle indices each
have an explicit budget below the portable 128 MiB binding limit. Overflow is
a failed tile, not silently truncated success.

The two actors currently repeat their typed stage wiring. Algorithm and
retention implementations are shared in `quilting_atlas_webgpu`; future actor
composition should consolidate this wiring without manufacturing Fe source or
moving scheduling policy into JavaScript.

Completion requires every canonical job ready and zero failures. Per-tile
receipts distinguish the generation stages. The diagnostic directory oracle
is separate from production and checks coverage, identities, epochs, range
overlap and completion counts. Its synthetic tests are not GPU proof.

## Current gate

Both actors pass `fe check`. Full quad precompilation now succeeds with Fe
`ca65833f7` pinning Sonatina `1d206a52`. Two compiler regressions covered the
job cursor's success/exhaustion join and a helper value between sequential
loops. All 134 shader integration tests passed in release on llvmpipe.
These compiler tests do not prove complete atlas generation.

The full quad browser run is a correctness diagnostic, not a passing result.
An independently checked partial snapshot contained 121 ready tiles, 114
failed tiles and 800 pending. All 114 failures reported converged sampling,
valid initial topology, remaining repair violations and zero repair invariant
failures. These are the algorithm's receipts, not independent geometric proof.
The run already exceeded seven minutes. Neither full coverage nor the requested
subsecond generation target has been achieved.

Captures are in `/laboratory/quilting/scratch/atlas-job-state-20260905/`.
Fixed sampling/insertion/repair budgets still need full-corpus convergence
testing; exhausted tiles must remain failed.

## Bounded diagnostic replay

`readback.verify.mjs` reads only job state, directory, completion and receipts
(at most 1 MiB). Its observation kernel copies storage into separate readback
buffers; it does not change production buffers or feed scheduling decisions.
`tile-replay.verify.mjs` reuses the compiled pipelines with private scratch
buffers and a recorded job. It preserves inner pass order and observes repair
at increasing round counts before running the unmodified final certificate.
Neither diagnostic is imported by the application. Their JavaScript is test
orchestration, not a replacement implementation of sampling or triangulation.

Checkpoint violation counts before the final certificate describe the last
round's pre-flip proposals. Only the final certificate recomputes violations
after the last mutation. Replays sharing a GPU with the corpus run are not
independent timing measurements; benchmark separately after correctness.

The first replay, key `(0,0,0,8)`, reproduced the corpus's 635 vertices,
1,009 triangles and 2,180 flips at round 128. It converged at round 248 with
2,836 flips and a successful final certificate. Later checkpoints retained
that state. This establishes insufficient budget for this tile, not a global
repair bound or an independent geometry validation. Receipt evidence:
`/laboratory/quilting/scratch/atlas-job-state-20260905/quad-0008-repair-replay.json`.

Use the shared release toolchain:

```sh
/laboratory/fe-stuff/fe-worktrees/mb2/target/release/fe web dev fe/web/atlas-build/index.html --port 8785
node --test fe/web/atlas-build/*.test.mjs
```
