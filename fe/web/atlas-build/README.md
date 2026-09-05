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

Both actors pass `fe check`. The first full quad precompile failed in the
typed job cursor during Naga aggregate lowering. The smaller `atlas_jobs`
fixture reproduced it. Sonatina commit `2e3cc1f5` fixes shared fallback phi
transport and inconsistent success/exhaustion control-flow classification.
Its six-case cursor regression executes on llvmpipe, and all 133 shader tests
and 29 control-flow tests pass in release mode. Shared Fe is rebuilding with
that exact pin. These compiler tests do not prove complete atlas generation.

Captures are in `/laboratory/quilting/scratch/atlas-job-state-20260905/`.
Neither a successful full build nor subsecond timing is yet established.
Fixed sampling/insertion/repair budgets still need full-corpus convergence
testing; exhausted tiles must remain failed.

Use the shared release toolchain:

```sh
/laboratory/fe-stuff/fe-worktrees/mb2/target/release/fe web dev fe/web/atlas-build/index.html --port 8785
node --test fe/web/atlas-build/verify.test.mjs
```
