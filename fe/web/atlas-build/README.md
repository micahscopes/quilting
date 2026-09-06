# Full GPU atlas construction

## Methodology audit — 2026-09-05

This is currently a **custom fixed-candidate conflict-graph sampler**, followed
by parallel insertion and Delaunay repair. It is not an implementation of
PixelPie or Wei's phase-group sampler. Exhausting its finite candidate set does
not certify continuous coverage or blue-noise statistics. The triangle
candidate generator also folds both trials into the same half of each interior
grid square; a direct Fe probe has confirmed that missing proposal support.

The latest independent review recommends preserving this pipeline as a measured
baseline and parking the uncommitted retirement-occupancy experiment. A native
wgpu comparison must separate generated code, handwritten equivalent code,
scheduling changes, and a faithful published algorithm. The existing handwritten
sampling module covers only proposal/retirement, not that end-to-end control.

Current full-corpus results are **not a complete certified atlas**. Historical
receipts contain 49 failed triangle tiles and 295 failed quad tiles due to
unfinished repair. No subsecond full-atlas result has been demonstrated.

Reference candidates: [Wei's original code](https://github.com/1iyiwei/noise/tree/master/BlueNoise),
[PixelPie](https://github.com/salivian/pixelpie), and the authors'
[gDel2D distribution](https://www.comp.nus.edu.sg/~tants/gdel3d.html).
These have distinct algorithmic, numerical, dependency and licensing conditions;
none has yet been incorporated. Detailed evidence and review disposition live
in `/laboratory/quilting/scratch/atlas-job-state-20260905/`.

## Existing baseline

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

Sampling proposal/retirement dispatches use a GPU-written indirect command
derived from the current key's candidate count. Initialization still clears the
whole reusable sampling arena, including both parity tails. This skips idle
lanes without removing candidates, changing the 64-round budget, or reducing
the atlas key set. For all quad keys, scheduled sampling invocations decrease
from 69,457,674,240 to 29,995,204,608 (56.8%). This is a static schedule census,
not GPU instructions or a measured time saving.

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
Optional dispatch overrides, whole-buffer hashes and poisoned initial storage
test active-prefix equivalence against the original compiled shaders. These
are test interventions, never production host-side dispatch decisions.

Checkpoint violation counts before the final certificate describe the last
round's pre-flip proposals. Only the final certificate recomputes violations
after the last mutation. Replays sharing a GPU with the corpus run are not
independent timing measurements; benchmark separately after correctness.

The first replay, key `(0,0,0,8)`, reproduced the corpus's 635 vertices,
1,009 triangles and 2,180 flips at round 128. It converged at round 248 with
2,836 flips and a successful final certificate. Later checkpoints retained
that state. This establishes insufficient budget for this tile, not a global
repair bound. A second replay retained the actual geometry and passed the
independent `verifyQuadSnapshot` oracle: exact boundary spacing, disk incidence,
positive orientation, exact square area, no crossing edges, and BigInt incircle
checks for every flippable interior edge (635 points, 1,009 triangles, 1,643
edges). Geometry and receipt evidence:
`/laboratory/quilting/scratch/atlas-job-state-20260905/quad-0008-repaired-geometry.json`;
`/laboratory/quilting/scratch/atlas-job-state-20260905/quad-0008-repair-replay.json`.

Keys `(0,0,0,3)` and `(0,1,3,4)` produce identical full sampling-buffer SHA-256,
receipts, and geometry with original versus active-prefix dispatch. Both pass
the independent geometric oracle. The latter also matches when the entire
sampling arena is poisoned before initialization. Evidence is saved alongside
the captures as `sampling-dispatch-equivalence.json` and
`sampling-poison-equivalence.json`. A production browser gate also passed for
`(0,1,3,4)` using the actual Fe-generated
`sample_schedule` and indirect proposal/retirement dispatches, with poisoned
initial storage. The full sampling hash and exact geometry match the original
fixed-dispatch baseline. See `production-indirect-sampling-gate.json` beside
the captures. Full-corpus indirect dispatch validation remains outstanding.

The full-build actors now select `PermutedRoundFlipClaims`; other callers
retain the original slot-order wrappers for controlled comparison. Both orders
are strict u32 permutations and preserve the closed-neighborhood ownership
rule. On the saved `(0,0,0,8)` fixture, an exact offline replay matches the GPU
slot-order baseline (248 rounds, 2,836 flips), while permutation reaches the
same geometric certificate in 122 rounds and 2,951 flips. This is not yet a
GPU performance claim or a corpus-wide repair bound. The Fe priority smoke
test executes successfully. The type-policy WebGPU build passes two exact
replays: `(0,0,0,8)` converges in 122 rounds (slot baseline 248), and
`(0,0,3,7)` converges in 60 (slot baseline 196). GPU round counts, flip counts
and the exact face arrays match the independent replay; both resulting meshes
pass the independent geometric oracle. See `typed-repair-gpu-cases.json` in
the capture directory. These are not full-corpus or timing guarantees.

The first full-corpus run also exposed presentation resize replaying the
compute-only graph after `.live()` finished, overwriting its final receipts.
Shared mb2 `6f30a3d02` fixes that host lifecycle bug; 58 runtime tests pass.
The current release CLI predates this runtime change, so its served assets do
not yet carry the fix. The old page was frozen rather than left rebuilding.

Use the shared release toolchain:

### Full triangle run, 2026-09-05

Source `7994d1b`, release Fe executable SHA-256
`99403c22267674c5e0d54e59a9f8b950ac3753be68ba846b8b56f8fdd113cd19`:
all 165 canonical LoD 0–8 jobs terminated, but only 116 tiles were retained;
49 failed. This is **not a complete atlas**. The saved completion receipt is
`[1,116,49,1,0]`. Independent directory/receipt consistency checks pass.
Every failed tile reports unsettled Delaunay repair after the 128-round budget:
1,330 remaining edge violations in total, at most 83 on one tile. Sampling
and insertion report success and no invariant failures. These are GPU-reported
gates, not an independent geometric proof of those 165 meshes.

Generation plus bounded receipt readback took 292,155.3 ms; preparation plus
generation/readback took 328,350.4 ms. Another atlas run shared the GPU, so
these are diagnostic wall times, not isolated performance measurements. They
do not satisfy the sub-second goal. Retained geometry occupies 3,075,348 bytes;
that excludes failed tiles and working buffers.

Evidence: `/laboratory/quilting/scratch/atlas-job-state-20260905/triangle-prefilter-full-receipts.json`.
`summarizeTileReceipts` classifies the individual gates and rejects a ready
directory entry whose receipt fails or reports different geometry counts.
The priority-prefilter optimization still needs a controlled GPU equivalence
comparison: finishing this run does not establish equivalence to its baseline.

An isolated compiled-Fe replay of failed triangle key `(3,3,7)` settles at
round 134, with 5,098 flips, versus 5,093 flips and three remaining violations
at round 128. The final certificate is valid and unchanged through 512
scheduled rounds. An independent BigInt replay reproduces the exact final
face array, round count and flip count. Evidence:
`/laboratory/quilting/scratch/atlas-job-state-20260905/triangle-337-repair-replay.json`.
The replay explicitly uses the equilateral metric `x²+y²+xy` on barycentric
coordinates `(b,c)`. Initially comparing with the quad's square metric gave
different faces; coordinate charts alone do not specify a Delaunay metric.
This small-case agreement does not prove a corpus-wide repair bound.

### Scheduling cost, separately from shader size

The full quad run also terminated without complete certification: 740 ready,
295 failed, no pending jobs. All 295 failures report unsettled repair, not
sampling/insertion/invariant errors. Its completion is `[1,740,295,1,0]`;
directory and receipt consistency checks pass. Saved evidence:
`/laboratory/quilting/scratch/atlas-job-state-20260905/quad-typed-full-receipts.json`.
Wall time was 1,547,367.9 ms under contention, including this run's surrounding
execution/readback overhead; it is not an isolated GPU timer measurement.
Retained geometry is 19,224,732 bytes, excluding failed tiles and scratch.

The captured quad manifest `fe-render-f3ec742143766412.json` contains 54 pass
definitions. Multiplying their nested Fe-authored repeats gives 2,047,232
dispatch commands for all 1,035 keys: 1,059,840 in repair, 728,640 in insertion,
198,720 in sampling, 31,050 in candidate-index reduction, 28,980 in other
per-tile work, and two global commands. This is a static schedule census,
not GPU instruction counts or a measured runtime attribution. Converged
repair tiles already publish zero-work indirect commands, but the repeated
commands are still encoded. Raising every tile's repair budget compounds
that scheduling cost; batching jobs needs investigation alongside shader size.
`atlasScheduleCensus` in `schedule.verify.mjs` reproduces the static count from
the compiled manifest and separately reports 1,061,910 indirect commands.
It rejects tapered graphs rather than guessing their iteration-dependent
counts. This diagnostic is not imported by the application and supplies no
production scheduling decisions.

### Compiler-failure observation is a separate gate

The checked-arithmetic compiler now emits per-invocation trap buffers. Atlas
directory/geometry receipts alone do not prove that the compiler reported no
failure. Repeated dispatches can overwrite earlier invocation status, and an
indirect dispatch can exceed the statically allocated invocation-status span.
In the captured triangle manifest `fe-render-dc98be48069b9cd1.json`, indirect
repair passes have a 256-byte trap buffer. A final zero snapshot is therefore
not a whole-run no-trap certificate.

Typed graph-epoch failure status is being developed upstream. Integration must
preserve failures across repetitions, cover indirect work, expose terminal
status, and fit the portable eight-storage-binding budget: repair proposal,
repair certification, and retention already reach eight bindings. Neither an
extra ninth binding nor CPU readback after every dispatch is an acceptable
solution. The private tile replay currently rejects compiler-output bindings
rather than silently ignoring this missing observation contract.

### Commands

```sh
/laboratory/fe-stuff/fe-worktrees/mb2/target/release/fe web dev fe/web/atlas-build/index.html --port 8785
node --test fe/web/atlas-build/*.test.mjs
```
