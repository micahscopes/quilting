# Quad generator integration gate

This standalone is a bounded diagnostic for the generator, not a replacement
for the **Tessellation Warp** demo. The latter is where triangle and quad tiles
share boundary skew, twist, focus, concentration, wires, and responsive rendering.

Run with the release compiler:

```sh
.toolchains/fe/target/release/fe web dev fe/web/quad-atlas/index.html --port 8781
node --test fe/web/quad-atlas/*.test.mjs
```

Four edge exponents select a square-domain key. D4 canonicalization preserves
adjacency; the renderer applies the inverse symmetry. Point selection uses the
existing immutable-generation independent-set algorithm with square-distance
exclusion. Convex seed insertion and indexed Delaunay repair share their
implementations with the triangle provider via typed coordinate/domain/metric
policies. Q14 boundary vertices remain fixed. All generated geometry remains in
GPU storage and drives an indirect draw; no CPU readback selects topology.

## Limits, not completion claims

The current live generator covers **0–3**, not the requested complete 0–8 atlas.
Both triangle and quad GPU selection now query a conservative candidate-cell
window. Uniform-density jobs visit at most 50 slots per query through LoD 8;
mixed-density windows can still be large. Triangle and square insertion now
share parallel, immutable-generation rounds with local child-face relocation.
Arbitration and face planning now use per-face atomic claims with bounded
work per point/face. Edge repair now executes in conflict-safe parallel rounds;
the one-time neighbor-index initializer is still scalar. Neither
the all-keys batch scheduler nor a complete resident quad atlas exists yet.
The full triangle atlas used by Tessellation Warp is a precomputed artifact,
not evidence that this live GPU generator already scales to level 8.

The delivery requirement is **device-side Fe/WebGPU generation**, not shipping
the 24,871,552-byte Rust-generated triangle artifact. That artifact is a legacy
baseline, not the intended startup path. Generation latency, first usable
frame, full-atlas readiness, peak memory, and downloaded bytes need separate
measurements; none is established by the small jobs below.

This is planar reference-domain Delaunay, not surface-metric uniformization.
Subsequent warps can invert triangles and do not preserve Delaunay legality.

## Parallel repair checkpoint, 2026-09-05

The live 34-pass build is `fe-render-ff07de52cc3cfbfb.json`. Triangle and quad
consumers now share parallel edge repair. A flip owns its two triangles and
all neighboring faces it touches; winner selection finishes before any
mutation. Separate final certification rejects an unfinished tile even when
the configured round budget expires. The actor uses 17 global resources but
no stage exceeds the device's eight-storage-binding limit.

`repair-browser-snapshots.json` contains fresh GPU output for all 55 canonical
preview keys plus the alternate seed. Independent geometry checks pass, every
neighbor link is reciprocal and correct, and the points/final triangle sets
match the serial reference. Repair order and flip counts need not match it.
31 jobs demonstrate multi-flip rounds; the maximum observed count is 38 rounds.
The dense all-3 job repairs 91 edges, with five flips in its first round.
`repair-budget.verify.mjs` replays the actual compiled stages into independent
diagnostic buffers: zero/one repair rounds leave drawing disabled, while a
sufficient budget publishes the mesh. These probes are not page assets.

Both consumers compile in release mode. The triangle consumer also runs in
Chromium: 19 points and 22 triangles pass exact barycentric-boundary, area,
incidence, crossing, reciprocal-neighbor, and equilateral-metric Delaunay
checks. Its evidence is under `../classic-quilting-generated/`.

This is **still preview LoD 0–3**. Higher levels, parallel index initialization,
mixed-density sampling costs, batched full-atlas residency, and startup timings
remain open. No complete-atlas speedup is inferred from the small-job results.

## Blocked compaction checkpoint, 2026-09-05

The subsequent 24-pass topology-scan build is
`fe-render-ec8c1583c88b231c.json`. Insertion now also uses independent block
scans and parallel offset assignment, sharing ownership checks with the
ordered GPU reference. Successful rounds scan only block totals on a single
invocation. Invalid plans use that GPU reference to preserve exact failure and
ordered overflow counts; no host processing is involved. This is not yet
parallel Delaunay edge repair.

`topology-scan.verify.mjs` runs the actual emitted stages on separate diagnostic
buffers. All 46 valid/error fixtures match the reference across every workspace
word, including both point generations, block tails, stale summaries, ownership
failures, and exhausted capacity. `topology-scan.browser.json` records those
observations. `topology-browser-evidence.json` records the separate complete
preview geometry comparisons. Both triangle and quad release bundles compile;
only the quad has this live browser gate.

The final total-scan shader is 22,737 bytes rather than the first build's
93,500: combining total scalar conditions with eager Boolean union avoids
duplicated error paths. Guarded buffer accesses still use short-circuit checks.
This measures shader size, not GPU generation latency. The page still does
not download or upload geometry as part of generating its tile.

Bundle `fe-render-c2445ee8cc903f07.json` has 22 passes: sample packing now uses
independent local ranks, a small block-total scan, and parallel scatter. The
block size is computed in Fe CTFE; accepted candidates retain their exact IDs.
At the LoD 8 quad candidate capacity this gives 1,024-item local scans and 512
block totals, rather than one 524,288-item scan. This is a work bound, not a
measured startup speedup, and the remaining topology scan/repair are still serial.

`compaction-browser-evidence.json` records hashes from fresh Chromium captures
of all 55 canonical preview keys plus the alternate-seed case. Each passed the
independent geometry oracle and exactly matched the previous point coordinates,
ordered triangle indices, and receipts. The diagnostic reads are not part of
production generation. `compaction.verify.test.mjs` also exhausts short state
sequences, block sizes, output clipping, and reversed dispatch order. The Fe
allocation test passes; both triangle and quad release bundles compile. Only
the quad bundle has this new browser gate. Full LoD 8 generation remains open.

## Browser evidence, 2026-09-04

Release Fe `8b7ff12ce`, shared Sonatina pin `ef13a656`, Chromium WebGPU:
four captured jobs passed their GPU sampling/insertion/repair receipts and the
independent `verify.mjs` oracle:

| Requested exponents | Seed | Points | Triangles |
| --- | --- | --- | --- |
| 2,3,2,3 | 42 | 34 | 42 |
| 0,0,0,0 | 42 | 4 | 2 |
| 3,3,3,3 | 42 | 58 | 82 |
| 0,3,1,2 | 143 | 16 | 15 |

The oracle checks canonical boundary coordinates, distinct points, strict CCW
triangles, boundary/interior edge incidence, exact square area, Euler's disk
relation, crossing edges, and local Delaunay legality using independent BigInt
incircle arithmetic. It accepts either cocircular diagonal. These four cases
are evidence, not an exhaustive all-key gate.

The candidate-window implementation was rerun on these four jobs and produced
identical complete point/index buffers and receipts. The release Fe-to-Wasm
`atlas_candidate_windows_cover_conflicts_and_bound_uniform_work_through_lod8`
gate independently checks conservative cell coverage and the uniform-query
bound. This is a work-count guarantee, not a GPU startup timing.

Snapshots were copied using a diagnostic-only WebGPU storage-copy pass because
the production buffers intentionally lack COPY_SRC. Direct buffer copies from
those buffers are invalid and initially yielded unusable zero readbacks; those
were discarded, not counted as passing receipts. No diagnostic shader or
readback is included in the demo's generated pipeline.

The initial seed-loop early-return form exposed a compiler structurization bug
(FE-QF-014 in the shared upstream report). Seed failures now join the same
checked error counters/final receipt as boundary and interior failures. That
refactoring is **not** a claim that the compiler bug was fixed.

## Parallel insertion checkpoint, 2026-09-05

The 18-pass release bundle replaces scalar
square insertion with the shared triangle/square insertion provider. After the
seed round, a pending point checks only the children of its one or two previous
containing faces (at most six), using the previous immutable face-plan offsets.
This is valid only during split-only insertion; Delaunay flips remain later.
No square coordinates are reinterpreted as triangular barycentrics.

`parallel-browser-snapshots.json` now records those four jobs from the reconciled
Fe compiler `bfd1d9e3c`, bundle `fe-render-804bb629066462c8.json`. Point buffers,
unordered triangle sets, and final receipts match the saved serial baseline;
triangle buffer ordering differs. This supersedes the earlier overly strong
byte-identical triangle claim. Each capture now records the actual manifest
and pass count and rejects a resource-generation change during readback.
The independent crossing/area/incircle oracle also passes. Source
checks pass for both demo domains. The release Fe-to-Wasm child-range test
checks valid, overlapping, out-of-range, and sentinel parent ranges.

The browser needed an explicit reload to pick up this build; the first reads
still showed the old ten-pass manifest and were not counted as new-path proof.
The verified page had eighteen passes and eleven allocated logical resources.
These remain small integration jobs, not a full-atlas startup benchmark.

Ownership validation now checks both directions in O(points + faces): faces
must name a winning proposer that requested them, and winners must own their
one or two requested faces. The former scan of every face for every winner is
removed. The new release bundle passes all four independent geometry checks.
Arbitration, face planning, compaction, and Delaunay repair still need scaling.

## Atomic-claim checkpoint, 2026-09-05

Fe `59d9a1a21` / Sonatina `b24f5bd6` compile the 20-pass bundle
`fe-render-00742decdf120500.json`. Claims use actual atomic initialization,
minimum, and observation, with separate dispatches for publication. Reversed
ID-bit priorities balance insertion along ordered boundaries without moving
or renumbering samples. Candidate queries stop once their immutable neighbor
facts decide the result. Source details and limits are in
`../../ingots/gpu/quilting_atlas_webgpu/PIPELINE.md`.

`claims-browser-snapshots.json` records the four baseline jobs. All pass the
unchanged independent geometry oracle, and their point buffers and unordered
final triangle sets match the original baseline. Flip counts may differ
because insertion priority intentionally changed.

`claims-canonical-browser-snapshots.json` records **all 55 canonical quad keys
through LoD 3**, at seed 42, captured from the actual running WebGPU pipeline.
All 1,017 resulting triangles pass boundary, area, incidence, crossing, and
Delaunay checks. The test independently enumerates the complete D4 key set to
detect missing or duplicate cases. These JSON files are diagnostic evidence,
not assets fetched by the page and not a production generation dependency.

The full LoD 0–8 pipeline is still incomplete. Parallel compaction/scans,
mixed-density indexing, parallel Delaunay repair, bounded full-key batching,
and startup/memory measurements remain. In particular, a successful small-key
coverage gate must not be reported as a complete-atlas performance result.
