# Quad generator integration gate

This standalone is a bounded diagnostic for the generator, not a replacement
for the **Tessellation Warp** demo. The latter is where triangle and quad tiles
share boundary skew, twist, focus, concentration, wires, and responsive rendering.

Run with the release compiler:

```sh
.toolchains/fe/target/release/fe web dev fe/web/quad-atlas/index.html --port 8781
node --test fe/web/quad-atlas/verify.test.mjs
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
mixed-density windows can still be large. Square insertion and edge-flip repair
currently execute on one GPU invocation. Neither
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
