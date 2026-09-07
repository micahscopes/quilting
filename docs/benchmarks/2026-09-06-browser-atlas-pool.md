# Complete Fe/Wasm browser-worker atlas acceptance

The four-worker Fe pool generated all 1,200 canonical tiles through exponent 8
in Chrome: 165 triangular tiles and 1,035 quadrangular tiles. There are no edge
ratio restrictions. Generation, job ownership, cancellation and packed retention
are Fe; the JavaScript acceptance harness observes and audits the compiler's
normal worker runtime.

| Domain | Tiles | Points | Triangles | Packed bytes |
| --- | ---: | ---: | ---: | ---: |
| Triangle | 165 | 464,872 | 901,309 | 7,267,416 |
| Quad | 1,035 | 1,705,134 | 3,173,138 | 25,859,956 |
| Total | 1,200 | 2,170,006 | 4,074,447 | 33,127,372 |

Observed elapsed time was **125.531 seconds**, including 17 ms coordinator
setup, request transfer, host audits and retention. This acceptance run overlapped
a two-job release compiler build, so it is **not a clean throughput benchmark**
or evidence of a speedup over Rust, Wasmtime, or one browser worker.

A second warm run, with this session's compiler builds stopped, passed in
**110.511 seconds** (12.2 ms setup), with identical tile/point/triangle/byte totals
and the same coordinator memory high-water mark. Workers completed 309, 290,
315 and 286 tiles. Other machine workloads were not controlled; this is one
warm timing, not a multi-round median or a cold-start result. Its per-tile
receipt is `/laboratory/quilting/scratch/atlas-pool-browser-warm-20260906.json`.

Four requests were active concurrently. Workers completed 282, 277, 305 and 336
jobs respectively. Their summed round-trip times were 499.933 seconds; these
overlap and must not be added to wall time. Host mesh audits totalled 622.6 ms.
The coordinator's linear memory reached 67,960,832 bytes, including its reserved
64 MiB packed store. This excludes child-worker memories, JavaScript allocations
and any GPU memory. Total process peak memory remains unmeasured.

The response audits checked packed length, indices, coordinate bounds, positive
triangle winding, exact summed domain area and use of every vertex. The pool
checked 1,200 distinct ordinal receipts, completion count and retained byte count.
These checks are not a new proof of Delaunay optimality, boundary permutations,
non-overlap, or the byte-for-byte correctness of every retained tile. Those need
their corresponding algorithmic tests and retained-buffer/rendering gates.

## Cancellation and ownership

Cancellation after 100 ms also passed: four in-flight results arrived, zero tiles
were published, and each response allocation was returned to arena checkpoint
67,196,569. Completion took 11.253 seconds because generation already in flight
was allowed to finish; this is not immediate preemption. The test finally closes
its own supervisors/workers.

Two genuine shared-runtime/compiler blockers were uncovered:

1. Concurrent ready completions lowered their values before an async helper
   returned, allowing allocations A/B/C/D followed by an invalid release of A.
   Shared mb2 `b65a27dc0` keeps lowering, resume and release synchronous. A
   deterministic four-task regression reproduced the failure before the fix;
   all 47 browser-runtime tests passed afterwards.
2. Shared mb2 `c1e75094a`: the continuation lifetime analysis rejected an empty range-bound placeholder
   and the copied actor-notification path. This prevented temporary storage
   reclamation above the response. The narrow follow-up admits the zero-sized
   value and nominal SendBegin alongside AskBegin under the existing closed-call
   and non-escaping-result/frame checks. The browser allocation trace changes
   from a trapped release with 180 extra bytes to exact checkpoint restoration.
   All 14 resident-actor regressions pass, including a dynamic-range/send test
   that verifies caller storage contents, exact arena cursor recovery and the
   persistent reducer's state.

The first issue alone did not fix browser acceptance. No allocator checks were
disabled and no demo-specific rewind was added.

## Reproduction and remaining gates

Build/serve instructions: [worker acceptance](../../fe/web/atlas-worker/README.md).
The full machine-readable receipt is in the project scratch area at
`/laboratory/quilting/scratch/atlas-pool-browser-acceptance-20260906.json` (1,200
per-tile receipts plus cancellation and allocation traces). The browser was
Chrome 152 on Linux, reporting 16 hardware threads.

Next: clean one-versus-multiple-worker timings, worker/total memory and finer
phase accounting, verify retained buffers and upload, then connect the atlas to
the standalone Fe/WebGPU triangle/quad warp explorers. The current page remains
a diagnostic, not a finished rendering demo. GPU atlas generation stays parked.
