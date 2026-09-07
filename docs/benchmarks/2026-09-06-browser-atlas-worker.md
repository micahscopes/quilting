# Fe atlas: actual Chrome module Worker acceptance

Verified via the installed Chrome MCP server on loopback, using `fe web dev` at
`http://127.0.0.1:42999/` (diagnostic page65 in this browser session).
Compiler-generated child/interface + unmodified Fe actor runtime, not a custom
worker shim. One worker, sequential requests, seed42. Runtime generation and
packing are entirely Fe/Wasm. These are observed samples, not benchmark medians.

Reproducible driver: `fe/web/atlas-worker/browser-acceptance.mjs`.
Artifact recipe and serving instructions: adjacent README.

| Domain / exponent key | Points | Triangles | Payload bytes | Roundtrip | Host audit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Triangle[0,0,8] | 1615 | 2970 | 24280 | 198.0ms | 2.4ms |
| Triangle[8,8,8] | 47366 | 93962 | 753236 | 7627.0ms | 18.5ms |
| Quad[0,0,8,0] | 626 | 991 | 8452 | 189.7ms | 0.1ms |
| Quad[8,8,8,8] | 39714 | 78402 | 629268 | 5801.4ms | 12.6ms |

Compile/fetch4.4ms (warm caches possible); worker startup28.7ms. Returned values
are actual Uint32Array payloads. All checked indices in range, all triangles
positive orientation, all vertices used, exact total domain area, request epoch
preserved. Invalid exponent9 returns status2 and an empty list. The test closes
its worker. This does not independently establish no overlap or global Delaunay;
the geometry audit suite supplies separate evidence for generation.

Earlier cold probes also passed (dense triangle8355ms/quad6093ms including audit).
The browser dense jobs are considerably slower than prior Wasmtime timings.
Do not explain this as copying or browser-tier effects without profiling, and
do not advertise a browser speedup based on the native/Wasmtime comparison.
Payload audits are too small to explain the gap. The prior Wasmtime benchmark
excluded packing and canonical transport, so it is not an identical timed path.

Not yet verified: full1200tile browser corpus, multiple workers, pool scheduling,
cancellation/stale publication under concurrent edits, separate phase/copy/upload
timing, render integration, or final memory budget. This page is diagnostic only.

## Follow-up: execution vs messaging

The canonical actor compiler enables optimization. Executing its dense calls in
Wasmtime gives triangle1.542s / quad1.091s (timed canonical entry, payload read
excluded). The same worker artifact called directly in Chrome, without Worker
messaging, gives triangle7.846s / quad5.655s. Messaging is therefore not the main
explanation. Log: `scratch/dense-canonical-worker-timing-20260906.log`.

Raw Fe validation exports in Chrome, triangle[8,8,8]: sampling4.646s,
sampling+CDT8.374s, packed generation8.343s. Separate executions, not additive
phase timings; this points to geometry execution, not packing. No host mesh
audit is included. Rust/Wasm triangle[8,8,8] directly in the same Chrome instance
took770/723/725ms over three calls (53511points/106252triangles), so the engine
gap is not a universal equal slowdown. These are different generated algorithms
and output densities; no compiler defect or V8-tier explanation is proven.

The validation artifact is produced via the same optional artifact-directory
environment variable in `cpu_atlas_wasm_exports_packed_geometry`. Both diagnostic
artifacts are release builds. Do not change browser flags or resume speculative
compiler optimization on the strength of these measurements alone.
