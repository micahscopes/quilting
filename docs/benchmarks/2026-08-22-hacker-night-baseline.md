# Hacker-night baseline — 2026-08-22

Baseline commit: `dbb67dd` on branch `rust`.

This is the pre-migration baseline for the Tuesday release work. Browser data
was read from Micah's already-running `localhost:8888` tabs through the
installed Claude Chrome DevTools MCP. The audit did not reload, navigate, or
otherwise change either application page.

## Environment

- Chrome 149.0.0.0 on Linux x86_64
- AMD Radeon 780M, ANGLE OpenGL ES 3.2
- 16 logical CPU threads, 32 GiB reported device memory
- device pixel ratio 1.25
- observed canvas 1260 × 826
- not cross-origin isolated
- Trunk/WASM development build served by the user on port 8888

Both application tabs were backgrounded. Chrome therefore clamped their frame
callbacks to approximately 250 ms. The accumulated frame-rate values below are
not foreground FPS measurements. LOD job timings, startup phase timings, data
sizes, counters, and heap observations remain useful.

## Correctness baseline

| Surface | Result |
| --- | --- |
| Native Rust workspace, excluding wasm-only crates | 283 tests passed plus doc tests |
| `quilting-wasm`, `wasm32-unknown-unknown` | `cargo check` passed |
| Browser-independent JavaScript | 34 tests passed |
| Inline Hyperscope module and worker/module syntax | passed |
| Blender Hyperscape codec/conformal helpers | 11 tests passed |
| Lean 4.29 conformal-mereology project | 2,238 build jobs passed |

The first exact Lean run found that commit `4531409` did not compile despite
its reported validation. The query and descendant-pruning proofs were repaired
in `dbb67dd`; this table records the repaired baseline.

## Canonical atlas

The browser's default reachable atlas contains:

- 22 patches;
- 25,551 vertices;
- 49,150 triangles;
- 147,450 triangle indices;
- 294,900 line indices;
- 2,076,628 transferred packed bytes; and
- 1,589,416 serialized cache bytes.

The existing browser pages both hit IndexedDB. Atlas phase time was 220–222 ms,
including 28–35 ms GPU upload. A development-profile native benchmark gave:

| Maximum edge LOD | Patches | Vertices | Triangles | Median topology build |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 22 | 25,551 | 49,150 | 3.537 ms |
| 256 | 25 | 100,242 | 196,606 | 17.084 ms |
| 512 | 28 | 397,077 | 786,430 | 82.882 ms |

This confirms that default atlas topology generation is no longer the startup
bottleneck. Browser packing, serialization/cache access, transfer, and upload
are separate costs and must stay separately instrumented.

## Animated horse tab

The page reported animation `horse_A_` and no authored Hyperscape scene.

| Measurement | Value |
| --- | ---: |
| Navigation to completed render phase | 5,858.5 ms |
| WASM phase | 3,448.8 ms |
| Worker phase | 935.6 ms |
| Atlas phase, cached | 220.5 ms |
| Model phase | 356.7 ms |
| Initial render phase | 262.2 ms |
| Environment cache read + upload | 406.3 ms |
| Average LOD job | 19.29 ms |
| Average worker portion | 18.35 ms |
| Average GPU fence wait | 17.38 ms |
| Average changed faces | 275.9 |
| Average sparse transfer | 7,724.7 bytes |
| Observed JS heap | 5.8 MB |

There were 8,369 completed animated LOD jobs in the retained lifetime sample,
with no recorded cancellations or errors. Their maximum values include browser
background/suspension outliers and are not useful optimization targets.

The important signal is that ordinary horse classification is dominated by
waiting for the worker GPU fence, not by readback (`0.04 ms` lifetime average)
or main-thread batch handling (`0.94 ms`). Continuous animation still submits
one classification for nearly every background frame.

## Chess tab

The page loaded `classic_chessboard.glb` with 94,628 source faces and no
authored Hyperscape scene.

| Measurement | Value |
| --- | ---: |
| Navigation to completed render phase | 24,435.9 ms |
| WASM phase | 3,630.7 ms |
| Worker phase | 1,026.7 ms |
| Atlas phase, cached | 222.1 ms |
| Model phase | 9,890.6 ms |
| Initial render phase | 8,946.9 ms |
| Environment cache read + upload | 269.9 ms |
| Full changed-face payload | 94,628 faces |
| Full LOD transfer | 2,271,072 bytes |
| Latest LOD job | 274.9 ms |
| Latest worker portion | 135.0 ms |
| Observed JS heap | 637.5 MB |

The atlas is not the chess problem. Model preparation, first render/batch
construction, and retained/transient CPU memory are the immediate measured
targets. `performance.memory` reports JavaScript heap only; it does not include
GPU allocations or all browser-process memory.

## Console qualification

The MCP's preserved-message query returned an empty console for the chess tab.
Because DevTools attached after the page had loaded, this is not treated as
proof that startup emitted no messages. The explicit runtime counters reported
zero LOD errors and all startup phases reported `failed: false`.

## Next measurements

1. Add phase subdivisions around glTF fetch/parse, texture decode, instance
   construction, renderer upload, initial batch preparation, and first draw.
2. Record allocation/byte counts for each chess typed array and identify which
   copies survive after upload.
3. Measure a foreground 10-second horse trace rather than using backgrounded
   lifetime frame averages.
4. Add the small authored Hyperscape Blender demo and a two-asset presentation
   scene to the repeatable matrix.
5. Preserve these counters while browser authority migrates into Rust so a
   refactor cannot hide regressions by renaming or removing telemetry.
