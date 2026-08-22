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

## Chess retained-payload follow-up

Commit `0a32e22` removed two browser-only retention paths without changing the
renderer-owned source buffer: decoded RGBA staging images are released after
their synchronous WebGL upload, and node-focus spheres use two linear scans
instead of retaining one JS coordinate array per unique vertex. The worker's
transferred face material/node arrays are also reused rather than copied.

The installed Chrome DevTools MCP then hard-loaded the same 94,628-face chess
asset in the disposable test tab. Geometry and materials rendered, object
selection resolved node 14, every startup phase completed, and the console had
no warnings or errors. Exact payload telemetry reported:

| Payload | Bytes retained or processed |
| --- | ---: |
| Source face instances retained for browser interaction | 19,682,624 |
| Initial face-LOD transfer | 2,271,072 |
| Stable face-node identities retained | 378,512 |
| Avoided duplicate face metadata copies | 757,024 |
| Decoded texture staging uploaded | 603,979,776 |
| Decoded texture staging retained after upload | 0 |

The focus pass covered 19 nodes and 54,562 unique vertices in 52.6 ms while
retaining no per-vertex JS maps or coordinate arrays. `performance.memory`
reported 24,881,603 bytes of used JS heap after startup, versus 637.5 MB in the
baseline observation (approximately 96% lower). That browser API is
garbage-collection-sensitive, but the 603,979,776-to-zero staging counter is
deterministic and explains the scale of the reduction.

The follow-up page had a user-carried fuzzy-focus URL, so its 6,551 ms model
phase and 4,653 ms render phase are recorded as a smoke result, not a controlled
startup comparison. A clean repeatable chess trace is still required for CPU
timing work.

## Chess direct browser-image upload follow-up

Commit `5bd1223` subdivided the retained worker/model phases. The next measured
change transferred nine decoded `ImageBitmap` handles from the loader worker
and uploaded them directly through the Rust WebGL texture cache. The existing
RGBA upload remains as a compatibility fallback. This removes worker canvas
readback, the 603,979,776-byte cross-thread RGBA transfer, the equally large
main-thread concatenation, and the WASM slice staging path without changing
texture dimensions or sampling quality.

On the first completed 94,628-face chess reload after the change, telemetry
reported `textureUploadPath: image-bitmap` and:

| Measured phase | RGBA staging trace | Direct ImageBitmap trace |
| --- | ---: | ---: |
| Worker browser texture decode | 3,381.1 ms | 2,134.4 ms |
| Main texture upload | 2,273.5 ms | 1,516.3 ms |
| Model finalization total | 6,968.0 ms | 3,283.0 ms |
| Settled observed JS heap | 25.0 MB | 30.4 MB |

These are single browser observations rather than a statistical benchmark;
the total includes unrelated noisy phases. The direct-path identity and exact
texture byte count are deterministic. A second warm crisp-view reload took
1,937.7 ms total, including 928.0 ms browser image decode and 916.1 ms texture
upload. Chrome DevTools inspection found sharp chess geometry, plausible wood
grain and UV orientation, no corruption, and no console warnings or errors.

## Foreground animated-horse follow-up

After the upload changes, the installed Chrome DevTools MCP kept the animated
horse tab selected and visible for ten seconds. The retained counters included
1,254 foreground frames and 1,248 completed LOD jobs (the page was already
running while the timed observation was prepared):

| Measurement | Value |
| --- | ---: |
| Average / maximum frame delta | 16.67 / 50.1 ms |
| Average / maximum main render CPU submission | 0.156 / 0.6 ms |
| Average / maximum LOD round trip | 7.16 / 35.4 ms |
| Average worker portion | 6.82 ms |
| Average GPU fence wait | 6.10 ms |
| Average changed faces | 8.03 |
| Average sparse transfer | 221.6 bytes |
| LOD errors / cancellations | 0 / 0 |

There were 43 coherent classifications with no changed records, not failures.
The active foreground fence wait is therefore much lower than the earlier
17.38 ms background-lifetime observation. The renderer held 60 fps apart from
one 50.1 ms hitch, and the console remained clean. This does not establish GPU
time—the worker fence includes queueing—but it bounds the current CPU and
transfer costs for the representative 984-face animation.

## Two-asset presentation follow-up

The opt-in `?presentation=1` adapter loaded the 984-face animated horse as its
primary asset and the 3,280-face Blender-authored Hyperscape fixture as a static
secondary asset. Both remained semantically distinct while the WebGL backend
packed 4,264 face records for depth-correct shared submission. The secondary
fetch returned 258,432 bytes in 14.6 ms in the observed local run.

Chrome DevTools inspection verified both cues. Cue one retained a recognizable
animated horse while hiding the resident secondary layer. Cue two showed the
horse separately above-left and the authored ground, traveler, and landmarks
to the right with plausible depth ordering. Ten authored guide nodes were
suppressed from presentation rendering, including the diagnostic wall spheres
that otherwise obscured the shot. The text card did not cover essential
geometry. Diagnostics reported two resident assets, 4,264 packed faces, both
active layer identities, no pending composition, no pending opacity, and no
error; the console contained no warnings or errors.

This establishes the checked-in fixture path, not general multi-model parity.
Secondary animation, secondary textures, per-layer fractional opacity, and a
second live authored ECS runtime remain explicit adapter limitations.
