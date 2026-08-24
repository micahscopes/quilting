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

## Rust application asset-effect shadow follow-up

On 2026-08-24, the opt-in `appshadow=1` adapter began mirroring actual browser
asset acquisition into `hyperscope-app` without changing fetch, file, parsing,
or renderer authority. Startup, IndexedDB, drag/drop, the authored Blender
demo, and presentation GLBs now produce typed request/completion traces.
Replacing an in-flight request emits `cancel_asset_load` before the new
`fetch_asset`; a late completion is retained as a bounded diagnostic and does
not replace the active generation.

The checked-in `scripts/smoke-hyperscope-app-shadow.mjs` oracle exercised the
generated WASM boundary and observed:

- one fetch effect for the initial request;
- cancel-then-fetch for a replacement;
- `ignored_stale` for the first completion while the replacement stayed
  loading;
- an exact 181,808-byte ready projection for the active completion; and
- rejection of a future-timestamped effect input.

The optimized main WASM grew from 5,843,798 to 5,905,317 bytes (61,519 raw
bytes, about 1.05%) when the app reducer/FRP dependency and adapter entered the
browser graph. The current gzip stream is 2,065,554 bytes. This is accepted for
the shadow gate but must be measured again before a Leptos view or additional
application adapters are linked. Chrome runtime inspection remains pending
because no remote-debug Chrome process was running for this checkpoint.

## Rust canonical-route shadow follow-up

The next 2026-08-24 slice moved all 65 current Hyperscope URL keys, defaults,
value classes, and serialization order into `hyperscope-app::ControlSpec`.
`HyperscopeRoute` now uses numeric default equivalence, so alternate spellings
such as `3`, `3.0`, and `3.00` do not create distinct links. It retains the
browser's first-value behavior for duplicate query keys while diagnosing the
duplicate, malformed known values, and unknown keys.

The checked-in `scripts/smoke-hyperscope-route-shadow.mjs` oracle verifies that
the Rust registry exactly matches the browser serializer's 66-key order, then
checks default omission, canonical ordering, duplicates, invalid toggles,
non-finite numbers, and unknown keys through generated WASM. The opt-in
`routeshadow=1` browser path compares bounded-rate browser and Rust query
strings but does not yet write the Rust result.

This route adapter increased the optimized main WASM from 5,905,317 to
5,923,934 bytes (18,617 raw bytes, about 0.32%); gzip increased from 2,065,554
to 2,073,745 bytes. The full protocol/app/asset/route migration shadow is now
80,136 raw bytes above the 5,843,798-byte pre-app binary (about 1.37%). Chrome
runtime comparison remains pending while the remote-debug browser is closed.

## Backend-neutral render-submission shadow follow-up

The 2026-08-24 renderer slice introduced a backend-neutral retained scene,
frame-command stream, exact indexed-submission counters, and bounded parity
observer in `quilting-core`. WebGL2 now records patch draw calls, zero-instance
draws, invalid signed counts, instances, triangles, and lines immediately next
to each actual indexed submission in both its shared non-PBR path and custom
two-pass PBR path. Picking, highlighting, and fullscreen post-processing remain
explicit auxiliary work rather than being conflated with patch submission.

The observer is disabled by default and retains no cloned scene while disabled.
With `rendershadow=1`, a scene snapshot is extracted only when retained batch,
transform, visibility, or material classification changes. Each frame is then
compared entirely inside WASM; JavaScript receives only the bounded aggregate
and last comparison when `globalThis.__hyperscopeRenderShadow.refresh()` is
called. Asset, pose, scene, and frame revisions remain distinct.

`scripts/smoke-render-shadow.mjs` verified the generated exports, the route
smoke verified all 66 controls and browser serializer order, and the native
workspace suite (excluding the target-specific WASM crate), `wasm32` check,
strict touched-code clippy, app/route smokes, and 40 browser-independent tests
passed. The combined render-contract/submission/shadow slice increased the
optimized WASM from 5,923,934 to 5,944,514 bytes (20,580 raw bytes, about
0.35%); gzip increased from 2,073,745 to 2,083,582 bytes.

Live parity remains pending. The Chrome DevTools MCP was available, but the
only service on port 8888 was an unrelated `Walkie Songie` application and no
Quilting Trunk process was running. No server was started, stopped, or replaced
to manufacture a runtime result.

## Rust presentation application authority follow-up

The next 2026-08-24 slices made presentation cue links durable and moved
presentation state into the `hyperscope-app` reducer without cutting over the
working browser controller. The route registry now has 67 keys: `cue` is an
optional, non-nil UUID, every committed cue is reflected into the URL, and
startup reaches it through Rust's `jump_to_cue`. The generated-WASM smoke
verified six fixture cues, direct entry to cue four, malformed/unknown rejection,
and preservation of the preceding cue after a rejected jump.

Inside `hyperscope-app`, presentation load and start/advance/reverse/jump/clear
are reducer events/actions. Cue activation operates transactionally on cloned
presentation/navigation state, then commits both together; rejected actions do
not advance the application revision. A low-rate futures-signals presentation
projection carries the document and active resolved cue while high-rate frame
snapshots remain limited to camera/focus state. Tests cover validation
atomicity, future-action rejection, commit-fence coherence, and equal final
camera/focus state for one 1.2-second tick versus twelve 0.1-second ticks.

Together, the cue route and application presentation authority increased the
optimized WASM from 5,944,514 to 5,960,031 bytes (15,517 raw bytes, about
0.26%); gzip increased from 2,083,582 to 2,089,818 bytes. Generated app,
presentation, and route smokes; 14 `hyperscope-app` tests; 53 `hyperscape`
tests; WASM checking; strict app Clippy; Rustdoc; and raw/staged release
preflights passed. Live browser comparison remains pending because the only
available Chrome page is still the unrelated application on port 8888.

The subsequent `appshadow=1` adapter added presentation document admission and
exact resolved-snapshot comparison at cue boundaries. It sends no frame ticks
across the browser/WASM boundary and does not alter the existing controller on
error. The generated app smoke covered load/start/jump plus malformed-jump
atomicity. This adapter increased optimized WASM from 5,960,031 to 5,965,208
bytes (5,177 raw); gzip increased from 2,089,818 to 2,093,926 bytes.

The presentation pose gate then added an atomic, clock-preserving navigation
synchronization event and frame-by-frame camera/focus comparison. Its
generated-WASM oracle matched the incumbent controller exactly at the midpoint
and endpoint of the opening transition, then across twelve 0.1-second samples
of the inversion cue, including basis vectors, semantic target, focus sphere,
reflection state, and remaining transition clocks. The browser gate uses one
`tickPresentation` call only during an active transition and returns at most 25
numeric scalars, three booleans, and one reflection tag; it emits no traffic
while settled or with `appshadow=0`.

Navigation synchronization plus the pose observer increased optimized WASM
from 5,965,208 to 5,971,073 bytes (5,865 raw); gzip increased from 2,093,926 to
2,097,167 bytes. Live browser samples remain pending until a user-run Quilting
tab is available.

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

## Animated round-index shadow follow-up

On 2026-08-23, the opt-in `roundshadow=1` observer was extended to capture the
exact joint matrices and morph weights associated with each asynchronous worker
LOD result. The main Rust renderer reconstructed the corresponding 984 posed
source patches, atomically refit 984 leaves and 983 internal nodes, and only
then compared the index query with that completed classification. The ordinary
path does not clone or transfer this pose payload when the observer is off.

The installed Chrome DevTools MCP observed these release-build cases on the
animated horse tab:

| View / transform | Index candidates | GPU survivors | Visited / pruned nodes | Sampled false negatives |
| --- | ---: | ---: | ---: | ---: |
| Centered identity, 8 changing poses | 984 | 984 | 1,967 / 0 | 0 |
| Partially clipped identity, changing poses | 539–645 | 984 | 1,159–1,357 / 34–41 | 0 |
| Partially clipped inversion, pose 1.316 | 374 | 389 | 783 / 18 | 0 |
| Partially clipped inversion, pose 0.730 | 307 | 367 | 685 / 36 | 0 |
| Fully off-camera identity | 0 | 0 | 1 / 1 | 0 |
| Fully off-camera inversion | 0 | 0 | 17 / 9 | 0 |

The centered identity sample advanced from comparison 1,138 to 1,273 while
pose time changed and wrapped normally. Two later partial-identity observations
advanced from comparison 9,229 to 9,633 while candidate membership changed
from 539 to 645 with the pose. Warm observed refits were usually about
7–11 ms, with a 6.4–18.7 ms range in the retained samples; queries were about
0.7–2.7 ms. After separating pose reconstruction from refitting in telemetry,
one final partial-identity sample measured 0.3 ms to reconstruct 984 morph-posed
controls, 5.7 ms to refit the hierarchy, and 0.5 ms to query it. This single
sample is a boundary check rather than a new distribution. These are opt-in
main-thread validation costs, not costs paid by the normal renderer and not yet
an optimized production culling path.

Pausing initially exposed a lifecycle bug: the UI stopped animation without
submitting a final rest-pose LOD classification. After scheduling one
coalesced recompute on the animation signal transition, the same live observer
changed from `animated: true` to `animated: false`, refit all 984 leaves and
983 parents to the rest pose in 9 ms, retained partial-frustum pruning, and
reported zero sampled false negatives. The final console was clean.

`falseNegativeFaces` is deliberately a red-alert sample test over seven points
of each rejected rational QB patch; it is evidence against integration errors,
not a numerical proof by itself. Conservativeness instead comes from the
complete-patch source bound, certified parent containment, fail-closed query
predicates, and atomic-refit tests. The observer still never changes WebGL draw
membership. WebGL already performs current-pose vertex rejection; eliminating
rejected instance invocation is reserved for a compacted/indirect WebGPU path
or another measured submission strategy.
