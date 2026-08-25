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

The next checkpoint added a versioned native application replay trace with
semantic inputs, commit/rejection outcomes, compact committed state, exact
decimal `f64` round trips, cadence-invariance coverage, and a checked six-cue
golden fingerprint. Linking exact replay JSON support into the browser was
measured and rejected: it raised optimized WASM to 6,094,559 bytes (+123,486
raw, +28,054 gzip). The replay module and CLI are therefore behind the native
`hyperscope-app/replay` feature. With that feature absent from the browser
graph, optimized WASM is 5,971,066 bytes and 2,096,849 bytes gzip—effectively
unchanged from the pose-observer checkpoint (-7 raw, -318 gzip in the final
source-coherent build).

The 15 default app tests, 19 replay-feature app tests, 53 Hyperscape tests, 40
browser-independent JavaScript tests, four generated-WASM smokes, strict
Clippy in both feature configurations, Rustdoc, WASM checking, the replay
golden, and raw/staged preflights passed. The build receipt was
`255d51aa55039588583c7d4699f8a5f5`. Live browser proof remains pending; no
server or unrelated browser tab was changed for this checkpoint.

Replay version 0.2 then replaced the process-local Bevy handle in
`FocusAnchor` with a non-nil stable UUID and added a portable navigation
fixture covering all 17 current semantic navigation action variants. It
records preset/queue/sequence state, stable focus ownership, surface-transition
clock and hop, and navigation diagnostics alongside camera/focus state. The
fixture includes camera and focus edits, inversion transport, camera and
surface transitions, animated surface retargeting, cancellation, selection
anchor/detach, and an atomically rejected nil identity. Both presentation and
navigation traces round-trip exact JSON and have checked fingerprints.

The expanded replay remains entirely behind the native feature. The optimized
browser WASM is 5,969,847 bytes, 1,219 bytes smaller than the preceding
5,971,066-byte build; gzip is effectively flat at 2,096,949 bytes (+100). The
source-coherent build receipt is `dd8eb4d56f3db0ba9a1e12bd24872a1e`.

Replay version 0.4 extends that oracle across every current application event
lane. Its third portable fixture proves complete asset descriptors and exact
fetch/cancel effects, supersession, loaded/failed/cancelled state, stale
completion, presence order and exact TTL expiry, all three durable authored
commands, stale authored revisions, and atomic protocol rejection. Serialized
byte lengths and queue counts are fixed-width `u64`, and exhaustive source-enum
matches make new application, effect-completion, semantic-action, navigation,
or authored-command variants fail the oracle suite until deliberately covered.
CI now runs the three replay goldens, scoped strict Clippy/Rustdoc, 40
browser-independent JavaScript tests, and all four generated-WASM smoke tools.

The expanded oracle and CI remain outside the shipped browser runtime. The
optimized WASM remains 5,969,847 bytes; gzip is effectively flat at 2,096,933
bytes (+84 from the preceding 2,096,849-byte baseline). The source-coherent
build receipt is `bf2cf66ce7924641196d602d9a35daa3`.

The first single-application-boundary slice then exposed the incumbent
device-neutral navigation action set through `HyperscopeAppShadow` and
`AppStore`, with queue-authoritative sequencing and application-owned virtual
time. Version 0.4 records navigation admission separately from integration so
pre-tick pending/pose state matches the compatibility facade. Frames are the
usual integration boundary; transactional cue activation also performs its
required zero-time integration in shared sequence order.
Its compact frame
packet now includes preset, pending work, quaternion/basis, camera and surface
transition clocks/hop, focus/reflection, and navigation diagnostics. The
generated-WASM smoke proves exact state parity against the compatibility
`HyperscopeNavigation` facade through flight, focus/inversion, and animated
surface re-anchor/retarget/cancel. Browser authority and `navshadow=1` remain
unchanged pending live target-browser evidence.

This temporary dual-facade checkpoint grows optimized WASM from 5,969,847 to
5,983,991 bytes (+14,144); gzip is effectively flat, growing from 2,096,933 to
2,096,971 bytes (+38). The raw overhead is explicitly budgeted for removal when
the compatibility facade is backed by or retired in favor of the single
application adapter. The source-coherent build receipt is
`f0a54c89d9df0016f5c0a0d40adf2535`.

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

## Rust SpaceMouse camera-input boundary follow-up

On 2026-08-24, the offline application gate added a validated
`SpaceMouseCameraInput -> NavigationFrame` conversion in `hyperscape` without
changing the live browser controller. The adapter supplies six filtered axes,
virtual delta, the linear speed frozen when the current translation gesture
began, and user move/rotation gains. Rust owns preset normalization, Y/Z swap,
all inversion masks, exact translation and rotation response, object-mode
dolly, and horizon-lock policy. WebHID reports, response shaping, smoothing,
stale decay, button layers, gesture-speed registration, surface walking, and
focus/inversion modifiers remain browser concerns at this checkpoint.

The generated-WASM smoke compared the incumbent JavaScript mapping with the
pure Rust boundary across 7,168 combinations: four presets, two swap states,
all eight pan masks, all eight rotation masks, and 14 zero/basis/mixed axis
vectors. It separately covered 648 delta/speed/gain/horizon policy cases, four
targetful/target-free `AppStore` camera initial states, a 120-frame
deterministic trace, and atomic rejection of malformed, coercible, and
overflowing samples. The
queue adapter emits `SetPreset` then `ApplyFrame` under the shared sequence
authority and does not integrate until an ordinary tick.

This slice increased optimized WASM from 5,983,991 to 5,989,628 bytes (+5,637
raw); gzip increased from 2,096,971 to 2,098,831 bytes (+1,860). The
source-coherent build receipt is `ec1e4e53ac2f082b20cf110ced5c822d` over 153
files and 38,378,234 bytes. Native camera tests, strict core Clippy and Rustdoc,
the WASM target check, the 16 SpaceMouse adapter tests, and all four
generated-WASM smokes passed. No runtime or cadence-invariance claim is made
until the user-run Chrome adapter is explicitly wired and measured.

## Rust surface-walk response boundary follow-up

On 2026-08-24, the next offline gate moved the semantic response above the
existing animated topology walker into `hyperscape::SurfaceWalkController`.
Rust now owns scene-radius/body-scale speed, combined-input unit-disc limiting,
displayed-chart tangent velocity, contact and normal response, retained
surface-relative pitch, tangent pull, eye height, and the scale-aware near
plane. The renderer-independent contact-frame API deliberately does not yet
own attachment lifecycle, animation scheduling, LOD scheduling, or UI.

The incumbent JavaScript camera response was factored into the pure
`resolveSurfaceWalkView` helper still used by the live page. The generated-WASM
smoke compared that actual oracle against Rust for 600 animated curved-contact
frames, in addition to 2,160 scene/pace/body/height/input mappings, pitch
recapture, position-only conformal updates, reset, and atomic rejection. Exact
vector arity, nested unknown fields, non-finite values, negative deltas, and
finite-input overflow are rejected at the WASM boundary. The generated
TypeScript declaration exposes structural request/response types instead of
`any`. The live page also now detaches the Rust topology runtime when an
attached contact cannot form a valid camera frame, preventing split attachment
state while browser walking remains authoritative.

This slice increased optimized WASM from 5,989,628 to 6,029,193 bytes (+39,565
raw); gzip increased from 2,098,831 to 2,118,125 bytes (+19,294). The
source-coherent build receipt is `cc7baa878a0ef914f3515eb6379940fa` over 155
files and 38,424,153 bytes. The 61 native Hyperscape tests, 25 shared JavaScript
focus/navigation tests, WASM target check, generated-WASM surface walk smoke,
and release Trunk build passed. The next authority gate must compose this view
response, `SurfaceWalker` advancement, height/scale intent, and the existing
surface-anchor transition into one application action before deleting browser
walk state.

## Atomic Rust surface-walk aggregate follow-up

On 2026-08-24, `hyperscape::SurfaceWalkRuntime` composed the response boundary,
animated `SurfaceWalker` topology, physical-side choice, body/eye scale,
scale-relative clipping, animated material-point velocity, recovery/detach,
and the surface-anchor transition into one copy-on-commit owner. Invalid
semantic input leaves the entire aggregate and camera untouched; a topology
failure resets topology, response history, and transition together. The
surface transition also changed from generic quaternion interpolation to the
incumbent browser oracle's independent forward/up spherical smoothing followed
by Gram-Schmidt recovery, with target lens and control distance applied
immediately.

The production WASM adapter borrows the existing posed QB controls, adjacency,
and conformal state rather than duplicating mesh data. `walkimpl=shadow` mirrors
right-click attachment and per-frame semantic walking input, records topology
and camera drift in the page, and never changes the rendered result. The
default remains `js`; a requested `rust` mode is deliberately redirected to
shadow. The remaining authority blockers are target-browser measurements of
the incumbent absolute anchor clock, follower state across reflection/chart
transport, Float32-sensitive near-edge crossings, and animation pose-time
velocity.

This slice increased optimized WASM from 6,029,193 to 6,053,590 bytes (+24,397
raw); gzip increased from 2,118,125 to 2,128,315 bytes (+10,190). The
source-coherent build receipt is `d2a0b8f45a03f2be97f772a88819605e` over 156
files and 38,481,078 bytes. The release Trunk build, 69 Hyperscape tests, 41
browser-independent JavaScript tests, five generated-WASM smokes, and all
three replay lanes passed after deliberately updating only the navigation
golden for the reviewed surface-anchor interpolation semantics. Live browser
authority remains unchanged.

## Shared surface-anchor clock and typed composed boundary follow-up

On 2026-08-24, the live JavaScript oracle stopped deriving the surface
re-anchor glide from absolute `performance.now()` and now advances it from the
same explicit, clamped frame delta passed to `SurfaceWalkRuntime`. Both sides
snap normalized progress within `1e-12` of the endpoint, so ten 0.1-second
steps and one 1-second step reach the identical terminal state. This removes a
timing-only source of shadow drift during uneven rendering or background-tab
stalls without changing the default `walkimpl=js` authority.

The composed production exports now retain a structural
`ComposedSurfaceWalkResult` TypeScript union through wasm-bindgen. The
generated-WASM surface smoke calls both exports before renderer initialization
to prove their inert boundary behavior and rejects a regression back to
`any`. A live initialized-renderer trace is still required before cutover.

The optimized WASM changed from 6,053,590 to 6,053,594 bytes (+4 raw); gzip
changed from 2,128,315 to 2,124,101 bytes (-4,214). The source-coherent build
receipt is `3e4b0bf87e58c17a179f4cebab91d902` over 156 files and 38,484,379
bytes. The complete native workspace suite, 69 direct Hyperscape tests, 25
all-feature application tests, 42 browser-independent JavaScript tests, 19
Blender interchange tests, strict application Clippy/Rustdoc, all three replay
goldens, the WASM target check, release Trunk build, all five generated-WASM
smokes, and the ordinary release preflight passed. Reflection/chart follower
transport, Float32-sensitive edge crossings, animation pose-time velocity,
and target-browser composed diagnostics remain explicit cutover gates.

## Atomic reflection-chart surface walking follow-up

On 2026-08-24, `ReflectionTransport` became the shared exact map for camera
and surface point/direction transport between Euclidean and spherical
reflection charts. `SurfaceWalkRuntime` now stages the follower, filtered
contact frame, physical-side parity, previous posed positions, and active
anchor transition before committing any of them. A pole in any staged value
rejects the complete navigation edit. A chart-parity change flips the physical
side exactly once, and an animated chart edit cancels the incumbent
partition-dependent anchor glide before it consumes another time slice.

The production WASM adapter applies that transaction to both the legacy and
composed walkers. It rebases their cached position once after a successful
chart edit so the semantic f64 reflection followed by the renderer's f32
Mobius packing cannot appear as false surface velocity. The browser now treats
manual or authored-camera reflection edits as one outer transaction: camera,
JavaScript follower, Rust aggregate, previous reflection state, and renderer
Mobius state either all advance or all remain unchanged. The generated
TypeScript boundary exposes a structural seven-field transport diagnostic.

The optimized WASM changed from 6,053,594 to 6,057,645 bytes (+4,051 raw);
gzip changed from 2,124,101 to 2,131,143 bytes (+7,042). The source-coherent
build receipt is `258938e4bb3576c8ac3b1838073dd2a8` over 156 files and
38,518,088 bytes. The complete native workspace suite, 75 direct Hyperscape
tests, 25 all-feature application tests, 42 browser-independent JavaScript
tests, 19 Blender interchange tests, strict Hyperscape and application Clippy,
application Rustdoc, all three replay goldens, WASM target check, release Trunk
build, five generated-WASM smokes, ordinary release preflight, and two
executable Node/WASM reflection-transport tests passed. An initialized target-
browser trace, Float32 edge-crossing parity, and explicit pose-time sampling
remain cutover gates before Rust surface walking can become authoritative.

## Semantic animation pose-clock follow-up

On 2026-08-24, animated surface walking stopped dividing an asynchronously
delivered pose displacement by the next browser render delta. Each issued pose
request now carries its clip time, a monotonic semantic sample time, a caller-
owned revision, and a continuity epoch. The worker echoes that stamp; the
browser retains one in-flight evaluation plus only the newest pending request,
rejects old epochs after clip/model changes, and WASM rejects a stale or
malformed stamp before either CPU or GPU pose state changes. Pause, paused
scrub, clip switch, model reset, and clip wrap explicitly rebase continuity.
Legacy and composed walkers retain independent stamped contact histories, use
the semantic sample delta for the first step that observes a new pose, emit
zero for a held revision, and preserve the separate one-shot reflection-chart
rebase.

The executable Node/WASM adapter suite now runs three tests. It proves the same
surface velocity with render deltas of 1/1000 and 1/3 second, a 0.2-second pose
sample delta, a coalesced two-upload secant over 0.3 seconds, zero velocity for
a repeated revision, zero on an epoch change, reflection rebasing for both
walkers, pole rollback, and atomic rejection of duplicate, stale, non-finite,
zero-revision, and non-monotonic packets. The generated declaration exposes
the seven-argument stamped upload and typed pose diagnostics; the ordinary
surface smoke still covers 2,160 mapping cases and 600 incumbent-parity frames.

A fresh initialized horse trace used Claude's installed Chrome DevTools MCP on
the user-run release server. Right-click attached both walkers to face 16. Over
120 animated frames crossing continuity epochs 60–62, every discontinuous
sample had zero velocity, continuous sample deltas stayed between 0.0165 and
0.0168 seconds, topology drift remained 0, camera drift remained 0, and the
maximum accumulated camera comparison error was 1.36e-7. Pausing produced an
epoch-60 rebase on the first observed frame followed by 29 held frames with
exactly zero surface/projected velocity. A paused scrub from clip time 1.0956
to 0.555 advanced epoch 77→78, rebased exactly once, then remained at zero for
the held pose. The page reported no warnings or errors.

The optimized WASM changed from 6,057,645 to 6,061,392 bytes (+3,747 raw);
gzip changed from 2,131,143 to 2,133,266 bytes (+2,123). The source-coherent
build receipt is `5114a03cbd013eebf39e44a446b18320` over 156 files and
38,535,660 bytes. The full native workspace, 75 direct Hyperscape tests, 25
application/replay tests, 42 browser-independent JavaScript tests, 19 Blender
interchange tests, all three replay goldens, strict Hyperscape/application
Clippy, application Rustdoc, three executable Node/WASM adapter tests, all
five generated-WASM smokes, release Trunk build, and ordinary release
preflight passed.

## Transactional pole rollback and opt-in Rust authority gate

On 2026-08-24, the remaining outer pole-rejection split was closed. The
camera, JavaScript follower, Rust aggregate, renderer Möbius state, URL, and
the public transform/focus controls now form one transaction. A rejected
center, radius, or transform request restores its signal controls in one
suppressed batch; programmatic focus-sphere edits restore their preceding
geometry, and wheel, interpolation, double-tap, and SpaceMouse callers do not
commit derived margin or transition state after rejection.

The deterministic initialized-browser oracle used Patch Lab at identity with
camera eye and proposed reflection center both exactly `[0, 0, 3]`. Clicking
Inversion preserved the identity button, all four center/radius controls, URL,
surface state, and Rust transport counters. Before the change, only the
renderer/URL transaction rolled back and the Inversion button remained active.

The executable WASM suite now runs four tests. Its new adapter oracle starts at
a pick-like `f32::EPSILON` distance from a shared edge, crosses that edge under
both identity and non-binary-exact Float32 sphere-reflection charts, and covers
all three cyclic source-corner by three cyclic neighbor-corner permutations:
18 cases total. The legacy Float32 boundary and composed f64 runtime agree on
attachment, face, one edge crossing, mapped shared-vertex weights, and output
position; projected velocity remains within one `f32::EPSILON` after the
intentional adapter rounding.

Those gates enabled a real `walkimpl=rust` mode. Rust's composed snapshot now
exports its filtered contact position, normal, tangent, relative pitch, camera,
and anchor phase. The browser applies that packet as authority while retaining
the legacy walker as a rollback diagnostic. In Rust mode only, that diagnostic
receives the same-frame Rust semantic velocity instead of planning from the
previous animated normal; `walkimpl=shadow` remains the unchanged JavaScript
oracle and the release default remains `walkimpl=js` during soak.

On the user-run release server, a fresh animated horse trace right-clicked into
Rust authority and then held forward for 120 frames. All 120 frames remained
attached and finite while crossing nine faces; topology drift and camera drift
were both zero, maximum legacy-shadow barycentric error was
`1.1526815446583072e-8`, and maximum camera error was
`8.913883969841052e-8`. An attached reflection edit committed exactly one Rust
transport, stayed finite for 60 frames, and produced no topology or camera
drift. Chrome reported no warnings or errors.

The optimized WASM changed from 6,061,392 to 6,061,935 bytes (+543 raw); gzip
changed from 2,133,266 to 2,133,257 bytes (-9). The source-coherent build
receipt is `d628592c228e87fd661257cc9e1258f9` over 156 files and 38,549,761
bytes. The full native workspace, 75 direct Hyperscape tests, 25
application/replay tests, 42 browser-independent JavaScript tests, 19 Blender
interchange tests, all three replay goldens, strict Hyperscape/application
Clippy, application Rustdoc, four executable Node/WASM adapter tests, all five
generated-WASM smokes, release Trunk build, ordinary release preflight, the
initialized pole rollback, and the final Rust-authority Chrome trace passed.

## Rust focus-chart transaction gate

The next 2026-08-24 checkpoint closed the equivalent split inside the shared
Rust navigation controller. Each ordered `NavigationAction` and the reflection
reconciliation it requests are staged together. If camera, in-flight camera
transition, or surface-walk transport reaches a reflection pole, camera,
focus sphere and enablement, active reflection, and surface state return to the
preceding coherent snapshot. The rejected sequence is still consumed exactly
once and produces one diagnostic. If an already-running inversion-sphere
transition reaches a pole later, it returns to the last active sphere and
stops instead of retrying an inconsistent desired chart every frame.

The application reducer and generated-WASM facade now execute the same exact
camera-eye pole oracle: identity camera eye `[0, 0, 3]`, proposed center
`[0, 0, 3]`, and radius `2`. The action leaves reflection identity and
`inversion_enabled=false`, preserves the complete camera/focus packet, empties
the queue, records sequence zero as consumed, and exposes the pole diagnostic.

Browser route hydration now batches inversion center, radius, and transform
mode as one chart request. Radius is normalized to the shared positive minimum
of `0.011` world units (`0.11` control units), eliminating the prior state in
which a zero-radius deep link displayed sphere reflection while the renderer
used identity. A live Chrome-MCP route from `mr=0` settled at `mr=0.11`; URL,
active button, and Rust navigation all reported sphere reflection. The
initialized camera-eye pole click preserved the identity URL/button, all four
sphere controls, and renderer surface counters.

Validation passed for the native workspace excluding the target-specific WASM
crate, 77 Hyperscape tests, 26 application/replay tests, 42 JavaScript tests,
strict no-dependency Hyperscape/application Clippy, all three replay goldens,
four executable Node/WASM tests, all five generated-WASM smokes, release Trunk
build, and ordinary offline preflight. The plain native workspace command is
not a valid WASM check because `glow::Context::from_webgl2_context` exists only
for `wasm32`; `wasm-pack test --node` is the executable gate for that crate.

The optimized WASM is 6,063,048 bytes raw and 2,133,614 bytes gzip. The
source-coherent build receipt is `69447c5415089632bf249e09db1231a5` over 156
files and 38,556,122 bytes. Preflight retains the expected local-GLB and horse
distribution-policy warnings.

## Selected-focus source/output projection gate

The following 2026-08-24 checkpoint completed the Rust read model needed for
selection cutover. `FocusAnchor` now retains one non-nil stable entity ID, its
ordinary-space bound, the exact clicked/object source pivot, and the fitted
margin. The application derives the selected pivot and bound radius in the
active output chart from that source state. Identity and sphere-reflection
views therefore cannot disagree about which point is selected, and a pivot at
the reflection pole clears only the two derived output values; source
selection remains intact. Detach removes ownership and its transition without
resetting the settled sphere, focus mode, inversion, or chart.

Both generated-WASM navigation facades expose matched `anchorFocus` and
`detachFocus` oracle calls. Their snapshots include the same selected-focus
packet, but the application reducer remains the intended authority. The Node
smoke compares both facades through identity, reflection, detach, and a
selected-pivot pole. Both adapters reject a nil entity UUID before queue or
revision mutation. A live isolated Chrome-MCP probe on page 3 reported
identity pivot/radius `[4,0,0]`/`2`, reflected pivot/radius `[1,0,0]`/`0.5`,
no packet after detach, and a pole packet retaining only source identity,
bound, pivot, and margin.

Application replay moved to version 0.5. It records selected source
bound/pivot/margin and derived output pivot/radius, while accepting version
0.4 anchor actions by defaulting an omitted pivot to the bound center. The new
goldens are `2ac1c692f6a6bfadf5bc5f5565d54823` for the presentation,
`b51c6b71ebc2031b7ca0c21239db6eed` for navigation, and
`4a0661ea377eaf22dd1130f60fc6b5cd` for orchestration.

Validation passed for the full native workspace, 82 direct Hyperscape tests,
34 application/replay tests, 42 browser-independent JavaScript tests, 19
Blender interchange tests, strict no-dependency Hyperscape/application
Clippy, Rustdoc, all replay checks, four executable Node/WASM tests, all five
generated-WASM smokes, release Trunk build, ordinary offline preflight, and
the live Chrome-MCP projection oracle. The optimized WASM is 6,070,198 bytes
raw and 2,137,294 bytes gzip, increases of 7,150 and 3,680 bytes respectively.
The source-coherent build receipt is `e89e73b62f6843b97c6d3df54df7de4d`
over 156 files and 38,578,327 bytes. The remaining browser cutover blockers are
durable asset-scoped node identity, dispatching renderer picks through the one
`AppStore` queue, effective focus-enable parity, lens/FOV transport, and
explicit free-tangent versus semantic-target camera policy.

## Rust-authoritative lens and aim-policy gate

The next 2026-08-24 checkpoint moved the complete perspective lens and camera
aim representation through the same ordered Rust navigation queue. A validated
`SetPerspectiveLens` action changes FOV, near, and far together and rewrites
both endpoints of active camera or surface-anchor transitions, so a glide
cannot restore stale projection values. Invalid lenses are consumed with a
diagnostic and no partial mutation. `SetSemanticTargetEnabled` changes only the
camera representation: enabling captures the current view target without
moving the camera, disabling restores free-tangent transport, and authored
transition endpoints retain any deliberate target they already had. Attached
surface walking rejects point-target mode transactionally because its camera
contract is target-free. A successful-action sequence fence, observed by the
presentation runtime, ensures that only an integrated manual aim edit preempts
a temporarily deferred authored target; future and rejected edits do not.
Enabling finite-target mode during a live camera glide solves a virtual target
endpoint whose current-clock sample equals the live view target, preserving the
existing eye/orientation/lens path without a next-frame aim discontinuity.

Application replay is now version 0.6. It accepts 0.4 and 0.5, performs the
omitted-selection-pivot migration only for 0.4, and rejects lens or aim-policy
actions under either legacy version. The checked fingerprints are
`5f1f3eb11b992f90f2653481d0d4fc5c` for the six-cue presentation,
`144908e4a979c1a599cfb99f5ca7d4ef` for navigation, and
`b6f2ab91e8eaa933b056768bc20cdb56` for orchestration.

Both generated-WASM facades now synchronize and report identical non-default
lenses and explicit target presence. The browser retains a target-presence bit
independent of inversion, sends the 75-degree default through Rust, restores
the bit from presentation snapshots, and queues slider edits through the
application shadow. Its independent transport oracle now rejects a target
pole instead of silently changing from point-target to free-tangent semantics.
An isolated live Chrome-MCP probe on page 3 observed 75 degrees,
`near=0.01`, `far=10000`, exact free-target parity, then exact `[0,0,0]`
point-target parity after enabling the mode. The page console had no warnings
or errors. A post-audit isolated probe also confirmed that WASM's `undefined`
free target and the browser's `null` free target both normalize to absent,
with no false mismatch.

Validation passed for 89 direct Hyperscape tests, 38 application/replay tests,
43 browser-independent JavaScript tests, strict no-dependency Hyperscape and
application Clippy, Rustdoc, all three replay checks, WASM target checking, the
five generated-WASM smokes, four executable Node/WASM tests, release Trunk
build, and the live Chrome probe.
The optimized WASM is 6,072,171 bytes raw and 2,137,056 bytes gzip, an increase
of 1,973 raw bytes and a decrease of 238 gzip bytes. The source-coherent build
receipt is `ce5f01b5fe1935b14ae6c6689ce8d7ac` over 156 files and
38,610,321 bytes.
Remaining cutover blockers are effective focus-enable parity, durable
asset-scoped selection identity/pick dispatch, and the raw ECS
`ProjectionCamera` lens/target path.

## Effective spheroidal-focus authority gate

The following 2026-08-24 checkpoint separated three concepts that the browser
prototype had allowed to drift together: existence of the retained shared
sphere, enablement of spherical inversion, and enablement of the spheroidal
depth-of-field effect. Rust `FocusNavigation.focus_enabled` now corresponds
only to fuzzy post-processing enabled in mode 3. Legacy modes 0–2 remain
renderer settings, while inversion or sphere editing can retain an active
sphere without silently enabling focus. URL-style boolean values are parsed
strictly, so `"0"` cannot become enabled through JavaScript truthiness.

The browser observes only semantic focus enablement, shell coordinate, and
angular aperture. One synchronous signal burst is coalesced into one microtask
and queues `SetFocusEnabled` plus `SetFocusField` exactly once through each
active Rust parity controller. Renderer-only blur radius, strength, quality,
and normalization do not create navigation traffic. Applying an authored Rust
presentation snapshot suppresses this adapter and cannot echo the same actions
back into Rust. Initial `AppStore` setup now synchronizes the complete browser
camera, lens, aim policy, focus field, inversion, and sphere rather than
retaining constructor defaults until a later presentation action.

An isolated Chrome-MCP probe used
`?animate=0&lab=triangle&fuzzy=1&fmode=3&appshadow=1&navshadow=1&rendershadow=1`.
At startup, the incumbent controller and `AppStore` agreed exactly on the
75-degree lens, `near=0.01`, `far=10000`, absent semantic target, sphere center
`[0.5,0,0]`, radius `2`, focus coordinate `0.62`, angular aperture `0.1`,
focus enabled, and inversion disabled. Disabling focus and changing its shell
to `0.35` in one synchronous burst produced one synchronization: four total
boundary calls across the navigation and application controllers, with two
application calls. Re-enabling focus produced the second synchronization and
the expected cumulative counts of eight and four. Both snapshots retained
exact parity, all eight observed render-shadow frames matched, the canonical
URL retained focus-only state, and Chrome reported no warnings or errors. The
disposable page was then closed without disturbing the user tabs.

Validation passed for 89 direct Hyperscape tests, 38 application/replay tests,
44 browser-independent JavaScript tests, all three unchanged replay goldens,
four executable Node/WASM tests, all five generated-WASM smokes, the release
Trunk build, ordinary offline preflight, and the live Chrome probe. The
optimized WASM remains 6,072,171 bytes raw and is 2,136,978 bytes gzip, a
78-byte gzip reduction. The source-coherent build receipt is
`a6e814abb309e8300e31906bff4b26ed` over 156 files and 38,615,656 bytes.
Remaining selection cutover blockers are durable asset-scoped node identity,
renderer-pick dispatch through the single `AppStore` queue, and Rust-owned
selection transition clocks. The raw ECS `ProjectionCamera` lens/target path
also remains before the application projection can become the sole camera
authority.

## Atlas topology-policy oracle

The 2026-08-24 audit found that the live runtime does not independently
blue-noise sample every resident LOD triple. The direct Bridson/constrained-
Delaunay builder remains available, but production calls the hierarchical
subset builder. Under 2:1 grading it independently samples only the three
irreducible families `[1,1,1]`, `[1,1,2]`, and `[1,2,2]`; all higher patches
are exact four-way midpoint descendants. The 2:1 within-face rule is a
separate resident-LOD policy, not a seam or atlas requirement.

`TessellationAtlas::build_for_keys` now compares direct and hierarchical
construction over the identical ratio-bounded key set. Both modes preserved
every requested boundary count at 2:1 and 4:1. The benchmark also runs the
production fixed-point reconciler with an explicit, power-of-two policy ratio
on a 24×24-cell triangulated grid containing one central LOD-64 peak. The live
wrapper remains fixed at 2:1.

The measured LOD-64 results from three post-warm-up release rounds were:

| Topology / ratio | Atlas keys | Atlas triangles | Serialized bytes | `2/64/64` resident triangles | Promoted halo faces | Grid requested → resident triangles | q p01 | Median build |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hierarchical 2:1 | 19 | 12,286 | 401,936 | 3,072 | 63 | 5,247 → 16,362 | 0.600 | 1.484 ms |
| Hierarchical 4:1 | 34 | 19,106 | 626,532 | 2,816 | 18 | 5,247 → 8,742 | 0.400 | 2.658 ms |
| Direct blue-noise 2:1 | 19 | 18,234 | 592,272 | 4,386 | 63 | 7,751 → 22,412 | 0.696 | 132.862 ms |
| Direct blue-noise 4:1 | 34 | 26,742 | 870,884 | 3,192 | 18 | 7,751 → 12,200 | 0.686 | 197.729 ms |

The conservative-waste concern is therefore real: on this fixed topology,
4:1 reduced the hierarchical promotion halo from 63 to 18 faces and resident
scene triangles by 46.6%, while increasing the complete LOD-64 atlas by 55.9%.
Fresh direct blue-noise topology materially improved first-percentile triangle
shape, but used more triangles and took roughly two orders of magnitude longer
to build. These numbers support testing 4:1 as a scene-work reduction and
keeping direct topology as an offline/background quality candidate; they do
not yet authorize a live default change. Browser cache/startup and representative
animated-scene measurements remain the cutover gate.

Validation passed for 122 direct core tests, 14 integration tests, four atlas
example tests, 44 browser-independent JavaScript tests, all three unchanged
replay goldens, four executable Node/WASM tests, and all five generated-WASM
smokes. The optimized WASM is 6,074,303 bytes raw and 2,138,023 bytes gzip:
a 2,132-byte raw / 1,045-byte gzip increase (0.035% / 0.049%) for the explicit
measurement API, with no atlas payload or live grading-policy change. The
source-coherent build receipt is `674ec7cc32a6f39663aadc42ccd8edca`
over 156 files and 38,631,448 bytes.

## Asset-scoped selection identity gate

The next 2026-08-24 checkpoint made durable selection identity an explicit
`(asset ID, entity ID)` pair from the protocol boundary through
`FocusNavigation`, the ordered application queue, its compact selected-focus
snapshot, replay, and both generated-WASM navigation facades. A glTF node
index or a renderer composition offset remains a transient handle and cannot
be mistaken for authored identity. Tests cover the same entity UUID in two
assets and prove that the application does not expose a queued selection until
the next frame integration boundary.

Application replay is now version 0.7. Current focus actions serialize
`asset_id` and `entity_id`; versions 0.4, 0.5, and 0.6 remain readable, but a
pre-0.7 anchor without explicit asset scope fails closed rather than acquiring
a fabricated durable asset. Version 0.4 alone retains its omitted-pivot
migration, and version 0.6 retains the complete lens and semantic-target
policy. The checked fingerprints are
`4d8598faf9db62e8500d49d94ead89ed` for the six-cue presentation,
`4b6f0b82cf471af7af17b99ed37317d4` for navigation, and
`2cb74a642b3d4fc40b4eda777addb833` for orchestration.

Validation passed for 89 direct Hyperscape tests, 6 protocol tests, 42
application/replay tests, 44 browser-independent JavaScript tests, four
executable Node/WASM tests, the five generated-WASM smokes, WASM target
checking, strict no-dependency application Clippy, Rustdoc, release Trunk
build, all three replay checks, and ordinary offline preflight. The optimized
WASM is 6,075,989 bytes raw and 2,138,520 bytes gzip,
increases of 1,686 and 497 bytes respectively. The source-coherent build
receipt is `cc8d7e32d05df457d4cb79ae738ae135` over 156 files and
38,642,396 bytes.

This checkpoint does not claim browser pick cutover. The next bounded step is
to export authored stable node IDs from validated glTF metadata, retain an
explicit durable asset ID through primary/secondary composition offsets, and
dispatch only mapped picks through the application shadow. Unmapped ordinary
GLBs must remain usable without synthesizing a durable UUID; renderer node and
face indices remain presentation handles.

## Durable browser-pick shadow gate

The following 2026-08-24 checkpoint connected the asset-scoped Rust contract
to browser picks without changing renderer authority. The validated
Hyperscape glTF loader now exports a dense `node_stable_entity_ids` table for
authored assets; null preserves an unbound source-node slot. Ordinary GLBs
return an empty table, avoiding one cloned null per node on large assets. A
generated one-triangle GLB smoke exercises both cases through the real WASM
loader rather than a mocked result.

Presentation composition builds a transient packed-node lookup containing the
durable asset ID, authored entity ID, and asset-local source node. It includes
only nodes that actually occur in pickable faces, so a high-index camera or
guide cannot collide with the next asset's face-based namespace. Each append
is staged and validated before the shared map changes. Tests cover primary and
secondary offsets, the same entity UUID in different assets, non-pickable
stable nodes, collision rejection, and atomic malformed-input rejection.

Asset provenance fails closed. Only an exact fetch from a validated
presentation manifest URI can authorize the primary asset ID. IndexedDB,
drops, basename matches, local fallback URLs, and session-generated load IDs
remain renderer-only. Secondary assets retain the explicit manifest identity
that initiated their fetch. A mapped click queues `AnchorFocus` and integrates
it at a zero-time application boundary; clear or replacement by an unmapped
pick queues `DetachFocus` only when an application selection exists. Source
identity, bound, clicked pivot, and margin are compared against the Rust
snapshot. Ordinary/unmapped picks keep the incumbent tint/focus behavior and
never synthesize a UUID.

Validation passed for 24 glTF metadata tests, 46 browser-independent
JavaScript tests, the inline browser-module syntax check, the five
generated-WASM smokes, four executable Node/WASM tests, WASM target checking,
release Trunk build, all three replay checks, and ordinary offline preflight.
The optimized WASM is 6,076,832 bytes
raw and 2,138,790 bytes gzip, increases of 843 and 270 bytes respectively.
The source-coherent build receipt is `9f53971741562b58c6c397a7e8b73e86`
over 156 files and 38,653,415 bytes.

This remains a shadow gate. The checked-in presentation meshes do not yet
provide representative pickable stable IDs, the AppStore transition clock is
not the renderer clock, and the renderer still consumes incumbent browser
focus state. Those are the next cutover gates; this checkpoint proves the
identity/provenance join and ordered dispatch without weakening ordinary GLB
loading.

## Runtime LOD-grading policy gate

The 2026-08-24 follow-up made the measured 4:1 policy an explicit,
rollback-safe runtime experiment rather than silently changing the 2:1
default. `FaceLodGrading` is the backend-neutral Rust authority and admits only
2:1 or 4:1. The main renderer uses it in the production sparse fixed-point
reconciler; the educational Patch Lab calls the same policy; the WASM atlas
builder restricts residency to its reachable keys; and the Rust route registry
validates `lodratio=2|4`. Unsupported ratios fail closed without replacing the
last valid atlas.

The browser control and route are deliberately reload-to-apply. This prevents
a partial live switch from trying to demote topology that was previously
promoted or requesting keys absent from the resident atlas. IndexedDB keys and
payload metadata include the active ratio and maximum exponent. Patch Lab
diagnostics compare the policy returned by its worker with the renderer's
active policy in addition to checking shared-edge equality.

The default exponent-7 release probe reported:

| Policy | Atlas keys | Atlas triangles | Serialized bytes | Independent blue-noise seeds | Promoted halo faces | Grid requested → resident triangles | Median build |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hierarchical 2:1 | 22 | 49,150 | 1,589,448 | 3 | 84 | 17,535 → 62,214 | 16.225 ms |
| Hierarchical 4:1 | 40 | 76,450 | 2,475,216 | 6 | 30 | 17,535 → 31,584 | 19.066 ms |

Thus 4:1 reduced this sparse-peak resident workload by 49.2% and the promotion
halo by 64.3%, while increasing complete-atlas triangles by 55.5% and bytes by
55.7%. Both policies had zero boundary mismatches. The normalized first-
percentile triangle quality changed from 0.600 to 0.400, which is why 4:1
remains an experiment pending representative horse/chess image and frame-time
soak rather than becoming the default from a synthetic workload alone.

Validation passed for 123 core tests, 14 integration tests, 27 application
tests, four executable Node/WASM tests, 46 browser-independent JavaScript
tests, WASM target checking, inline browser and worker syntax, all five
generated-WASM smokes, the release Trunk build, the ordinary offline preflight,
and both exponent-6 and exponent-7 policy probes. The optimized WASM is
6,082,036 bytes raw and 2,140,433 bytes gzip, increases of 5,204 and 1,643
bytes respectively. The source-coherent
build receipt is `69f31fa6cabdf7a881b5dca9c521a467` over 156 files and
38,661,952 bytes.

## Application-frame selection-transition shadow gate

The next 2026-08-24 checkpoint removed the presentation-only clock gap without
changing browser or renderer authority. `HyperscopeAppShadow::advanceFrameQuiet`
dispatches the ordinary validated `AppEvent::Frame` lane without serializing an
`AppCommit`. The browser calls it once per rendered frame when `appshadow=1`,
but requests a navigation snapshot only while presentation or mapped-selection
parity is active. `appshadow=0` still allocates no application controller and
returns before crossing WASM.

A selection event between animation frames first advances the application to
the event timestamp before queuing `AnchorFocus`. Active selected-focus frames
then use the incumbent focus transition's wall-clock cadence rather than the
render loop's 250 ms clamp, including after background throttling. Presentation
frames preserve their established clamped delta. The shadow compares center,
radius, anchor, focus/inversion enablement, and remaining transition time. A
compact renderer diagnostic exposes the already-retained CPU focus sphere,
enablement, and selected node; it performs no GPU synchronization or readback.

Chrome DevTools MCP 1.7.0 recorded no-reload steady traces of the same triangle
lab with the shadow enabled and disabled. Each trace contained 300 target-page
animation callbacks:

| Gate | Frame median | Frame p95 | Frame p99 | Application-lane CPU sample |
| --- | ---: | ---: | ---: | ---: |
| `appshadow=1` | 0.420 ms | 0.725 ms | 1.015 ms | 2.1 us/frame self; 9.6 us/frame inclusive |
| `appshadow=0` | 0.372 ms | 0.600 ms | 0.957 ms | no samples |

The 48 us median and 125 us p95 whole-frame differences are conservative
single-run measurements and include ordinary profiler/system noise; they are
not a broad rendering benchmark. A separate fresh-page probe observed 226
application frame calls for 226 rendered frames, zero settled-frame snapshots,
zero frame errors, zero mismatches, and no console warnings or errors. The raw
traces were intentionally kept outside the repository under `/tmp`.

The generated-WASM oracle proves quiet-frame cadence parity and atomic invalid
time rejection against the standalone navigation controller. Live mapped-pick
transition evidence remains unavailable because the checked-in presentation
meshes do not yet contain representative pickable stable IDs. Accordingly this
is still a shadow gate: the renderer continues consuming incumbent browser
focus state, and no selection authority changed.

Validation passed for 43 application/replay tests, 46 browser-independent
JavaScript tests, four executable Node/WASM tests, all five generated-WASM
smokes, WASM target checking, the inline browser-module syntax check, strict
no-dependency application Clippy, Rustdoc, all three unchanged replay goldens,
the release Trunk build, ordinary offline preflight, and the Chrome MCP probes.
The optimized WASM is 6,083,489 bytes raw and 2,136,370 bytes gzip: a
1,453-byte raw increase and 4,063-byte gzip reduction relative to the runtime
LOD-grading checkpoint. The source-coherent build receipt is
`e922f7cbd2b2e2dc6b3edafea6d4543e` over 156 files and 38,671,378 bytes.

## Persistent Blender identity and live mapped-selection gate

The next 2026-08-24 checkpoint closed the release-asset half of the selection
authority gate. Blender object bindings now store a persistent stable entity
UUID, expose explicit UUID generation in the object panel, emit it into
Hyperscape glTF metadata, and restore it on import. The reproducible demo
exporter creates the `.blend`, `.gltf`, `.bin`, and `.glb` variants from one
authored scene and asserts five deterministic IDs. Four belong to pickable mesh
nodes; the projection camera is intentionally non-pickable.

The headless fixture generator now removes Blender's factory Cube and Camera
before running the otherwise additive demo operator. The unbound factory Cube
had been exported into the checked fixture, visually occluded authored nodes,
and correctly produced an unmapped pick. Removing it from fixture generation
keeps the interactive operator non-destructive while preventing unrelated
startup objects from becoming release content.

Chrome DevTools MCP loaded the final two-asset presentation and selected the
authored nested landmark. The application reported one mapped pick, one
selection dispatch and comparison, two sampled transition frames, two retained
renderer-focus comparisons, zero selection/transition/renderer mismatches,
zero frame errors, an empty global mismatch list, and no warning or error
console messages. A direct pick also disables the now-inactive presentation
pose observer before focus/lens signal synchronization; a later cue activation
resynchronizes that observer explicitly. This prevents the selection action's
additional sequence from being compared with a presentation controller which
no longer owns the transition.

Validation passed for 19 dependency-free Blender tests, the Blender 5.1.1
headless export/import round trip, 46 browser-independent JavaScript tests,
all five generated-WASM smokes, and the live Chrome MCP gate. Renderer focus
state remains incumbent in this checkpoint; the next authority step is to
apply the already-compared Rust selected-focus packet to the renderer behind a
rollback flag.

The release Trunk build and ordinary offline preflight also passed. The
optimized WASM remains 6,083,489 bytes raw and is 2,136,342 bytes under
deterministic `gzip -9 -n`. The source-coherent build receipt is
`3cb56e606c892c44d590e223d11c4797` over 156 files and 38,669,610 bytes.
Preflight retained the expected warnings for the untracked local GLB directory
and the noncommercial horse fixture; neither local asset was added to source.

## Rust selected-focus renderer authority gate

The next 2026-08-24 checkpoint introduced the explicit
`selectionimpl=js|shadow|rust` rollback boundary. The Rust mode admits only a
mapped durable selection: JavaScript supplies the backend-local packed node and
the expected `(asset, entity)` IDs, `HyperscopeAppShadow` verifies those IDs
against the selected `AppStore` focus, and Rust applies the complete
focus-sphere/enablement/node packet to the resident renderer atomically. The
sphere itself does not serialize through JavaScript. Detached selection uses
the same checked path; malformed identity changes nothing. Ordinary GLBs and
manual/free sphere edits remain on incumbent authority, and `js` remains the
default.

The first live cutover probe exposed a real event-clock edge. Chrome may run an
already-queued RAF callback after a pick while retaining a timestamp older than
the pick's `performance.now()`. Clearing the selection event fence on that
stale frame made the following raw frame delta integrate the pre-event interval
twice. The adapter now retains the fence until RAF time reaches the event and
clamps the remaining-time oracle to the authored duration. A dependency-free
regression proves that the stale frame advances zero and preserves the fence,
then the first post-event frame advances exactly from the event.

Chrome DevTools MCP selected the authored landmark in the final two-asset cue
from a fresh release bundle. The trace reported one mapped pick, one dispatch,
one identity comparison, two transition frames, two renderer packet
comparisons, two Rust renderer-authority writes, zero unmapped picks, zero
selection/transition/renderer mismatches, zero authority misses/errors, zero
frame errors, an empty mismatch list, and no console warning or error. The
disposable tab and server were removed after the gate.

Validation passed for 28 `hyperscope-app` tests, strict no-dependency
application Clippy, WASM target checking, 47 browser-independent JavaScript
tests, all five generated-WASM smokes, the release Trunk build, and the live MCP
gate. The source-coherent build receipt is
`07acda5d054f4427057f9dd5c5646a6e` over 156 files and 38,677,409 bytes.

## Session-scoped ordinary selection gate

The 2026-08-25 follow-up extended the same Rust selection/focus path to
ordinary startup, IndexedDB, and dropped GLBs without laundering runtime
handles into durable identity. `hyperscope-app::session_node_identity` derives
an injective entity UUID from a non-durable load-lane asset ID and the exact
`u32` glTF node index. The generated WASM boundary releases a deduplicated
identity batch only after the Rust reducer reports that asset ready. A stale
completion remains loading and is explicitly denied selection identity.
Authored manifest/node UUIDs continue through the separate durable path, and
neither session selection nor any other presence state is admissible to HHHS.

The generated-WASM smoke covered stale denial, ready admission, deterministic
node 0/7 values, deduplication, and negative-node rejection while retaining the
7,168 SpaceMouse mappings, 648 response-policy cases, four camera states, and
120-frame trace. All 45 replay-enabled native `hyperscope-app` tests passed.
Chrome DevTools MCP then selected ordinary animated horse node 0 under `selectionimpl=rust`:
one mapped pick, one dispatch/comparison, two transition/renderer comparisons,
two renderer writes, and zero unmapped picks, semantic drift, packet drift,
misses, authority errors, or frame errors. A composed authored node repeated
the zero-drift result. The browser emitted no warnings or errors. The release
default remains `js`; this is a broader rollback-safe authority candidate, not
an unmeasured default flip. A separate release build staged 23 files / 30.98
MiB and passed strict `noncommercial-mixed` offline preflight with source/build
fingerprint `69e1ebbf196a09673865f3c981fed0b8`.

## HHHS 0.4.3 retained-history dependency gate

The 2026-08-25 dependency checkpoint moved every HHHS workspace crate from the
immutable 0.4.2 revision to tagged 0.4.3 commit
`35d2018cb5239f29a420eab26ed23f0bd3870b5b`. Upstream 0.4.3 replaces eager
capability reachability with linear-space lazy capture, shares immutable DAG
snapshot entries, extends staged snapshots without sorting complete history,
reads storage sequence directly during optimistic finalization, and avoids a
post-history commitment when a transaction carries no projection checkpoint.
Canonical encodings and authority semantics are declared unchanged upstream.

The local gate did not infer application speedup from those implementation
changes. It proved compatibility with 20 `hyperscape-hhhs` tests, 10 recovered
shadow/checkpoint tests, 6 browser durability codec tests, strict no-dependency
Clippy for all three integration crates, and wasm32 compilation of the browser
tests. The frozen authored-payload, project-archive, and source-checkpoint
digests remained unchanged. HHHS 0.4.3 still has no open-authority
co-transaction attachment API, so crash-atomic browser persistence of the
source cursor remains gated rather than approximated with a second write.

## Rust canonical-route write authority gate

The 2026-08-25 URL follow-up promoted the existing Rust route oracle behind an
explicit `routeimpl=js|shadow|rust` rollback boundary. The 71-entry
`hyperscope-app::ControlSpec` registry remains the sole authority for key
identity, value classes, default equivalence, and serialization order. Rust
mode commits the canonical pair sequence returned through WASM; a diagnostic
or bridge failure retains the complete browser-built query and increments a
bounded fallback counter. Shadow mode compares without writing, and the
default JavaScript mode performs no route WASM calls. Startup value application
was deliberately unchanged at this write-only checkpoint.

Chrome DevTools MCP verified a fresh Rust-authority load and a subsequent wheel
navigation update. Both writes had byte-identical browser, Rust, and committed
queries, with two comparisons, two authoritative writes, zero fallbacks, and
zero mismatches. A fresh shadow load produced one comparison and no authority
or fallback write; a fresh default load left the observer disabled with zero
comparisons. The generated-WASM oracle covers canonical order, numeric default
equivalence, duplicates, malformed values, unknown keys, and the adapter's
explicit fallback branch.

In the same Chrome process, 2,000 five-pair canonicalizations measured
0.038 ms/call median and 0.145 ms/call worst sampled batch average. URL writes
remain capped at 3.33 Hz by the existing 300 ms scheduler, so this path is not
frame-rate traffic. The optimized WASM at this checkpoint is 6,287,430 bytes
raw and 2,201,127 bytes under deterministic `gzip -9 -n`; no before/after size
claim is inferred because the preceding composed-LOD checkpoint was not
rebuilt as an isolated release artifact.

The paired read-side checkpoint then routed valid startup pairs through the
same Rust registry before model, Patch Lab, animation, or control state was
applied. A valid link containing the alternate default spelling `zoom=3.00`
reported `startupSource=rust`, omitted that default canonically, retained the
requested mode and rotation, and recorded no mismatch. Shadow mode compared
the same startup input while reporting `startupSource=browser`. A malformed
`smlock=yes` link produced the exact `invalid_value` diagnostic, selected
`startupSource=browser-fallback`, and preserved the incumbent interpretation;
the subsequent ordinary state write normalized it to `smlock=0`. Default mode
still performed zero startup or write-side route calls. This adds one
non-frame-rate WASM call only when shadow or Rust route mode is explicitly
selected.

## Atomic authored application materialization gate

The next 2026-08-25 checkpoint turned admitted protocol revisions into a
deterministic application read model instead of advancing only a checkpoint
number. `AppState` now retains key-sorted authored asset descriptors and latest
entity transforms. It validates every envelope before cloning and applying the
command batch, so one malformed command cannot partially update either map;
equal/older projection revisions remain diagnostic no-ops. Command order inside
an accepted revision is authoritative and a later command for the same key wins.
Runtime load requests remain a separate lane: an authored asset descriptor does
not implicitly fetch or replace renderer state.

`AppStore` publishes authored assets and entities as low-rate `SignalVec`
projections before setting its summary revision fence. Replay version 0.8 adds
both materialized collections to every compact state record. The orchestration
fixture proves that a valid asset/transform survives a stale removal and a
subsequent invalid revision; version 0.7 remains readable with its existing
asset-scoped focus semantics. The checked fingerprints are
`2123c41359d3187dbcbbff4334e069a0` for the six-cue presentation,
`9b68dd4542773115658cfb78282feb41` for navigation, and
`71b484d19c93d0171d9c4996831b2542` for orchestration.

Validation passed for 47 replay-enabled application tests, strict
no-dependency application Clippy, warning-free Rustdoc, wasm32 compilation of
`quilting-wasm`, 47 browser-independent JavaScript tests, all five generated
WASM smokes, and all three replay checks. This checkpoint deliberately stops
before renderer extraction or a Blender transport: it establishes the atomic
materialization boundary those adapters can consume without giving network or
frame traffic reducer authority.

## Generated authored-checkpoint admission gate

The transport-neutral boundary now crosses generated WASM without selecting a
network. `HyperscopeAppShadow::applyAuthoredRevision` accepts an atomic array of
the same protocol-v0.1 JSON envelopes used by Blender and carries the projection
revision as decimal text. The executable gate used revision
`9007199254740993`, proving that the fence does not round through JavaScript's
safe-integer limit. The bounded application snapshot exposes sorted authored
assets and entity transforms alongside that exact fence.

The generated-WASM oracle admitted an asset plus transform, ignored a stale
removal, then rejected a batch containing a valid update followed by an invalid
zero-scale transform. Revision, asset projection, and transform projection were
byte-for-byte equivalent before and after rejection; an out-of-range `u64`
fence was also rejected before dispatch. wasm32 checking, focused formatting,
the full browser-independent JavaScript suite, all five generated/script WASM
smokes, and all three replay fingerprints remained green. Package-wide strict
Clippy is not claimed here: the renderer has 42 pre-existing warnings under the
current toolchain, outside this boundary's files. No socket, Blender callback,
renderer matrix, or HHHS operation is inferred from admission alone.

## Rust packed-scene extraction contract

The first authored-scene extraction slice remains a pure function in
`hyperscape`; it does not call a renderer, browser, or transport. It joins
asset/layer/node runtime records to the application materialization by stable
entity UUID, normalizes `(w,x,y,z)` quaternions, composes
`presentation layer × (authored absolute source-world TRS or glTF world)`, and
checks the final matrix at the `f32` backend boundary. Output is sorted by
scene-wide packed node handle and separately reports valid authored entities
whose assets are not resident.

Five focused cases cover absolute replacement rather than accidental delta
composition, outer presentation transforms, non-unit quaternion input,
negative/nonuniform scale, deterministic ordering, repeated layers of one
asset, unmatched edits, cross-asset identity ambiguity, duplicate packed
handles, non-finite source matrices, malformed unmatched edits, and `f32`
overflow. The checkpoint passed 95 Hyperscape tests, 6 protocol tests,
crate-local strict Clippy, warning-free rustdoc, and a wasm32 library check.
Transitive strict Clippy still reports one pre-existing unnecessary cast in
`quilting-mesh`; it is not folded into this semantic checkpoint.

## Browser packed-scene rollback gate

The generated application facade now accepts packed layer/node metadata and
extracts against its accepted authored checkpoint. Its generated-WASM oracle
proved two node records, packed-node sorting, exact projection-fence retention,
absolute authored replacement, outer layer composition, and duplicate-handle
atomic rejection. The canonical URL registry grew from 71 to 72 controls with
`sceneimpl=js|shadow|rust`.

Chrome DevTools MCP loaded the final two-asset cue through a temporary Trunk
server. In `shadow`, two low-rate applications compared 18 matrices across the
horse and Blender-authored asset: zero mismatches, zero fallbacks, maximum
absolute error `4.768371586472142e-8`, 4,252 resident faces, and 9 topology
domains. `rust` then made 18 authoritative node/LOD writes with the same error,
zero fallbacks, and no console messages. A final isolated test admitted an
absolute transform for the stable Blender ground entity, reactivated the cue,
and observed one authored override, 17 fallback-matrix comparisons, an
intentional maximum authored delta of `8`, exact authored fence `"1"`, 18 Rust
writes, zero mismatches/fallbacks, and no presentation error. The temporary
servers were stopped and the existing listener on `10.0.0.1:8888` was not
touched.

## Transport-neutral local peer ingress

The first local Blender bridge checkpoint selects no WebSocket, HTTP server,
or relay. `hyperscape-protocol::LocalPeerEnvelope` freezes distinct authored
and presence lane shapes, and `hyperscope-app::LocalPeerIngress` is the only
direct-demo admission policy. Authored messages receive a monotonic local
projection fence only after validation; duplicate IDs, sender-stale sequences,
and consumed local echoes produce no reducer commit. Presence goes through the
existing receipt-time TTL path and never advances the authored projection.

Four application cases prove retry/echo/stale rejection, bounded-memory
eviction, invalid-frame atomicity, expiry, and sequential single-writer
convergence between independent stores; the protocol case proves round trips
and rejects a presence payload relabeled as authored. The gate passed 51 app
tests, 7 protocol tests, strict crate-local Clippy, warning-free
rustdoc, and wasm32 app checking. This intentionally does not claim
multi-writer convergence: that remains the HHHS adapter's job.

## Generated local peer admission

`HyperscopeAppShadow` now retains the Rust ingress next to its application
store. Generated WASM accepts canonical `LocalPeerEnvelope` JSON, exposes a
separate ephemeral presence sample, and records already-applied local authored
envelopes solely for echo suppression. The low-rate application snapshot keeps
its existing commit-fence meaning; presence carries its own peer sequence,
receipt-relative expiry, and sampled application time.

The generated oracle proved authored apply, exact duplicate rejection,
sender-stale rejection, invalid-then-corrected atomic admission, consumed echo
and repeated-echo behavior, presence admission, presence stale rejection, and
TTL expiry. A release `wasm-pack build` succeeded, `wasm-pack test --node`
passed four tests, and all five generated Node smokes passed. A wasm32 library
check also passed. Ordinary crate-local Clippy attributed no warning to
`app_shadow.rs`; strict whole-crate Clippy remains blocked by 42 existing
renderer/runtime findings, and strict whole-crate rustdoc by eight existing
broken-link findings. This checkpoint does not conceal those separate cleanup
debts or select a Blender/browser carrier.

## Optional local peer relay

The first carrier infrastructure lives behind the non-default
`hyperscope-web/local-peer-relay` feature. Its state retains opaque raw JSON so
unknown future fields and exact integers survive forwarding. Bounded eviction,
pagination, process generations, decimal delivery cursors, and explicit gaps
are transport evidence only; the relay has no application reducer, authored
projection fence, persistence, or repair claim.

Eight Rust cases passed for bounded gaps, opaque preservation, pagination,
restart detection, invalid-input atomicity, cursor overflow, secure CLI
defaults, and query validation. Strict crate-local Clippy and rustdoc passed.
The checked Node smoke launched the real
binary on an ephemeral loopback port and proved bearer rejection (`401`),
origin rejection (`403`), exact-origin preflight (`204`), authenticated health,
canonical authored-frame delivery, invalid-JSON rejection, and future-cursor
gap reporting. The test relay was stopped; the user-run `:8888` server was not
touched.

## Blender-to-browser local relay

The disabled-by-default browser carrier now preserves application frames as
exact JSON text, validates delivery generations/cursors with `BigInt`, and
hands semantic admission directly to generated Rust/WASM. Eight carrier unit
tests passed for configuration rejection, exact `u64` delivery, semantic
failure atomicity, restart recovery, bounded-history degradation, ordered
retry, presence validation, and clean stop. The real end-to-end smoke launched
an ephemeral Rust relay plus an isolated Blender 5.1 process, published one
absolute object edit and one presence frame, admitted both in Rust/WASM, and
resolved translation `[3, 4, 5]` through packed-scene extraction. Its actual
Blender sequence was `1787631571532014228`, safely above JavaScript's exact
integer range. All temporary processes were stopped and `:8888` was untouched.

## AppStore presentation authority gate

The presentation adapter now exposes `presentimpl=js|shadow|rust` as the 73rd
canonical route control. Default mode retains the browser-orchestrated
standalone Rust controller; shadow mode retains the existing full cue/pose
comparison; Rust mode allocates only `HyperscopeAppShadow`, dispatches cue
intent through the application reducer, and consumes its application-frame
navigation snapshot. The low-rate presentation read model now includes the
validated asset catalog, eliminating a second semantic manifest parse and the
standalone controller from Rust-authority mode.

The gate passed 51 replay-enabled application tests, 55 browser adapter tests,
five generated-WASM tests, strict application Clippy, warning-free application
rustdoc, the 73-control route smoke, the six-cue presentation smoke, and the
generated application, render-shadow, and surface-walk smokes. An isolated
release artifact built successfully and ordinary offline preflight returned
`PASS` with an exact source/build fingerprint. Those oracles prove
manifest/cue equality and cadence-invariant transition state without a DOM or
GPU. The default remains `js` until the Rust-authority URL is visually and
temporally rechecked in the target Chrome renderer.

## Active presentation composition authority

The second packed-scene slice removes semantic layer input from the browser
join. `PackedPresentationLayerBinding` contains only layer/asset identity and
renderer-resident node metadata. Rust validates every active layer exactly
once against `PresentationSnapshot`, applies cue-owned TRS/visibility/opacity,
then composes the accepted authored absolute transform under the outer layer.
`AppStore::extract_active_presentation_scene` samples the application revision,
active cue/scene, and authored projection under one lock without publishing a
signal or mutating state. The generated WASM input rejects unknown fields, so
an adapter cannot smuggle a `layerTransform` through the binding.

Eight focused Hyperscape cases cover authoritative transform/render state,
repeated assets with distinct renderer handles, sorted output, authored
replacement, and missing/duplicate/unknown/mismatched bindings. The
application suite proves a coherent non-mutating read; generated WASM proves
exact decimal revision fences, active cue/scene identity, authored replacement,
effective visibility/opacity, semantic-input rejection, and atomic failure.
The browser comparison now checks both matrices and layer render state and
commits a shadow AppStore cue before rendering it, preventing extraction of a
new browser cue against the previous application cue.

The checkpoint passed 98 Hyperscape tests, 52 replay-enabled application
tests, 55 browser-adapter tests, five generated-WASM tests, strict native
Clippy, warning-free native rustdoc, and the presentation, route, application,
render-shadow, and surface-walk smokes. An isolated release build passed
offline preflight with identical source/build fingerprint
`c1ab30d96589b0ad1b8abe69baf94082`. Target-Chrome visual timing remains a
separate cutover gate; renderer-wide wasm32 Clippy still reports the same 42
pre-existing warnings outside this composition boundary.

## Active presentation composition cutover

The target-Chrome gate subsequently exercised explicit `sceneimpl=rust` on the
release artifact before changing the canonical default. The composed final cue
held both assets and 4,252 packed faces while 567 matrix and semantic checks
reported zero mismatches, zero fallbacks, maximum matrix error
`4.768371586472142e-8`, zero opacity error, and no console messages. Reverse,
advance, and an explicit cue jump increased the evidence to 1,336 comparisons
without changing those bounds or losing an asset. `sceneimpl=rust` is therefore
the canonical default; `sceneimpl=js` remains a URL-persistent rollback and
`sceneimpl=shadow` remains the non-authoritative comparison path. Extraction
failure still returns the complete incumbent scene atomically and records the
fallback rather than partially installing Rust output.

## Primary-scene asset effect authority

The asset adapter now exposes `assetimpl=js|shadow|rust` as the 74th canonical
route control. The default `js` path is unchanged. Shadow mode runs the Rust
request oracle but never aborts or suppresses incumbent browser work. Rust mode
classifies startup, drag/drop, and the authored demo as one primary-scene load
scope; presentation resources remain independent per-asset jobs. Replacing a
primary request cancels across different asset IDs, aborts an in-flight fetch,
and prevents an already acquired but obsolete result from entering renderer
installation.

The thin browser effect host owns only `AbortController`, logical-URI
acquisition, and a serialized dynamic install lane. The Rust reducer owns
request generations, cancellation, completion disposition, and replay
semantics. The install fence is checked after model parse and every asynchronous
skinning, tessellation-parameter, rest-pose, compute-upload, and animation
boundary. This also prevents two dynamic loads from interleaving through the
worker's single retained glTF state. Initial glTF parsing remains parallel with
atlas generation; dynamic requests wait for the initial render phase rather
than slowing ordinary startup.

Validation passed 55 replay-enabled application tests, 61 browser-independent
JavaScript tests (including six effect-host cases), the generated application
WASM smoke, the 74-control route smoke, inline-module syntax checking, and an
isolated optimized Trunk build. Release staging and noncommercial-mixed offline
preflight passed over 23 files / 31.30 MiB with matching source/build receipt
`3cf9e463dd7839f9325146154c2fa640`. Replay schema 0.9 preserves 0.8 input
meaning and renews the presentation, navigation, and orchestration fingerprints
to `08f7953320c733fbab99cbe12d5e81a7`,
`2656516995573de63986647d4196c478`, and
`0dfe524f3c0a022dc4507d51e87679fb`. Target-Chrome drag/drop timing remains a
separate cutover check because the Chrome DevTools MCP had not yet been
connected to this agent at that checkpoint; no Playwright substitute was used.

## Single-pass composed-scene LOD

The final two-asset cue was live-checked through the installed Chrome DevTools
MCP against an isolated optimized artifact. The Blender-authored asset is not
outside adaptive LOD: it owns 3,268 of 4,252 resident faces, and changing
`minpx` from 16 to 1 reclassified all 3,268 while visibly changing the floor
and object LOD colors. The formerly ambiguous `lastLodScope` is now accompanied
by separate scene/primary-animation counters and timestamps.

The WebGL2 classifier now uploads immutable face-to-node ownership once and a
compact ten-texel record per active node. One vertex pass selects each face's
Möbius and Euclidean matrices before shared edge coherence. On this cue the
full-scene workload fell from one baseline plus nine whole-mesh subject
classifications/readbacks to one classification/readback carrying nine node
records; the animated prefix carries one horse record. This removes 90% of the
whole-mesh GPU passes and staged readback bytes for the measured scene. A
20-sample alternating `minpx=1/16` target-Chrome check measured 21.0 ms median
before and 17.6 ms after end-to-end scene-update latency (16.2% lower); p95 was
27.3 ms before and 28.2 ms after, so no tail-latency improvement is claimed for
this small 4,252-face fixture. The architectural win is bounded per-scene work
instead of multiplication by authored node count.

Native renderer tests prove full-scene versus face-prefix domain selection,
last-record duplicate resolution, ownership-scoped adjacency, and the shader
subject-table contract. The wasm32 library check, release Trunk build, and live
scene/primary classifications completed with one GPU pass and no browser
console warnings or errors. Noncommercial-mixed offline preflight passed over
23 files / 31.32 MiB with matching source/build receipt
`372352f4e9592ec6a4bc978fab6adb80`.

## Rust asset authority default cutover

The measured `assetimpl=rust` lane is now the canonical default. An absent or
invalid browser implementation value selects Rust, while `assetimpl=js`
remains an explicit serialized rollback. Effective Rust authority no longer
injects the legacy `appshadow=1` observer flag into synchronized URLs. The
browser effect host also requires an explicit policy at construction, avoiding
a second hidden JavaScript default.

The installed Chrome DevTools MCP exercised a fresh isolated optimized build
on `127.0.0.1:8891`; the user-run `:8888` was untouched. Default startup
finished with one Rust request and completion, no failure, stale completion,
prevented install, or mismatch, and synchronized to the clean
`?animate=0&anim=0` route. An isolated `assetimpl=js` page rendered the same
984-face horse with the application adapter disabled and preserved the
rollback value in its URL. The final two-asset Rust-presentation cue completed
two Rust requests, retained both adaptive-LOD assets and 4,252 packed faces,
and reported no pending composition, application mismatch, presentation
error, or console warning.

A real pair of same-turn drop events (`ant.glb` followed by `horse.glb`)
produced exactly two requests, two load completions, one requested
cancellation, one stale completion, and one prevented install. The horse was
the final 984-face model, the URL retained no default authority flags, and the
console remained clean. This is the expected distinction between semantic
completion and renderer installation: the superseded result is admitted as
stale evidence but cannot mutate the resident scene.

Validation passed 56 replay-enabled application tests, 62
browser-independent tests, the three unchanged replay fingerprints, route,
application, presentation, render, and 600-frame surface-walk smokes,
crate-scoped strict Clippy, and application rustdoc. The exact staged release
passed strict noncommercial-mixed preflight over 23 files / 31.34 MiB with
matching source/build fingerprint `f2e8c4358314964e728205f947f6925e`.
Transitive strict Clippy still stops on the previously existing unnecessary
cast in `quilting-mesh`; it is not attributed to this cutover.

## AppStore presentation default cutover

The measured `presentimpl=rust` path is now the canonical presentation mode.
An ordinary `presentation=1` route allocates AppStore as the sole semantic
controller; it does not instantiate or tick the standalone
`HyperscopeNavigation` controller and it omits `presentimpl=rust` plus implied
`appshadow=1` from synchronized URLs. Explicit `presentimpl=js` and
`presentimpl=shadow` remain linkable rollback/comparison modes.

The installed Chrome DevTools MCP exercised a fresh isolated optimized build
on `127.0.0.1:8891`, again without touching `:8888`. The unflagged presentation
reported `implementation: rust`, `authority: hyperscope-app`, and a
`HyperscopeAppShadow` controller identical to the application controller. A
real advance from cue 1 to cue 2 exposed 350 ms of transition time at the
mid-sample, settled to zero over 28 application frames in the observed
353.1 ms window, and added no application or pose mismatch. Jumping to the
final cue settled over 73 frames with both 984- and 3,268-face assets resident,
4,252 packed faces, no pending composition, and no presentation error.

The isolated `presentimpl=js` rollback instantiated
`HyperscopeNavigation`, remained distinct from the AppStore controller,
preserved its URL flag, and completed one AppStore comparison with zero pose
mismatch. Both the default and rollback pages had clean warning/error consoles.

Validation passed 57 replay-enabled application tests, 62
browser-independent tests, the unchanged presentation/navigation/orchestration
fingerprints, presentation, route, application, render, and 600-frame walking
smokes, crate-scoped strict Clippy, and application rustdoc. The staged
artifact passed strict noncommercial-mixed preflight over 23 files / 31.34 MiB
with matching source/build fingerprint
`e874d7f5250d8df19552cb36c8a05c6f`.

## Presentation-freeze acceptance audit

The 2026-08-25 freeze was checked from a detached, clean worktree at
`c874fe8`, served on an isolated loopback port. The user-run `:8888` service
and application tabs were not changed. Strict noncommercial-mixed preflight
passed over the final 19-file / 23.46-MiB offline set. Its 173 build inputs
total 20,688,441 bytes and have matching source/build fingerprint
`10cfa8064da53dffab9fa4e005f0c356`.

The installed Chrome DevTools MCP traversed all six presentation cues. Each
cue retained both assets, 4,252 resident faces, nine topology domains, and one
GPU LOD pass, with the expected visualization and no application mismatch,
presentation failure, console warning, or console error. A direct reload of
cue six restored the same two-asset PBR composition. The final exact artifact
then reloaded cue one with Rust presentation and scene-extraction authority,
the expected 984-face horse, build fingerprint above, and a clean console.

A real `ant.glb` drag/drop while the deck was active exposed a release defect:
the file was admitted while the old cue composition remained active, so the
single-primary animation invariant rejected the result. The repaired behavior
enters a clean ordinary-model route; Chrome observed the 9,768-face ant with
one Rust request, one completion, no failure, and no warning or error. Browser
Back remains the route-level return to the deck. Default JavaScript selection
selected the horse and fitted its focus sphere, and default surface walking
attached to animated horse face 57 without a console error.

The real Blender/browser relay smoke received three browser frames, applied
two Rust frames, projected translation `[3, 4, 5]`, and retained exact
sequence `1787645284226591060` plus presence sequence
`18446744073709551614`. Poll timers now detach settled abort listeners, so the
smoke no longer emits the prior listener-leak warning.

The final gate passed 63 browser-independent Node tests, all six checked
browser/relay smoke programs, and all three deterministic Rust replay
fingerprints. A foreground frame-rate claim is deliberately omitted: the
isolated audit page stayed backgrounded to avoid selecting or disturbing the
user's active tab, and Chrome throttled its animation callbacks accordingly.

Four reviewable commits close the failures found during this freeze:

- `bb0d493` removes retired WASM packages from release preflight;
- `5ce9ba2` makes presentation drag/drop enter a clean model route;
- `5d145cf` releases relay abort listeners after polling; and
- `c874fe8` fingerprints the browser adapters copied into the release.

The uncommitted selection/inversion bridge experiment was deliberately parked
as `stash@{0}` rather than shipped. Release defaults remain the proven
selection and walk paths; post-presentation work should replace the browser
bridge with one typed Rust semantic-action ingress before changing authority.
