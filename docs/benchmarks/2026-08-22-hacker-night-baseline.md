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
