# Hyperscope hacker-night release architecture

Target: Tuesday, 2026-08-25.

This document is the execution contract for turning the current browser
prototype into a rehearsable Quilting/Hyperscope presentation without losing
the longer-term Hyperscape architecture. It consolidates the repository
roadmaps, the `61106329-8039-4e62-853b-8bf6c86005e5` Claude session, the
conformal-mereology work, and the HHHS/Hyperscape integration review.

The release is not a rewrite. The current application is the behavioral oracle
until a Rust subsystem has parity tests and a thin browser adapter consuming
it. Each migration must leave a runnable, committed checkpoint.

## Product boundary

The names describe layers, not competing applications:

- **Quilting** owns quaternionic-Bezier surface evaluation, tessellation,
  conformal LOD, mesh topology, and reusable rendering algorithms.
- **Hyperscape** owns stable scene identity, ECS state, conformal frames,
  constraints, camera/navigation state, semantic interaction, presentation
  state, and authored interchange.
- **Hyperscope** is a browser presentation and rendering client. It adapts DOM,
  WebHID, files, WebGL2, and eventually WebGPU to Rust-owned state.
- **Blender** is an authoring peer. Ordinary geometry and PBR remain ordinary
  glTF; Hyperscape metadata is versioned glTF data.
- **HHHS** is durable replicated history and reconciliation. It is not a frame
  loop, renderer, input event bus, or authority policy.

Hyperscape uses Bevy ECS without depending on Bevy's renderer. WebGL2 and
WebGPU consume the same extracted logical view and render-command data.

## Rust application migration status

The first application boundary is now explicit:

- `hyperscape-protocol` owns version `0.1` wire headers, validated stable IDs,
  asset descriptors, ordinary authored transform commands, and a distinct
  TTL-bounded presence envelope. Only `AuthoredEnvelope` is eligible for a
  future HHHS admission adapter; camera, selection, focus, cue, and animation
  presence have no conversion into the durable lane.
- `hyperscope-app` owns `AppEvent -> AppCommit + AppEffect`, deterministic
  navigation scheduling, asset job generations, stale completion rejection,
  presentation loading/cue actions, local presence expiry, diagnostics, and
  futures-signals read models. Cue activation and its navigation transitions
  commit transactionally; rejected cue/pole/reference operations preserve the
  preceding revision. Effect-producing and presentation future inputs are
  rejected until a real application event scheduler exists rather than being
  executed at the wrong time.
- `appshadow=1` now feeds real startup, IndexedDB, drag/drop, authored-demo,
  and presentation asset acquisition plus presentation load/cue intent into
  that reducer without changing the browser loader, cue controller, or
  renderer. Superseding a load emits cancel-then-fetch; late completions remain
  observable but cannot replace the active request. Each cue action compares
  the complete resolved presentation snapshot. A separate opt-in pose gate
  synchronizes settled navigation through the reducer and makes exactly one
  compact comparison call per active cue-transition frame; it makes no calls
  while settled or when shadowing is disabled. Bounded diagnostics are exposed at
  `globalThis.__hyperscopeAppShadowDiagnostics` until this lane earns browser
  authority. The same adapter now accepts the incumbent navigation shadow's
  device-neutral actions through `AppStore`; the shared navigation queue owns
  their sequence and the application owns virtual time. The adapter exposes a
  parity-complete frame snapshot. This is an offline
  cutover gate only: `navshadow=1` and the browser camera remain unchanged until
  live Chrome parity is measured.
- Effective spheroidal-focus authority now crosses that same ordered boundary.
  Rust `focus_enabled` denotes fuzzy post-processing enabled specifically in
  mode 3; modes 0–2 remain renderer-only blur choices, and the retained shared
  sphere may stay active for inversion or editing without enabling focus.
  Browser signal changes to enablement, shell coordinate, and angular aperture
  coalesce into one microtask and enqueue exactly two semantic actions per
  active parity controller. Initial `AppStore` synchronization now includes
  the complete camera lens, aim policy, focus field, inversion, and sphere, so
  a focus-only route cannot temporarily retain Rust defaults. Applying an
  authored presentation snapshot suppresses the reciprocal signal adapter and
  cannot feed the same focus edit back into the queue. The browser renderer is
  still live authority while this measured gate soaks.
- The normalized SpaceMouse camera gate freezes samples at
  a platform-neutral Rust boundary. Browser code retains only WebHID/report
  acquisition, device shaping/smoothing, button layers, and the
  screen-relative linear speed frozen at gesture start. Rust validates axes in
  `[-1, 1]`, computes translation/rotation/object-dolly response from virtual
  delta and user gains, applies preset/swap/inversion/horizon policy, and queues
  the resulting ordinary navigation actions through `AppStore`. It adds no
  device-specific event to the application vocabulary and does not tick or
  change live browser authority. Generated WASM matches the incumbent mapping
  over 7,168 exhaustive mapping cases, 648 response-policy cases, four
  `AppStore` camera initial states, and a 120-frame deterministic trace.
  Modifier layers, surface-walk
  input, and authored-transition arbitration remain explicit later cutovers.
- Surface walking now has one backend-neutral Rust aggregate:
  `SurfaceWalkRuntime` atomically owns `SurfaceWalker` topology and physical
  side, `SurfaceWalkController` metric locomotion and view response, animated
  material-point velocity, body/eye scale and near plane, recovery/detach, and
  the sole `SurfaceAnchorTransition`. Invalid semantic input rolls the complete
  aggregate and camera back; topology failure commits one coordinated detach.
  The production WASM adapter borrows the incumbent posed QB controls,
  adjacency, and conformal transform rather than cloning a chess-scale mesh.
  `walkimpl=shadow` mirrors pointer attachment and each semantic walking frame
  through this aggregate and exposes topology/camera drift at
  `globalThis.__hyperscopeSurfaceWalkRustShadow`; the default remains `js`,
  while `walkimpl=rust` now consumes the aggregate's contact frame, camera, and
  transition as authority and keeps the incumbent walker on the same semantic
  velocity as a rollback diagnostic. The transition now matches the incumbent's
  independent forward/up direction smoothing, Gram-Schmidt basis recovery,
  and immediate scale-relative lens/control-distance update. Generated-WASM
  gates retain the 2,160 mapping cases and 600-frame response oracle, while
  native aggregate tests cover atomic admission, animated velocity, scale,
  recovery side, coordinated detach, view recapture, and locomotion cadence.
  The live oracle and Rust candidate now advance the re-anchor glide from the
  same explicit clamped frame delta, including the same endpoint snap, so
  replay and background scheduling cannot create timing-only drift. The direct
  generated-WASM smoke also preserves a structural
  `ComposedSurfaceWalkResult` boundary instead of `any`. Reflection edits now
  transport the camera, stable attachment side, retained contact follower, and
  previous posed-contact samples transactionally through the exact chart
  differential; a pole rolls every participant back, and a successful edit
  cancels the old-chart anchor glide to match the browser oracle. Initialized
  Chrome traces now validate the shared clock, successful reflection
  transport, explicit pose-time velocity, pause/scrub rebasing, active
  locomotion, the initialized camera-pole rollback, and zero topology/camera
  drift. The executable WASM gate also covers pick-like Float32 near-edge
  crossings under all cyclic source/neighbor permutations in identity and
  non-binary-exact reflection charts. Those gates enable explicit
  `walkimpl=rust` authority without changing the default. Node/WASM aggregate
  tests additionally cover both walkers, one-shot velocity rebasing, and the
  first real animated-pose sample; native replay proves that an animated chart
  edit cancels an old-chart anchor independently of tick partition.
- `hyperscope-app::ControlSpec` is the canonical registry for all 68 currently
  linkable controls and migration flags. `HyperscopeRoute` owns default
  equivalence, first-value duplicate semantics, stable ordering, and explicit
  malformed/unknown diagnostics. With `routeshadow=1`, the browser still writes
  its URL but compares every bounded-rate serialization against Rust at
  `globalThis.__hyperscopeRouteShadowDiagnostics`.
- `quilting-core::render` owns retained scene snapshots, logical frame
  commands, indexed submission accounting, and the bounded backend-parity
  observer. `rendershadow=1` extracts WebGL state only when the retained scene
  changes and compares every subsequent frame inside WASM; the browser can
  explicitly query `globalThis.__hyperscopeRenderShadow` but receives no
  per-frame diagnostic traffic.
- High-rate frame, navigation, and presence events advance authoritative state
  without forcing DOM-rate notifications. `SignalVec` asset/diagnostic views
  and the low-rate presentation projection are published as a batch and an
  `AppSummary` revision is set last as the consumer commit fence; adapters
  explicitly flush at their UI cadence. Presentation transitions reconcile on
  the frame lane without cloning cue assets/layers into render snapshots.
- `hyperscope-app` exposes a versioned, adapter-independent replay format. A
  replay contains semantic events, each commit/rejection outcome, and a compact
  camera/focus/cue/asset/presence/diagnostic snapshot; it contains no DOM
  events, device reports, renderer handles, or wall clock. Decimal JSON uses
  exact `f64` round trips.
  The native `replay` feature is excluded from browser builds;
  `hyperscope-replay` version 0.7 walks every checked-in cue, every current
  semantic navigation action, and every current application event lane.
  Version 0.7 makes selected identities explicitly asset-scoped, so the same
  entity UUID in two composed assets cannot alias. Legacy unscoped focus
  anchors fail closed instead of receiving a fabricated asset identity.
  Version 0.6 added complete validated perspective-lens edits and an explicit
  semantic-target-presence policy without inferring aim mode from inversion.
  Version 0.5 retains selected source bounds and clicked pivots and derives
  output-chart pivots/radii in the application snapshot; a projection pole
  clears only those derived values. The reader accepts 0.4, 0.5, and 0.6
  inputs, but only 0.4 migrates an omitted source pivot to the bound center.
  Versions 0.4 and 0.5 reject 0.6-only actions rather than silently changing
  their meaning; every pre-0.7 unscoped focus anchor is rejected. Action
  admission and integration remain distinct: same-time
  navigation input remains pending until the next integration boundary. That
  is normally a frame event; transactional cue activation also integrates at
  zero time so its own queued transitions and any preceding due input commit
  in sequence. This exactly matches the standalone controller's observable
  queue contract. The
  navigation oracle covers camera frames,
  focus/inversion, camera transitions, surface re-anchor/retarget/cancel,
  stable-identity selection, detach/free edits, and rejected-input atomicity.
  The orchestration oracle covers asset effects, stale completion, failures,
  cancellation, presence TTL/order, authored revisions, and rejected wire
  input. Tests prove exhaustive current event/action coverage, JSON round trips,
  atomic rejection, and transition cadence invariance. The six-cue golden is
  `fnv1a-128-json:4d8598faf9db62e8500d49d94ead89ed`;
  the navigation golden is
  `fnv1a-128-json:4b6f0b82cf471af7af17b99ed37317d4`; the orchestration
  golden is `fnv1a-128-json:2cb74a642b3d4fc40b4eda777addb833`.
- `hyperscape::StableEntityId` converts explicitly to the validated wire
  `EntityId`, so the protocol wrapper is an interchange type rather than a
  second identity authority.
- Blender's dependency-free `protocol.py` validates and canonically roundtrips
  the same checked-in Rust fixtures. It implements only receipt-relative
  presence expiry/order and bounded authored-message echo suppression; it does
  not yet select a transport or admit ephemeral state to HHHS.

This layer is not yet the browser authority. It is the target behind the same
shadow-and-rollback policy used for navigation; browser loading and URL state
move only after adapters can compare existing behavior against reducer traces.
The selection adapter now joins validated authored node UUIDs to explicit
presentation asset IDs across packed composition offsets and mirrors mapped
picks/detaches through the AppStore. Session-generated load IDs, IndexedDB,
drops, basename matches, and ordinary GLBs cannot acquire durable selection
scope. Renderer focus transitions remain incumbent until the shared Rust clock
and selected-focus packet complete the cutover gate.

The offline release gate also has source provenance now. A Trunk pre-build hook
uses Rust to fingerprint the authoritative crate/shader, HTML/module, manifest,
and copied-asset inputs into `pkg/hyperscope-build.json`. Filesystem preflight
recomputes that bounded receipt and rejects missing, malformed, unsupported, or
stale fingerprints before considering the bundle releasable. The deterministic
FNV-1a-128 receipt is drift detection, not signing or adversarial integrity.

## Three graphs, never one overloaded hierarchy

1. The **ownership graph** describes entities, ordinary node parenting,
   assets, and presentation grouping.
2. The **conformal frame forest** describes charts and composable Möbius maps.
   A subject/view pair receives one relative map; shared ancestry cancels.
3. The **constraint graph** describes tracking, paths, focus anchors, surface
   attachment, and authored relationships between entities or frames.

An ordinary non-uniform glTF scale is a leaf deformation, not a conformal frame
edge. Möbius transitions animate meaningful generators, control geometry, or
versors; they never linearly interpolate the 16 raw matrix coefficients.

## State ownership

### Rust-authoritative

- stable entity and asset identity;
- ordinary scene topology and conformal frame topology;
- quaternion camera orientation, eye, semantic target or free sight tangent;
- scale-independent fly, orbit, drone, and surface-walk policies;
- selection identity and the shared focus/inversion sphere;
- deterministic transitions and their clocks;
- semantic input actions and recorded/replayed action streams;
- presentation deck, cue, view, layer, and transition state;
- current animation pose identity and backend-neutral render extraction;
- conservative spatial-index queries and surface attachment state.

### Browser-adapter state

- DOM controls and accessibility;
- WebHID permission/device acquisition and raw report delivery;
- dead-zone shaping, temporal smoothing, button interpretation, and
  per-gesture screen-relative SpaceMouse speed registration;
- drag/drop, file handles, IndexedDB, and network fetches;
- canvas sizing and browser scheduling;
- WebGL2 resource handles and backend implementation details.

JavaScript may cache a projection of Rust state for display, but it must not
silently own a second camera, selection, focus sphere, or transition timeline.

## Semantic action boundary

Device adapters produce timestamped actions. Examples include:

```text
SelectEntity { stable_id, source_bound }
DetachSelection
TranslateCameraLocal { right, up, forward }
RotateCameraLocal { pitch, yaw, roll }
OrbitSelection { pitch, yaw }
TranslateFocusLocal { right, up, forward }
ScaleFocus { log_delta }
SetFocalShell { coordinate }
SetAngularAperture { aperture }
ToggleInversion
ReframeSelection
AttachToSurface { entity, face, barycentric }
AdvancePresentation | ReversePresentation | JumpToCue { cue }
```

Mouse, keyboard, SpaceMouse, touch, gamepad, XR, replay, networking, and game
code all target this vocabulary. Device-specific axis normalization is policy
at the adapter edge; integration and camera geometry live in Rust.

## Durable and ephemeral lanes

Durable authored state uses stable UUIDs and small atomic scene operations.
Vectors, quaternions, generator words, and keyframes are atomic values so
concurrent edits cannot produce torn transforms. A completed authored gesture
may become one HHHS commit.

Pointer hover, live camera motion, SpaceMouse reports, transient selection,
physics snapshots, and interpolation samples are ephemeral. They do not enter
permanent history unless explicitly promoted to an authored cue or scene edit.

The first Blender/Hyperscope slice must work without HHHS: export or reload a
versioned scene/presentation description with stable IDs. HHHS 0.4 then adds
offline-repairable replication to the same operation vocabulary; it does not
replace it.

## Presentation model

A presentation is data consumed by Hyperscape, not imperative JavaScript. The
minimum logical model is:

```text
Presentation
  assets[]       stable ID, URI or embedded reference, load policy
  scenes[]       entity composition and authored conformal frames
  views[]        camera rig, focus sphere, visibility/layer state
  cues[]         text, active scene/view, animation and diagnostic state
  transitions[]  duration, easing, semantic camera/frame/focus edits
```

A cue can display text while the 3-D view remains interactive. A transition may
move the camera, change a conformal frame generator, fit a focus sphere, cross
fade scene layers, or combine those operations. A Möbius transition is used
when it explains the material or improves continuity, not as a mandatory slide
effect.

Multiple GLBs remain distinct scene entities. They are not merged merely to
satisfy the renderer. This preserves material, animation, node, selection,
frame, and presentation identity.

## Surface walking

The walker keeps a stable source address `(entity, face, barycentric)` while
motion is evaluated in the displayed output chart:

```text
Y(q,t) = F_t(X(q,t))
J = [dY/du dY/dv]
q_dot = (J^T J)^-1 J^T v_output
```

Speed, gravity, eye height, and contact are Euclidean in that output chart.
Animation and conformal-frame motion contribute surface velocity. Adjacency is
the ordinary local path; the conformal round index is recovery, reattachment,
and broad-phase support. Near poles or ill-conditioned parameterizations the
walker takes conservative substeps or detaches explicitly rather than
teleporting.

## Spatial index and culling rollout

`quilting-round-index` begins as a shadow oracle:

1. derive conservative bounds for complete posed rational QB patches;
2. refit animated leaves while retaining stable topology;
3. pull a finite output-chart frustum query into the source index;
4. compare indexed results to a conservative brute-force/reference path;
5. record false negatives, unknowns, traversal cost, and surviving patches;
6. enable culling only for certified-disjoint results after zero-false-negative
   evidence on static, animated, affine, inversion, and pole-adjacent cases.

Unknown, tangent, pole-touching, and uncertified bulge cases remain visible.
WebGL2 vertex rejection saves raster/fragment work but not vertex invocation;
WebGPU later performs visible-instance compaction and indirect submission.

The browser's first observer is opt-in with `roundshadow=1`. It builds a
stable-topology `StaticPatchIndex`, compares its candidates with coherent
completed GPU LOD classifications, and exposes build, refit, and query counts
at `globalThis.__hyperscopeRoundShadow`. For ordinary animated glTF scenes, the
worker optionally captures the exact joint matrices and morph weights used by
that asynchronous LOD job. Rust reconstructs the posed source controls and
atomically refits all bounded leaves before comparing the result; this adds no
mesh-sized readback and is disabled with the observer. Returning to a static
pose explicitly refits the rest controls so animated bounds cannot linger.

GPU-only survivors measure how much more conservative the current classifier
is; they are not mislabeled as visible geometry. A separate seven-point
rational-QB sample check records a red-alert false negative only when a
rejected patch has a point strictly inside the clip frustum. The observer never
changes a draw call. Authored per-node transforms still report `unsupported`;
advancing those cases requires structured frame chains coherent with each
classification.

## Conformal QB optimization boundary

The new optimizer prototype is an offline research input, not a live-renderer
dependency. The next useful stages are coarse boundary construction,
fit-driven connected clustering, animation-pose envelopes, shared boundary
constraints, and trustworthy denominator/bulge bounds. Existing historical
fitters are evidence and test material, not an architecture to preserve.

Any optimized output must retain stable source provenance, material and
attribute domains, watertight boundaries, and a measurable advantage over the
flat baseline under representative conformal views.

## Backend-neutral rendering boundary

Shared Rust data describes pose, prepared patch records, visibility state,
resident LOD, atlas keys, material/node keys, and logical render commands.
Backend code owns actual buffers, textures, VAOs, transform feedback, storage
buffers, pipelines, bind groups, framebuffers, and submission.

WebGL2 keeps asynchronous classification and resident crack-free topology.
WebGPU will replace transform feedback with compute preparation, reconcile LOD
in storage, compact visible instances, and emit indirect draw arguments. CPU
readback becomes optional telemetry rather than a frame dependency.

## Tuesday cut

### Release gates

- drag/drop, URL, bundled, and Blender-exported GLB loading all work;
- static and animated models retain materials, selection, picking, and
  crack-free LOD behavior;
- camera, focus, and transition behavior has deterministic Rust tests before
  JavaScript authority is removed;
- presentation data can compose at least two model assets, named views,
  authored transitions, and textual cues;
- at least one walk/attach path is demonstrable, with a safe detach fallback;
- selected legacy Quilting examples or modes run through the current renderer;
- a preflight reports missing assets and unavailable optional capabilities;
- the demo has an offline-friendly launch path and a checked-in runbook;
- every accepted milestone is committed and the release worktree is clean.

### Strong stretch goals

- Blender live reload or one-way edit sync during the presentation;
- shadow-index visualization and measured culling comparison;
- a conformal transition between presentation sections;
- an educational patch/tessellation inspector tied to the selected face;
- browser/device input capture and renderer-image replay atop the completed
  semantic presentation replay oracle.

### Explicitly deferred unless the gates are already safe

- full HHHS peer-to-peer browser/Blender replication;
- a production WebGPU backend;
- replacing all WebGL2 submission with compacted indirect draws;
- higher-order QB surfaces;
- making the experimental conformal optimizer part of asset loading;
- general rigid-body physics in transformed space.

## Measurements and evidence

For each performance change, record the relevant subset of:

- model parse, texture decode, atlas topology, atlas packing/transfer, upload,
  and time-to-first-render;
- frame CPU time, GPU time when available, and frame-time percentiles;
- source faces, prepared patches, visible patches, submitted instances,
  atlas vertices, triangles, and draw calls;
- LOD classification frequency, readback bytes, sparse update bytes, and batch
  rebuild count;
- spatial-index visited nodes, certified rejects, unknowns, reference mismatch,
  and animated refit time;
- interaction-to-visible latency and transition determinism;
- asset bytes, peak transient bytes where observable, and offline cache hits.

The representative matrix is horse animation, chess-scale high face count,
one small static asset, a pole-adjacent inversion, a mixed-material scene, and
a two-GLB presentation scene. A result is not generalized beyond the path and
browser actually measured.

## Migration sequence

1. Preserve the green baseline and add presentation/demo fixtures.
2. Split `hyperscape` into focused modules without changing its public behavior.
3. Add `CameraRig`, semantic actions, and deterministic transition clocks.
4. expose one compact Rust runtime packet through the WASM boundary;
5. switch browser camera/focus integration to that packet, retaining a parity
   diagnostic until the duplicate JavaScript path is deleted;
6. add presentation state and multi-asset scene composition;
7. add surface attachment/walking and shadow round-index queries;
8. port the chosen legacy examples and educational views;
9. optimize only measured bottlenecks and rehearse the exact release path;
10. freeze the demo, document recovery paths, and tag the release candidate.

This order intentionally creates a useful presentation before finishing the
long-term networking or WebGPU work, while ensuring the presentation itself is
built on the Rust ownership model rather than becoming another disposable UI.
