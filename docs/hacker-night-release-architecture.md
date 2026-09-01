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

### HHHS release boundary

The exact HHHS `v0.4.4` dependency is pinned to its immutable release tag.
HHHS now provides the browser journal/worker placement and local co-transaction
attachment seams, but Hyperscope does not imitate or automatically adopt them
inside its renderer. Camera, focus, selection, animation, cue time, asset jobs,
and peer presence remain ephemeral application state. Explicit authored edits
may enter the durable Replica adapter only after semantic validation and a
whole-`AuthoredRevision` atomicity decision; the opt-in local Blender relay
remains a disposable transport with no durability or repair claim. This keeps
the renderer independent of HHHS release cadence while leaving one deliberate
durable authored-state seam.

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
- The application adapter feeds real startup, IndexedDB, drag/drop,
  authored-demo, and presentation asset acquisition plus presentation load/cue
  intent into that reducer. `appshadow=1` explicitly enables the observer when
  no Rust authority lane already requires it; implicit enablement is not added
  to canonical URLs. `assetimpl=js|shadow|rust` is the separate acquisition
  rollback boundary. Rust is now the canonical default; an explicit
  `assetimpl=js` remains the serialized rollback. Shadow mode observes effects
  without changing incumbent behavior; Rust mode gives startup/drop/demo
  requests one mutually exclusive primary-scene scope while presentation
  layers retain per-asset concurrency.
  Cross-asset replacement emits cancel-then-fetch, aborts obsolete browser
  acquisition, serializes the global model worker's dynamic parse/upload lane,
  and checks the current primary request after every asynchronous upload
  boundary. Late completions remain diagnostic but cannot reach renderer
  installation. Each cue action compares
  the complete resolved presentation snapshot. A separate opt-in pose gate
  synchronizes settled navigation through the reducer and makes exactly one
  compact comparison call per active cue-transition frame. The application
  clock now advances once per browser frame through a non-serializing boundary;
  it takes a navigation snapshot only during an active cue, mapped-selection,
  or selected-camera transition and makes no call when no Rust lane is
  active. Mapped selection
  interpolation uses the incumbent wall-clock timestamp, including throttled
  frames, and compares both browser focus state and the renderer's retained CPU
  focus packet without a GPU readback. Bounded diagnostics are exposed at
  `globalThis.__hyperscopeAppShadowDiagnostics` until this lane earns browser
  authority. The same adapter now accepts the incumbent navigation shadow's
  device-neutral actions through `AppStore`; the shared navigation queue owns
  their sequence and the application owns virtual time. The adapter exposes a
  parity-complete frame snapshot. `navimpl=js|shadow|rust` is the navigation
  rollback boundary. SpaceMouse, pointer turntable, and selected-camera aim and
  reframe
  have generated-WASM and live Chrome gates; JavaScript remains the default
  until the remaining camera gestures and arbitration paths carry equivalent
  evidence.
- `presentimpl=js|shadow|rust` is the presentation-orchestration rollback
  boundary. AppStore is now the canonical default, while an explicit
  `presentimpl=js` remains the serialized rollback. Shadow mode advances both
  Rust implementations and compares their complete cue and navigation
  projections. Rust mode allocates only `hyperscope-app`, obtains asset
  metadata from its low-rate read model, dispatches cue intent through
  `AppEvent`, and consumes the application frame snapshot; it performs no
  standalone controller tick or semantic manifest parse in the browser.
  In that Rust-authority lane, the Leptos presentation card first asks the
  platform adapter to clear transient selection and synchronize incumbent
  camera/focus state, then commits `Advance` or `Reverse` directly through
  `AppStore`. Its commit callback only adapts the already-committed cue,
  navigation, render, animation, and composition projections. Shadow and JS
  modes deliberately retain the HTML card so their standalone controller
  remains the incumbent rollback authority.
  Keyboard, startup, and deep-link cue inputs now use the same store-allocated
  semantic sequence through a direct generated-WASM boundary. The older
  explicitly sequenced `present` method remains only for shadow/replay parity,
  so switching rollback lanes cannot erase that oracle while JavaScript no
  longer allocates Rust-authority cue sequence numbers.
  Active cue snapshots also resolve exclusive presentation overlays to
  Quilting's shared backend-neutral `RenderStyle`. The browser no longer owns
  an overlay-to-render-mode policy; it only adapts the legacy `matcap_wire`
  control spelling and reports unsupported orthogonal overlay capabilities.
  Rust likewise owns cue tessellation defaults and bounds. The browser applies
  the complete validated policy without coercion or clamping and fails visibly
  if the generated bridge violates that typed snapshot boundary.
- Effective spheroidal-focus authority now crosses that same ordered boundary.
  Rust `focus_enabled` denotes fuzzy post-processing enabled specifically in
  mode 3; modes 0–2 remain renderer-only blur choices, and the retained shared
  sphere may stay active for inversion or editing without enabling focus.
  Browser signal changes to enablement, shell coordinate, and angular aperture
  coalesce into one microtask and enqueue one atomic `SetFocusField` action per
  active parity controller. The optional enabled-state field preserves the old
  coordinate-only replay/API surface while ensuring an invalid aperture cannot
  partially toggle focus. Initial `AppStore` synchronization now includes
  the complete camera lens, aim policy, focus field, inversion, and sphere, so
  a focus-only route cannot temporarily retain Rust defaults. Applying an
  authored presentation snapshot suppresses the reciprocal signal adapter and
  cannot feed the same focus edit back into the queue. Under the default
  `selectionimpl=rust` gate, browser values are intent only: WebGL retains the
  last committed field until AppStore integrates the action, then the Rust
  projection updates renderer, Leptos view, controls, and URL together.
  `selectionimpl=shadow` observes without renderer writes and
  `selectionimpl=js` bypasses this boundary.
- `AppStore` now publishes a throttled navigation projection before its summary
  revision fence, separate from the immediate render/input frame snapshot. A
  read-only Leptos CSR island consumes that FRP signal to show anchor, chart,
  focus sphere, field, and lens status. The browser selects only the low-rate
  flush boundary and mounts the host; it neither polls navigation nor owns a
  second view model. Selection and presentation publish their settled endpoint
  explicitly, while active frame integration remains allocation-light.
- The focus-transition and surface-walk preference panel is likewise a Leptos
  CSR island in the canonical `navstateimpl=rust` lane. It consumes the separate
  committed navigation-settings signal and dispatches one complete replacement
  through `AppStore` per edit. JavaScript applies only the committed packet to
  legacy runtime consumers and URL state. Live six-control Rust/shadow/JS
  evidence on 2026-08-31 promoted Rust to the route default; `js`, `shadow`,
  and mount failure retain the incumbent HTML controls as measured rollbacks.
- The vertical field-of-view control is a separate Leptos CSR island in the
  explicit `navimpl=rust` lane because projection lens state belongs to the
  application camera, not to the preference packet. It derives its exact
  35–110 degree integral domain from `ControlSpec`, queues
  `NavigationAction::SetPerspectiveLens` through `AppStore`, and preserves the
  committed near/far planes. The browser callback advances that same
  application controller at zero elapsed time and projects only the integrated
  Rust FOV into the legacy renderer/URL signal. The HTML control remains the
  fallback for JavaScript/shadow navigation and mount failure.
- The normalized SpaceMouse camera gate freezes samples at
  a platform-neutral Rust boundary. Browser code retains only WebHID/report
  acquisition, device shaping/smoothing, button layers, and the
  screen-relative linear speed frozen at gesture start. Rust validates axes in
  `[-1, 1]`, computes translation/rotation/object-dolly response from virtual
  delta and user gains, applies preset/swap/inversion/horizon policy, and queues
  the resulting ordinary navigation actions through `AppStore`. It adds no
  device-specific event to the application vocabulary. Under `navimpl=rust`
  its retained camera packet is authoritative; shadow and JavaScript routes
  remain explicit measurement and rollback lanes. Generated WASM matches the
  incumbent mapping over 7,168 exhaustive mapping cases, 648 response-policy
  cases, four `AppStore` camera initial states, and a 120-frame deterministic
  trace.
  Modifier layers and surface-walk retain their own semantic routes;
  authored-transition arbitration remains an explicit later cutover.
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
- `hyperscope-app::ControlSpec` is the canonical registry for all currently
  linkable controls and migration flags. `HyperscopeRoute` owns default
  equivalence, first-value duplicate semantics, stable ordering, and explicit
  malformed/unknown diagnostics. `routeimpl=js|shadow|rust` is the rollback
  boundary for URL writes, with Rust now canonical and `routeimpl=js` retained
  as the serialized rollback. Rust commits the validated canonical pair order;
  a bridge error or Rust diagnostic falls back to the unchanged browser query
  and records the fallback. Old `routeshadow=1` links are admitted once by the
  browser bootstrap and normalized to `routeimpl=shadow`; the duplicate legacy
  flag is no longer part of Rust state or canonical URLs.
  The same switch admits valid startup pairs through Rust before model, Patch
  Lab, animation, or control state is applied; malformed startup input retains
  the incumbent browser path. DOM assignment and control-specific clamping
  remain browser adapter work while that rollback is soaking. Camera drafts
  normalize rounded signed zero before comparison, so Rust numeric-default
  equivalence does not create false parity failures.
  Render mode, resolution level, tessellation density, pixel floor, and atlas
  exponent now carry their actual enum or bounded numeric contracts in that
  registry rather than masquerading as generic text or numbers. Invalid URL
  values therefore produce a Rust route diagnostic and preserve the incumbent
  fallback path; valid values are not silently admitted for a later JavaScript
  clamp. A successful Rust startup admission now also carries the complete
  typed `RenderSettings`, with omitted controls resolved from the Rust
  registry. The browser projects that value exactly into controls; its old
  defaults and clamps run only for explicit JavaScript authority or a recorded
  Rust startup fallback. A missing typed projection fails over visibly rather
  than being mistaken for successful Rust authority.
  `ControlSpec` now exposes reusable numeric-domain metadata—minimum, maximum,
  integrality, and preferred view step—rather than burying those constraints
  in route-specific validators. It also carries closed choice vocabularies for
  transform, focus, SpaceMouse, and Patch Lab modes. Render URL admission and
  Leptos controls share one range contract, while camera, focus, walk,
  SpaceMouse, animation-clock, and Patch Lab startup values use the same
  machinery. Patch Lab exponent requests are additionally validated against
  the resolved resident-atlas exponent as one cross-field route rule.
  The same successful result carries every resolved control value in
  registry order, so omitted URL controls no longer regain authority through
  a browser projection default after WASM starts. The browser retains ten
  values for its inert pre-WASM model/Patch-Lab/camera preview, verifies them
  against the Rust registry during initialization, and then installs Rust's
  complete defaults even for the explicit JavaScript route rollback. The
  separately parity-gated implementation/backend switches remain one explicit,
  registry-checked 18-entry bootstrap policy until WASM startup itself moves
  ahead of those decisions.
  After initialization, the route rollback therefore preserves the incumbent
  parser and serializer, not a second application-default table. Its decoded
  startup object is likewise projected by iterating that installed registry;
  the browser no longer repeats the control-key list merely to read values.
  Compact canonical pairs remain a separate provenance channel: only an
  explicitly linked `animtime` or `animspeed` may arm deferred clock
  restoration. A resolved default can populate state without masquerading as
  authored URL intent.
  Once that complete route passes Rust admission, bounded camera, conformal
  sphere, animation clock, SpaceMouse, walk, focus, and Patch Lab numbers are
  converted without another fallback, clamp, or rounding policy. The legacy
  helpers now run only for explicit JavaScript authority or recorded Rust
  fallback. Patch Lab installs the admitted atlas maximum before assigning its
  exponent controls, preventing the DOM's old maximum of seven from silently
  truncating an atlas-nine link.
  Camera/conformal-transform state, SpaceMouse policy, and surface-walk policy
  additionally leave Rust as one typed `RouteNavigationSettings` value.
  Centiseconds and percentages become semantic seconds and fractions before
  crossing the WASM boundary; default walking pace, eye height, fast factor,
  and near-plane policy come from `hyperscape::SurfaceWalkControls` rather than
  a second set of frame-loop constants. The browser only projects that value
  into existing DOM signals and retains raw-query decoding for the inert
  pre-WASM preview and explicit rollback lane.
  A valid selected-object route also carries its protocol-native
  `AssetEntityId` as a typed projection. Rust performs atomic pair admission,
  nil rejection, UUID parsing, and canonical lowercase formatting; successful
  startup no longer re-runs the browser regex. The old helper remains for the
  explicit JavaScript and runtime fallback lanes.
  Explicit clip-relative animation time and speed are likewise projected as a
  typed partial value. Rust retains the difference between an omitted field
  and an explicitly linked default; the browser only defers that value until
  the requested clip range is resident, then commits it through the existing
  application animation action.
  Animation clip indices are integral Rust-domain values as well; a
  successful route converts them exactly, while fractional legacy inputs can
  only take the explicit browser-fallback path that retains `parseInt`.
  Camera links carry explicit `aim=1` when `px/py/pz` is a finite semantic
  target; omitting it means the same visible pose has a free sight tangent.
  Rust validates and canonically orders that policy, and selection or model
  cleanup cannot silently rewrite it. Selected-object links carry the stable
  `(selasset, selentity)` UUID pair atomically; a partial pair is invalid and
  no transient face or packed-node index enters the route. The browser retains
  a valid unresolved pair only while the requested asset is loading, then
  resolves it against Rust-issued packed identities after node bounds exist.
  Missing or ambiguous identities fail closed instead of selecting a nearby
  face. These session selection IDs are ephemeral URL state, not durable HHHS
  commands; authored Blender identities remain stable across export while
  ordinary runtime assets receive deterministic IDs only for that load.
  Animation links likewise carry clip-relative `animtime` and signed
  `animspeed` rather than exposing the application's unwrapped clock. Explicit
  values remain pending until the requested clip is resident, then one Rust
  `SetAnimationClock` commit maps them into that clip and updates the browser
  pose controls. Startup awaits this final clip selection before enabling URL
  writes, preventing a default clip-reset from erasing an admitted route.
  Wrapped time is intentional: a copied link restores the same visible pose
  and playback direction without serializing an irrelevant loop count.
- `quilting-core::render` owns retained scene snapshots, logical frame
  commands, indexed submission accounting, and the bounded backend-parity
  observer. `rendershadow=1` extracts WebGL state only when the retained scene
  changes and compares every subsequent frame inside WASM; the browser can
  explicitly query `globalThis.__hyperscopeRenderShadow` but receives no
  per-frame diagnostic traffic.
- High-rate frame, navigation, and presence events advance authoritative state
  without forcing DOM-rate notifications. `SignalVec` runtime-asset,
  authored-asset, authored-entity, and diagnostic views plus the low-rate
  presentation projection are published as a batch and an `AppSummary`
  revision is set last as the consumer commit fence; adapters explicitly flush
  at their UI cadence. Presentation transitions reconcile on the frame lane
  without cloning cue assets/layers into render snapshots.
- The generated `HyperscopeAppShadow` facade admits one transport-neutral
  authored checkpoint as a decimal-text `u64` projection revision plus the
  canonical protocol JSON command array shared with Blender. This avoids
  JavaScript integer truncation, validates the whole batch before mutation,
  and exposes the key-sorted authored asset/entity projection in its bounded
  snapshot. It still chooses no socket and has no renderer authority; those
  remain separate extraction and transport cutovers.
- `hyperscape::extract_packed_scene` now owns the first renderer-independent
  durable-edit join. A protocol-v0.1 entity transform is an absolute ordinary
  TRS in its source asset's world chart: it replaces that node's flattened
  glTF world matrix, and the presentation-layer TRS remains outermost. The
  result is packed-node sorted and backend neutral. Multiple layers may
  instance one asset and receive the same edit; one entity UUID crossing asset
  boundaries is rejected because the current command carries no asset ID.
  Unmatched valid edits remain explicit so a later-resident asset can converge.
- Generated WASM exposes that join through the application facade, consuming
  the app's active cue and accepted authored projection rather than a
  socket-local or browser-semantic cache. The browser supplies only stable
  layer/asset identity plus renderer-local node/source metadata; AppStore
  samples cue, authored projection, and application revision under one lock
  and Rust supplies layer TRS, effective visibility/opacity, and sorted node
  matrices. Unknown, missing, duplicate, mismatched, or semantic-bearing
  bindings fail atomically. The browser has a canonical
  `sceneimpl=js|shadow|rust` rollback gate, with `rust` as the canonical default
  and explicit `js` rollback. `shadow` compares fallback matrices
  and layer render state while retaining JavaScript authority; authored
  absolute edits are counted as intentional overrides. `rust` feeds Rust
  matrices and layer state into both presentation rendering and LOD state,
  falling back visibly on extraction failure. The join runs on low-rate
  layer/scene changes over node records, never per face or per animation frame.
  Diagnostics live at `globalThis.__hyperscopeSceneExtraction`.
- The direct local-peer ingress is transport neutral and explicitly
  single-writer. `LocalPeerEnvelope` has disjoint `authored` and `presence`
  variants; application ingress turns only validated authored frames into
  monotonically fenced `AuthoredRevision` events and sends presence through
  receipt-relative TTL handling. Bounded message-ID memory, sender-local
  sequence tracking, and consuming local-echo memory reject retries without
  giving a relay reducer authority. Multi-writer authored convergence still
  requires `hyperscape-hhhs`; arrival order is never presented as causality.
- Generated WASM owns one `LocalPeerIngress` beside its `AppStore`.
  `receiveLocalPeerEnvelope` admits canonical JSON, while
  `recordLocalAuthoredEnvelope` marks an already-applied outbound edit for echo
  suppression and `recordLocalPresenceEnvelope` validates an outbound sample
  while consuming its later relay echo without admitting this process as its
  own remote peer. Ephemeral peers are sampled through `peerPresenceSnapshot`, a
  deliberately separate high-rate lane whose sender sequences and local
  expiries do not masquerade as the throttled UI read-model revision. These
  methods select no carrier and add no browser state authority.
- The opt-in `quilting-wasm/durable-history` route now exposes
  `openDurableAuthoredPeer` on the same application facade. Project identity
  and proposal role now enter the `hyperscope-app` reducer as semantic session
  intent; the browser resource future executes the returned typed open effect
  and completes its exact job with recovered history/projection evidence.
  IndexedDB handles, HHHS objects, archive bytes, relay endpoints, and bearer
  credentials remain platform resources. The default peer is a replica and
  rejects raw `authored` proposals; promoting those requires the explicit
  `openDurableAuthoredPeerWithRole(..., "admission_authority")` path. The peer
  object shares the Rust `AppStore` while owning its IndexedDB-backed
  `DurableAuthoredSession`; dropping it executes the matching close lifecycle.
  Authored input is persisted before the result returns, and an applied result
  includes both carrier-ready encoded `ReplicaRecord` bytes and canonical
  Rust-produced `AuthoredRecordFrame` JSON. Incoming record frames atomically
  persist the public entry with a receiver-local cursor, defer missing
  predecessors without writes, and replace the AppStore authored read model
  from canonical HHHS materialization rather than arrival order. Presence
  stays on the ephemeral lane. A per-application writer lease rejects a second
  live durable peer. An RAII open reservation reports cancelled opening as a
  retryable completion against the exact reducer job and releases the writer
  lease, while an RAII session lease prevents reentrant calls without holding
  a `RefCell` borrow across writes; write cancellation restores the session.
- Protocol v0.1 now has an additive optional `authoring_leases` presence field.
  A claim combines a stable lease UUID with an asset-scoped entity identity;
  the containing peer envelope supplies ordering and receipt-relative expiry.
  The app derives key-sorted `held` or explicit `contended` read models from
  live presence. It never chooses a winner from arrival order, and omission or
  TTL expiry releases a claim. These values are advisory coordination—not
  capabilities, authorization, durable locks, or HHHS history. Blender/browser
  acquisition and edit gating remain a measured adapter cutover.
- `hyperscope-web` offers an optional `local-peer-relay` binary, excluded from
  default features. It is an authenticated, exact-origin, loopback-default
  HTTP adapter over bounded opaque JSON delivery. Process generations and
  decimal cursors make restart or retention gaps explicit. It neither parses
  protocol commands nor allocates authored projection revisions; a gap is
  degraded delivery, never counterfeit repair or convergence.
- The browser carrier is disabled by default and owns only authenticated HTTP
  delivery, a bounded ordered outbound queue, and restart-aware decimal
  cursors. Relay batches carry application frames as exact JSON text, so
  Blender's nanosecond-scale `u64` sequence never crosses JavaScript's lossy
  numeric representation. Its role is explicit: `legacy` retains the direct
  single-writer rollback, `ignore` makes a durable replica refuse to promote
  raw proposals, and `admit` selects exactly one durable proposal authority.
  Only that authority may turn a Blender `authored` envelope into an HHHS
  record; every durable replica may consume the resulting Rust-authored
  `authored_record` frame. Async admission reserves outbound capacity before
  persistence so a committed record cannot lose its announcement slot.
  Generated Rust/WASM performs semantic admission and packed-scene extraction.
  The renderer receives only resolved ordinary-world node matrices, including
  the presentation layer outside an authored absolute transform, so the same
  result overrides active conformal source packets in drawing, LOD, focus
  bounds, picking, and walking. Project/session selection and automatic repair
  remain separate application lifecycle work.
- The same opt-in carrier now publishes the live browser viewport in the
  opposite direction. Until the general navigation cutover, the incumbent
  browser controller supplies one semantic eye/forward/up, selection, focus,
  cue, and animation sample; generated Rust/WASM parses UUIDs and decimal
  `u64`, validates the complete protocol value, and emits canonical presence
  JSON. Changed samples are bounded to 20 Hz and settled samples refresh every
  500 ms under a 1,500 ms TTL. Receipt time and `AppEvent::Frame` use the same
  application-clock epoch, so a disconnected received peer expires rather
  than retaining a page-uptime offset. This is an explicit migration seam, not a
  claim that default mouse navigation is already AppStore-authoritative.
  Outbound echo suppression also avoids a second reducer event per sample.
  Blender exposes detached TTL-filtered remote samples to a transient
  `POST_VIEW` draw adapter. It renders peer camera glyphs, focus/inversion
  spheres, and selected-entity wire bounds without creating helper datablocks
  or saving presence in the `.blend`. An inverted output-chart camera is
  reflected back through the shared sphere, including its tangent frame,
  before it is drawn in Blender's ordinary source chart. The real
  Blender/browser smoke proves both directions, at least 166 overlay segments,
  no added objects, and near-`u64::MAX` sequences preserved as exact JSON.
- `hyperscope-app` exposes a versioned, adapter-independent replay format. A
  replay contains semantic events, each commit/rejection outcome, and a compact
  camera/focus/cue/asset/presence/diagnostic snapshot; it contains no DOM
  events, device reports, renderer handles, or wall clock. Decimal JSON uses
  exact `f64` round trips.
  The native `replay` feature is excluded from browser builds;
  `hyperscope-replay` version 0.23 walks every checked-in cue, every current
  semantic navigation action, and every current application event lane. It
  records the key-sorted authored asset/entity materialization as well as its
  atomic projection revision, so stale or invalid Blender-style checkpoints
  cannot silently mutate the scene oracle.
  Version 0.23 records the atomic device-independent navigation-settings
  packet and includes that packet in every compact committed-state witness.
  Transition duration and surface-walk behavior therefore survive replay and
  future session transport without admitting browser HID maps or window-focus
  policy into shared application state. A 0.22 script cannot introduce the new
  event and retains the historical default packet.
  Version 0.22 records the ephemeral presentation-animation residency that
  joins an authored presentation asset to an exact process-local renderer
  scene. Presentation and resident asset IDs may differ; the request/asset
  fence must still match the installed scene exactly. Cue animation names are
  resolved in Rust, duplicate names and out-of-range authored times fail
  explicitly, and a successful binding emits the ordinary exact clip job. The
  generated WASM adapter binds the primary residency before multi-asset scene
  packing. Both direct and Leptos cue controls carry committed clip effects to
  the renderer adapter; they do not repeat name lookup in JavaScript. A clip
  change requested after the current WebGL packed scene exists is completed as
  an explicit failed job until per-layer animated residency is implemented.
  Version 0.21 makes the installed animation catalog operational application
  state. Installation derives the renderer's initial clip, selection allocates
  a Rust job and emits an exact scene/request/asset-scoped effect, and only its
  matching completion changes the active clip. Duplicate intent is inert,
  returning to the incumbent cancels pending work, replacements cancel old
  clip jobs, and stale or failed completions preserve the active clip. The
  browser retains `animclipimpl=js|shadow|rust`; live multi-clip replacement,
  switch-back, cancellation/repair, stale-completion, and URL evidence promoted
  Rust to the canonical default on 2026-08-31. JavaScript and shadow remain
  explicit rollback/measurement routes. A 0.20 replay retains its
  historical installed catalog without gaining application-owned active-clip
  state or asynchronous clip jobs.
  The browser records a separate scalar renderer-residency witness only after
  the clip's worker, skin/morph textures, rest instances, LOD compute model,
  and same-context residency are coherent. This is platform resource state,
  not a second semantic controller. `__hyperscopeAnimationClipDiagnostics`
  compares it with Rust's active clip after install, pending selection,
  completion, failure, and explicit same-clip repair; a Rust no-op cannot hide
  missing renderer residency. Returning to the incumbent while another clip
  switch is in flight consumes Rust's cancellation effect and performs an
  ordered incumbent reinstall, so the already-dispatched worker request cannot
  make a canceled clip resident later.
  Under the default Rust lane, a Leptos selector projects the installed
  catalog and active/pending clip through the application summary revision
  fence. It dispatches directly through `AppStore`; the platform callback
  receives the exact committed selection and cancellation effects rather than
  returning user intent to JavaScript. The incumbent HTML selector remains the
  explicit JavaScript/shadow rollback, and the shadow oracle remains available.
  `AppFrameSnapshot` also carries an allocation-free active-clip sample with
  exact scene/request/asset identity, clip index, authored range, and wrapped
  time. The four-`f64` WASM packet omits repeated IDs but derives entirely from
  that snapshot; measured clip lanes no longer need to pass a browser-owned
  `{time_min, duration}` pair back into Rust every frame.
  The exact multi-clip shadow/Rust soak and acceptance counters are frozen in
  the [2026-08-29 animation clip cutover model](benchmarks/2026-08-29-animation-clip-cutover-model.md).
  Packed presentations no longer rebuild the primary prefix when a cue changes
  clips. The platform adapter supplies its exact retained primary vertex/face
  witness to Rust; Rust validates it before replacing the evaluator, and a
  successful switch retains the composed instance, face-domain, worker-LOD,
  and same-context resources. This keeps secondary presentation assets intact
  without moving renderer resource policy into the application reducer.
  Version 0.20 separates a successfully fetched/decoded primary candidate from
  a renderer-resident primary scene. Decode emits an explicit
  `InstallPrimaryScene` effect; only a matching, validated completion can
  publish topology and animation-clip facts as installed. A newer request
  cancels an in-flight installation, stale results are diagnostic-only, and a
  failed replacement preserves the preceding resident scene. The replay
  adapter removes this new job when reading 0.19 and older scripts, so opening
  an old trace does not manufacture renderer work that its schema never had.
  Version 0.19 added the last successfully decoded primary candidate, including
  exact request/asset identity, byte length, digest, and provenance, without
  claiming that candidate had reached a renderer.
  Version 0.18 makes render style, resolution override, tessellation policy,
  resident-atlas exponent, and face-edge grading one atomic reducer value and
  includes it in every committed replay state. Cue activation replaces only
  its authored style and tessellation subset while preserving session atlas
  and grading policy. The generated application facade admits the same
  complete value, rejects unknown backend-neutral styles and invalid ranges
  atomically, and exposes its revision-fenced projection; browser control
  signals remain a measured rollback behind
  `renderstateimpl=js|shadow|rust`. The explicit shadow lane coalesces one
  browser signal batch into one complete reducer comparison, while the
  Rust lane reapplies the committed projection and rolls invalid intent back
  to the preceding Rust value. Live browser parity evidence recorded on
  2026-08-31 promoted Rust to the canonical route default; JavaScript and
  shadow remain explicit, linkable rollback/measurement lanes.
  The Rust lane mounts a Leptos control island directly over the
  reducer's `render_signal`. Each edit reads the current reducer snapshot,
  allocates a direct semantic-input sequence under the same application lock,
  and commits one complete replacement value without returning action intent
  through JavaScript. Only the resulting committed projection crosses the
  platform-effect callback to drive the incumbent renderer signals; its
  allocated sequence also advances the compatibility counter so a later
  rollback-path input cannot reuse it. A separate read-only error callback
  retains browser diagnostics if Rust rejects an edit. The incumbent HTML
  controls remain available for the explicit JavaScript lane and automatically
  reappear if the Rust view cannot mount.
  Replay 0.26 separates semantic render style from focus-compositor inspection:
  Composite, Weight, Distance Field, and Firmness are a typed diagnostic value,
  while the underlying scene remains PBR. Legacy `mode=fz-*` links normalize to
  `mode=pbr&fdebug=*`; replay 0.25 rejects non-composite diagnostics rather than
  assigning the new meaning retroactively. Unsupported focus diagnostics cause
  an explicit WebGPU-to-WebGL presentation fallback, and returning to Composite
  requests a fresh device LOD epoch even when animation is paused.
  Shadow/Rust startup no longer seeds the reducer by reading those projected
  browser signals back. The admitted typed route value is committed directly
  to `AppStore` before WASM readiness publishes control effects; controls then
  consume the same commit. Route startup and browser shadow synchronization
  share the one compatibility sequence/effect adapter; Leptos dispatches
  directly through `AppStore`. This eliminates duplicate browser
  `setRenderSettings` call sites without weakening the rollback lane.
  Version 0.19 retains the last successfully loaded/decoded primary asset as a
  distinct FRP read model, including its request, descriptor, byte length,
  digest, and provenance. A pending or failed replacement cannot erase that
  candidate, including same-asset reloads. This does not claim renderer
  residency: scene installation and active animation clip selection remain
  separate asynchronous completion boundaries.
  Version 0.17 added the renderer-independent primary animation clock: playing,
  unwrapped scene time, and signed finite speed. Frame events advance it from
  explicit deltas, cue activation replaces all three fields atomically, and
  invalid clock edits preserve the preceding application revision. Version
  0.16 added selected-object aim without reinterpreting 0.15. Generated WASM
  exposes atomic clock/seek/speed actions and a fixed three-`f64` write packet,
  so a high-rate browser parity lane need not allocate or serialize a frame
  object. Browser clip wrapping remains the incumbent until that lane is
  measured and cut over. `animclockimpl=js|shadow|rust` is that independent
  rollback boundary; the measured default is now `rust`. The Leptos playback
  toggle allocates and commits `TogglePlaying` through `AppStore` directly. Its
  browser callbacks observe only the committed playing value, sequence, and
  revision—or a rejection—so renderer adaptation and diagnostics remain thin
  effects without regaining action authority. Live ordinary-horse evidence
  recorded 416 shadow comparisons with zero mismatches and `9.45e-14` maximum
  error, then 432 Rust authority writes with no fallback or errors, including
  pause and seek. The cutover gate subsequently passed reverse wrapping on a
  75.8-second clip, a real hidden-tab interval, clamped foreground resumption,
  pause/seek/resume, and presentation-driven play/pause transitions. Explicit
  `animclockimpl=js|shadow` routes remain available for rollback and parity.
  Spacebar, startup, and retained HTML playback intent now also allocate their
  semantic sequence inside `AppStore`; JavaScript consumes only the committed
  playback receipt. Explicitly sequenced WASM methods remain as replay/shadow
  oracles while the independent high-rate clock lane continues to soak.
  In `animclockimpl=rust`, route/presentation clock restoration and paused
  scrubbing likewise allocate inside the store. JavaScript/shadow modes retain
  their explicit sequence path, so this narrows Rust authority without erasing
  the comparison oracle or prematurely changing the default clock lane.
  The Rust lane now also exposes a compact `AppAnimationSnapshot` and dedicated
  throttle flush, separate from the full summary and navigation projections.
  A Leptos timeline consumes that FRP signal, disables seeking during playback,
  converts authored clip time to the reducer's unwrapped relative clock, and
  dispatches `Seek` directly through `AppStore`. The 50-ms browser cadence only
  publishes this compact read model and adapts committed sample time into the
  incumbent renderer; it no longer writes the timeline DOM or clones unrelated
  application read models. The original HTML timeline remains the `js|shadow`
  and mount-failure rollback.
  Version 0.9 records whether an asset request is independent or replaces the
  primary scene. A 0.8 trace keeps its historical per-asset meaning; it cannot
  smuggle a `primary_scene` request into the older schema.
  Version 0.7 makes selected identities explicitly asset-scoped, so the same
  entity UUID in two composed assets cannot alias. Legacy unscoped focus
  anchors fail closed instead of receiving a fabricated asset identity.
  Version 0.6 added complete validated perspective-lens edits and an explicit
  semantic-target-presence policy without inferring aim mode from inversion.
  Version 0.5 retains selected source bounds and clicked pivots and derives
  output-chart pivots/radii in the application snapshot; a projection pole
  clears only those derived values. The reader accepts 0.4 through 0.23 inputs,
  but only 0.4 migrates an omitted source pivot to the bound center.
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
  atomic rejection, and transition cadence invariance. The eight-cue golden is
  `fnv1a-128-json:4855accdee4f28b48ae954ebb4ab99cb`;
  the navigation golden is
  `fnv1a-128-json:a89d8fdeeb12474d28ae4bf38faf5c01`; the orchestration
  golden is `fnv1a-128-json:8e6b83ab648451471bd246457cc790a6`.
- `hyperscape::StableEntityId` converts explicitly to the validated wire
  `EntityId`, so the protocol wrapper is an interchange type rather than a
  second identity authority.
- Blender's dependency-free `protocol.py` validates and canonically roundtrips
  the canonical fixtures from `hyperscape-protocol`. Its shared bounded-memory
  primitives match Rust's duplicate, sender-sequence, and consuming local-echo
  policy for both authored edits and receipt-relative-TTL presence. The relay
  remains a separate delivery adapter and ephemeral state has no HHHS admission
  path.

This layer is not yet the complete browser authority. Primary asset request,
cancellation, completion, and stale-install policy have crossed to Rust after
their live cutover gate; file/network acquisition and renderer installation
remain browser adapters. Presentation cue and transition orchestration have
also crossed to AppStore, while the browser still adapts the committed read
model to DOM and renderer state. Navigation, selection, scene extraction, and
URL state retain their separate shadow-and-rollback gates until each default
cutover has equivalent evidence.
The selection adapter now joins validated authored node UUIDs to explicit
presentation asset IDs across packed composition offsets and mirrors mapped
picks/detaches through the AppStore. Ready IndexedDB, dropped, startup, and
ordinary GLB loads now acquire deterministic session node IDs from Rust after
the application reducer commits their load. These pairs are selection/runtime
scope only: they cannot masquerade as durable Blender identity or enter HHHS.
Stale and unresolved loads fail closed to the browser path. The checked Blender
release fixture now carries five persistent stable IDs, four joined to pickable
mesh nodes, and a live Chrome MCP composition gate selects one with zero
browser-transition, renderer-packet, or observer-sequence mismatch. The shared
clock and CPU-retained packet remain measurable in every mode.
The next semantic layer is now explicit in `hyperscape::interaction`.
Renderer, browser, XR, or replay adapters resolve their own rays into one
validated `InteractionHit`: asset-scoped entity identity, source bound and
pivot, displayed-chart distance, and optional face/barycentric detail. The
virtual-time reducer owns hover/press/release/cancel and focus-radius-aware
reach. Its state is deliberately ephemeral and contains no selected field; a
same-identity press/release emits the existing `AnchorFocus` navigation action,
and snapshots derive selection from `FocusNavigation::anchor`. This keeps
WebGL2/WebGPU picking, selection tint, and DOM event shape outside semantic
authority. The generated `HyperscopeAppShadow` facade now validates and queues
entity-level or face/barycentric hover plus primary press/release/cancel, and
projects an interaction snapshot without mutating focus directly. Renderer
residency is registered through an atomic `InteractionTargetTable`: transient
packed nodes carry source bounds and optional stable asset/entity identity,
while WebGL2/WebGPU samples carry only the packed node, source pivot, displayed
distance, and optional surface coordinate. The Rust join rejects unknown or
unmapped nodes before semantic dispatch. Each replacement advances a checked
residency epoch, preventing delayed WebGPU results from resolving a reused
packed handle. The table remains adapter state rather than durable `AppState`.
Browser query plumbing now has a shadow-only first slice:
the WebGL2 pick evaluates the exact
animated QB point in source and displayed charts only on explicit picks, while
`selectionimpl=shadow` sends hover/press/release and compares the resulting
stable identity. Exact ray hits bypass proximity range; explicitly named
proximity hits use focus-radius-aware reach. Any facade/version error falls back
to the incumbent direct observer. Live evidence and Rust-default promotion
remain separate work; this boundary does not yet claim live picking authority.
Replay schema 0.25 is the offline interaction oracle: it records the complete
semantic action stream and exact hover/active/derived-selection state, including
optional face/barycentrics. Its golden demonstrates cadence-independent
activation through the existing navigation owner and rejects the same events
under 0.24 rather than assigning historical meaning retroactively.
The staged WebGPU backend now has the corresponding opt-in prepared-patch
query packet. A clip-space remap renders one requested viewport pixel into
retained 1x1 identity/surface/depth attachments and asynchronously maps two
WebGPU-aligned rows. It preserves semantic node, source face/barycentrics,
source-chart point, displayed distance, and the interaction-target residency
epoch without allocating a viewport-sized ID buffer. The exported query does
not publish interaction state; its result must enter the existing Rust packed
target join. Resident-root plus sparse-overlay frames reject this first adapter
until both draw domains populate one shared query target, so WebGL2 remains the
live authority rather than accepting stale ordinary-scene geometry.
`selectionimpl=js|shadow|rust` is the rollback boundary: the Rust mode verifies
the application identity against `AppStore`, joins only the backend-local packed
node, and applies the selected focus packet directly to the resident renderer
without round-tripping the sphere through JavaScript. Delayed RAF callbacks
retain the selection event fence until their timestamp reaches the event, so
background/main-thread backlog cannot double-integrate pre-selection time.
The selected-object inversion gesture is now one
`RefitFocusAndToggleInversion` action: Rust restarts an existing anchored fit,
toggles the reflection chart, and transports camera, in-flight transitions,
and surface-following state under one rollback boundary. A pole rejects that
whole action. Shadow mode compares the incumbent gesture; Rust mode consumes
the same coherent camera/focus snapshot adapter as presentation while retaining
the selected identity and pole-safe derived pivot. No separate browser focus
transition runs in the Rust branch.
Selected recovery framing is likewise one replayable `ReframeSelection`
action. It resolves the selected source sphere in the active conformal chart,
uses the current aspect/FOV and narrower perspective axis, enables a semantic
target without changing the initial visible pose, and interpolates target plus
positive camera distance exactly as the incumbent did. `navimpl=shadow`
measured 40 transition frames with zero mismatches and `9.1e-12` maximum error;
`navimpl=rust` completed with 39 Rust camera writes, exact independently
recomputed distance, zero fallback, and no console errors. Invalid framing or
a selected pivot at the reflection pole leaves the camera unchanged.
Selected-object aiming is the companion replayable `AimAtSelection` action.
It projects the selected source pivot into the active chart, preserves
orientation, lens, and control distance, and uses the same target-orbit path as
the incumbent Object-mode transition. Shadow mode measured 43 transition
frames with zero mismatches and `9.1e-12` maximum error. A nontrivial Rust gate
selected the horse, shift-panned through the real pointer path, retained the
selected identity, and returned to the pivot in 45 Rust camera writes with
zero fallback or diagnostics and an independently measured distance of
`3.0000000000000013`. That gate also separated idle-HID state from camera
staleness: neither an idle SpaceMouse nor a pointer press now destroys the
selected anchor through an unnecessary full-state synchronization.
Unresolved assets and free/manual focus-sphere geometry deliberately remain on
the browser path. Mapped selected-object focus and the semantic spheroidal
field crossed to the Rust default on 2026-08-27 after exact live
selection/inversion, commit-order, and explicit JavaScript/shadow rollback
gates; `selectionimpl=js|shadow` remains available for rollback and
observation. Renderer-only blur modes 0–2 are not reinterpreted as scene
semantics.

The offline release gate also has source provenance now. A Trunk pre-build hook
uses Rust to fingerprint the authoritative crate/shader, HTML/module, manifest,
and copied-asset inputs into `pkg/hyperscope-build.json`. Filesystem preflight
recomputes that bounded receipt and rejects missing, malformed, unsupported, or
stale fingerprints before considering the bundle releasable. The deterministic
FNV-1a-128 receipt is drift detection, not signing or adversarial integrity.

## Three graphs, never one overloaded hierarchy

1. The **ownership graph** describes entities, ordinary node parenting,
   assets, and presentation grouping.
2. The **conformal-frame graph** describes charts and composable Möbius maps.
   Each evaluated revision has exactly one active spatial parent per frame, so
   it reduces to a forest with a unique local-to-world map. Stable authored
   frame IDs map to dense runtime `FrameId` values; array positions are never
   durable identity. A subject/view pair receives one relative map and shared
   ancestry cancels.
3. The **constraint graph** describes tracking, paths, focus anchors, surface
   attachment, and authored relationships between entities or frames.

This separation still supports deeply nested conformal scenes. A typed
frame-to-surface pin references a stable source entity plus face/barycentric
address, normal side, tangent direction, scale, and local conformal offset.
At each animation pose it evaluates the surface point and differential into a
conformal tangent similarity, which becomes the pinned frame's active
local-to-parent edge before descendants are extracted. Descendants therefore
inherit animation and every ancestor's conformal map without baking geometry.
Composed `orientation_sign` and attachment `normal_sign` remain distinct:
authoring can request an inherited chart side or an ambient right-side-in
orientation without confusing either choice with material back-face policy.

For example, a reflected outer object supplies one frame; a second frame can
be pinned to a stable material point on its animated reflected surface; an
inside-out child surface occupies that frame; and a final right-side-in object
can occupy a descendant frame with an explicit orientation/side policy. LOD,
culling, selection, and walking evaluate the same complete frame path and its
differential. Blender should expose this as frame parenting, surface picking,
and side/orientation controls rather than requiring authors to edit generator
arrays manually.

Constraint/reference edges may cross-link this evaluated forest. Multiple
simultaneously active spatial parents are rejected as ambiguous; an authored
cycle is admitted only if a future solver explicitly validates its holonomy
and publishes one deterministic evaluation tree. Until then cycles fail
atomically with a path diagnostic.

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

### Blender conformal preview and sync dependency

Blender conformal visualization is a separate reusable dependency, not hidden
inside the Hyperscope bridge or made conditional on HHHS. Its stable boundary
is a backend-neutral conformal/GA expression IR plus an explicit LOD contract.
A Blender-side compiler may lower that versioned IR into managed Geometry Node
groups, while Quilting/Hyperscape evaluate the same operations through their
runtime renderer. The compiler owns generated-node identity, upgrades, and
capability diagnostics; authored parameters and stable scene identities remain
the synchronized values rather than generated meshes or transient node IDs.

The Blender preview must share fixtures for transforms, conservative patch
bounds, projected error, shared-edge grading, and animation time with the
runtime LOD implementation. Geometry Nodes need not reproduce the renderer's
atlas or submission strategy, but an advertised LOD-aware result must satisfy
the same visible error and crack-free edge invariants. Unsupported operations
fall back visibly to bounds, control cages, or a baked preview instead of
quietly claiming conformal/LOD parity.

Before the low-latency sync API is frozen, inspect the then-current HHHS work
used by Tutti and Walkie Songie. In particular, measure its worker-owned state,
batched projection, event notification, transaction, persistence, reset, and
recovery paths against Blender's main-thread constraints and interactive edit
cadence. Reuse generic mechanisms where the evidence supports them and improve
HHHS upstream where the need is shared. The versioned operation protocol must
still pass a dependency-free one-way Blender/Hyperscope fixture first, and
camera, selection, focus, cue, and time presence remain ephemeral even when an
HHHS carrier is active.

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
the staged WebGPU backend already performs same-device visible-instance
compaction and indexed-indirect submission for both prepared adaptive patches
and source-indexed resident roots. Promoting that path to the browser default
still requires live image/cadence parity; it does not require a new CPU
visibility loop.

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
Composed scenes upload immutable face-to-node ownership once and refresh a
compact node transform table. The LOD vertex pass selects Möbius and Euclidean
state per face, so one coherent two-pass classification and one staged
readback cover the scene; the animated-primary prefix uses the same contract
without touching static authored assets. This table is backend-neutral input,
not a WebGL transform-feedback abstraction.

Current-view adaptive selection also consumes a backend-neutral one-byte risk
ranking. The classifier derives it from within-patch projected metric variation
and normalized pole proximity, then carries it in the high byte of the existing
packed word. Resident topology remains entirely in the low 24 bits, so priority
changes do not become topology deltas or expand readback traffic. Shadow mode
admits the hint only after exact worker parity; Rust authority admits it only
after pose-continuity acceptance. The pure-JavaScript implementation remains a
rollback oracle with its historical stable ranking.

The staged WebGPU backend replaces transform feedback with compute preparation,
reconciles LOD in storage, compacts visible instances, and emits indexed
indirect draw arguments. Both the adaptive replacement layer and the
source-indexed resident-root layer consume those retained device buffers. CPU
readback is optional conformance/telemetry rather than a frame dependency.
Selection highlighting is also a retained post-style indirect pass: the stable
source-face ID survives preparation and dyadic restriction, and the fragment
shader discards nonselected patches without a pick-texture allocation or
readback. Its diagnostic parity oracle rerenders the incumbent pick-texture
overlay into the WebGL evidence target, allowing complete selected-face image
comparison without changing either visible backend. The incumbent PBR path
now composes the same post-style highlight before capture as well. Focus
postprocessing remains a separate capability gap. Browser-default promotion
remains gated on live WebGL2/WebGPU image and interaction parity.

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
browser actually measured. The current staged presentation artifact is recorded
in the
[2026-08-27 hacker-night release evidence](benchmarks/2026-08-27-hacker-night-release.md).

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
