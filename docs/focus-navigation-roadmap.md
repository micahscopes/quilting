# Focus, selection, and navigation roadmap

The release-oriented ownership and migration contract is maintained in
[`hacker-night-release-architecture.md`](hacker-night-release-architecture.md).
This document remains the detailed behavior oracle for focus and navigation.

Status: the browser prototype is functional, the render contract is GPU-side,
and `hyperscape` now owns `FocusNavigation`, quaternion `CameraRig`, named
navigation policies, sequence/timestamp-ordered `NavigationAction`s, and
deterministic camera/focus transitions. The backend-neutral interaction reducer
now owns sequence/timestamp-ordered hover, press, release, and cancel semantics;
selection remains the existing `FocusNavigation::anchor` rather than a second
interaction-state copy. `SurfaceWalkRuntime` is the single Rust
owner of surface topology, physical side, metric locomotion, animated contact
response, view following, scale/height/near policy, recovery/detach, and the
surface re-anchor glide. That glide matches the browser oracle's independent
forward/up spherical smoothing and Gram-Schmidt reconstruction rather than
using a generic quaternion camera interpolation. The default `walkimpl=js`
remains the rollback oracle, `walkimpl=shadow` measures the composed aggregate,
and `walkimpl=rust` now makes the Rust contact frame, camera, anchor transition,
and topology packet authoritative while retaining the legacy walker as a
same-input diagnostic shadow.

## One sphere, several meanings

Selection focus and spherical inversion share one positive ordinary-space
`FocusSphere`. They must not acquire parallel center/radius state.

- Selecting an object attaches an optional `FocusAnchor` and smoothly fits the
  sphere to the object's source bound with a default 10% margin.
- Detaching selection removes ownership only. The sphere, focus effect, and
  inversion state persist and the sphere becomes freely editable.
- An anchored sphere keeps the object's center and edits only a 1x–4x margin.
  A detached sphere accepts translation and scale edits.
- Focus and inversion can be enabled independently, but both consume the exact
  same sphere.
- Center interpolation is linear with smootherstep timing; positive radius is
  logarithmic so transitions remain smooth across very different scales.

The sphere is classified before the subject/view Möbius map. Möbius maps send
spheres to spheres or planes, but source-space classification is both cheaper
and exact for the sphere used by inversion. The posed pre-Möbius surface point
is carried to PBR, which writes the compactified radial coordinate
`u = 2/pi atan(distance/radius)` to the spare B channel of the existing weight
MRT. This is the normalized round-S3 geodesic radius: the center is 0, the
sphere is 1/2, and infinity is 1. Sphere reflection sends `u` to `1-u` exactly.

Selection focus is therefore spheroidal depth of field rather than a binary
inside/outside mask. A focal-shell coordinate chooses what is sharp and an
angular aperture controls falloff on both sides. The current circle-of-confusion
response is `c/(1+c)`, where `c = abs(u-focus)/aperture`; one aperture away is
50% defocused and the response approaches full blur without a hard cutoff.
This dense field bypasses JFA seed propagation and reuses the retained
variable-blur passes.

## Prototype controls

| Input | Anchored selection | Detached sphere |
| --- | --- | --- |
| Primary click | Select object, tint it, enable spheroidal focus, animate fit without moving the camera | Click empty to detach without resetting |
| SpaceMouse right double tap | Select at view center; an empty hit reframes the retained selection | Select at view center |
| SpaceMouse left double tap | Fit shared sphere and toggle inversion | Toggle inversion around retained sphere |
| SpaceMouse left hold | Center locked; twist edits bounded margin | Translate center; twist edits radius |
| SpaceMouse right hold | Push/pull focal shell, lift aperture, twist blur radius | Edit the retained focus field |
| `F` + wheel | Edit bounded margin | Edit radius |
| Shift + `F` + wheel | Edit angular aperture | Same |
| Ctrl/Meta + `F` + wheel | Edit blur radius | Same |
| Escape / empty click | Detach selection and retain sphere | No reset |
| Focus/navigation transition slider | Set fit/reframe/surface re-anchor duration from 0.05–5.00 seconds in 0.01-second increments | Same |

Mouse selection uses a four-pixel drag threshold so an orbit gesture cannot
become a pick on release. Device mappings are prototype adapters, not domain
state: keyboard, mouse, SpaceMouse, touch, gamepad, XR, and game code should
all emit the same semantic actions.

Changing an active inversion sphere transports the manual camera through
`F_new o inverse(F_old)`. The eye follows the point map exactly. An unanchored
fly/orbit camera has a sight tangent rather than an arbitrary finite target, so
its forward/up frame and control distance follow the local conformal
differential. When an object pivot, authored target, or deliberate reframe
establishes a semantic target, the two-point transport can instead map eye and
target exactly and parallel-transport roll between the resulting sight
directions. FOV and lens parameters remain unchanged. Selection itself never
pans the camera; object-pivot navigation and explicit reframe remain deliberate
camera actions.

An attached surface follower now crosses the same chart boundary as one atomic
Rust operation. Its stable source face/barycentric address and eye height do
not change; the filtered output point, normal, tangent, and prior posed-contact
sample are transported by the exact conformal differential; relative pitch is
retained; and physical-side parity flips exactly once. A pole rejects the
camera, topology side, follower, and velocity history together. The current
browser oracle cancels an active surface re-anchor glide when its output chart
changes, so the Rust aggregate deliberately does the same until rebase
semantics are introduced as an explicit behavior change. The first sample in
the new chart rebases animation velocity instead of differencing f64 semantic
sphere transport against f32-packed renderer coefficients; the following
sample resumes ordinary pose-time velocity measurement. This trades one
deliberate zero-velocity sample at a rare chart edit for freedom from a
frame-delta-amplified false impulse.

## Target Rust ownership

The intended flow is:

```text
device adapter -> semantic interaction actions -> FocusNavigation + CameraRig
              -> Hyperscope extraction -> WebGL2 or WebGPU backend
```

`hyperscape::FocusNavigation` owns the sphere, optional asset-scoped stable-UUID
entity anchor, transition, focus/inversion enablement, constrained/free
translation, and radius policy. `InteractionController` consumes renderer-
independent `SetHover`, `PressPrimary`, `ReleasePrimary`, and `CancelPrimary`
actions. A successful release emits the existing `AnchorFocus` navigation
action; it does not mutate a second selected field. `NavigationController` and
the ECS plugin then consume semantic actions such as:

- `AnchorFocus` and `DetachFocus`;
- `TranslateFocus`, `ScaleFocusLog`, and `SetFocusField`;
- `SetFocusEnabled` and `ToggleInversion`;
- `ReframeSelection`; and
- camera-local translate/rotate actions independent of any device axes.

Actions should be timestamped against `Time<Virtual>` and consumed in a fixed
system set before constraints and extraction. This makes input recordable,
replayable, networkable, and testable without a browser or HID device. Rust
quaternions should own camera orientation; Euler values remain UI projections
only. The browser should eventually retain no authoritative camera, selection,
or sphere state.

## Migration stages

1. **State core — complete.** Keep the tested `FocusNavigation` resource and
   GPU focus-field contract backend-neutral.
2. **Camera/action core — complete.** Rust owns the quaternion camera,
   inversion transport, four navigation policies, explicit easing, virtual
   time, and frame-rate-independent action replay.
3. **WASM authority bridge — active.** `HyperscopeNavigation` exposes the shared
   controller without a Bevy `App`, while `HyperscopeAppShadow` routes the same
   actions through application sequence and virtual-time authority. Use
   `navimpl=js|shadow|rust` to retain the browser rollback, compare both paths,
   or apply the Rust camera respectively; inspect
   `globalThis.__hyperscopeAppShadowDiagnostics`. The surface observer separately
   reports samples, drift frames, maximum error, and the last Rust/browser
   pose pair. `HyperscopeAppShadow` now admits the same action set through the
   application reducer and an offline generated-WASM smoke compares both Rust
   boundaries through ordinary frames, focus/inversion edits, and animated
   surface re-anchor/retarget/cancel. Drift is recorded without changing the
   rendered camera in shadow mode; `navimpl=js` is the immediate rollback. Remove
   remaining duplicate JavaScript authority only after representative all-mode,
   all-scale parity runs are clean.
   The normalized SpaceMouse camera boundary is now frozen in Rust as
   `SpaceMouseCameraInput -> NavigationFrame`. The browser still owns WebHID
   permission/acquisition, report decoding, dead-zone and quadratic response,
   stale decay, smoothing, buttons/modifier layers, and one screen-relative
   speed registered and held for each translation gesture. Rust validates the
   normalized sample and owns Blender/Hyperscope axis policy, swap/inversion
   preferences, delta/gain integration, object dolly, and horizon policy. The
   queue-only application adapter emits `SetPreset` followed by `ApplyFrame`
   through the shared sequence authority and never ticks implicitly. An
   offline generated-WASM oracle covers all four presets, both swap states, all
   64 pan/rotation mask pairs, 14 axis vectors (7,168 combinations), 648
   response-policy combinations, four `AppStore` target/basis initial states,
   and a deterministic 120-frame trace. Surface walking, focus/inversion
   modifiers, authored-camera
   arbitration, and live browser cutover remain outside this slice; rotation
   cadence invariance is not inferred from same-trace parity.
   SpaceMouse and pointer turntable camera integration now use the retained
   Rust packet under `navimpl=rust`; shadow mode has exact live Chrome evidence.
   Selected-object recovery framing is a semantic `ReframeSelection` action:
   Rust projects the selected source pivot/radius into the active chart, fits
   the narrower viewport axis using the live perspective lens, and follows the
   established target-orbit/log-distance path. Replay 0.15, generated WASM,
   invalid-geometry and reflection-pole rollback oracles, 40 clean live shadow
   frames (maximum error `9.1e-12`), and 39 no-fallback Rust authority writes
   gate that route. The browser's old implicit 60-degree framing assumption was
   removed; both paths now consume the actual vertical FOV and an explicit 15%
   camera framing margin.
   Focus/inversion actions and their reflection transport are now one staged
   Rust transaction: a camera, transition, or surface-follower pole consumes
   the ordered input exactly once while restoring the preceding camera, focus
   sphere/mode, active reflection, and walk state. The app boundary and
   generated-WASM smoke exercise the exact camera-eye pole oracle. Browser URL
   restoration likewise batches center, radius, and transform mode as one
   request, so it cannot reject a safe linked sphere against transient default
   geometry. Effective spheroidal focus now has the same one-queue boundary:
   Rust `focus_enabled` means fuzzy post-processing is enabled in mode 3,
   while legacy blur modes and the retained shared sphere remain separate
   renderer/interaction state. Focus coordinate and angular aperture are
   coalesced with enablement into one browser signal burst and one atomic
   `SetFocusField` action per navigation/`AppStore` parity controller. The
   default Rust path retains the prior renderer packet and URL until AppStore
   integrates that action; the committed projection then updates WebGL,
   controls, and the Leptos read model. An invalid field rolls enablement,
   coordinate, and aperture back together. `selectionimpl=shadow` measures
   without authoritative writes and `selectionimpl=js` bypasses the boundary.
   Initial application synchronization includes the complete lens,
   aim, focus, inversion, and sphere packet, so a focus-only deep link no
   longer starts from Rust defaults. Presentation snapshot application is
   suppressed at the signal boundary and cannot echo semantic focus actions
   back into Rust. The selection shadow now exports validated authored glTF
   node IDs, joins pickable primary/secondary nodes to their explicit manifest
   asset scope, and dispatches mapped picks/detaches through the application
   queue. Ordinary URL, IndexedDB, local-fallback, and drop loads remain
   deliberately unmapped. The checked Blender release asset now carries five
   persistent stable IDs, four of which are joined to pickable mesh nodes; its
   reproducible exporter rejects a fixture which loses that identity. Live
   selection cutover still requires renderer consumption of the Rust packet.
   Rust retains the selected source bound and clicked pivot
   beside its asset-scoped stable identity, and the application
   snapshot derives output-chart pivot/radius without destroying selection at
   a reflection pole. The same application queue now accepts complete,
   validated perspective-lens edits and explicit semantic-target enablement.
   Lens changes persist through active camera and surface-anchor transitions;
   point-target mode is rejected transactionally while surface walking, whose
   camera is deliberately target-free. Both WASM facades round-trip the same
   FOV/near/far and target-presence packet, and the browser's 75-degree default
   survives Rust synchronization without using inversion as an aim proxy.
   A successful-action sequence fence makes presentation target preemption
   integration-time exact: future and rejected manual aim edits cannot clear a
   deferred authored target. Mid-glide enablement preserves the existing pose
   trajectory and solves a continuous finite-target path at the current clock.
   The semantic surface-walk response and topology are now composed in
   `SurfaceWalkRuntime`: scene-relative pace and avatar scale, tangent
   velocity, current animated material-point velocity, contact/normal
   smoothing, relative-pitch retention, tangent pull, eye height, scale-aware
   near plane, surface advancement, recovery/detach, and re-anchor transition
   commit atomically. `walkimpl=shadow` feeds right-click attachment and each
   semantic walking frame to the candidate while JavaScript remains live
   authority. The production adapter shares the same posed QB geometry and
   adjacency as the incumbent walker and exposes bounded topology/camera drift
   diagnostics. The oracle and candidate now advance re-anchoring from one
   explicit virtual frame delta with identical endpoint snapping, and the
   generated boundary exposes a typed composed result including its retained
   filtered contact frame. Initialized Chrome traces cover the shared clock,
   successful and pole-rejected reflection transport, animation pose
   sampling, pause/scrub, active locomotion, and transition completion. The
   Float32 adapter gate crosses a pick-like near-edge address through all 18
   identity/reflection and cyclic source/neighbor permutation cases.
   `walkimpl=rust` is therefore a real rollback-safe authority mode; the
   default stays `js` until it has seen broader interactive soak time.
4. **Selection bridge — mapped renderer authority available behind rollback.**
   Protocol, navigation, AppStore snapshots, replay 0.8, and the generated WASM
   facades now carry an explicit `(asset ID, entity ID)` pair. Pre-0.7
   unscoped replay anchors fail closed. Validated glTF bindings now cross the
   loader as a dense authored identity table. Presentation composition maps
   only pickable nodes, preserves source-node offsets, and admits asset scope
   only from an exact validated manifest fetch; its append is transactional.
   Mapped picks and clears traverse `AppStore::dispatch_navigation` in shadow.
   Authored assets retain their durable UUIDs; ready ordinary loads receive
   deterministic Rust-derived session node identities scoped by the load-lane
   asset ID. Those session pairs are runtime keys only and cannot enter authored
   commands or HHHS history. Stale, loading, or otherwise unresolved assets
   retain incumbent browser selection. The application clock now advances
   through an
   allocation-light boundary once per browser frame, snapshots only active
   parity windows, and compares mapped focus interpolation with both the
   incumbent browser state and the renderer's retained CPU packet. It performs
   no GPU readback. `selectionimpl=js` keeps the incumbent renderer path,
   `selectionimpl=shadow` enables the same AppStore oracle, and
   `selectionimpl=rust` identity-checks the selected `(asset, entity)` pair and
   transfers the complete Rust focus/selection packet directly into the
   resident renderer without serializing its sphere through JavaScript.
   Only genuinely unresolved assets remain on JavaScript authority. The
   selected-fit
   clock also retains an event fence across delayed RAF callbacks whose
   timestamp predates the pick, preventing the pre-event interval from being
   integrated twice. As of 2026-08-27, mapped selected-object focus defaults to
   `rust`; `selectionimpl=js` remains the explicit rollback and unresolved
   assets still fall back locally. Free/manual focus edits remain a separate
   browser-owned migration lane rather than a second selected-focus transition.
5. **Interaction layer — Rust semantic core complete; query adapters active.**
   `InteractionHit` carries one validated asset/entity identity, source bound,
   source pivot, displayed-chart distance, and optional face/barycentric detail.
   `InteractionState` retains only ephemeral hover/active state; its snapshot
   derives selected identity from `FocusNavigation::anchor`. Focus-radius-aware
   proximity reach is explicit; exact screen/XR ray hits remain selectable at
   any distance so a tiny detached focus sphere cannot reject visible geometry.
   Virtual-time ordering, cadence invariance, cross-entity release cancellation,
   and invalid-hit atomicity are covered natively. The ECS plugin
   routes a successful activation into the established `AnchorFocus` action in
   the same ordered interaction set. `AppState` integrates interaction before
   navigation on the same virtual frame, and `HyperscopeAppShadow` exposes
   validated entity-level or face/barycentric hover plus primary
   press/release/cancel and a read-only interaction snapshot. The retained
   WebGL2 pick now also evaluates the exact animated QB point in source and
   displayed charts on demand. A backend-neutral `InteractionTargetTable`
   atomically joins transient packed nodes to optional stable identity and
   source bounds. WebGL2 and WebGPU queries therefore supply the same compact
   packed-node/surface sample instead of independently rebuilding semantic
   identity. `selectionimpl=shadow` routes mouse and
   view-center picks through hover/press/release, compares the activated stable
   identity, and falls back to the incumbent direct observer without changing
   renderer authority. Replay 0.25 records exact interaction actions and state,
   including face/barycentric detail, and rejects those events under 0.24.
   Live shadow evidence, Rust-default promotion, broader hover cadence, shape
   queries, and visualization policy remain. Selection tint remains
   presentation; neither it nor a backend-local face index becomes selection
   authority.
6. **Persistence and replay — native oracle complete.** Replay version 0.25
   serializes asset-scoped stable entity references, selected source
   bounds/pivots, derived output-chart pivots/radii, detached spheres, camera
   rig state, the complete high-level navigation vocabulary, semantic
   interaction actions, and exact ephemeral interaction observations. The
   interaction golden proves JSON round-trip, surface identity retention,
   activation through navigation authority, cadence invariance, legacy
   rejection, and invalid-hit atomicity. Navigation and orchestration fixtures
   cover transition cadence, asset effects, presence/authored lanes,
   deterministic authored asset/entity materialization, and atomic rejection.
   Older supported schemas remain readable with their original semantics;
   specifically, 0.24 is not reinterpreted as containing pointer actions.
   Browser input capture and networking must carry semantic actions or
   authoritative state deltas, never raw HID reports.
7. **WebGPU backend.** Upload the extracted focus packet as frame/view data,
   classify the field in WGSL, and retain the same weight-channel meaning. GPU
   visibility compaction and indirect draws remain independent of interaction
   ownership.

## Blender / HHHS 0.4 live editing

The simplified Blender bridge should treat stable object identity and source
bounds as authored scene facts. Blender selection may emit `Select`, but it
must not own a second runtime focus sphere. Hyperscape remains authoritative
for the live sphere, camera, inversion, and interaction state.

The first live-sync slice should exchange versioned transactions for ordinary
node transforms, mesh/material revisions, conformal frame generators, camera
constraints, and stable IDs. A changed selected-object bound retargets the
existing anchored transition; deleting the object detaches the sphere in
place. Runtime-only focus edits should not dirty the Blender file unless the
user explicitly promotes them to authored data.

## Game-readiness constraints

- Input mappings must be remappable and scale-independent.
- Selection and focus transitions must be deterministic under fixed deltas.
- Picking must report stable entity identity, not transient draw-batch IDs.
- Animated bounds need a declared policy: conservative authored/rest bounds
  first, optional current-pose GPU bounds later.
- Focus/inversion state changes must never synchronously read GPU geometry.
- Transparent materials and focus blur retain authored PBR sidedness and alpha
  semantics; selection does not silently enable OIT or double-sided rendering.
- Rendering may interpolate presentation, but authoritative gameplay queries
  use the current Rust sphere and entity anchor.

This turns the prototype into a primary navigation and interaction mechanism
rather than a collection of UI shortcuts, while keeping the WebGL2 and future
WebGPU renderers as consumers of the same game-layer state.
