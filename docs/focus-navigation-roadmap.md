# Focus, selection, and navigation roadmap

The release-oriented ownership and migration contract is maintained in
[`hacker-night-release-architecture.md`](hacker-night-release-architecture.md).
This document remains the detailed behavior oracle for focus and navigation.

Status: the browser prototype is functional, the render contract is GPU-side,
and `hyperscape` now owns `FocusNavigation`, quaternion `CameraRig`, named
navigation policies, sequence/timestamp-ordered `NavigationAction`s, and
deterministic camera/focus transitions. `SurfaceWalkRuntime` is the single Rust
owner of surface topology, physical side, metric locomotion, animated contact
response, view following, scale/height/near policy, recovery/detach, and the
surface re-anchor glide. That glide matches the browser oracle's independent
forward/up spherical smoothing and Gram-Schmidt reconstruction rather than
using a generic quaternion camera interpolation. The prototype remains the
live behavior oracle while `walkimpl=shadow` measures the composed aggregate
before authority is removed from `hyperscope.html`.

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

`hyperscape::FocusNavigation` owns the sphere, optional stable-UUID entity anchor,
transition, focus/inversion enablement, constrained/free translation, and
radius policy. `NavigationController` and the ECS plugin consume semantic
actions such as:

- `Select { entity, source_bound }` and `DetachSelection`;
- `TranslateFocus`, `ScaleFocus`, `SetFocalShell`, and `SetAngularAperture`;
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
3. **WASM shadow bridge — active.** `HyperscopeNavigation` exposes the shared
   controller without a Bevy `App`. Add `navshadow=1` to a Hyperscope URL to
   mirror SpaceMouse camera actions and surface re-anchor glides, then inspect
   `globalThis.__hyperscopeNavigationShadow`. The surface observer separately
   reports samples, drift frames, maximum error, and the last Rust/browser
   pose pair. `HyperscopeAppShadow` now admits the same action set through the
   application reducer and an offline generated-WASM smoke compares both Rust
   boundaries through ordinary frames, focus/inversion edits, and animated
   surface re-anchor/retarget/cancel. Drift is recorded without changing the
   rendered camera; removing `navshadow=1` is the immediate rollback. Remove
   duplicate JavaScript authority only after representative all-mode,
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
   generated boundary exposes a typed composed result. Cutover still waits for
   clean target-browser traces across that clock and the now-atomic
   reflection/chart transport, plus animation pose sampling and
   Float32-sensitive edge crossings;
   `walkimpl=rust` therefore resolves back to shadow for now.
4. **Selection bridge.** Map stable glTF node identities to Hyperscape
   entities, send pick results as semantic selection actions, tick sphere
   transitions in Rust, and extract one compact focus packet per view.
5. **Interaction layer.** Add ray/shape queries, hover/active/selected states,
   focus-aware interaction range, and explicit visualization policies. The
   selection tint remains presentation; selection identity belongs to ECS.
6. **Persistence and replay — native oracle complete.** Replay version 0.4
   serializes stable entity references, detached spheres, camera rig state, and
   every current high-level navigation action. Checked navigation and
   orchestration fixtures cover transition cadence, asset effects,
   presence/authored lanes, and atomic rejection. Browser input capture and
   networking must carry semantic actions or authoritative state deltas, never
   raw HID reports.
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
