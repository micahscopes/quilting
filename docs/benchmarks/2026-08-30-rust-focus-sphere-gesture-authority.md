# Rust focus-sphere gesture authority — 2026-08-30

## Outcome

Continuous `F`+wheel and SpaceMouse inversion-layer sphere edits now enter one
application-owned operation under `selectionimpl=rust`.

`AppStore::dispatch_focus_sphere_edit` samples the current focus state and
allocates the navigation sequence under the same lock. It resolves the device-
independent request as either:

- an absolute detached `SetFreeFocusSphere`, which clears any old anchor; or
- an anchored `ScaleFocusLog`, which preserves selected identity, fixes the
  center to the selected source bound, and edits only its bounded margin.

The normal navigation transaction then transports the camera, any in-flight
camera transition, reflection chart, and surface follower together. A pole or
invalid anchored move consumes the semantic input without partially changing
those states. The browser projects the committed Rust camera/focus snapshot
and applies the exact application focus packet to the resident renderer.

The high-rate success path does not flush or serialize the full application
read model. DOM/FRP publication remains on the existing throttled application
cadence.

## Rollback boundary

- `selectionimpl=rust`: mapped selections and detached spheres use the Rust
  operation.
- `selectionimpl=js|shadow`: the incumbent browser geometry remains intact.
- An unmapped local selection deliberately stays on the browser fallback so it
  cannot be mistaken for a durable or session-mapped Rust identity.
- The four direct legacy Möbius sliders are still a named browser-owned seam;
  moving their view into a Rust/Leptos control is a later cut.

## Verification

- Native `hyperscope-app` test covers detached replacement, anchored scaling,
  anchor preservation, rejection without sequence/state mutation, and an
  explicit anchored-to-free replacement.
- Generated-WASM compilation exposes `editFocusSphere` as a thin delegation to
  `AppStore`.
- Route/source oracle proves both browser gestures reach that boundary and
  project committed state rather than redispatching browser intent.
- Inline Hyperscope module syntax check passes.
- No browser tab or user-run server was touched.

## Architectural audit

The nearby navigation split is now:

- **Rust semantic authority:** quaternion camera integration, perspective
  lens, selected and free focus geometry, inversion transport, spheroidal
  field, navigation settings, animation clock, presentation cues, and typed
  canonical URL admission.
- **Thin browser adapters:** pointer/HID acquisition, SpaceMouse report
  shaping and foreground policy, renderer/canvas handles, URL/history writes,
  and projection into retained compatibility signals.
- **Remaining migration debt:** direct sphere sliders, broader gesture
  arbitration, default promotion after live soak, and replacing compatibility
  camera/URL projections with Rust/Leptos consumers.

This is independent of the WebGPU implementation: WebGL2 and WebGPU consume
the same committed focus/navigation semantics and `RenderFrame` contract.
