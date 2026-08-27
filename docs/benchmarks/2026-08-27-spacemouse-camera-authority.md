# SpaceMouse camera authority gate — 2026-08-27

## Scope

The browser continues to own WebHID permission, report decoding, stale-input
decay, response smoothing, focus-relative speed registration, and the
surface-walk modifier layer. `HyperscopeAppShadow::stepSpaceMouseCamera` owns
the ordinary camera layer after those platform concerns: axis mapping, preset
policy, semantic-target policy, quaternion integration, and the committed
camera packet.

`navimpl=js|shadow|rust` is explicit and linkable. JavaScript remains the
default during the cutover. The legacy `navshadow=1` spelling canonicalizes to
`navimpl=shadow` and is not part of the Rust route registry.

## Generated-WASM gate

`node scripts/smoke-hyperscope-app-shadow.mjs` passed against the release WASM
package. The oracle covered 7,168 mapping combinations, 648 response-policy
combinations, four camera/target states, and the retained 17-number output
packet. Invalid packet lengths and non-finite input changed neither the camera
nor the caller's packet.

## Live Chromium gate

Chrome DevTools MCP drove temporary horse routes while leaving the existing
chess route unchanged.

- `navimpl=shadow`: 13 synthetic six-axis frames, 13 comparisons, zero
  mismatches, zero errors, zero authority writes. Maximum component error was
  `1.5387264884481056e-7`, below the `2e-6` gate.
- `navimpl=rust`, Hyperscope: 9 steps, 9 authority writes, 9 post-apply
  comparisons, exact zero drift, zero fallback writes, and zero errors.
- The Rust route then exercised Object, Fly, and Drone. Each ended with the
  requested preset, no pending actions, and no reducer diagnostics. Object
  retained a semantic point target; Fly and Drone retained free-flight tangent
  semantics. Across 29 cumulative frames there were 29 authority writes, zero
  mismatches, zero fallbacks, and zero errors.

The two temporary routes were closed after the gate. The user-owned chess tab
remained at its original URL.
