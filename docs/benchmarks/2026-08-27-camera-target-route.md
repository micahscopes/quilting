# Camera target-policy route gate — 2026-08-27

## Invariant

The camera's numeric pose is not enough to reconstruct navigation semantics.
`px/py/pz` may be a finite semantic aim target or only the control pivot used
to encode a free sight tangent. Canonical routes now carry `aim=1` for the
finite case and omit `aim=0` as the free-camera default.

Selection detach no longer disables this independent policy. Model installs
call selection cleanup, so coupling the two previously erased a valid target
during ordinary startup even after the Rust application had admitted it.

## Evidence

- `hyperscope-app` has 77 canonical control specifications; `aim` is a validated
  toggle ordered directly after the camera pivot.
- 70 all-feature application tests passed, including explicit default,
  non-default, invalid-value, and canonicalization cases.
- The generated-WASM route smoke passed with all 77 Rust/browser defaults and
  serialization positions covered.
- Live Chrome loaded `aim=1&px=2&py=1&pz=-1`. Rust was the startup and write
  authority; the canonical URL retained `aim=1`, and the application camera
  exposed semantic target `[2, 1, -1]` with zero mismatches or fallback writes.
- Live Chrome then loaded the same pose with `aim=0`. Rust omitted the default
  from the canonical URL and exposed no semantic target, again with zero
  mismatches or fallback writes.

The temporary acceptance tab was closed without selecting or modifying the
user's chess tab.
