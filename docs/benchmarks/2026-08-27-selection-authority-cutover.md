# Rust selected-focus authority cutover — 2026-08-27

Mapped selected-object focus now defaults to `selectionimpl=rust`. Picking
remains a browser platform adapter, but the AppStore reducer owns the single
scoped `(asset, entity)` anchor and eased sphere transition. The browser no
longer starts a competing focus transition in Rust mode.

The live Chrome DevTools gate used the ordinary horse load. Selection followed
by inversion produced 68 authoritative renderer writes with zero selection
transition mismatches, renderer-packet mismatches, authority errors, frame
errors, or diagnostic mismatches. The explicit `selectionimpl=js` rollback
produced 39 incumbent transition frames with zero mismatches; its maximum
numeric error against the Rust observer was `9.33e-15`.

A fresh bare-URL load then omitted `selectionimpl` during canonicalization and
still produced 26 Rust renderer-authority writes with zero transition,
renderer, authority, frame, or diagnostic mismatches. This proves the default
rather than merely the explicit opt-in.

The cutover also passed all 66 browser-independent JavaScript tests, all 43
`hyperscope-app` tests, the WASM target check, and generated-WASM route/default
parity. Unmapped assets retain the incumbent local fallback; free/manual focus
editing remains browser-owned and independent of this selected-focus authority
boundary.
