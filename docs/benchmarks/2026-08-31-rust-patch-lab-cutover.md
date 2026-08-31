# Rust Patch Lab authority cutover — 2026-08-31

## Outcome

`patchlabimpl=rust` is the canonical route default. Explicit `js` and `shadow`
routes remain the rollback and parity lanes. Rust owns the Patch Lab controls,
normalization, animation phase, geometry/LOD job generations, coalescing,
stale-completion policy, status read model, and Leptos view. JavaScript remains
only the platform effect host for workers and renderer buffer installation.

Every platform completion carries the exact Rust geometry and LOD job IDs.
The host checks those IDs, the installed geometry generation, and Rust's dirty
fence before changing renderer batches. A superseded result completes the
reducer fence so its queued replacement can start, but never flashes through
the renderer or status model.

## Defects found by the gate

- Superseded worker results could be installed for one frame before the
  coalesced newest LOD request ran. Rust now withholds a dirty completion from
  the read model and the platform host withholds it from renderer residency.
- Completion summaries sampled the renderer immediately before installing the
  corresponding batches, making `renderedTriangles` lag by one job. Current
  jobs now install inside the exact job fence before Rust records the summary.
- An inactive render-settings commit could project default Patch Lab controls
  over still-pending URL values. Inactive, effect-free commits no longer touch
  the route bootstrap projection.
- The 2:1/4:1 control admitted a new Rust policy while the renderer required a
  reload. Atlas exponent and grading now share one debounced replacement. The
  renderer accepts the policy switch only after atlas upload has retired every
  old batch, then the typed LOD job rebuilds residency under the new policy.

## Live evidence

`scripts/audit-rust-patch-lab.mjs` uses an isolated Chrome target and leaves the
user's page untouched. The Rust lane proved:

- deep-link restoration of manual `2 / 64 / 64` requests;
- exact 2:1 reconciliation to `32 / 64 / 64` and live 4:1 reconciliation to
  `16 / 64 / 64`;
- a debounced live atlas change from exponent 7 to 6;
- Rust-owned animated field sampling with a superseded install fenced out;
- triangle-to-plane geometry replacement and a grid change from 8 to 10;
- matching Rust/renderer triangle counts, zero shared-edge mismatches, and
  zero reducer, effect-host, view, or console errors.

The explicit shadow lane repeated browser controls with zero projection
mismatches, and the explicit JavaScript lane rendered normally with every Rust
Patch Lab effect counter at zero. A 4:1 startup rerun preserved the complete
`2 / 64 / 64` URL request and rendered 2,304 triangles. The final release gate
also runs the Rust audit with no `patchlabimpl` query parameter to prove the
compact route selects the new canonical default.
