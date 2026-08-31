# Rust animation-clip authority cutover — 2026-08-31

## Outcome

`animclipimpl=rust` is the canonical route default. `animclipimpl=js` and
`animclipimpl=shadow` remain explicit, linkable rollback and measurement lanes.
This cut does not change `animclockimpl`; the high-rate animation clock remains
JavaScript-owned until its separate background-cadence and long-clip gates pass.

Those independent gates passed later the same day and are recorded in
`2026-08-31-rust-animation-clock-cutover.md`; the current clock default is Rust.

Rust owns the installed catalog, active and pending clip identity, job
allocation, cancellation, stale-result rejection, and the Leptos selector.
JavaScript remains the platform effect host for worker evaluation and GPU
resource installation, and records renderer residency only after that work is
coherent.

## Defect found by the gate

Replacing the horse with a new model reused clip index zero. The reducer and
renderer both committed the new `Action` clip, but the Leptos keyed list had
used only the renderer-local numeric index and retained the horse label. The
key now includes both index and projected label, so a replacement catalog
cannot reuse stale option content.

The shadow diagnostics also retained the canceled job ID after a rapid return
to the incumbent. Cancellation-only requests now clear that witness when the
returned cancellation matches it.

## Live evidence

`scripts/audit-rust-animation-clip.mjs` streams the 5.3 MB
`still_life_based_on_heathers_artwork.glb` fixture through a short-lived
loopback endpoint, then exercises the real drag/drop, decode, upload, worker,
selector, URL, and completion paths in an isolated Chrome target. The fixture
contains `Action` and `EmptyAction` clips.

The explicit Rust lane passed with:

- exact replacement catalog and visible Leptos authority;
- clip 0 → 1 → 0 with two completions and two authority writes;
- rapid 1 → 0 without yielding, producing one cancellation and one ordered
  incumbent repair;
- zero mismatches and zero errors throughout;
- a direct obsolete completion returning `ignored_stale` while clip 0 remained
  active and no job remained pending;
- canonical URL selection advancing to `anim=1` and returning to `anim=0`.

The shadow lane produced the same resident/active states and cancellation
repair with zero mismatches or errors and no authority writes. The explicit
JavaScript lane switched and restored the renderer while all Rust clip
dispatch/comparison/authority counters remained zero. The final no-flag run
against the generated package selected Rust, exposed the Leptos control,
repeated all four clip operations with zero mismatches/errors, and kept
`animclipimpl` out of the canonical URL.

Native verification passed 150 `hyperscope-app` tests and 41
`hyperscope-web` tests, strict no-dependency library Clippy for both crates,
the `quilting-wasm` test build for `wasm32-unknown-unknown` with
`leptos-ui,webgpu-backend`, the route/default smoke, and the focused
animation-clip boundary smoke.
