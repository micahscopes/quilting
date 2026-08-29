# Animation clip cutover model — 2026-08-29

This is the live-browser acceptance gate for moving installed animation-clip
selection from its JavaScript default to Rust authority. It does not claim
that the live gate has passed yet.

## Frozen implementation boundary

- `hyperscope-app` owns the validated installed catalog, active clip, pending
  job, cancellation, and matching/stale/failed completion decisions.
- `AppFrameSnapshot` maps the application clock into the active installed clip
  and retains exact request, asset, and clip identity without cloning names per
  frame.
- `writeInstalledAnimationSample` writes only
  `[playing, clip_index, wrapped_clip_time, speed]`. Measured clip lanes do not
  send a browser-owned clip range back into Rust each frame.
- The browser owns worker calls and GPU resource installation. Its scalar
  `rendererAnimationClipIndex` witness changes only after the required worker
  and GPU state is coherent. An ordinary model switch refreshes skin/morph
  textures, rest instances, the LOD compute model, and same-context residency.
  A packed presentation switch instead sends its exact retained primary
  vertex/face witness to Rust; Rust checks it before replacing the evaluator,
  and the adapter retains every composed buffer.
- Explicit `animclipimpl=rust` mounts the Leptos selector. It dispatches through
  `AppStore` and sends only committed selection/cancellation effects to the
  platform adapter. `animclipimpl=js` remains the default rollback.

The implementation checkpoints are `182e76e`, `b687968`, `c28f929`,
`6c04c92`, `bc76e63`, and `42d0f39`.

## Deterministic evidence already green

- 97 `hyperscope-app --all-features` tests, including pending, cancellation,
  stale, failure, success, and reverse-wrapped installed-frame sampling.
- 28 native `hyperscope-web --all-features` library/relay tests.
- strict native application and native/WASM web-control Clippy.
- `quilting-wasm` with `leptos-ui,webgpu-backend` on
  `wasm32-unknown-unknown`.
- generated application and route smokes.
- all three replay `0.22` fingerprints.

Presentation animation now has the same exact gate. Rust joins the authored
presentation asset to the installed request/session asset through an ephemeral
residency event, resolves the cue's clip name and relative time, and emits the
ordinary clip selection effect. The browser binds before multi-asset packing
and executes only that committed effect. Later clip changes preserve the
packed scene: `set_active_animation_preserving_topology` rejects a stale
vertex/primary-face witness before mutating the evaluator, while a successful
switch retains skin/morph sources, composed instances, face domains, worker
LOD residency, and same-context residency.

The generated WASM smoke explicitly performs this race:

1. install a two-clip primary scene;
2. select clip 1 and complete it;
3. request clip 0;
4. return to incumbent clip 1 and observe the exact cancellation effect;
5. deliver the obsolete clip-0 completion and require `ignored_stale`.

## Live fixture

Drag `/home/micah/Downloads/bird_animations_alex.glb` into Hyperscope. Its GLB
JSON declares ten clips:

```text
fly1_bird
fly2_bird
fly3_bird
fly_endA_bird
fly_endB_bird
fly_startA_bird
fly_startB_bird
idleA1_bird
idleA2_bird
idleB1
```

`RobotExpressive.glb` in the downloaded three.js fixtures is a secondary
14-clip oracle if a second rig is needed. These facts were read directly from
the GLB JSON chunks; neither asset was copied or modified.

## Shadow gate

Open an otherwise ordinary route with:

```text
animclipimpl=shadow&animclockimpl=shadow
```

After dropping the bird:

1. let `fly1_bird` animate for several loops;
2. switch to `fly2_bird`, `idleA1_bird`, and `fly3_bird` normally;
3. rapidly request `fly_endA_bird`, then `fly_endB_bird`, then return to the
   currently resident clip before the intermediate switch settles;
4. pause, scrub near both ends, resume, and repeat with `animspeed=-1`;
5. leave it running across several background/foreground transitions;
6. replace it with another GLB while a switch is pending, then reload the bird.

Acceptance requires:

- `__hyperscopeAnimationClipDiagnostics.state === "parity"`;
- `rendererResidentClip === rustActiveClip` and `rustPendingClip === null` after
  every settled operation;
- `mismatches === 0`, `errors === 0`, and no repeated sample-mismatch key;
- at least one selection effect, cancellation effect, completion, no-op, and
  ordered repair observed;
- `__hyperscopeAnimationClockDiagnostics.mismatches === 0` and finite maximum
  error;
- no `animation_clip_*` entry in
  `__hyperscopeAppShadowDiagnostics.mismatches`;
- no wrong-clip pose, rest-pose flash, stale texture/skin state, LOD blip, seam
  discontinuity, or stalled animation after settling.
- for presentation cues that change clips after packing,
  `__hyperscopePresentation.compositionPreservingClipSwitches` advances while
  resident asset count, packed face count, and LOD topology domains remain
  unchanged.

## Rust-authority gate

Repeat the same sequence with:

```text
animclipimpl=rust&animclockimpl=rust
```

Additionally require:

- the Leptos selector is visible and
  `animationClipControlAuthority === "hyperscope-web"`;
- the hidden HTML selector remains only a renderer/platform mirror;
- `authorityWrites > 0` for both clip and clock lanes;
- clock `fallbackWrites === 0` outside the intentionally blocked resource
  installation interval;
- canonical URL state preserves the explicit Rust gates and selected clip;
- returning to the incumbent during a pending switch visibly settles on the
  incumbent, not the canceled worker result.

Only after both gates pass on the same generated package may the default move
from `js` to `rust`. The JavaScript and shadow paths remain until a subsequent
soak demonstrates the same behavior under presentation cue changes and primary
scene replacement.
