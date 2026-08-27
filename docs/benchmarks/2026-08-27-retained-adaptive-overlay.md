# Retained adaptive overlay measurement

Date: 2026-08-27

This checkpoint first measured the exact sparse replacement layer introduced by
`AdaptiveRenderOverlay`, then connected the same representation to an explicit
opt-in WebGL2 publication gate. The default remains the complete adaptive
frontier while the retained path is benchmarked. When enabled for an
order-insensitive scene, the live renderer publishes baseline roots with an
exact suppression mask plus independently retained overlay batches.

The measurement is fail-closed. Rust reports only a fully published adaptive
epoch whose reconciliation generation and batch-layout revision match the live
GPU grouping. Staged candidates, fallbacks, and rolled-back publications are
rejected. Calling the diagnostic does not schedule frame work or mutate GPU
resources.

The second shadow revision also constructs retained baseline membership and
compares baseline-minus-suppression plus overlay against the live complete
grouping in two ways:

- members sorted by stable `(source face, dyadic leaf)` identity, which proves
  semantic/topological equivalence;
- the literal baseline-draw-then-overlay-draw sequence, which exposes ordering
  changes that can matter to alpha compositing.

## Pathological inverted chess view

Source view:

```text
?glb=classic_chessboard.glb&mode=wire&xform=sphere_reflection
&mx=7.700333893299103&my=-0.2627578377723694
&mz=7.668757736682892&mr=1.7725382092756496
&minpx=35.4&zoom=0.37&rx=0.355&ry=0.081&rz=-0.317
&px=0.655&py=-0.167&pz=0.640&lodimpl=rust
```

Current-view policy: ceiling 64 px/subtriangle, depth 5, eight selected
faces, 64 selected leaves per face, 512 partition leaves, and two million
triangles.

| Metric | Complete frontier | Retained composition |
| --- | ---: | ---: |
| Source faces | 94,628 | 94,628 |
| Drawable members | 94,880 | 94,555 retained roots + 325 overlay members |
| Groups requiring complete grouping | 52 | 24 overlay groups |
| Suppressed source roots | — | 73 |
| Overlay roots / dyadic leaves | — | 69 / 256 |
| Members avoided during overlay publication | — | 94,555 (99.657%) |
| On-demand extraction time | — | 19.4 ms |

The active plan selected eight distortion-sensitive faces and produced 260
selected leaves. Shared-edge and grading closure expanded the replacement set
to 73 source faces, which is why suppressing only the eight selected roots
would be incorrect.

Complete planning remained 430.8 ms: 18.3 ms mesh planning, 321.8 ms frontier
construction, 64.5 ms reconciliation, 3.9 ms atlas accounting, and 20.9 ms
complete grouping. The overlay result proves that retained publication is
worth implementing, but it does not address the dominant complete-frontier
construction yet.

A follow-up ordering shadow on the same view found exact semantic membership:
zero mismatched groups and zero mismatched members. Drawing the two physical
layers consecutively changed the member order in 15 of 52 buckets, all using
material 0. That chess material is opaque, so depth-tested color is insensitive
to this order, but the result rules out treating the same composition as
automatically equivalent for alpha-blended or transmissive materials. The
follow-up diagnostic took 33.8 ms for overlay extraction and 33.2 ms for the
full semantic/order comparison in that cold browser sample; neither operation
runs per frame.

## Animated horse

Source view:

```text
?glb=horse.glb&mode=wire&xform=sphere_reflection&mr=2
&minpx=16&animate=1&anim=0&lodimpl=rust
```

The same bounded current-view policy produced:

| Metric | Complete frontier | Retained composition |
| --- | ---: | ---: |
| Source faces | 984 | 984 |
| Drawable members | 1,089 | 901 retained roots + 188 overlay members |
| Groups requiring complete grouping | 9 | 9 overlay groups |
| Suppressed source roots | — | 83 |
| Overlay roots / dyadic leaves | — | 76 / 112 |
| Members avoided during overlay publication | — | 901 (82.736%) |
| On-demand extraction time | — | 0.5 ms |

The accepted animation pose had revision 3 in continuity epoch 3. Eight faces
were selected, including 480 nonzero-priority candidates in the current view;
the plan installed without fallback.

## Persistent shadow publication

The follow-up retained-shadow gate stages the sparse overlay during adaptive
planning and commits it only with the matching complete GPU epoch. A failed GPU
upload leaves both the former complete grouping and former published overlay
intact; fallback and disable publish roots before retiring the overlay. It still
does not change draw submission.

A fresh static reflected-horse browser run used 984 source faces and a 1,038
member complete frontier. The committed shadow contained 909 retained roots,
75 suppressed roots, and 129 overlay members in eight groups, avoiding 87.57%
of complete-frontier publication membership. The existing measurement oracle
reused that exact committed shadow with zero overlay extraction work and found
zero semantic group or member mismatches. Baseline-then-overlay order differed
in one material-0 group, retaining the opaque-only cutover restriction. The
parity comparison took 4.1 ms, and Chrome reported no warning or error.

## Opt-in live GPU publication

`mr_setAdaptiveRetainedPublicationEnabled(true)` now performs a transactional
physical cutover. It forces persistent shadow staging on, refuses an already
staged adaptive transaction, and publishes the new root and overlay resources
before changing the active suppression mask. Upload or mask failure restores
the former GPU resources and CPU membership. Disabling the gate transactionally
returns to complete batches and clears the inactive mask.

The cutover is deliberately restricted to scenes whose active PBR buckets are
all opaque. A blended or transmissive bucket retains complete ordered
publication and exposes the reason in the adaptive diagnostic. This is a
material-semantics gate, not a render-mode shortcut: wire and diagnostic passes
still inherit the safety classification of the underlying PBR material.

### Static reflected horse

The complete path submitted 1,038 instances in eight draws. The retained path
submitted 984 physical roots plus 129 overlay members in nine draws; the
visibility pass suppressed 75 roots, leaving the same 1,038 logical members.
The retained composition reused 909 memberships, or 87.57% of the complete
publication membership. Render-shadow observation remained exact before,
during, and after cutover, with no browser warning or error.

Complete, retained, and returned-complete viewport captures had the same
SHA-256 digest:

```text
cc8fa578253a5e0cc154978385a16819d862989caecfa5434b16305fb103b27b
```

### Pathological inverted chess view

The cold current-view plan took 825.2 ms in the final validation browser:
50.0 ms mesh planning, 636.0 ms frontier construction, 96.0 ms reconciliation,
6.5 ms atlas accounting, and 33.9 ms complete grouping. This reinforces that
bounded frontier construction is the next performance target; atlas accounting
is not the bottleneck in this sample.

Enabling retained publication reused the existing frontier and reconciliation
caches. The resulting refresh took 36.1 ms, including 24.3 ms mesh planning,
2.1 ms frontier lookup, 2.9 ms reconciliation lookup, 5.0 ms atlas accounting,
zero complete-grouping time, and 21.1 ms to stage the retained overlay during
that transaction.

The complete path contained 94,880 logical and physical members in 52 draws.
The retained path submitted 94,628 physical roots plus 325 overlay members in
72 draws; suppressing 73 roots left the same 94,880 logical members. The exact
parity oracle reported:

- 94,555 retained memberships, avoiding 99.657% of complete-frontier
  publication membership;
- zero semantic group mismatches and zero semantic member mismatches;
- 16 baseline-then-overlay ordering differences, all in opaque material 0;
- 633 of 633 observed frames matching the active extraction contract after the
  reverse cutover, with zero extraction or observation errors.

Complete, retained, and returned-complete viewport captures had the same
SHA-256 digest:

```text
4469402745bb9a394fb6a972cf7990f7ce5fe1536e82d03efc647f990329a0d8
```

The complete submission returned to its original 52 draws, 94,880 instances,
636,474 lines, and ordered fingerprint `9296deab9943fcc5`. Chrome reported no
warning or error throughout the forward and reverse transition.

## Decision

Use two retained renderer layers for order-insensitive passes:

1. Stable baseline root batches, filtered by the exact suppressed-face set.
2. Independently owned adaptive overlay batches for every affected source face.

Keep `RenderBatchKey` backend-neutral. WebGL2 and WebGPU may represent the two
layers differently, but extraction must expose the same composition and draw
ordering contract. Opaque PBR, matcap, wire, normals, LOD, and stretch now have
an opt-in retained path with exact image parity. Alpha-blended and transmissive
buckets remain on the complete ordered path until an order-preserving
compaction strategy or OIT makes their equivalence explicit. Cut over behind
shadow parity, then replace the diagnostic's complete scan with a bounded local
closure that expands or falls back whenever adaptive influence reaches its
boundary.

This optimization reduces CPU grouping, allocation, and upload work. It does
not reduce the number of rendered patches or triangles; composed output remains
exactly equal to the full adaptive frontier.

## GPU suppression primitive

The renderer now has the inert-by-default mechanism needed for the baseline
layer. It owns a one-byte-per-source-face mask texture, updates only changed
row-contiguous runs, and binds it as an explicit memoized shader resource. The
camera-dependent visibility pass reads source face identity and applies the
mask only when a batch is marked as a retained baseline-root batch. Adaptive
overlay batches leave that flag disabled, including replacement leaves whose
dyadic identity is the source root.

Complete batches keep suppression disabled. Retained-root batches enable it;
overlay batches do not, including replacements whose dyadic identity is the
source root. The generated visibility shader, binding plan, one-float output
ABI, native renderer tests, WASM32 build, and live WebGL2 cutover were verified
together.

The backend-neutral extraction contract now names the physical publication
role of every batch as `Complete`, `RetainedRoot`, or `AdaptiveOverlay` and
carries the exact sorted root-suppression set at scene scope. Validation treats
suppressed roots as physical dispatches but not logical patches, requires every
suppressed root to exist in the baseline and have an overlay replacement,
rejects unmasked replacements, and forbids mixing the incumbent complete path
with retained layers. This permits WebGL2 to use a visibility mask while a
future WebGPU backend compacts the same logical scene without inventing a
different topology contract.

WebGL2 GPU maps are now keyed by that shared batch identity as well. The
behavior-preserving migration published every incumbent resource as
`Complete`, leaving suppression disabled. A live animated horse render-shadow
sample rebuilt three scene snapshots and observed four frames with exact
submission parity: five draw calls, 984 submitted instances, 3,306 lines, the
same ordered submission fingerprint, and zero extraction or observation
errors. Retained-root and overlay resources can therefore be introduced
without a second ad hoc key space.
