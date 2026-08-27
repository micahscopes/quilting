# Retained adaptive overlay measurement

Date: 2026-08-27

This checkpoint measures the exact sparse replacement layer introduced by
`AdaptiveRenderOverlay`. It does not change rendering. The live renderer still
publishes the complete adaptive frontier; the new `measureOverlay()` diagnostic
extracts the equivalent baseline-minus-suppression plus overlay composition on
demand.

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

## Decision

Use two retained renderer layers for order-insensitive passes:

1. Stable baseline root batches, filtered by the exact suppressed-face set.
2. Independently owned adaptive overlay batches for every affected source face.

Keep `RenderBatchKey` backend-neutral. WebGL2 and WebGPU may represent the two
layers differently, but extraction must expose the same composition and draw
ordering contract. Opaque PBR, matcap, wire, normals, LOD, and stretch can use
the retained composition after image parity. Alpha-blended and transmissive
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

Existing batches all keep suppression disabled, so this checkpoint changes no
draw output. The generated visibility shader, binding plan, one-float output
ABI, native renderer tests, WASM32 build, and live WebGL2 initialization were
verified before the resource is connected to retained publication.

The backend-neutral extraction contract now names the physical publication
role of every batch as `Complete`, `RetainedRoot`, or `AdaptiveOverlay` and
carries the exact sorted root-suppression set at scene scope. Validation treats
suppressed roots as physical dispatches but not logical patches, requires every
suppressed root to exist in the baseline and have an overlay replacement,
rejects unmasked replacements, and forbids mixing the incumbent complete path
with retained layers. This permits WebGL2 to use a visibility mask while a
future WebGPU backend compacts the same logical scene without inventing a
different topology contract.
