# Adaptive screen-priority gate — 2026-08-27

## Observation

A temporary Chrome tab replayed the saved 94,628-face pathological inverted
chess view with a 35.4-pixel floor, 64-pixel ceiling, depth five, eight selected
faces, 64 leaves per face, and a two-million-triangle transaction cap. The
user's existing chess tab and server were not touched.

The pure-JavaScript rollback path retained its historical stable-ID ranking.
It published faces `1, 11, 34076, 34441, 34442, 34443, 34452`, reported no
priority candidates, and produced 224,110 triangles. This is the expected
baseline because `lodimpl=js` deliberately does not run the renderer-context
classifier.

With `lodimpl=shadow`, the worker and renderer-context classifiers compared
94,628 faces exactly: zero raw, semantic, visibility, culled-count, or batch
group mismatches. The renderer-context pass found 37,727 visible candidates
with nonzero screen-metric risk and selected a maximum priority of 196/255.
The selected faces changed to
`3, 258, 272, 275, 15044, 15045, 15054, 15072`; the adaptive transaction used
260 leaves and 212,158 triangles. The packed readback remained 378,512 bytes,
exactly four bytes per source face.

With `lodimpl=rust`, renderer-context authority selected the same eight faces
and the same priority population. Its first full packed publication contained
94,628 records; an identical second classification was admitted as a no-op.
The authority path performed no legacy six-float decode. The measured cold
adaptive planning totals varied from 432.0 ms in shadow to 734.4 ms in Rust
authority, so this gate establishes selection correctness and ABI traffic, not
a speedup. Whole-scene frontier construction remains the dominant cost.

The animated 984-face horse was then run under spherical reflection with Rust
authority. A later diagnostic projection showed 359 adaptive attempts and 359
atomic installs, zero adaptive fallbacks, the current animation continuity
epoch, 525 nonzero-priority candidates, and selected priority 60/255. Its last
adaptive plan took 17.5 ms. The renderer had accepted 664 pose-coherent LOD
publications, 663 of which were topology no-ops; it reported zero authority
failures. Thirty-two completions that lost a race with a newer animation pose
were rejected before publication as designed.

## Decision

The renderer-context classifier now emits a quantized ranking hint derived
from within-patch projected metric variation and normalized inversion-pole
proximity. Visible pass-one alpha stores `1 + priority`; culled faces retain
zero. Pass two stores the priority in the high byte of the existing packed
`u32` classifier record.

The low 24 bits remain the complete resident-topology identity. Sparse topology
comparison masks off the priority byte, so a ranking-only change neither
rebuilds batches nor adds transfer traffic. Shadow mode retains priorities only
after exact worker parity. Rust authority retains them only after pose
continuity acceptance. The current-view selector ranks priority first, resident
root cost second, and stable source identity last. The JS rollback path remains
unchanged.

This improves which bounded faces receive exact dyadic screen-space analysis;
it does not solve the remaining cold whole-scene publication cost. The next
performance slice must retain the baseline and replace only changed adaptive
neighborhoods instead of regrouping every root.
