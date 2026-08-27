# 2026-08-27 animated LOD cadence and delta-stream gate

## Scope

This gate freezes the publication and pose identity that a same-context WebGL2
or future WebGPU LOD implementation must preserve. It covers commits:

- `8d3f35c` — backend-neutral composed LOD model packing;
- `e603a51` — sequenced full/sparse classifier publications;
- `512c3a6` — Rust receiver admission and full-snapshot recovery;
- `bd25fd0` — exact accepted-renderer-pose classification.

It does not switch classifier authority away from the worker and does not claim
that the worker fence is fast enough.

## Frozen contract

Every classifier completion carries a nonzero `delta_epoch`, its exact
`delta_base_revision`, and the next `delta_revision`. Empty sparse results still
advance the stream. The Rust receiver accepts a sparse result only when it
extends the exact resident base in the same epoch. A gap is rejected before any
record reaches renderer state, arms a worker reset, and makes the next result a
self-contained revision-one full snapshot.

Animated classifications additionally carry the exact clip time, monotonic
sample time, pose revision, and continuity epoch accepted by
`mr_uploadAnimationPose`. The browser no longer samples `animTime` independently
from the render pose on every RAF. Each accepted renderer pose requests one LOD
classification; the existing scheduler retains at most one in-flight request
and one newest follow-up. A completion from a retired continuity epoch is
rejected and recovered through a full snapshot. Lag within the same epoch is
measured explicitly.

Static or not-yet-posed assets use the rest path. Pausing does not: the paused
model keeps classifying its last accepted deforming pose.

## Deterministic gates

The following passed against the source above:

- `cargo test -p quilting-core --lib`: 200/200;
- `cargo check -p quilting-wasm --target wasm32-unknown-unknown --features leptos-ui`;
- `node scripts/smoke-hyperscope-presentation.mjs`;
- `node scripts/smoke-surface-walk.mjs`;
- worker and extracted browser-module syntax checks;
- `git diff --check`.

The core tests cover initial full publication, an empty but sequenced sparse
publication, a changed-face sparse publication, a missing-base rejection,
explicit epoch recovery, and malformed record rejection. The presentation
smoke executes the generated Rust receiver and asserts the browser/worker pose
and recovery wiring. The surface-walk smoke verifies that the same stamped pose
accepted by the renderer is the source of the LOD request.

Package-wide strict Clippy is not a green gate yet because `quilting-core`
retains pre-existing warnings in quaternion arithmetic, incidence, atlas,
evaluation, and older batch code. The new manual-multiple lint was removed;
unrelated lint debt was not folded into this change.

## Chrome/WebGL2 observation

The installed Chrome DevTools MCP opened a temporary horse LOD tab on the
user-run `localhost:8888` server. During playback the first measured window
reported:

- 8 submissions, 8 completions, and 8 applied classifications;
- zero empty/error completions;
- delta epoch 10, revision 7, and two explicit full-stream resets;
- submitted/completed pose revision 7 in continuity epoch 3;
- zero pose-revision lag;
- zero pose mismatches and zero delta-sequence mismatches.

A paused canonical load then reported:

- 3 submissions, 3 completions, and 3 applied classifications;
- submitted/completed pose revision 1 in continuity epoch 4;
- zero pose-revision lag;
- zero pose mismatches and zero delta-sequence mismatches.

The paused control read `Play animation`, proving the classifier was observing a
paused accepted pose rather than treating playback intent as geometry. Chrome
reported no warnings or errors. The temporary tab was closed, the original
chess tab and URL were preserved, and the user's server was not restarted.

The temporary tab was backgrounded and its mean worker/fence timings were much
worse than the prior foreground baseline, so they are deliberately excluded as
performance evidence.

## Remaining promotion work

The worker still owns a separate OffscreenCanvas/WebGL context and fence. The
next candidate must reuse this exact model payload, delta sequence, and pose
stamp on the main renderer context, first in shadow. Promotion requires:

- identical full and sparse records, including invisible standby topology;
- identical seam reconciliation, permutation, and grading behavior;
- zero retired-epoch publications and zero sequence mismatches;
- a foreground moving-camera/animated-horse distribution of pose lag;
- measured GPU dispatch, fence, readback, transfer bytes, and batch work;
- an explicit rollback to the worker until the same-context path wins those
  gates over a sustained run.
