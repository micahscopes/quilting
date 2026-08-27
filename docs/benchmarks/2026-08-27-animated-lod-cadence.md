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

- `cargo test -p quilting-core --lib`: 202/202;
- `cargo test -p quilting-renderer --lib`: 55/55;
- `cargo check -p quilting-wasm --target wasm32-unknown-unknown --features leptos-ui`;
- `cargo test -p quilting-wasm --target wasm32-unknown-unknown --features leptos-ui --no-run`;
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

## Follow-on same-context shadow gate

Commit `3d38877` (`Shadow same-context LOD dispatch`) installed the classifier
on the renderer's own WebGL2 context behind `lodimpl=shadow`. It borrows the
exact pose retained by `mr_uploadAnimationPose`, consumes the same prepared
model, atlas lookup, subject table, camera matrix, and tessellation parameters,
stages an independent GPU copy, polls a fence without blocking, and compares
the complete six-float record for every classified face in Rust. It never
applies its result to batches. The worker remains the explicit effective
authority even when `lodimpl=rust` is requested.

The first composed-scene comparison found a real context-state bug rather than
being waved through. The renderer normally left alpha blending enabled. An
invisible pass-one record has alpha zero but must still write its bounded
standby exponents; inheriting blend state instead retained the clear sentinel
and yielded `0.5` edge values. The shadow reported 1,513 mismatched faces and
4,539 mismatched fields. Making the classifier pass explicitly disable
blending restored exact worker parity. The transform-feedback output was also
changed from the incorrect `DYNAMIC_READ` usage hint to `DYNAMIC_COPY`: it is
GPU-written and GPU-copied, while only the fenced staging buffer is
CPU-readable. This removed Chromium's repeated discarded-shadow-copy warning.

The final foreground presentation run covered the 4,432-face composed scene,
12 topology domains, full-scene classifications with 12 subject records, and
animated 984-face primary-prefix classifications with one subject record. Its
final diagnostics reported:

- 55 same-context dispatches and 55 completions;
- 54 exact comparisons and zero mismatched comparisons or fields;
- nine intentional busy skips and one topology-lifecycle cancellation;
- zero shadow failures and zero browser diagnostic errors;
- zero worker pose or delta-sequence mismatches;
- zero scene-extraction semantic mismatches;
- no Chrome warning or error messages.

The last observed main-context dispatch enqueue was 0.4 ms, the
scheduling-inclusive fence-poll latency was 12.4 ms, and the signaled readback
was 0.1 ms. WebGL2 cannot reveal when the fence became signaled, so poll latency
is not labeled GPU time. This was a correctness run that deliberately executed
both classifiers and retained a comparison snapshot; it is not evidence that
shipping both is cheaper than the worker. Renderer tests passed 54/54, the
WASM32 Leptos build check passed, the presentation/render/walk smokes passed,
and the temporary tab was closed with the user's original chess URL preserved.

### Exact publication and resident-batch gate

The 54-comparison run above was not long enough to prove the asynchronous
publication lifecycle. The extended gate therefore fingerprints the complete
immutable prepared model, the worker's full GPU result before sparse encoding,
the Rust-reconstructed authority snapshot, and the same-context full result.
The prepared-model fingerprint includes exact float or integer bits for source
positions and faces, joints and weights, morph deltas, face nodes, derived face
indices, scoped adjacency, morph-target count, and mesh radius. It is a stable
textual ABI so JavaScript cannot truncate a 64-bit fingerprint.

The first long animated run appeared to find classifier divergence despite
bit-exact pose and model fingerprints. The mismatch count was exactly equal to
the number of worker publications that disagreed with Rust's reconstructed
authority snapshot. The worker is allowed to publish while the same-context
classifier is still busy, but the browser had recorded only publications that
also obtained a shadow dispatch. The next sparse result therefore extended the
worker's current revision while Rust applied it to an older baseline. Every
worker publication now advances the immutable shadow authority; request ID
zero explicitly means baseline-only observation and cannot create a parity
candidate.

That exposed a second, downstream timing artifact. A matching worker result
could update live resident batches, a later unmatched publication could update
them again, and only then could the earlier same-context batch candidate become
available. Comparing that delayed candidate with mutable live batches produced
transient semantic and group mismatches even though both classifiers agreed.
The observer now takes an authority batch snapshot only in this delayed case
and compares the candidate with the exact matching publication. Immediate
comparisons retain the allocation-free live path.

After both lifecycle corrections, a foreground standalone animated-horse run
reported:

- 5,288 exact classifier comparisons and zero classifier mismatches;
- 5,288 exact pose-payload and raw-result fingerprint comparisons;
- 5,384 exact publication-reconstruction fingerprint comparisons;
- 5,288 exact resident batch-semantic and batch-group comparisons;
- zero requested, resident, visibility, culling, or group mismatches;
- zero failures and no bounded diagnostic errors.

The complete worker and renderer model fingerprints were bit-exact. A separate
composed-scene run retained all five presentation assets, 4,432 faces, 7,953
vertices, and 12 topology domains. Its static full-scene classification was
exact. The animated primary-prefix run then accumulated:

- 2,950 exact classifier, pose, raw-result, and batch comparisons;
- 3,024 exact reconstructed-publication comparisons;
- 75 intentional busy skips and six topology/cue lifecycle cancellations;
- zero classifier, publication, resident, visibility, culling, batch-semantic,
  or batch-group mismatches;
- zero classifier failures and zero scene-extraction semantic mismatches.

Full-scene work used 12 subject records in one GPU pass; animated primary-prefix
work used one subject record in one GPU pass. The worker remains live authority,
so this proves semantic equivalence and lifecycle discipline rather than a
performance win or authority cutover.

## Renderer-context authority cut

The next opt-in cut makes `lodimpl=rust` materially different from shadow mode.
It dispatches only the classifier resident on the renderer's WebGL2 context,
polls its fence without blocking, and publishes the completed full scene or
animated primary prefix directly through retained Rust admission,
reconciliation, adaptive planning, batch grouping, and GPU upload. The
classifier vector never crosses the WASM boundary. JavaScript receives only a
bounded diagnostic snapshot. `lodimpl=shadow` retains the exact dual-context
gate above, while `lodimpl=js` remains the serialized worker rollback.

An authority completion may lag the latest accepted animation revision, just
as the worker completion does. Rust therefore accepts bounded revision lag
inside the same continuity epoch, but rejects a completion after a clip wrap,
scrub, model reset, or static/animated domain change. Requiring the exact latest
revision was tested and rejected: poses advance faster than a fenced LOD pass,
so that rule starved publication. A rapid paused-polytope/playing-horse cue
round trip subsequently dropped four obsolete completions, published 298 valid
successors, retained one-revision pose lag, and reported no failures.

A foreground warm 15-second animated composed-scene window reported:

- 854 renderer-context publications, approximately 57 per second;
- 16.78 ms mean scheduling-through-publication time;
- 13.73 ms mean scheduling-inclusive fence-poll latency;
- 0.027 ms mean signaled readback call and 3.02 ms mean retained batch work;
- 9.34 changed source faces per publication on average;
- one accepted-pose revision of lag, zero failures, and zero LOD bytes crossing
  JavaScript.

These are one Chrome/WebGL2 observation, not cross-device performance claims.
In particular, WebGL2 still reads the complete 984-face animated classifier
prefix into WASM before sparse retained batch admission. Eliminating or
compacting that GPU-to-CPU transfer remains separate from eliminating the
worker-to-browser mesh payload.

Rollback was exercised by forcing the prepared-model fingerprint gate false.
The route changed effective authority to `worker`, armed both delta endpoints
for one coherent full reset, transferred a 106,368-byte 4,432-face fallback
snapshot, then resumed sparse worker publications with zero sequence or pose
errors. An authority fence/readback/publication failure uses the same fallback
boundary. A later matching model upload clears the failure latch.

## Remaining promotion work

The renderer-context implementation is now a real opt-in authority, but the
default and release promotion still require:

- a foreground moving-camera distribution across the saved inverted-chess
  views and animated horse, with frame percentiles rather than last values;
- explicit accounting for duplicate-shadow overhead, full readback bytes,
  retained comparison memory, sparse batch updates, and skipped submissions;
- GPU-side sparse result compaction or resident publication that reduces the
  remaining full-prefix WebGL2 readback;
- moving-camera screenshot gates for seams, permutations, and LOD grading;
- longer all-cue, all-scale, inversion, adaptive, pause/scrub, and background
  soak before changing the default away from `js`;
- retention of the explicit worker rollback until the promoted path wins those
  gates.
