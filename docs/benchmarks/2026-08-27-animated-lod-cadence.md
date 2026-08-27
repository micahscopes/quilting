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

## Packed WebGL2 classifier readback

Both worker and renderer-context classifiers now emit one lossless `u32` per
face from pass-2 transform feedback instead of six `f32`s. The low 24 bits
carry three four-bit edge exponents, the three-bit S3 permutation, a visibility
bit, and the eight-bit atlas index. Rust validates reserved bits and field
ranges and derives parity from the permutation. The unit oracle exhaustively
roundtrips all 30,000 combinations of the current exponent, permutation,
visibility, and representative atlas-index domains and freezes one exact
layout golden.

Live Chrome evidence after the cut:

- the 4,432-face composed-scene classification read 17,728 bytes instead of
  106,368 bytes;
- the 984-face animated-primary prefix read 3,936 bytes instead of 23,616
  bytes;
- shadow mode completed 13 exact worker/renderer comparisons with zero raw,
  pose, publication, resident, visibility, cull, batch-semantic, or batch-group
  mismatches and zero failures;
- a foreground 10-second Rust-authority window published 593 classifications,
  retained one accepted-pose revision of lag, transferred zero LOD bytes
  through JavaScript, and reported zero failures or runtime errors.

This is an exact sixfold byte-volume reduction at the GPU-to-CPU boundary, not
a claim that the corresponding wall-clock call is six times faster. Driver
scheduling, fence cadence, validation, CPU expansion, and retained batch work
remain separately visible in telemetry. `gpuReadbackBytes` records this GPU
boundary while `lodTransferBytes` continues to mean WASM/JavaScript payload.

### Direct packed authority admission

Renderer-context authority now keeps the validated `u32` words packed through
retained admission. `quilting_core::batch::FaceLodClassification` is the
backend-neutral semantic record, and the admission boundary consumes one
decoded record at a time without constructing a mesh-sized six-float vector.
The historical expansion remains only in worker rollback and `lodimpl=shadow`,
where exact worker-result fingerprints and field-by-field parity require it.

For the 984-face animated prefix this removes a 23,616-byte CPU decode vector
and its complete write pass from each authoritative publication. For the
4,432-face composed scene it removes 106,368 bytes. The packed readback vector
itself remains 3,936 or 17,728 bytes respectively and is retained in a separate
pool. Diagnostics expose `legacy_float_decodes`,
`last_legacy_float_decode_bytes`, `readback_vectors`, and `decoded_vectors`, so
a Rust-authority run can prove that the rollback-only expansion stayed cold.
This is a CPU traffic/allocation cut; it does not remove the WebGL2 fence or
the four-byte-per-face GPU readback.

The live Chrome gate made the distinction observable. A Rust-authority horse
run reached 407 publications with one packed vector creation, zero decoded
vector creations, zero legacy float decodes/bytes, matching model
fingerprints, and no failures or console errors. The same temporary tab was
then reloaded in shadow mode: 240/240 classifier comparisons were exact, the
raw/publication/resident/visibility/cull/batch checks had zero mismatches, and
the separate decoded pool was created once and reused. Each shadow decode was
the expected 23,616 bytes for 984 faces. This proves both the cold production
path and the still-live rollback oracle rather than inferring either from
source structure.

### Packed sparse admission

Rust authority now retains the previous complete packed publication below the
WASM boundary. Equal-length successors are diffed as `u32` words before
semantic decoding: changed words and their monotonically increasing face IDs
enter the same atomic sparse admission used by worker rollback, while exact
no-ops skip source admission and, in the ordinary non-adaptive path,
reconciliation, grouping, and batch upload.
Model or classification-scope shape changes remain explicit full snapshots.
The retained current/scratch vectors are swapped and reused rather than cloned.

A six-second animated-horse Chrome window reported 390 publications: one full
snapshot, 369 sparse packed publications, and 20 exact no-ops. Only 5,555
records entered semantic admission in total—14.24 per publication rather than
984, or 1.45% of the former full-prefix record workload. The last publication
changed two records. Model fingerprints matched; failures, legacy float
decodes, decoded-vector creations, and console errors remained zero. This does
not reduce the four-byte-per-face GPU readback; it removes the subsequent
per-face CPU semantic work when quantized topology is stable or sparse.

The no-op branch was also exercised with automatic current-view adaptive
partitioning enabled. During the four-second window exact-root no-ops advanced
from 32 to 46 while the adaptive planner completed 230/230 installations, with
no fallbacks, no transition left pending, and no failures or errors. Thus
view-dependent leaf refresh remains live even when root source-face admission
is skipped.

### Rejected WebGL2 scatter-upload experiment

A retained source-face-to-slot map was prototyped for non-adaptive root
batches so that only changed or swap-moved 40-byte topology records needed
upload. Adaptive leaves deliberately remained on their transactional full
publication path. The semantic algorithm and its removal/addition invariants
were unit-tested, but the browser measurements rejected the design for the
WebGL2 backend.

The pre-experiment animated-horse profile uploaded about 947 of 984 records per
batch build (465,066 records across 491 builds), with 0.83 ms mean WebGL upload
work. Exact sparse ranges reduced traffic to about 76 records per build, but
required about 21 `bufferSubData` calls per build and raised mean upload work to
1.73 ms. Coalescing each changed bucket to one bounding span restored roughly
one call per bucket, but the span covered about 893 records per build and still
measured 1.27 ms mean upload work. Both variants reported zero shared-edge
mismatches and zero GPU-batch failures; the rejection is about measured cost
and complexity, not correctness.

No implementation from this experiment is retained. WebGL2 keeps one compact
full-prefix upload for each changed bucket. True sparse/scatter publication is
reserved for the WebGPU compute/compaction/indirect-draw path, where it can be
expressed without multiplying JavaScript/WebGL driver crossings. This also
keeps the resident tessellation atlas unchanged: WebGPU work surrounds the
atlas with GPU-side classification, reconciliation, culling, compaction, and
submission rather than regenerating microgeometry every frame.

## Bounded distributions and rejected root fusion

Runtime cadence and publication timing now use bounded 2,048-sample Rust
windows. Recording is constant-time; p50/p90/p95/p99 are computed only when a
diagnostic snapshot is requested. The browser no longer sorts per-frame timing
arrays. Separate distributions cover frame interval, render CPU time, LOD
dispatch, fence-poll latency, readback, retained publication, reconciliation,
vertex-density reconstruction, render-node resolution, root-member grouping,
and upload.

A settled animated-horse window reported a 16.70 ms frame median, 17.20 ms p95,
and 18.20 ms p99. Render CPU p95 was 0.30 ms. The signaled readback remained
3,936 bytes and below 0.20 ms at p99; the roughly one-frame fence-poll latency
was asynchronous rather than a blocking CPU wait. No publication, batch, or
semantic failures were reported.

The pathological 94,628-face inverted chess view exposed the retained CPU
cost more clearly. Before the attempted fusion, a moving-camera sample reported
16.7 ms frame p50, 37.1 ms p95, and 49.9 ms p99. Retained publication measured
37.2 ms p50 and 49.2 ms p95. Within grouping, vertex-density reconstruction was
8.1/10.6 ms p50/p95, render-node resolution 1.0/1.2 ms, root-member grouping
11.5/19.7 ms, and changed-bucket upload 10.3/13.9 ms. The classifier readback
was 378,512 bytes. Shared-edge mismatches, roundtrip failures, and GPU-batch
failures remained zero.

Removing one source-face walk by materializing corner densities during root
grouping was then tested with the on-demand `mr_measureRootGrouping` oracle.
The oracle runs the historical separate path and proposed fused path over the
exact same resident state, primes retained scratch first, alternates execution
order, and compares compact-vertex maxima, every face-corner density, and every
ordered batch member.

Across 64 paired rounds on the same 94,628-face state, all 64 results were
exact. The separate path measured 21.8 ms p50, 24.7 ms p95, 28.0 ms p99, and
21.41 ms mean. The fused path measured 22.2 ms p50, 25.6 ms p95, 27.3 ms p99,
and 21.75 ms mean. Thus the fused path was about 1.8% slower at the median and
1.6% slower by mean, with no useful tail improvement. Production returned to
the separate path in `d8cb375`; the exact oracle and fused reference remain so
future layout changes can be judged without cross-run scheduler or camera
confounding. Cached render-node resolution and its explicit layout revision
invalidation remain independent wins.

The next WebGL2 target is therefore not another whole-scene loop fusion. It is
an incremental root-membership representation driven by the already bounded
changed component/neighborhood, while retaining one compact full-prefix upload
per changed bucket. That design must account for vertex-corner density
propagation and deterministic member order before it can replace the complete
rebuild.

## Incremental retained-root grouping shadow

Commits `f9e9515` through `5234f89` implement that representation without
changing authority. Reconciliation exposes its complete twin-connected changed
component. A retained compact-vertex-to-face CSR then expands that seed through
all shared vertices, recomputes each touched maximum from every incident face,
and returns every source face whose corner-density observation can change. A
source-ordered root index applies removals and additions only to dirty batch
buckets and rematerializes their compact member prefixes. This preserves the
exact `BTreeMap<RenderBatchKey, Vec<RenderBatchMember>>` result expected by the
current WebGL2 uploader; it is not a scatter-upload revival.

`rootgroupshadow=1` runs this candidate after the incumbent complete grouping
and compares every ordered bucket member. The candidate is non-authoritative.
Its timer includes incremental corner-density refresh and retained grouping,
but stops before the complete reference comparison. The diagnostics separately
report seed/closure faces, dirty buckets, rebuilt members, and retained vector
payload capacities.

The deterministic gate now includes 256 rounds of sparse resident, corner,
material, and node churn over 257 faces. Every round matches a complete ordered
rebuild. Token lifetime is bounded by simultaneously active keys plus one
refresh's churn rather than session history. The current gate passes 224 core
unit tests, 15 conformal integration tests, and the Leptos-enabled WASM32
check.

### Browser evidence and retained memory

The first pathological chess run covered 94,628 faces and 20 settled camera
gestures. All 21 comparisons were exact, with 20 incremental refreshes and no
shared-edge, roundtrip, or GPU-batch failures. The last closure contained 6,266
faces and rebuilt 47,172 members across 21 buckets. Candidate refresh measured
4.8 ms p50 and 6.5 ms p95, compared with approximately 13.2 ms p50 for the
incumbent vertex reconstruction plus root-member grouping in that run.

A broader 64-gesture orbit/zoom run completed 65/65 exact comparisons and 64
incremental refreshes. Before compacting retained keys, its index vector
payload reached 12,595,952 bytes. The compaction work in `111abf0`, `eb995a0`,
and `04384ec` then:

- replaced per-face and per-dirty-entry `RenderBatchKey` copies with `u32`
  tokens;
- reclaimed inactive key tokens instead of retaining session history;
- corrected capacity accounting for compact removal/addition entries; and
- kept the one large merge-scratch allocation local instead of diffusing its
  capacity through small buckets.

The same saved route and 64-gesture script after compaction again completed
65/65 exact comparisons. Initial full-build index payload was 1,307,728 bytes,
down from 3,189,408 bytes before compaction. After churn it was 5,414,744 bytes,
a 57.0% reduction from the earlier 12,595,952-byte run. These numbers cover
reported vector payload only; allocator/B-tree nodes and the shadow's duplicate
reference member map are deliberately excluded.

The final sample's last closure contained 21,650 faces, dirtied 36 buckets, and
rebuilt 47,456 members. Candidate refresh was 8.6 ms p50. Its 76.6/77.2 ms
p90/p95 tail shows that synthetic orbit/zoom states can still make the exact
closure expensive; it is not presented as a universal frame-time win. The
incumbent vertex and root-member stages in the same run measured 4.0 and 7.1 ms
p50 respectively. Changed-bucket upload remained separate at 6.2 ms p50 and
13.8 ms p95. Shared-edge mismatches, missing residents, roundtrip failures,
same-density jumps, and GPU-batch failures were all zero.

The animated horse supplies the necessary counterexample. Its 984 faces form
one connected component, so every changed seed expands to all 984 faces and all
five buckets. A 332-comparison run was entirely exact, but the incremental path
measured 0.4 ms p50 while the incumbent vertex-plus-group work was about 0.2 ms.
Incrementality is therefore not promoted globally.

The authority design must select between complete and retained grouping from a
pre-work estimate, retain the complete path for small or full-closure models,
and continue sampled complete comparisons after promotion. A selector needs
evidence across both closure fraction and rebuilt-member fraction; face count
alone is insufficient because the chess run rebuilt roughly half the scene's
members while still sometimes winning. WebGL2 continues to upload one compact
prefix per changed bucket. WebGPU remains the intended path for storage-buffer
reconciliation, visible compaction, and indirect submission.

Commit `3ee45cf` adds the first non-authoritative work classifier. A forced or
uninitialized index uses the complete path. Scenes below 4,096 faces likewise
stay complete. Larger scenes enter the retained candidate only when the exact
reconciliation seed is at most one eighth of the scene; post-work certification
also requires the shared-vertex closure to stay at or below one quarter and
rebuilt membership at or below three quarters. These thresholds are promotion
policy, not correctness invariants, and are tested at their exact boundaries.

The same 64-gesture chess trace then separated into 29 certified-incremental
and 36 complete-recommended samples. The candidate's certified-incremental
samples measured 6.6 ms p50 and 8.1 ms p95. The samples rejected by the seed
gate—still executed because this remains a shadow—measured 38.6 ms p50 and
100.9 ms p95. All 65 ordered-group comparisons were exact. This cleanly
separates the cheap and pathological populations that were mixed in the
aggregate tail above.

An animated-horse run reached 122/122 exact comparisons. All 122 were classified
complete before incremental work because the 984-face scene is below the size
gate; none were recommended incremental. Candidate work measured 0.3 ms p50,
while incumbent vertex reconstruction plus grouping measured approximately
0.2 ms p50. The selector therefore preserves the known small/full-component
counterexample. It remains diagnostic-only until a rollback-safe authority path
can skip rejected candidate work and periodically sample the complete oracle.

### Post-selector WebGL2 submission boundary

The selector trace also makes the next physical cost explicit. Its last chess
classification conservatively rejected 83,062 of 94,628 source faces (87.78%),
leaving 11,566 current-view candidates. WebGL2 nevertheless submitted all
94,628 resident patch instances in 37 indexed bucket draws. Preparation marks
the rejected patches invisible and the main vertex shader makes them
out-of-clip, so this does not imply fragment or raster work for every patch; it
does mean their resident atlas vertices are still invoked.

Across 66 changed batch builds, WebGL2 uploaded 4,209,023 instance records and
168,360,920 bytes: exactly 40 bytes per record, approximately 63,773 records
and 2.55 MiB per build. Upload measured 8.5 ms p50 and 17.9 ms p95 in this run.
The final wire submission represented 1,160,286 line primitives before
current-pose vertex rejection. Shared-edge and GPU-batch failure counters
remained zero.

This is the backend boundary rather than another invitation to multiply
`bufferSubData` calls: the measured WebGL2 scatter experiment above already
reduced bytes while increasing upload time. The next backend slice should
consume the existing prepared-patch visibility and `RenderBatchKey` contract,
compact survivor instance IDs in storage, and emit per-bucket indexed-indirect
arguments. A CPU oracle must freeze stable compaction order, suppressed-root
handling, counts, and overflow before WGSL becomes authoritative. WebGL2 keeps
the current degenerate-vertex fallback; it cannot provide the same submission
contract without a count readback or a universal maximum-topology draw.

## Remaining promotion work

The renderer-context implementation is now a real opt-in authority, but the
default and release promotion still require:

- a foreground moving-camera distribution across the saved inverted-chess
  views and animated horse, with frame percentiles rather than last values;
- explicit accounting for duplicate-shadow overhead, full readback bytes,
  retained comparison memory, sparse batch updates, and skipped submissions;
- optional change detection that can skip an unchanged full-prefix WebGL2
  readback without weakening the explicit WebGPU compaction target;
- moving-camera screenshot gates for seams, permutations, and LOD grading;
- longer all-cue, all-scale, inversion, adaptive, pause/scrub, and background
  soak before changing the default away from `js`;
- retention of the explicit worker rollback until the promoted path wins those
  gates.
