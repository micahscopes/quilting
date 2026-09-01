# HHHS 0.4.5 downstream integration audit

Date: 2026-09-01

Reviewed candidate through: `hhhs-rs`
`e68402fe3efe6a22212c804f84c038ee3838d499`

Initial audit baseline: `2bfba3662cfb175b9700393acc38ee09c507ccc4`

Status: integration audit, not a release qualification or an HHHS design
specification. This document names behavior Hyperscope needs and leaves the
upstream API shape deliberately open.

## Directional conclusion

The 0.4.5 candidate is a substantial improvement over the immutable 0.4.4
release. In particular, `DurableReplicaHost` now owns the asynchronous
persist-before-publish boundary, checks an exact content-bound recovery state,
requires a real writer lease, fences ambiguous failure and cancellation, and
refuses aliased Replica/storage ownership. `hhhs-session` cleanly separates
receipt, causal readiness, prediction, effects, durable confirmation, repair,
and renewal. `hhhs-web-browser` provides a real IndexedDB transaction journal,
Web Lock exclusion, worker generations, bounded request lanes, and explicit
snapshot/revision/reset semantics.

Hyperscope should migrate to those boundaries instead of preserving its 0.4.4
storage handles, direct sink access, or application-specific causal retry
machinery. The strict host encapsulation is valuable and should not be
weakened to ease that migration.

The audit did reveal several generic composition gaps. Upstream resolved both
initial P0 requests in subsequent pushed commits. The remaining requests are
ergonomic, integration, or measured-performance work; the exclusive host's
read-only reactive boundary remains the one issue directly on Hyperscope's
0.4.5 migration path.

## Resolved P0 requests

### Restricted local-only durable transactions

Resolved upstream by `2ed092f` (`Persist receiver-local state through durable
hosts`). `DurableReplicaHost::prepare_local_transaction` and
`commit_prepared_local_transaction` retain host ownership, exact recovery-state
validation, lease fencing, persist-before-publish ordering, and ambiguous
failure handling without admitting a public history entry. Hyperscope should
use this path for projection checkpoints rather than preserving its 0.4.4
manual sink/storage commit.

The original issue statement follows for rationale and acceptance history.

`DurableReplicaHost` can persist and publish a prepared public admission, but
cannot advance receiver-local evidence, secrets, projection checkpoints, or
session lifecycle checkpoints without a public history entry.

Hyperscope needs this for its source/projection cursor. A restart-safe
`hhhs-session` host also needs it to persist sender nonce/progress checkpoints
and renewal floors before making the corresponding packet or replacement
overlay externally observable. Calling the asynchronous sink and synchronous
storage separately would bypass the host's recovery-state, lease,
cancellation, and publication fence.

Requested behavior:

- a host-owned, restricted local-state preparation/commit path;
- no ability to add or rewrite a public entry, predecessor, authority, or
  proof through that path;
- the same exact next-state preview, writer-lease validation, pre-await
  recovery fence, persist-before-publish rule, and reopen-on-ambiguity behavior
  as `commit_prepared`;
- an exact retry/no-effect rule and restart tests;
- a cancellation test showing that dropping the persistence future cannot
  restore an operational host.

The upstream API may be a prepared local transaction, a local-attachments
commit, or a narrower checkpoint operation. The behavioral boundary matters
more than the name.

### End-to-end bounded browser projection delivery

Resolved upstream by `e68402f` (`Bound browser projections with guarded pull
credits`). Protocol v5 gives each subscriber one worker-side retained payload
slot and one explicit pull credit, isolates subscriber generations, rejects
unsolicited/uncredited projection frames, accounts retained bytes, and covers
stalled consumers in real Chromium. This is the bounded confirmation and
correction path Hyperscope requested; it does not replace window-local
reversible prediction.

The original issue statement follows for rationale and acceptance history.

`WorkerEventBuffer` correctly bounds in-process application events and each
projection subscription independently. Its own documentation also correctly
states that this does not bound the browser's internal `postMessage` task
queue. Dedicated-worker projection events are currently posted directly.

Hyperscope is a concrete workload that meets the promotion condition recorded
for acknowledged subscriptions in the upstream 0.4.6 register: realtime
camera/control confirmation may run at 60--240 Hz while the window thread is
temporarily occupied by rendering, shader compilation, asset loading, or
layout. Rust-side queue bounds do not prevent an unbounded browser task/payload
backlog in that case.

Requested behavior:

- acknowledged credit/pull delivery or one coalesced dirty notification per
  subscription;
- independent slow-subscriber isolation;
- no lost wakeup between observing and clearing dirty state;
- generation and revision continuity checks;
- exact reset/snapshot recovery after coalescing or lag;
- a real-browser stress test with a deliberately stalled consumer proving a
  fixed bound on pending messages and retained bytes;
- no claim that worker delivery gives same-turn local feedback or peer
  acknowledgement.

Window-local reversible prediction remains the correct immediate-feedback
path. This request bounds its later confirmation/correction path.

The dedicated-worker close barrier also needs one explicit law. The current
server posts the correlated `Closed` reply before flushing deferred unsolicited
events. The window-side `close()` terminates the worker as soon as it observes
`Closed`, so later messages in that final flush can be lost even though the
in-process oracle returns reply and events together. Either a Close handler
must be unable to emit events, or accepted final events must cross an
acknowledged barrier before `Closed` becomes observable. Add browser parity
coverage proving that `Closed` is genuinely terminal.

Resolved upstream by `03e67df` (`Make browser worker close a terminal
barrier`). Final accepted events cross the close exchange before the correlated
terminal reply; once `Closed` is observed, worker termination cannot discard a
later accepted event.

## P1: high-value integration requests

### Reusable asynchronous durability test sink

At least HHHS Replica tests, browser-service tests, and Hyperscope currently
define separate `AsyncTransactionSink` fixtures. This is a correctness-heavy
trait and inconsistent test doubles can accidentally prove the wrong thing.

An `hhhs-testkit` sink should provide:

- cached exact `StorageRecoveryState`;
- inspectable accepted transactions and restart construction;
- an active/losable writer lease;
- deterministic fail-before-write and ambiguous fail-after-write modes;
- collision, stale-state, and non-advancing/lying-state modes where useful;
- shared conformance assertions for host fencing and reopen.

It belongs in testkit, not the runtime kernel.

### Exact-entry presented-authoring convenience

`ReificationPlan` intentionally constructs an entry with exact causal
predecessors. The convenient signed authoring methods intentionally construct
an entry at the Replica's current frontier. Combining the two therefore
requires repeated presentation-context, proof, and `AdmissionRequest`
ceremony.

Please promote the existing upstream candidate for exact-entry presented
authoring now that another downstream repeats it. Authority, requested right,
area, presented grants, and signing material must remain explicit; the helper
should only remove error-prone assembly and must never substitute the current
frontier for the supplied entry's predecessors.

A small `hhhs-session`/Replica helper that derives and records a
`SessionAdmission` directly from a `DurableCommit` would likewise reduce
ceremony without combining projection, effects, carrier acknowledgement, or
durability into one false outcome.

### Complete session composition vertical

The individual layers are well specified, but no example currently exercises
their complete public composition. Add one application-neutral example that
shows:

1. capability-founded establishment and a receiver-selected lease clock;
2. a replaceable transient prediction;
3. a durable-intent causal event and `ReificationPlan`;
4. exact-entry admission through `DurableReplicaHost`;
5. `SessionAdmission` flowing to projection and effect confirmation;
6. outbound record announcement plus later generation-2 repair;
7. projection notification loss followed by exact resynchronization;
8. sender/receiver and renewal-floor checkpoint persistence;
9. renewal, retired-packet fencing, restart, and late old-entry repair;
10. the same application service behind `hhhs-web-browser`'s application
    sideband and durable request lane.

Keep every stage and failure distinct. The example should not introduce a
universal session runtime or carrier policy.

### Runtime-negotiated participant counts

The const-generic session kernel is a good embedded and auditable core, but a
browser/Blender collaboration room learns its participant count at runtime.
Downstreams should not own a parallel session implementation or hand-maintain
an enum over every supported `SEATS` value.

Please provide a supported alloc-backed wrapper, bounded type-erased
dispatcher, or generated dispatch surface while retaining the const-generic
core. Also provide a bounded canonical manifest decoder/intermediate that can
validate the encoded seat count before conversion to a supported concrete
width. The exact mechanism is an upstream choice; requirements are bounded
allocation, identical canonical identity, explicit unsupported-width refusal,
and parity with the concrete kernel.

### One-pass browser recovery with bounded peak memory

`IndexedDbReplicaLog::open` retains every decoded `StorageTransaction`.
`recover_memory_storage()` then clones and replays those transactions while
the resulting `MemoryStorage` retains the reconstructed history. This keeps a
second history-sized decoded journal alive for the full session and raises
cold-start peak memory further because IndexedDB keys and values are fetched
in aggregate.

Please provide a take/stream/combined-open recovery path which validates the
same complete sequence but lets the live sink release decoded transaction
bodies after constructing the Replica store. It may retain compact row
commitments or query an old row on the rare exact-retry path. Add retained and
peak-memory measurements over a large history and payload corpus; do not trade
away corruption, gap, collision, or migration checks.

### Runtime-neutral reactive adapters

`hhhs-reactive::StateUpdate` already expresses snapshot, transition, and
reset. Small optional adapters should map:

- `SessionProjectionSnapshot`/`SessionProjectionTransition`; and
- browser-worker Snapshot/Revision/Reset events

into a coalescing `futures::Stream` or the existing runtime-neutral reactive
vocabulary. These adapters should live outside the kernel and have no Leptos,
DOM, Blender, executor, or application-state dependency. They must preserve
the exact sequence/generation reset law rather than converting a gap into an
ordinary delta.

The exclusive durable host also needs a safe source for those adapters.
`DurableReplicaHost::preparation()` can register a growth callback and
`DurableReplicaHost::snapshot()` can read, but the borrowed preparation facade
cannot become the non-publishing handle a polled stream needs. The old
Hyperscope path fed a cloned `Arc<MemoryStorage>` to
`hhhs_reactive::stream_view`; retaining that storage clone is now correctly
rejected as an aliased writer.

Please provide either a host-issued cloneable read-only Replica view or a
host-owned reactive adapter. Such a view may expose snapshot, growth epoch,
and subscription/DAG reads, but must be structurally unable to prepare,
publish, import, repair, or access durability. Issuing it after the durable
host claims ownership must not weaken writer exclusion or require downstreams
to put the writer behind a broad shared mutable lock.

## P2: papercuts worth resolving before the public API settles

- The `*_with` suffix names different attachment phases and closure shapes for
  open versus Ed25519/Sigma authoring. Provide consistent names for
  request-building hooks versus post-preparation restricted local
  attachments, with compatibility shims if appropriate.
- Consider a profile-specific, must-use value that pairs a newly sealed packet
  with its required post-seal sender checkpoint. This would make the
  persist-before-send obligation harder to accidentally separate while
  leaving persistence and carrier ownership application-defined.
- Expose small read-only diagnostic snapshots for worker generation/online
  state/pending lane counts and session gap/budget/retention state. Hyperscope
  can then render truthful diagnostics without mutable internals or parsing
  debug strings.
- Prefer structured reason accessors or stable reason codes at JS/IPC
  boundaries. Human `Display` text should remain diagnostic, not an
  application protocol.

## What Hyperscope should own

These are not upstream requests:

- scene, asset, entity, conformal-frame, camera, walking, selection, focus,
  inversion, animation, renderer, and presentation semantics;
- command vocabulary and materialization policy;
- WebRTC/Iroh discovery, topology, retry timing, and peer lifecycle;
- Leptos signals/components and Blender integration;
- local reversible camera/control prediction;
- the policy deciding which edits become durable commands versus transient
  presence;
- GPU work, frame scheduling, and mesh/LOD policy.

## Downstream migration consequences

Pending upstream feedback, Hyperscope should:

- remove its separately retained `Arc<MemoryStorage>` and read through
  `DurableReplicaHost::snapshot`;
- replace direct `host.replica()` preparation with `host.prepare*` or
  `host.preparation()`;
- replace custom browser IndexedDB durability with `IndexedDbReplicaLog`;
- redesign test failure injection around a shared control handle or upstream
  testkit sink instead of `durability_mut()`;
- not commit the 0.4.4-specific bounded carrier-record retry pool until it is
  compared against generation-2 `StepwiseRepairAttempt` and the browser
  repair path;
- use `hhhs-session` transient samples for camera/selection/control presence,
  and durable causal events only for explicit authored intent;
- keep the exact upstream revision pinned until the full dependency graph and
  wire generation migrate together.

## Exact downstream adoption probe

An isolated Quilting worktree at `6f84b52` was checked against the exact
`e68402f` dependency revision without modifying the production pin. The
resulting compile failures are deliberate 0.4.5 ownership breaks concentrated
in `hyperscape-hhhs`, not broad ecosystem churn:

- `AsyncTransactionSink` now reports writer-lease and cached exact recovery
  state and receives the expected state on every persist;
- `DurableReplicaHost::new` is fallible and rejects the separately retained
  `Arc<MemoryStorage>` used by the old adapter;
- direct `replica()`, `durability()`, and `durability_mut()` escape hatches are
  gone in favor of host preparation, snapshot, and commit methods;
- projection checkpoint persistence can move directly to the new restricted
  local transaction path;
- tests need a shared observation/failure-control handle or the upstream
  `hhhs-testkit` durability sink instead of mutable sink access.

The only unresolved architectural choice exposed by the probe is reactive
observation. The current `state_stream` and `state_signal_vec` methods clone the
same storage behind the durable writer. Their only current consumers are
adapter tests, while production application FRP already observes committed
`AppStore` read models. Hyperscope should neither keep that aliased storage nor
invent a second mutable replica. Adopt one of these explicit outcomes:

1. upstream provides a host-issued, structurally read-only reactive view or
   host-owned adapter; or
2. Hyperscope removes those unused durable-project reactive convenience
   methods and keeps application FRP downstream of committed host snapshots.

Until that choice is settled and the entire HHHS dependency graph moves
together, production remains pinned to immutable `v0.4.4`. This is a migration
containment decision, not a negative qualification of the 0.4.5 candidate.

## Audit evidence

At the reviewed revision:

- an isolated downstream `cargo check` against exact `e68402f` reached only the
  ownership/API migration points enumerated above; the production worktree and
  `v0.4.4` lock remained unchanged;
- upstream's protocol-v5 browser projection suite passed 20/20 in real
  Chromium after the retained-byte accounting and uncredited-frame rejection
  regressions were added;

- `cargo test -p hhhs-session --all-features --quiet`: green across the
  package's seven test/example suites (95 tests total);
- `cargo test -p hhhs-web-browser --quiet`: green (26 native tests);
- `cargo clippy -p hhhs-session -p hhhs-web-browser --all-targets
  --all-features -- -D warnings`: green;
- `cargo check --locked --target wasm32-unknown-unknown -p hhhs-session
  --features replica,xchacha20poly1305`: green;
- `cargo check --locked --target wasm32-unknown-unknown
  -p hhhs-web-browser`: green;
- the release-profile `session_hot_path` probe completed with 128 measured
  events, approximately 7.4 microseconds p50 end-to-end CPU in this local run,
  and reported explicit fixed-slot memory for every configured capacity.

These checks are focused integration evidence only. They do not replace the
upstream release matrix, real-browser slow-consumer test, downstream migration
gates, or exact-release-commit qualification.
