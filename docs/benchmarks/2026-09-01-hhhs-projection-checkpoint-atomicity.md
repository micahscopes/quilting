# HHHS authored-entry and projection-checkpoint atomicity

Date: 2026-09-01

## Why this seam exists

`hyperscope-hhhs-shadow` currently applies an authored revision to `AppStore`
before mirroring its commands into HHHS. That is useful as a diagnostic shadow,
but it is not an authority boundary: persistence may fail after the visible
projection has already changed.

HHHS 0.4.4 provides receiver-local attachments on a prepared admission.
`hyperscape-hhhs::DurableProject` now exposes the narrow primitive needed by a
future durable-first coordinator: one authored envelope and one opaque local
projection checkpoint can be staged against the same post-entry frontier and
history root, persisted in one storage transaction, and only then published.

## Proven behavior

- A failed external persistence write publishes neither the authored entry nor
  the local checkpoint.
- A successful write anchors the checkpoint to the exact admitted entry and
  post-entry history root.
- Replaying the durable transaction log restores both the materialized project
  and the intact checkpoint.
- The checkpoint remains receiver-local. It is absent from public replica
  records, repair streams, and portable project archives.

The focused adapter suite passes 21 tests, including an injected failure,
successful retry, and exact restart recovery.

## Durable-first application projection

`hyperscope-hhhs-shadow::DurableAuthoredCoordinator` now consumes this seam for
the common one-envelope browser/Blender revision:

1. Verify that the complete AppStore authored projection matches the durable
   cursor and HHHS materialization.
2. Persist the envelope and source cursor atomically in HHHS.
3. Compare-and-dispatch the AppStore projection against its pre-await revision.
4. Return the persisted public replica record regardless of the separate
   AppStore projection outcome, so a carrier never loses an announceable HHHS
   commit merely because its rebuildable UI projection raced.

An injected AppStore bypass inside the persistence future proves the race
fence: HHHS remains authoritative, AppStore does not overwrite the bypass, and
the coordinator enters an explicit poisoned/rebuild-required state. Stale
revisions are zero-write no-ops, and failed durability remains retryable.

The transport-neutral local peer boundary now reserves authored deduplication
across that asynchronous write. Dropping the reservation on persistence failure
does not advance message or sender-sequence memory, so the exact Blender/browser
envelope retries successfully. Completion happens only after HHHS admission.
Presence continues through the direct ephemeral AppStore lane and produces no
HHHS entry or checkpoint.

Restart restoration now follows the same authority split. Recovery validates
the receiver-local cursor against the current HHHS history root, derives one
canonical key-sorted `AuthoredProjectionSnapshot` from HHHS materialization,
and installs it through a dedicated atomic AppStore reducer event. The restore
does not manufacture authored envelopes, sender sequences, removals, or HHHS
history. Exact parity is an idempotent no-op; a non-empty store with either a
different cursor or different content is rejected without overwriting it.

The end-to-end coordinator test proves that restore leaves durable bytes,
history length, and materialized HHHS state unchanged, survives a second
idempotent call, refuses both revision and content divergence, and then accepts
the next local-peer envelope at the succeeding projection revision.

## Deliberate non-goals

This does not claim atomicity across HHHS and an in-memory UI projection.
Instead, HHHS is the authority and AppStore is a rebuildable projection with a
compare-and-dispatch fence. Multi-command revisions remain outside the
authoritative path until they have an explicit batch payload/protocol. A
recovered nonempty HHHS project must restore its fresh AppStore projection
before new ingress; the coordinator now supplies that explicit restore seam.

No browser, GPU device, WebGPU context, server, or Blender process was started
for this CPU-only slice while another workload was exercising the shared GPU.
