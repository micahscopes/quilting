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

## Deliberate non-goals

This does not claim atomicity across HHHS and an in-memory UI projection.
Instead, HHHS is the authority and AppStore is a rebuildable projection with a
compare-and-dispatch fence. Multi-command revisions remain outside the
authoritative path until they have an explicit batch payload/protocol. A
recovered nonempty HHHS project also requires its AppStore scene projection to
be restored before new ingress; cursor recovery alone intentionally does not
invent or duplicate authored commands.

No browser, GPU device, WebGPU context, server, or Blender process was started
for this CPU-only slice while another workload was exercising the shared GPU.
