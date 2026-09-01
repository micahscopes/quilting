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

## Deliberate non-goals

This does not yet make `AppStore` authoritative or claim atomicity across HHHS
and an in-memory UI projection. The next coordinator will support the common
one-envelope authored revision by committing HHHS first and rebuilding or
advancing `AppStore` only after success. Multi-command revisions remain outside
that authoritative path until they have an explicit batch payload/protocol.

No browser, GPU device, WebGPU context, server, or Blender process was started
for this CPU-only slice while another workload was exercising the shared GPU.
