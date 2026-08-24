# Hyperscape HHHS authored payload 0.1

This crate admits only `hyperscape-protocol` 0.1 `AuthoredEnvelope` values:
`UpsertAsset`, `SetEntityTransform`, and `RemoveEntity`.

The HHHS entry payload is frozen as:

1. the exact bytes `hyperscape authored operation v0.1\0`;
2. payload-schema major and minor as two little-endian `u16` values (`0, 1`);
3. protocol major and minor as two little-endian `u16` values (`0, 1`);
4. the stable project UUID as 16 network-order UUID bytes;
5. a bincode 1.3.3 body using fixed-width, little-endian integers and rejecting
   trailing bytes.

The complete payload is capped at the public `MAX_AUTHORED_PAYLOAD_BYTES`
(1 MiB), and the bincode decoder carries a matching body limit before it sees
untrusted record bytes.

The binary body uses a private untagged enum with frozen variant order:

0. `UpsertAsset(FrozenAssetDescriptor)`, whose fields are `id`, `uri`,
   `media_type`, and `content_digest`; both option discriminants are always
   encoded, independent of the protocol type's JSON omission policy
1. `SetEntityTransform(EntityId, WireTransform)`
2. `RemoveEntity(EntityId)`

`tests/adapter.rs::frozen_payload_round_trips_deterministically` pins a golden
BLAKE3 digest for the complete encoding of a representative value.

The project ID and schema coordinates are inside the content-addressed HHHS
payload. Admission rejects a mismatched project, domain, payload version,
protocol version, malformed body, or invalid protocol value.

Sender authentication, message-ID uniqueness, and sender-local sequence-policy
enforcement are deliberately deferred. HHHS entry identity and causality—not a
sender sequence—is the ordering authority in this adapter. Presence, cameras,
frame clocks, render state, and GPU resources have no durable API here.

Recovery replays only a locally trusted transaction log produced by the durable
sink. Untrusted peer records always enter through normal HHHS repair and the
application admission policy.

## Portable project archive 0.1

`ProjectArchive` is the offline/sneakernet boundary. It is intentionally not an
`hhhs-store` transaction log: storage logs contain local persistence framing and
are accepted only for trusted crash recovery. An imported project archive
carries public HHHS replica records and `DurableProject::import_archive` offers
every record to the ordinary authority, payload-policy, and causal-repair path.

The archive encoding is:

1. the exact bytes `hyperscape hhhs project archive v0.1\0`;
2. archive-schema major and minor as little-endian `u16` values (`0, 1`);
3. the stable project UUID as 16 network-order UUID bytes;
4. record count as a little-endian `u64`;
5. the expected 32-byte HHHS history root;
6. the expected 32-byte materialized project-state root;
7. each record as a little-endian `u64` byte length followed by the canonical
   `hhhs-replica` record bytes, in deterministic predecessor-first order with
   entry-hash tie breaking;
8. a 32-byte BLAKE3 digest of every preceding archive byte.

The complete archive is capped at `MAX_PROJECT_ARCHIVE_BYTES` (512 MiB), the
record count at `MAX_PROJECT_ARCHIVE_RECORDS` (1,000,000), and each record at
HHHS's `MAX_REPLICA_RECORD_BYTES`. Parsing checks declarations before reserving
record storage. Records must be unique, contain their complete predecessor
closure, decode as this project's frozen authored payload, and use the open
authority profile configured by this adapter.

The final digest detects corruption; it is not a signature or proof of author
identity. Successful import additionally requires zero refused/deferred
records and exact record-count, history-root, and state-root agreement after
admission. A failed in-memory import returns no partial project.

`DurableProject::apply_archive` also supports an exact retry or resuming a
locally durable prefix of the same archive. It refuses a project with any local
history hash absent from the archive before persistence, so this is an import,
not an implicit destructive replace or divergent-history merge.

On `wasm32`, `hyperscope-web::durable_history::import_project_archive` first
performs the complete in-memory import above, then applies that validated
archive to the dedicated strict-durability IndexedDB sink. JavaScript only
needs to obtain or save the opaque bytes; it does not decode project state.

The archive contains authored scene history, stable IDs, asset URIs, and asset
content digests. It does not embed GLB bytes, ephemeral camera/selection state,
render resources, routes, or local storage metadata. GLB transport therefore
remains an orthogonal content-addressed bundle concern.
