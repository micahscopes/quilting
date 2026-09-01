# Hyperscape HHHS authored payloads 0.1 and 0.2

This crate admits exact, frozen `hyperscape-protocol` 0.1 and 0.2
`AuthoredEnvelope` values. Protocol 0.1 contains `UpsertAsset`,
`SetEntityTransform`, and `RemoveEntity`. Protocol 0.2 retains those commands
and adds `SetConformalFrameTransform`, an atomic replacement of one stable
frame's complete local-to-parent generator word.

Every HHHS entry payload is frozen as:

1. the exact bytes `hyperscape authored operation v0.1\0` or
   `hyperscape authored operation v0.2\0`;
2. the matching payload-schema major and minor as two little-endian `u16`
   values (`0, 1` or `0, 2`);
3. the matching protocol major and minor as two little-endian `u16` values
   (`0, 1` or `0, 2`);
4. the stable project UUID as 16 network-order UUID bytes;
5. a bincode 1.3.3 body using fixed-width, little-endian integers and rejecting
   trailing bytes.

The complete payload is capped at the public `MAX_AUTHORED_PAYLOAD_BYTES`
(1 MiB), and the bincode decoder carries a matching body limit before it sees
untrusted record bytes.

The 0.1 binary body uses a private untagged enum with frozen variant order:

0. `UpsertAsset(FrozenAssetDescriptor)`, whose fields are `id`, `uri`,
   `media_type`, and `content_digest`; both option discriminants are always
   encoded, independent of the protocol type's JSON omission policy
1. `SetEntityTransform(EntityId, WireTransform)`
2. `RemoveEntity(EntityId)`

The 0.2 binary body freezes the same first three variants and appends:

3. `SetConformalFrameTransform(ConformalFrameId,
   Vec<FrozenConformalGenerator>)`

The positional generator enum has its own frozen variant order, independent of
the protocol's self-describing JSON tags:

0. `Translation([f64; 3])`
1. `Rotation([f64; 4])`, in quaternion `wxyz` order
2. `UniformScale(f64)`
3. `SphereReflection([f64; 3], f64)`

The protocol caps one frame word at 256 validated generators. Payload encoding
selects the schema from the envelope's protocol version; decoding accepts only
the exact matching domain, payload version, and embedded protocol version.
Tests pin independent BLAKE3 goldens for representative 0.1 and 0.2 payloads.

The project ID and schema coordinates are inside the content-addressed HHHS
payload. Admission rejects a mismatched project, domain, payload version,
protocol version, malformed body, or invalid protocol value. Replica namespace
derivation permanently retains the original 0.1 domain salt, so a payload
upgrade does not split one project's causal history.

Sender authentication, message-ID uniqueness, and sender-local sequence-policy
enforcement are deliberately deferred. HHHS entry identity and causality—not a
sender sequence—is the ordering authority in this adapter. Presence, cameras,
frame clocks, render state, and GPU resources have no durable API here.

Recovery replays only a locally trusted transaction log produced by the durable
sink. Untrusted peer records always enter through normal HHHS repair and the
application admission policy.

## Live authored-record carrier 0.1

`AuthoredRecordFrame` is the small live-transport wrapper around one public
`ReplicaRecord`. Its canonical compact JSON contains exactly:

- `lane: "authored_record"`;
- `version: {"major":0,"minor":1}`;
- the hyphenated `project_id` UUID;
- canonical unpadded URL-safe base64 in `record_base64`.

Unknown fields, an incorrect lane/version, nil or mismatched projects,
non-canonical base64url, oversized framing, invalid replica-record bytes, and
records whose authored payload belongs to another project are rejected. Relay
delivery grants no authority: the decoded record still enters ordinary HHHS
admission, causal-predecessor checking, and application policy.

This frame is deliberately distinct from the direct-demo `authored` envelope
and ephemeral `presence` lanes. A raw authored envelope is a proposal that
exactly one selected admission authority may turn into an HHHS record. Having
every receiving browser independently open the same proposal would create
distinct concurrent records and unnecessary durable history. Once admitted,
the resulting `authored_record` frame is safe for every replica to consume,
deduplicate, defer pending predecessors, and repair. Presence never converts
to either durable form.

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

Portable archives omit the receiver-local authored projection cursor. An
application that intends to continue editing must use
`import_durable_authored_session`: after validation/admission it durably
anchors a new local cursor, or advances a proven local prefix cursor once,
then restores the rebuildable AppStore projection. Ordinary restart remains
strict and never manufactures a missing or unrelated cursor. Exact re-import
at the current history horizon performs no checkpoint write.

The archive contains authored scene history, stable IDs, asset URIs, and asset
content digests. It does not embed GLB bytes, ephemeral camera/selection state,
render resources, routes, or local storage metadata. GLB transport therefore
remains an orthogonal content-addressed bundle concern.

For histories containing only 0.1 payloads, materialized state roots retain the
exact 0.1 domain and `(project, assets, entities)` encoding. Once a history
contains a 0.2 entry, its root uses the 0.2 domain and includes the key-sorted
conformal-frame register map. This preserves exact old archive verification
while ensuring frame edits are committed by the current state root.
