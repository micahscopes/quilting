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

0. `UpsertAsset(AssetDescriptor)`
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
