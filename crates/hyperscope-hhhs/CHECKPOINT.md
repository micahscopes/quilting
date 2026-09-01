# Hyperscope authored-source checkpoint 0.1

HHHS project history and a source replay cursor have different authority.
`ProjectArchive` contains portable authored scene history. The local source's
`projection_revision` is local ingest/echo-suppression state for one Blender or
application projection source, so it is not admitted as an authored HHHS
command and is not copied into a sneakernet project archive.

`AuthoredShadowCheckpoint` binds that local cursor to an exact durable horizon:

1. `hyperscope authored source checkpoint v0.1\0`;
2. little-endian schema version `u16` major and minor (`0`, `1`);
3. 16 network-order project UUID bytes;
4. a one-byte `projection_revision` presence tag;
5. a fixed little-endian `u64` projection revision (`0` when absent);
6. little-endian `u64` history length;
7. the 32-byte HHHS history root;
8. the 32-byte materialized project-state root;
9. a 32-byte BLAKE3 digest of every preceding byte.

The fixed-width decoder rejects a wrong length, domain, version, checksum,
project, history length, history root, state root, or non-canonical option
encoding. The digest detects corruption; it is not an authority signature.

## Restart and import

- `from_project_checkpoint` recovers a local source only when the checkpoint
  matches the exact durable horizon. `align_store` initializes a fresh
  AppStore cursor through the existing reducer and does not add HHHS history.
- `from_imported_project` and `import_archive` reconstruct the scene baseline
  but intentionally start with no projection cursor. The first revision from a
  newly selected local source establishes that cursor.
- A cursor mismatch against a non-fresh AppStore is rejected. It is never
  silently overwritten.
- `DurableAuthoredCoordinator::recover` validates the receiver-local HHHS
  cursor at the recovered history root. `restore_store` then installs the
  canonical materialized asset/entity snapshot into an empty AppStore without
  adding an HHHS entry or fabricating source envelopes. Exact parity is a
  no-op; divergent non-empty stores are rejected.
- `DurableAuthoredSession` owns that coordinator together with its
  `LocalPeerIngress`. Recovery decodes canonical authored HHHS history and
  rebuilds the configured bounded message/sender window before any new peer
  envelope can enter. Presence and pending local echoes remain ephemeral.

## Atomic-persistence gate

The pinned HHHS 0.4.5 candidate provides authority-neutral local co-transaction
attachments, including open-authority preparation, for one admitted entry.
`DurableAuthoredCoordinator` uses that seam to persist a one-envelope authored
admission and the receiver-local source cursor atomically before publishing its
rebuildable AppStore projection.

Inbound `ReplicaRecord` values use the symmetric atomic path. A new record and
the receiver-local cursor commit in one storage transaction; a missing causal
predecessor defers without writes, a refused record changes nothing, and an
exact duplicate ignores the proposed cursor bytes. After a new commit, the
AppStore reducer installs canonical HHHS materialization at the advanced local
projection revision rather than replaying a remote command in arrival order.
Concurrent records therefore converge in either delivery order. An AppStore
bypass can fault only the rebuildable projection: the public record remains
durable, announceable, and recoverable on strict restart.

Live transport uses `hyperscape_hhhs::AuthoredRecordFrame`, not an invented
JavaScript record format. Raw `authored` envelopes are proposals for exactly
one explicitly selected admission authority. Rust-produced
`authored_record` frames are safe for every replica; `presence` stays outside
HHHS. This role split prevents several browsers from independently opening the
same Blender proposal into redundant concurrent records.

A separate atomicity question remains: one source `AuthoredRevision` may carry
several commands while the diagnostic shadow admits one HHHS entry
per command. Attaching the cursor to the final entry would not prevent a
durable prefix if an earlier entry succeeded and a later one failed. Before
extending the authority path to that shape, choose and test one bounded
application-revision payload, a generic prepared batch, or an equivalent
resumable design. Until that lands, the authoritative coordinator deliberately
accepts exactly one command per revision; the older diagnostic shadow retains
its explicit partial-prefix reporting for multi-command experiments.

The recovered peer window has the same explicit bounded-memory contract as
live `LocalPeerIngress`. It prevents immediate relay replay and retains sender
sequence floors within that configured window; it is not an unbounded global
message-ID ledger. Stronger adversarial replay protection belongs in a future
protocol/policy checkpoint rather than being implied by this local demo lane.
