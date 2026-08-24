# Hyperscope authored-source checkpoint 0.1

HHHS project history and a source replay cursor have different authority.
`ProjectArchive` contains portable authored scene history. The shadow's
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

## Atomic-persistence gate

The current pinned HHHS 0.4.2 open-authority API can prepare an authored entry,
but unlike its presented-authority APIs it cannot attach a local projection
checkpoint to that same `StorageTransaction`. Therefore the browser adapter
must not yet claim crash-atomic persistence of the source checkpoint alongside
the last command of a multi-command revision.

Before this diagnostic coordinator becomes application authority, use an HHHS
open-admission co-transaction attachment API (or an equivalent atomic batch)
so the final authored entry and cursor checkpoint reach IndexedDB together.
Until then, the portable codec and exact-horizon validation are ready, but live
browser checkpoint persistence remains deliberately gated.
