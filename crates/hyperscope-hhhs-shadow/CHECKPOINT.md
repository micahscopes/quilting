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

The pinned HHHS 0.4.4 API now provides authority-neutral local co-transaction
attachments, including open-authority preparation, for one admitted entry.
That removes the upstream API blocker for binding a source checkpoint to an
exact entry and post-admission history horizon, but this adapter has not yet
wired that seam.

A separate atomicity question remains: one source `AuthoredRevision` may carry
several commands while the diagnostic shadow currently admits one HHHS entry
per command. Attaching the cursor to the final entry would not prevent a
durable prefix if an earlier entry succeeded and a later one failed. Before
this coordinator becomes application authority, choose and test one bounded
application-revision payload, a generic prepared batch, or an equivalent
resumable design, then persist its exact-horizon checkpoint through the 0.4.4
attachment seam. Until both pieces land, the portable codec and validation are
ready but live browser checkpoint persistence remains deliberately gated.
