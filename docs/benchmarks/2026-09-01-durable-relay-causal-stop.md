# Durable relay causal-stop boundary

Date: 2026-09-01

Audit target: the browser-owned delivery cursor between the local relay and
Rust `DurableAuthoredSession`.

## Finding

An inbound `authored_record` whose HHHS admission returned `deferred` or
`refused` was previously counted as ignored and the relay cursor advanced.
For a deferred child, especially after a bounded-retention gap, that could
discard the only retained delivery of the child before its missing causal
closure arrived.

## Corrected contract

- `applied` and exact `already_present` may advance the delivery cursor;
- `deferred` stops at the current cursor, exposes the missing entry hashes in
  the error, marks the carrier degraded, and requires causal repair;
- `refused` likewise stops rather than laundering a rejected record into
  delivery progress;
- a record that is durably applied but loses the rebuildable AppStore race may
  advance and be announced, while the carrier truthfully becomes degraded;
- raw proposal admission still reserves announcement capacity before the
  asynchronous durable write, so committed history cannot lose its outbound
  slot.

This follows the stage separation in HHHS candidate `7575510`: delivery,
durable admission, projection, announcement, and repair are not interchangeable
acknowledgements.

## Evidence

- 21 Node carrier/session lifecycle tests passed, including new deferred,
  refused, and projection-fault cases;
- the Blender/browser relay smoke passed with an exact 64-bit sequence,
  authored transform projection, canonical record traffic, presence, and
  overlay extraction;
- the carrier module passed Node's syntax check.
