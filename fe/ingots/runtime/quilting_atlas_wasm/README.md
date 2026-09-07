# Wasm atlas jobs

One implementation shared by the CPU acceptance exports and the typed worker
transport probe. Sampling, constrained triangulation and packing remain Fe.
No prebuilt atlas or Rust implementation is used by this ingot.

`packed_triangle` and `packed_square` generate one requested tile and return a
status, point/triangle counts and bounded `BrowserList<u32, 262144>`. Coordinates
are exact Q14 pairs (x in low16, y in high16); U16 index pairs follow the point
words. `packed_tile::tile_layout` is the layout authority. The present job scratch
holds65536points/131072faces; exhaustion is reported, never clamped to a lower
LoD. These are explicit job-memory envelopes, not restrictions on requested keys.

Internal Fe arrays are not assumed to have canonical browser stride. The sink
writes four-byte words through the core typed browser-memory API into an aligned
byte allocation. Canonical worker wrappers copy the returned list into response
storage; the raw-export probe must read before arena reset. Neither is a claimed
zero-copy worker transfer.

Current acceptance:

- All1200 keys passed generation/mesh audits before extraction into this module.
- Raw packed buffers pass index/winding/area/vertex-use checks for triangle and
  quad controls, mixed densities and uniform LoD8.
- `validation/atlas_worker_oracle` compiles a real typed Worker child from this
  implementation, with request epoch and a packed response.
- Its canonical entry executes mixed triangle/quad jobs in Wasmtime and returns
  aligned payloads. This is NOT yet browser worker or worker-pool acceptance.

Remaining: persistent bounded browser pool, job scheduling and cancellation,
strong stale-result handling, full packed-corpus audit, transfer/upload cost and
browser demos. Per-call memory is reused by allocator reset between canonical
jobs; a persistent worker process and retained scratch strategy still need actual
browser evidence.

`JobQueue<N>` is the placement-neutral ownership policy for that pool. It bounds
outstanding jobs by N, identifies leases by batch epoch/ordinal/slot, rejects
duplicate or stale completion, and stops publication after cancellation/failure.
Three release Fe tests pass, including1200jobs with four outstanding slots and
reverse-order completion. This is policy testing, not proof of parallel worker
execution. The current browser mailbox selects one scope per nominal actor type;
reusing that API for multiple instances still requires an explicit typed design.

Known separate limitation: EVM tests of `packed_tile` leave-output-untouched
guards fail while the equivalent actual Wasm tests pass. Retained tests must not
be represented as globally green. Also, a mutable local sink containing a
BrowserPtr field hits an unsupported compiler parameter representation; the Wasm
sink currently stores its wasm32 address and constructs typed pointers for writes.
Neither finding warrants resurrecting GPU generation or speculative optimization.
