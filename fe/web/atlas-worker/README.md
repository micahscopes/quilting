# Actual browser worker acceptance

This is a diagnostic page, not the finished atlas explorer. The browser acceptance
driver calls the compiler's unmodified module-worker actor runtime. All generation
and packing run in the compiled Fe child. No custom worker implementation, sampled
geometry fixture, or Rust geometry fallback is served.

Build compiler-derived child artifacts from `fe/tools/quilting-fe-fixtures`:

```sh
QUILTING_ATLAS_WORKER_ARTIFACT_DIR=/laboratory/quilting-fe/fe/web/atlas-worker/generated \
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=2 cargo test --release --features fe-oracle \
  cpu_atlas_compiles_typed_worker_payload -- --nocapture
```

Serve with Fe's development server from the Quilting-Fe root:

```sh
/laboratory/fe-stuff/fe-worktrees/mb2/target/release/fe web dev \
  fe/web/atlas-worker/index.html --port 0 --color never
```

On that page, the acceptance harness can run:

```js
await (await import('./browser-acceptance.mjs')).verifyAtlasWorker()
```

It creates one persistent module Worker, sequentially generates four triangle/quad
tiles including uniformLoD8, checks indices/winding/area/vertex-use and epochs,
verifies invalid-key rejection, and closes only its own worker in `finally`.
Returned timing separates compile, worker startup, request roundtrip and host
audit. It does not isolate sampling, triangulation, packing and transport yet.
`generated/` is compiler output, ignored by Git. The static page has no Fe
application script yet; `fe web dev` serves this diagnostic HTML and compiler
artifacts. This is not automatic compilation of a finished demo application.

## Full-pool acceptance (integration gate)

The four-worker driver is authored in `validation/atlas_pool_workers`. It uses
two `ScopedTaskFamily<WORKERS>` declarations, with one const pool size and no
numbered methods or host-owned job queue. This requires shared mb2 at
`b65a27dc0` or newer: the task-family support landed in `4fd919320`, and
`b65a27dc0` fixes concurrent response allocation/cleanup in the shared runtime.

Build its compiler-derived package from the fixture crate:

```sh
QUILTING_ATLAS_POOL_ARTIFACT_DIR=/laboratory/quilting-fe/fe/web/atlas-worker/generated-pool \
CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=2 cargo test --release --features fe-oracle \
  cpu_atlas_compiles_four_worker_pool -- --nocapture
```

Then the test driver is `pool-acceptance.mjs`, with `verifyAtlasPool()` for the
full 1,200-tile corpus or `{cancelAfterMs: 100}` for cancellation. It audits
responses, observes concurrency and reads Fe's retained-atlas summary. It does
not select jobs or implement the queue. Both generation and ownership remain
Fe. The pool reserves a 64MiB packed-output budget and reports exhaustion.

This gate is still being integrated: do not infer a passing browser run or
multi-worker speedup from these instructions. The separate Wasm ownership test
has passed retention and cancelled-publication checks on small generated tiles,
plus full canonical key enumeration. Full-pool rendering remains pending.

Current browser failure: four requests run concurrently, but response cleanup
traps after resume. After the shared runtime concurrency fix, one observed
325,860-byte response allocation ends at arena cursor 67,522,572; resume leaves
the cursor at 67,522,752 (180 extra bytes), so its checked post-return fails.
The optional `onMemoryEvent` callback observes allocator calls/checkpoints for
this investigation without replacing allocator behavior. Cancellation publishes
no tiles in that test, but cleanup has not passed. This is not a throughput
benchmark or a reason to bypass checked ownership.
