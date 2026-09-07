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
