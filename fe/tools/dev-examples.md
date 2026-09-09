# All examples with fe web dev

Run from any directory:

```sh
FE_BIN=/path/to/release/fe node fe/tools/dev-examples.mjs
```

The script discovers `fe/web/*/index.html`, runs each with the real `fe web dev`
command, and serves a link/status index at http://127.0.0.1:38300/.
Two initial builds run concurrently. Once a child starts serving, the next
queued build starts; successful servers remain running with their own watchers.
An initial compilation failure is recorded and does not stop the other builds.
Stop the launcher with Ctrl-C to terminate its children too.

Settings: `FE_BIN` (default `fe` from PATH), `FE_DEMO_JOBS` (default 2),
`FE_DEMO_PORT` (index port, default 38300), `FE_DEMO_LOG_DIR` (persistent log
parent). Example ports follow the index port in sorted directory-name order.
`--list` prints the discovered entries without starting anything.

The index does not embed or run all GPU demos. Open the ones you want to use.
“Serving” reports initial server readiness, not browser correctness or the
success of every subsequent rebuild. Child logs and each page's existing
diagnostics remain authoritative. Failed initial builds are not automatically
retried; restart the launcher after fixing them. Rebuild concurrency after
startup remains controlled by the independent Fe watchers, not this launcher.

This is development-process orchestration only, not a rendering runtime,
geometry implementation, source generator, or replacement for `fe web dev`.
