# Fe toolchain boundary

Use the shared worktree `/laboratory/fe-stuff/fe-worktrees/mb2` for Fe compiler
commands, standard ingots, and generic compiler changes. This is the explicit
project policy as of 2026-09-05. A matching branch name in a separate checkout
is not sufficient: both projects must see the same working files.

Quilting application ingots stay in this repository. Workspace dependencies
and Rust diagnostic-tool path dependencies refer to the shared worktree.
Use its `target/release/fe`; Fe analysis additionally uses `--profile release`.
Coordinate shared Cargo builds rather than launching redundant compiler builds.

Communication goes through the sole mailbox:
`/laboratory/fe-stuff/mb2/QUILTING_FE_UPSTREAM_BUG_REPORTS.md`.
Note that the mailbox location is not the compiler worktree location.
Respect ownership of in-flight hunks, preserve unrelated changes, and land
focused compiler fixes in the shared mb2 history. Do not independently advance
or rewrite a second mb2 history.

The old `.toolchains/fe` checkout is preserved only until upstream reconciliation
accounts for its unique commits and pending provider fix. Do not develop or
launch new builds there. Existing server artifacts are not proof that the shared
source has been rebuilt: record compiler revision, dirty state, generated artifact
identity and actual acceptance results when validating a demo.

## Historical evidence

Earlier checkpoints in Git and the demo evidence documents remain historical
measurements, not current pins. In particular `745044776d03a758471d2bf55de947d9e9f95d05`
was the first GPU-resident generated-atlas baseline; later arc/raster work needed
`672432abf`, `c2e7b346e`, and `8b7ff12ce`. These do not establish that the current
shared tree contains every subsequent fix or passes every release gate.
