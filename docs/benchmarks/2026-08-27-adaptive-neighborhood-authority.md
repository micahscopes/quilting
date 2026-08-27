# 2026-08-27 adaptive neighborhood authority evidence

## Scope

This record covers commits `47cb548` (`Certify bounded adaptive neighborhood
epochs`) and `973dce0` (`Bind neighborhood authority to stable layout`). The
publication lane remains opt-in and retains the complete whole-scene planner as
its periodic oracle and transactional fallback.

An exact bounded-neighborhood comparison certifies the immutable source/batch
layout. Every changed pose still rebuilds its current neighborhood against the
current crack-free root field; the certificate never reuses stale geometry.
The complete planner is sampled every sixteenth changed epoch. A fixed-boundary
escape, mismatch, staging error, source/layout change, or triangle-budget
failure revokes authority. Revocation remains sticky for that layout epoch.

The mutable root-LoD revision is deliberately not part of the certificate
identity. It advances during ordinary animation and camera-driven root LoD
changes, while the neighborhood planner consumes the new root field and checks
its fixed boundary on every publication. Binding authority to that revision
initially produced only 9 skips in 152 animated attempts; the live observation
identified the incorrect epoch definition before release.

## Automated gates

- `cargo check -p quilting-wasm --target wasm32-unknown-unknown`
- `cargo check -p quilting-wasm --tests --target wasm32-unknown-unknown`
- `cargo check -p quilting-wasm --target wasm32-unknown-unknown --features leptos-ui`
- `cargo check -p quilting-wasm --release --target wasm32-unknown-unknown --features leptos-ui`
- `cargo clippy -p quilting-wasm --target wasm32-unknown-unknown --tests`
- `wasm-pack test --node crates/quilting-wasm`: 21 passed, 0 failed
- `node scripts/smoke-render-shadow.mjs`
- `git diff --check`

Clippy completed successfully with the repository's existing warning debt. No
new warning was promoted to a release blocker.

The WASM suite includes both a state-machine test and a full
plan-to-overlay-to-commit test. The latter changes the mutable root revision on
every epoch, proves 15 bounded publications without advancing complete
frontier/reconciliation counters, forces the sixteenth complete sample, checks
empty-selection recovery, corrupts a sampled boundary, and proves that a later
same-layout exact result cannot silently re-certify the revoked epoch.

## Animated horse gate

Chrome ran the ordinary animated horse with current-view adaptive LoD,
retained publication, and neighborhood publication enabled. Over one
four-second interval:

| Counter | Start | End | Delta |
| --- | ---: | ---: | ---: |
| changed attempts | 73 | 313 | 240 |
| complete oracle samples | 5 | 20 | 15 |
| bounded oracle skips | 68 | 293 | 225 |
| bounded GPU installs | 68 | 293 | 225 |

The delta is exactly 15 bounded publications per complete sample. Mismatches,
boundary escapes, failures, revocations, and neighborhood publication
fallbacks all remained zero.

An ordinary mouse-event orbit changed the canonical camera URL to
`rx=0.415&ry=0.920`. Across that interval, attempts advanced from 1,013 to
1,178, samples from 64 to 74, and bounded skips from 949 to 1,104, again with
zero safety events.

Switching to spherical inversion produced 179 explicit empty-selection epochs.
They did not count as failures or revoke authority. Returning to identity
resumed exact periodic sampling and bounded publication automatically:

| State | Attempts | Samples | Skips | Empty selections |
| --- | ---: | ---: | ---: | ---: |
| before inversion | 1,426 | 90 | 1,336 | 0 |
| inverted | 1,606 | 90 | 1,337 | 179 |
| restored | 1,786 | 101 | 1,506 | 179 |

## Pathological chess gate

The disposable chess tab used the reported skinny/inverted view beginning with
`mx=7.700333893299103`, a 35.4-pixel floor, and the checked-in classic
chessboard. Its exact sample reported:

- 94,628 source faces and 93,158 visible faces;
- 94,754 complete dyadic leaves versus 631 bounded leaves;
- 341 reconciliation faces, 505 observed faces, and 164 fixed leaves;
- 204,456 baseline triangles, 11,732 suppressed-root triangles, and 31,386
  overlay triangles;
- 224,110 composed triangles, exactly equal to the complete publication;
- zero leaf, request, resident, composed-vertex, group, or triangle-budget
  mismatches.

Seventeen subsequent settled camera epochs produced 16 bounded publications
and one complete sample. The complete sample's planner/grouping diagnostic was
157.3 ms; bounded epochs were approximately 15.4–31.4 ms in this trace. The
end-to-end settled gesture averages, including the 100 ms input debounce and
render scheduling, were 376.2 ms for the sampled epoch and 201.5 ms for bounded
epochs. These are one-browser observations, not generalized latency claims.

The chess and horse consoles contained no warnings or errors. Both disposable
tabs were closed after measurement. The existing user chess tab and both
user-managed Trunk servers were not navigated, restarted, or stopped.

## Result

The bounded neighborhood is now a measured physical authority for 15 of every
16 eligible changed epochs, not merely a shadow observer. The complete oracle
remains frequent, exact, and immediately recoverable. The next performance
frontier is GPU-resident visibility/culling and removal of the remaining
classification/readback traffic; this certificate does not claim to solve
those costs.
