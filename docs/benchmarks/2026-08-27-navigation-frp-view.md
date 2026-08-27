# Navigation FRP view gate — 2026-08-27

`AppStore` now exposes one throttled `AppFrameSnapshot` projection for UI use.
It is published before the existing summary revision fence; renderer and input
adapters continue to sample the immediate frame state and never wait for FRP.
The first consumer is a read-only Leptos CSR navigation/focus status island.

An exact release build against parent commit `2293240` measured:

| Artifact | Parent | Candidate | Delta |
| --- | ---: | ---: | ---: |
| raw WASM | 7,135,905 B | 7,150,550 B | +14,645 B |
| gzip -9 WASM | 2,525,061 B | 2,529,410 B | +4,349 B |

The cost includes the new projection component, WASM mount/flush boundary, and
FRP subscription; Leptos itself was already resident. Native
`hyperscope-web --all-features` passed 21 tests and the full
`wasm32-unknown-unknown` Leptos build passed.

Chrome DevTools MCP verified one mount with no mount or application diagnostic
errors. The view began as `Free focus sphere · ordinary chart · focus inactive`,
then settled to `Selected-object anchor · ordinary chart · focus active` after
selection and to `inverted chart` after the atomic inversion gesture. Both live
transitions retained zero selection, renderer-packet, or frame mismatches. The
temporary test page was closed and the existing chess page was not touched.

A final integration gate opened the first polytope cue and jumped to the
inverted horse cue. The view moved from ordinary sphere radius `1.800` to the
authored inverted sphere radius `1.500`; the cue had no presentation error,
navigation diagnostic, application mismatch, pose mismatch, or frame error.
This also verifies that UI publication observes the transactional cue/chart
repair without becoming part of its frame clock.
