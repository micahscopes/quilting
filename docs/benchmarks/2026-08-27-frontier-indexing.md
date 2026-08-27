# Root-heavy adaptive frontier indexing

Date: 2026-08-27

This checkpoint reduces source-sized work inside
`ScreenMeshLeafFrontier::build` without changing its public topology or
reconciliation contract. A mostly-root frontier already carries canonical
source-vertex IDs, but the former builder rediscovered every corner through an
ordered map. It also allocated a temporary endpoint set for every welded line,
even when all spans on that line were identical and therefore could not contain
a hanging vertex.

The revised builder uses:

- a dense source-vertex-to-frontier index owned by the build;
- ordered lookup only for the sparse edge/interior vertices introduced by
  dyadic refinement;
- one reused endpoint vector for genuinely nonuniform lines;
- an allocation-free early exit for singleton and uniform-span lines.

These are internal representations. Leaf order, line order, reconciliation,
corner maxima, batch membership, and render extraction remain unchanged.

## Reproducible native oracle

The checked-in command constructs a 94,628-face disconnected all-root mesh. It
isolates source-sized frontier validation, line construction, and corner
indexing from glTF decoding and camera-dependent patch partitioning:

```sh
cargo run --release -p quilting-core --example bench_screen_frontier -- 94628 5
```

The accepted run reported:

```text
faces=94628 repetitions=5 topology_ms=75.967
frontier_median_ms=437.637
frontier_samples_ms=[436.478166, 455.967121, 437.385192, 437.637353, 437.782198]
```

Two exploratory cold samples from intermediate implementations measured
932.779 ms with ordered lookup for every corner and 853.718 ms after replacing
that lookup with a general-purpose hash map. They are not treated as a formal
benchmark distribution, but they rejected the hash-only approach and motivated
the direct source-identity representation.

## Exact pathological chess view

The browser gate used the same inverted chess URL and adaptive policy recorded
in `2026-08-27-retained-adaptive-overlay.md`: 94,628 source faces, eight selected
faces, 260 selected leaves, and 94,880 reconciled frontier members.

The immediately preceding fresh-tab build, which still used associative lookup
for every corner, measured:

| Stage | Before | Direct source index |
| --- | ---: | ---: |
| Total adaptive plan | 1,054.4 ms | 568.1 ms |
| Mesh plan | 58.8 ms | 36.6 ms |
| Frontier build | 779.4 ms | 370.9 ms |
| Reconciliation | 151.6 ms | 111.6 ms |
| Atlas accounting | 10.2 ms | 6.9 ms |
| Complete grouping | 50.9 ms | 40.0 ms |

The browser was not otherwise isolated, so these numbers establish a large
directional win rather than a frame-time percentile claim. More importantly,
the semantic gates were exact:

- the complete path retained 52 draws, 94,880 instances, 636,474 lines, and
  ordered fingerprint `9296deab9943fcc5`;
- render-shadow extraction matched 68 of 68 observed complete frames;
- the retained path matched 270 of 270 observed frames with its established
  72-draw physical submission;
- complete and retained screenshots matched the prior accepted digest
  `4469402745bb9a394fb6a972cf7990f7ce5fe1536e82d03efc647f990329a0d8`;
- Chrome reported no warning or error.

## Decision

Keep the direct source-identity index and uniform-line fast path as a standalone
optimization. They roughly halve the dominant build in this observed browser
sample, but 370.9 ms remains far too expensive for camera-rate reconstruction.
The next architectural step is still a bounded local frontier/closure that
uses the retained root classification as fixed boundary state and expands or
falls back when grading influence escapes its certified neighborhood.
