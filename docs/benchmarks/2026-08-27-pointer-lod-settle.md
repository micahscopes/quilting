# Pointer LOD settle gate — 2026-08-27

## Observation

A separate Chrome tab replayed the saved 94,628-face pathological inverted
chess view with bounded current-view adaptation: 35.4-pixel floor, 64-pixel
ceiling, depth five, eight selected faces, 64 leaves per face, and a
two-million-triangle transaction cap.

The first transaction examined all 94,628 roots, found 93,158 visible, selected
eight, and published seven faces plus 133 adaptive leaves. It completed in
689.0 ms. An exact repeat hit every retained cache and completed in 60.1 ms,
although the enclosing LOD turn still took 110.2 ms because retained batch
grouping cost 81.6 ms.

Five synthetic pointer-motion reports spaced 80 ms apart exposed the scheduler
rather than a correctness failure. The five reports caused five additional
adaptive installs. The last transaction took 302.6 ms and its enclosing LOD
turn took 411.4 ms, including 326.6 ms of batch grouping. Across the stress run
there were zero LOD errors, delta mismatches, adaptive fallbacks, pose
mismatches, or incomplete publications.

## Decision

Pointer motion now follows the existing SpaceMouse policy. Current-camera GPU
visibility remains live every rendered frame, while worker classification,
adaptive partitioning, seam reconciliation, and batch publication use a
100-ms trailing settle boundary. Mouse drag and wheel share one scheduler, so
neither path can retain the former five-millisecond transaction cadence.

Repeating the same five drag reports after the change produced exactly one LOD
submission, one adaptive attempt, and one atomic install. A separate burst of
five wheel reports also produced exactly one of each. Both runs retained zero
LOD errors, delta mismatches, adaptive fallbacks, warnings, and console errors.

This is a scheduling containment measure, not a claim that 60-ms warm adaptive
planning is inexpensive. Variation-aware candidate selection, sparse retained
reconciliation, and cheaper changed-batch publication remain required before
current-view adaptation can become a default renderer path.
