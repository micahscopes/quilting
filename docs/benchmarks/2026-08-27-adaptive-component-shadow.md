# Adaptive source-component shadow parity

Date: 2026-08-27

This checkpoint runs the complete adaptive planner and the independently
recomputable source-component planner from the same source requests, selected
patches, output-chart pose, camera, viewport, atlas cap, and grading policy.
It is an explicit observation gate and does not alter GPU publication.

The shadow compares:

- scene-wide plan diagnostics and selected source identities;
- stable dyadic leaf identities and raw requests;
- reconciled physical edge LoDs;
- welded and hanging-corner density overrides;
- suppressed retained roots; and
- ordered replacement groups and members.

Enabling the component shadow also enables the established retained-overlay
shadow so the last comparison is made against the exact complete publication.
Any planning, topology, reconciliation, or extraction error is retained as a
diagnostic and cannot change live batches.

## Pathological inverted chess view

The live Chrome gate used the established skinny-triangle view:

```text
http://localhost:8888/?glb=classic_chessboard.glb&mode=wire&xform=sphere_reflection&mx=7.700333893299103&my=-0.2627578377723694&mz=7.668757736682892&mr=1.7725382092756496&minpx=35.4&fuzzy=1&fmode=3&ffocus=50&zoom=0.37&rx=0.355&ry=0.081&rz=-0.317&px=0.655&py=-0.167&pz=0.640
```

Current-view adaptation selected eight candidates. One unresolved camera
boundary retained its source root, leaving seven published adaptive faces and
133 selected leaves.

| Metric | Value |
| --- | ---: |
| Source faces | 94,628 |
| Certified component faces | 2,956 |
| Unaffected faces | 91,672 |
| Avoidable source domain | 96.876% |
| Complete composed leaves | 94,754 |
| Component leaves | 3,082 |
| Component planning | 17.6 ms |
| Component frontier | 24.6 ms |
| Component reconciliation | 3.6 ms |
| Component overlay | 1.6 ms |
| Component work subtotal | 47.4 ms |
| Shadow-only full-plan comparison | 31.0 ms |

All diagnostic, selection, leaf, request, resident edge, corner-density,
suppression, group, and ordered-member comparisons matched exactly. The first
complete cold plan on this tab measured 535.8 ms; its immediately repeated,
fully cached plan measured 50.8 ms. The component result above was cold, so it
is evidence for the intended frontier-size reduction, not a generalized
frame-time claim.

The retained-cache follow-up reran the identical request without changing the
published view. Both component identities hit exactly:

| Repeated component phase | Value |
| --- | ---: |
| Component planning | 10.8 ms |
| Frontier lookup | 0.0 ms, cache hit |
| Reconciliation lookup | 0.2 ms, cache hit |
| Component overlay | 1.3 ms |
| Shadow-only full-plan comparison | 17.7 ms |
| Measured shadow total | 30.0 ms |

The corresponding complete cached path measured 32.7 ms. Excluding the
deliberate 17.7 ms complete-result comparison, the repeated component phases
cost 12.3 ms. Cache counters ended at one hit and one miss for both frontier
and reconciliation, with two exact component-publication matches and zero
mismatches.

Enabling and disabling the observation gate produced identical viewport PNGs:

```text
6e06eaf27721069225789b875bae007b0aeb053b387849321750ce069d4e9e19
```

Chrome reported no console messages. The temporary test tab was closed and the
user's existing chess tab was not selected or modified.

## Automated gates

- 214 `quilting-core` unit tests and 15 integration tests pass.
- 16 generated Node/WASM tests pass, including a disconnected-source component
  shadow with exact complete-publication parity.
- The generated-export/inert-before-renderer Node smoke passes.

## Decision

The component path has earned a live-but-observational parity lane. It has not
earned publication authority yet. The next gate is retained component caches
and repeated pose traces: a cutover must reuse stable component frontiers,
preserve scene-wide triangle/work budgets, and fall back transactionally to the
complete planner on any shadow mismatch or oversized connected component.
