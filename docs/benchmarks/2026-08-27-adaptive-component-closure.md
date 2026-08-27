# Certified adaptive source-component closure

Date: 2026-08-27

This checkpoint identifies the exact source subset on which a sparse adaptive
frontier may be reconciled independently. It does not yet change planning or
draw publication.

`ScreenMeshTopologyCache` now computes vertex-connected source-face components
once from canonical welded vertices. Vertex connectivity is intentionally
stronger than half-edge connectivity: resident edge LoDs reconcile through
welded lines, while physical corner-density overrides max-reduce every face at
a canonical source vertex. Two faces that meet only at a vertex must therefore
remain in the same independently recomputable component.

The retained query:

- accepts stable source-face seeds;
- unions their complete canonical components;
- returns globally source-ordered face identities;
- rejects invalid seeds and an explicit face budget before publishing output;
- leaves its caller-owned output empty on failure.

The browser exposes this as the explicit, read-only
`__hyperscopeAdaptiveScreen.measureComponentClosure(maxFaces)` diagnostic. It
does not replan, schedule work, or mutate the active adaptive epoch.

## Pathological inverted chess view

The established eight-face adaptive selection was:

```text
3, 258, 272, 275, 15044, 15045, 15054, 15072
```

The exact closure measured:

| Metric | Value |
| --- | ---: |
| Source faces | 94,628 |
| Component-closure faces | 3,878 |
| Certified unaffected faces | 90,750 |
| Avoidable source scan | 95.902% |
| Closure query | 0.4 ms |

A budget of 3,877 failed closed with
`adaptive source-component closure needs 3878 faces; budget is 3877`. The
published adaptive plan remained active and unchanged after the rejected
diagnostic.

## Horse control case

The horse is one vertex-connected 984-face component, so its eight selected
faces correctly produced a 984-face closure and zero source-scan reduction.
That is not a pathological fallback: the full horse adaptive plan measured
12.5 ms in the same browser, including 6.7 ms of frontier construction. The
component query itself measured 0.4 ms.

## Decision

Use complete touched components as the first sparse-frontier correctness
boundary. For the chess case this can remove more than 95% of the current plan,
frontier, reconciliation, and overlay-extraction scan without inventing a
heuristic halo. A future, tighter influence closure may operate inside a large
connected component, but it must prove fixed boundary state or fall back to the
whole component. Small single-component assets retain the established complete
path when sparse setup would not win.
