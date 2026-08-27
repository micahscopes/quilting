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

## Retained baseline-root grouping

The complete root grouping and its C0 corner field are now retained across
camera-, animation-, and adaptive-configuration-only refreshes. Its exact
identity is the pair of the crack-free root-topology revision and batch-layout
revision. A changed admitted root request, a reconciliation correction, new
geometry, or a material/node/atlas layout revision invalidates the cache.

The pathological chess gate produced one cold miss followed by two exact cache
hits:

| Baseline-root grouping counter | Value |
| --- | ---: |
| Cache misses | 1 |
| Cache hits | 2 |
| Exact component matches | 2 / 2 |
| Mismatched shared edges | 0 |
| Missing residents | 0 |
| Same-density seam jumps | 0 |

On the final repeated request, the component planner took 10.6 ms, retained
frontier and reconciliation lookup took 0.0 ms, component overlay extraction
took 1.1 ms, and the deliberate complete-result comparison took 16.9 ms. The
measured shadow total was 28.6 ms. This removes another source-sized rebuild
from an otherwise component-local adaptive refresh while keeping the complete
baseline available for exact rollback.

## Exact sparse triangle budget and atlas residency

The backend-neutral sparse-work oracle now derives physical scene work as:

```text
retained baseline - suppressed component roots + component overlay
```

It visits baseline render buckets, suppressed component identities, and
overlay buckets; it does not scan unaffected source faces. Missing atlas
patches, incomplete root coverage, unordered or invalid suppression identity,
and arithmetic overflow all fail closed before a result is accepted.

The pathological chess view produced:

| Sparse work term | Triangles |
| --- | ---: |
| Complete retained baseline | 204,456 |
| Suppressed component roots | -11,732 |
| Component overlay | +31,386 |
| Sparse composed scene | 224,110 |
| Complete adaptive planner | 224,110 |

The component overlay suppressed 55 roots and emitted 181 members in 28
render buckets. Every referenced atlas topology was resident. Sparse and
complete totals matched exactly on both the cold and repeated request, while
component-plan, selected-face, leaf, request, resident-edge, corner-density,
suppression, group, and ordered-member parity also remained exact. Chrome
reported no console messages and the temporary test tab was closed.

## Shadow-backed physical cutover

`setComponentPublication(true)` now opts into component-derived retained GPU
layers. The complete planner remains an oracle at this checkpoint: only an
exact component plan, overlay, atlas-residency result, and triangle budget may
supply the staged overlay. A component mismatch publishes the already-staged
complete overlay; a GPU staging failure retains the previous GPU epoch.

The pathological chess scene reached `componentPublicationState: active` with
one component install, zero component fallbacks, and retained physical layers
active. Disabling component publication transactionally returned to the
complete-derived retained overlay. Both physical paths produced the identical
viewport PNG:

```text
6e06eaf27721069225789b875bae007b0aeb053b387849321750ce069d4e9e19
```

The WASM gate also corrupts a component certificate after planning. The
component path is rejected, its install count does not advance, and the exact
complete overlay commits with `componentPublicationState: complete-fallback`.
This proves the cutover and fallback controls before the complete oracle is
removed from the hot path.

## Certified repeated-component hot path

An exact complete comparison now issues a reusable component certificate. Its
identity contains the component reconciliation generation, crack-free root
topology revision, batch-layout revision, scene-wide diagnostic, and exact
selected source identities. No hash or approximate pose comparison is used.
Any identity change runs the complete oracle again.

On an identical repeated pathological-chess request, the certificate allowed
the component planner to reuse the previously proven complete grouping and
overlay epoch:

| Repeated phase | Value |
| --- | ---: |
| Component planning | 20.2 ms |
| Component frontier lookup | 0.2 ms, cache hit |
| Component reconciliation | 0.4 ms, cache hit |
| Complete mesh/frontier/reconciliation/atlas/grouping | skipped |
| Component plan phases | 20.8 ms |
| Including overlay validation | 23.2 ms |
| Comparable complete cached path | 58.2 ms |

Complete frontier and reconciliation counters remained at zero hits and one
cold miss, proving that the repeated request did not quietly traverse the
complete path. Component certificate counters ended at one reuse and one cold
miss. The sparse work equation remained exactly 224,110 triangles, all seam
diagnostics remained zero, Chrome reported no messages, and the viewport hash
remained:

```text
6e06eaf27721069225789b875bae007b0aeb053b387849321750ce069d4e9e19
```

## Automated gates

- 216 `quilting-core` unit tests and 15 integration tests pass.
- 17 generated Node/WASM tests pass, including a disconnected-source component
  shadow with a real dyadic replacement, changed welded-neighbour C0 metadata,
  exact sparse triangle budget, and complete-publication parity.
- The generated-export/inert-before-renderer Node smoke passes.

## Bounded changed-component authority

After one exact oracle sample for a root-topology/batch-layout epoch, a changed
component may now publish directly from its certified welded closure. This lane
has a distinct component group identity, so it cannot masquerade as the stale
complete CPU grouping retained for rollback. It is eligible only when retained
publication is active, all active PBR buckets are opaque, the component does
not cover the whole scene, and it contains at most 4,096 faces. Above 4,096
source faces it must also cover at most one quarter of the scene.

The first 15 eligible changed epochs use component authority; the sixteenth
runs the complete planner and overlay oracle. Topology or batch-layout changes
require a new exact basis. Any sampled plan, resident LoD, corner-density,
overlay, atlas-residency, or triangle-budget discrepancy revokes component
authority for that exact topology/layout epoch. GPU publication remains
transactional, and explicit disable/re-enable resets the basis before retry.

On a 94,628-face pathological chess view, a controlled seven-face current-view
request produced:

| Work term | Value |
| --- | ---: |
| Welded component closure | 284 faces |
| Unaffected retained roots | 94,344 faces |
| Component frontier | 680 leaves |
| Baseline triangles | 143,786 |
| Suppressed root triangles | -24,388 |
| Component overlay triangles | +10,780 |
| Composed / complete triangles | 130,178 / 130,178 |
| Cold complete planning | 549.9 ms |
| Warm changed-component planning | 62.9 ms |
| Warm component overlay extraction | 0.6 ms |

The changed epoch reported `authoritative-unsampled`, one oracle skip, one
changed-authority install, zero mismatches, zero revocations, and zero
publication fallbacks. A clean complete-oracle tab reported exact plan,
resident, vertex-density, overlay-group, atlas, and triangle parity for the
same request. After revisiting that request through the changed-authority lane,
the full viewport PNG matched the complete-oracle image byte-for-byte:

```text
08c08b6fb5e62c32e79bcdd20dbb789db59e3c38ac9fe375d67cb50e724dba49
```

Chrome reported no warnings or errors, the temporary tabs were closed, and the
user's existing chess tab was not modified.

## Decision

The exact component overlay now has physical publication authority. Unchanged
certified epochs bypass the complete CPU planner, and bounded changed epochs
may do so between periodic exact samples. Whole-scene, oversized,
order-sensitive, uncertified, revoked, or failed candidates retain the complete
transactional path. The next gate is representative animated-pose evidence and
then moving the remaining component plan itself out of the frame-critical CPU
lane.
