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
- 18 generated Node/WASM tests pass, including a disconnected-source component
  shadow with a real dyadic replacement, changed welded-neighbour C0 metadata,
  exact sparse triangle budget, complete-publication parity, and a whole-scene
  closure that stays on the complete path without a false mismatch or fallback.
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

## Animated whole-component eligibility

The animated horse is one welded component containing all 984 source faces. A
component publication therefore cannot retain an unaffected root layer, and a
second component plan duplicates complete planning rather than reducing it.
Before the eligibility gate, a 440-install animated sample remained exact but
ended with 20.4 ms of complete planning plus a separate 12.3 ms component
comparison. It accumulated 370 cold component samples and 70 certified reuses,
with zero mismatches or fallbacks; correctness was not the problem.

Component planning now receives the same bounded authority budget used by the
publication policy. For a source of `N` faces the limit is at most `N - 1`; for
large scenes it is additionally capped at 4,096 faces and one quarter of the
scene. Exceeding that limit is recorded as `ineligible`, clears no valid
complete state, and increments neither mismatch, certificate-miss, nor
publication-fallback counters. Overlay comparison and component publication
are not staged for that attempt.

In a four-second foreground animation sample after the change:

| Counter | Delta |
| --- | ---: |
| Complete attempts / installs | 106 / 106 |
| Complete fallbacks | 0 |
| Component eligibility attempts | 106 |
| Expected ineligible outcomes | 106 |
| Component comparisons / mismatches | 0 / 0 |
| Component publications / fallbacks | 0 / 0 |

The terminal diagnostic was `complete-ineligible` with the explicit reason
`component closure needs 984 faces; authority limit is 983`. Complete planning
ended at 17.8 ms in that sample. A fresh rebuilt-browser observation ended at
11.5 ms with 70 complete installs, 55 expected ineligible component attempts,
zero certificate misses, and no component frontier or reconciliation work.
Those per-pose timings are not an A/B speed ratio, but the removed 12.3 ms
duplicate path is structurally visible in the zero component-work counters.
Chrome reported no warnings or errors; the temporary horse tabs were closed
and the user's chess tab and both user-run servers were left untouched.

## Bounded face-neighborhood sizing

An audit after the whole-component bypass found that component eligibility was
already topology-indexed. `ScreenMeshTopologyCache` computes canonical
vertex-connected components once and answers each closure query from component
IDs and stored ranges; it does not walk all 984 horse faces every frame.
Caching that answer again would duplicate authority without addressing the
remaining complete-plan work.

The next topology primitive therefore models a finer candidate boundary. It
preindexes half-edge face neighbors and canonical-vertex face incidence, then
reports two distinct sets:

- reconciliation faces: selected seeds plus a requested number of half-edge
  rings;
- observed faces: reconciliation faces plus one non-recursive vertex-incidence
  ring whose physical corner-density records may observe the change.

The distinction is intentional. Edge-density fixed-point propagation crosses
shared edges and within-face grading; a vertex-only peer observes the final
corner maximum but does not itself justify recursively reconciling its whole
vertex-connected component. The query retains allocation scratch, is globally
source ordered, fails closed on invalid identities or explicit face budgets,
and explicitly does **not** claim publication authority. A local planner must
still use fixed boundary state and match the complete oracle.

For one animated horse pose with selected faces
`[6,10,507,508,596,597,716,733]`:

| Edge hops | Reconciliation faces | Corner observers | Retained roots | Query |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 68 | 916 | 0.2 ms |
| 3 | 89 | 210 | 774 | 0.2 ms |
| 6 | 259 | 396 | 588 | 0.2 ms |

The whole-component answer remained 984 faces. Thus even a conservative
six-hop candidate avoids touching about 60% of the source roots, while a
three-hop candidate avoids about 79%.

For a fresh pathological chess pose with selected faces
`[0,1,4,10,11,169,201,279]`:

| Edge hops | Reconciliation faces | Corner observers | Retained roots | Query |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 41 | 94,587 | 0.2 ms |
| 3 | 51 | 104 | 94,524 | 0.2 ms |
| 6 | 123 | 180 | 94,448 | 0.2 ms |

The complete adaptive plan at that pose measured 984 ms, including 612.4 ms
of frontier construction and 186.6 ms of reconciliation. Its full welded
component contained 284 faces. The six-hop query is therefore smaller than the
component while retaining 99.81% of source roots, but this remains workload
evidence rather than a correctness shortcut. The 217 core unit tests and 15
conformal integration tests pass, the WASM target check and generated-export
smoke pass, Chrome reported no warnings or errors, and all temporary tabs were
closed without modifying the user's chess tab or servers.

## Fixed-boundary neighborhood shadow

The bounded neighborhood is now integrated as a separately toggleable,
read-only runtime oracle. Its edge radius is the worst-case grading chain:

```text
ceil(log(max atlas LoD) / log(max face-edge ratio))
```

Thus ratio 2 with atlas LoD 64 uses six edge hops, while ratio 4 uses three.
The live atlas exercised below required seven ratio-2 hops. Selected and
edge-neighborhood leaves remain mutable. The additional vertex-incidence ring
is fixed to already-reconciled root residency; any attempted promotion of a
fixed leaf is recorded as a boundary escape and cannot become authority.

The first live horse comparison found zero leaf, request, or resident-edge
differences but eight raw fixed-observer corner differences. This was useful
shadow evidence, not a rendering failure. A fixed observer's outer corners can
still receive maxima from retained faces outside the local frontier. The local
corner result therefore must be max-composed with the already-reconciled
baseline corner field. Raw differences remain counted diagnostically, while
the composed field is the parity requirement. This is the same retained-state
composition a future sparse overlay must perform.

After that correction, one paused horse pose produced:

| Horse term | Value |
| --- | ---: |
| Source / complete leaves | 984 / 984 |
| Reconciliation faces | 393 |
| Observed faces / local leaves | 585 / 585 |
| Fixed observer roots | 192 |
| Raw / composed corner mismatches | 8 / 0 |
| Complete edge/request/leaf mismatches | 0 / 0 / 0 |
| Boundary escapes / failures | 0 / 0 |
| Exact | yes |

The horse was then animated in the disposable tab. After 235 epochs the
counters were 235 matches, zero mismatches, zero boundary escapes, and zero
failures. The terminal selection used 354 reconciliation faces and 527 total
observed faces. Its bounded plan, frontier, reconciliation, and proof
comparison took 11.0 ms; the same-epoch complete path took 12.5 ms. Those warm
horse timings are parity evidence rather than a claimed speedup, because the
shadow deliberately pays for both paths.

The pathological 94,628-face chess scene showed the intended scale separation:

| Chess view | Reconciliation | Observed | Local leaves | Complete leaves | Neighborhood shadow | Complete path | Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Skinny inverted view | 341 | 505 | 631 | 94,754 | 77.5 ms | 581.3 ms | yes |
| Broad inverted board | 210 | 276 | 276 | 94,628 | 49.3 ms | 467.6 ms | yes |

The skinny view had eight raw fixed-observer corner differences and zero after
baseline composition. The broad view had zero raw differences. Both had zero
diagnostic, selected-face, leaf, request, resident-edge, or composed-corner
mismatches, and zero boundary escapes or failures. Repeating the skinny request
hit both the local frontier and fixed-boundary reconciliation caches and ended
at three matches out of three.

This checkpoint still grants the neighborhood no render or publication
authority. A successfully staged neighborhood explicitly suppresses the
certified-component hot-path shortcut so the complete oracle runs in the same
epoch. The generated Node/WASM suite now passes 19 tests, including the
worst-case radius derivation and a disconnected-source runtime comparison with
a real dyadic replacement and fixed vertex-only observer. The normal and
Leptos-enabled WASM target checks and generated-export smoke pass. Chrome
reported no warnings or errors; the temporary horse and chess tabs were
closed, and the user's chess tab and both user-run servers were untouched.

## Exact neighborhood overlay and triangle budget

The next checkpoint extends the comparison through the actual retained render
transaction. `group_resident_screen_neighborhood_overlay_into` extracts only
the mutable neighborhood leaves, while fixed observer roots max-compose their
already-published baseline vertex densities. It rejects a fixed dyadic leaf
rather than silently replacing retained topology. The shadow compares:

- the complete plan fields described above;
- the exact set of suppressed baseline roots;
- ordered sparse overlay groups and their batch keys;
- every referenced resident atlas topology; and
- `baseline - suppressed + overlay` against the complete rendered triangle
  budget.

A focused core regression uses three connected faces: one selected root, one
fixed observer, and one retained outside face that contributes a larger outer
corner maximum. The raw component overlay is deliberately unequal to the
complete result; baseline-composed neighborhood extraction is exactly equal.
This prevents a passing shadow from depending on the outside face happening to
have a low resident corner density.

One paused horse comparison used 345 reconciliation faces and 559 observed
faces. The complete and bounded overlays both rendered 1,160 triangles. The
overlay extraction took 0.4 ms, the complete local proof took 20.4 ms, and the
same-epoch complete path took 25.6 ms. The horse was then animated for 194
epochs:

| Animated horse result | Value |
| --- | ---: |
| Matches / attempts | 194 / 194 |
| Plan / overlay / triangle-budget mismatches | 0 / 0 / 0 |
| Boundary escapes / failures | 0 / 0 |
| Terminal reconciliation / observed faces | 345 / 559 |
| Terminal composed / complete triangles | 1,094 / 1,094 |
| Terminal overlay / bounded proof / complete path | 0.3 / 2.4 / 3.5 ms |

The pathological skinny inverted chess view retained the intended separation:

| Skinny chess term | Value |
| --- | ---: |
| Source / complete leaves | 94,628 / 94,754 |
| Reconciliation / observed / local leaves | 341 / 505 / 631 |
| Raw / composed corner mismatches | 8 / 0 |
| Baseline triangles | 204,456 |
| Suppressed root triangles | 11,732 |
| Sparse overlay triangles | 31,386 |
| Composed / complete triangles | 224,110 / 224,110 |
| Cold bounded proof / complete path | 81.3 / 689.2 ms |

Three repeated comparisons were exact and reused both frontier and
reconciliation state. The final warm bounded proof took 49.6 ms versus 68.0 ms
for the complete path, including 1.0 ms for sparse overlay extraction. The
broad inverted-board view used 210 reconciliation faces, 276 observed/local
leaves, and no replacement overlay because its 476,704-triangle baseline was
already exact. Its bounded proof took 33.0 ms versus 295.7 ms for the complete
path.

All horse and chess comparisons matched suppression, ordered overlay groups,
atlas residency, and triangle budgets, with zero boundary escapes or failures.
The 220 core unit tests, 15 conformal integration tests, 19 WASM runtime tests,
normal and Leptos-enabled WASM target checks, and generated-export smoke pass.
Both disposable Chrome consoles were clean. Those tabs were closed after the
measurements; the user's chess tab and both user-run servers were untouched.

The implementation still has no neighborhood publication authority. Full
overlay equality is now strong enough for the next deliberately reversible
gate: publish the neighborhood result only after same-epoch complete-oracle
equality, retain the complete transaction as fallback, and compare final image
output before allowing certified epochs to skip the oracle.

## Decision

The exact component overlay now has physical publication authority. Unchanged
certified epochs bypass the complete CPU planner, and bounded changed epochs
may do so between periodic exact samples. Whole-scene, oversized,
order-sensitive, uncertified, revoked, or failed candidates retain the complete
transactional path. Whole-scene and oversized closures are expected policy
outcomes rather than false failures. Component identity and size are already
preindexed, so another ineligibility cache is unwarranted. Fixed-boundary
neighborhood planning, resident edges, baseline-composed corners, sparse
overlay membership, atlas residency, and triangle budgets now match the
complete oracle across the initial static, animated, camera, and inversion
gates. The next gate is same-epoch exact physical publication with the complete
transaction retained as fallback, followed by image equivalence and sustained
sampling. Only then may certified neighborhood epochs skip the complete path.
