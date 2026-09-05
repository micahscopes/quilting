# Hand-written WGSL comparison

Requested as a separate experiment, not a hidden replacement for Fe.

`reference-sampling.wgsl` implements the quad sampler's proposal and retirement
passes by hand. Both call one shared candidate iterator. It retains exact Q14
distance tests, immutable generations, deterministic priority ties, the small
cell-window optimization, and conservative tree queries. Buffer lengths supply
capacities; the default request limit matches the current LoD 3 preview.

This is **not a complete hand-written atlas pipeline**. Candidate generation,
boundary rejection, bounds construction, compaction, insertion, Delaunay repair,
and rendering are still Fe-generated. Triangle sampling is not yet exercised by
this hand-written module. The production page does not load either probe file.

## First measured result

- Manual proposal/retirement together: **5,980 WGSL bytes** in one shared module.
- Generated proposal/retirement: **134,557 bytes** across two modules.
- Chromium: all 55 canonical quad preview keys plus an alternate-seed job retain
  exactly the same points, ordered triangles, twins, repair and completion
  receipts. Selected-point exclusion checks pass for all 56 jobs.
- Twelve alternating measurement pairs, after two warm-up pairs, for each of
  uniform 3/3/3/3 and mixed 0/3/0/3 at seed 42. Mixed submit-to-completion medians:
  generated **4.40 ms**, manual **3.35 ms**. Uniform results are approximately
  **4.1 ms** for both; no demonstrated uniform-case improvement.

The timing includes candidate/bounds initialization and all 64 sampling rounds,
not triangulation or rendering. CPU command encoding and initial shader/pipeline
compilation are excluded. The device did not enable timestamp queries, so these
are queue-completion measurements, not isolated GPU timestamps. Small preview
jobs, shared-machine noise, and dispatch overhead limit extrapolation. This
does not establish full-atlas startup performance.

## Reproduction

Use the current release `fe web dev` quad preview and an isolated Chrome MCP
diagnostic tab. Import/evaluate the two exported helpers in
`reference-sampling.probe.mjs` there; read the WGSL verbatim from its file.
`installReferenceSampling(surface, source)` validates both replacement pipelines
before installing either, retains their original pipelines and returns a
`restore()` function. No production runtime edits are needed.

Run `captureCandidateAtlas` with each key/seed from
`candidate-index-browser-evidence.json`. Hash its points, triangles, twins,
repair and receipt fields in that order, and compare every hash to the baseline.
Check selected-point independence too. Then use `compareSamplingQueueTime` on
each fixed key. Restore the original pipelines and render once when finished.

The captured evidence records the exact manual source hash. Its verifier tests
that association, all baseline comparisons, and the paired timing protocol;
it does not rerun a GPU or assert a speedup from saved observations.

Next comparisons: hand-written triangulation kernels and triangle-domain
sampling; larger jobs and GPU timestamp measurements; then use the differences
to improve Fe's lowering and helper sharing without sacrificing correctness.
