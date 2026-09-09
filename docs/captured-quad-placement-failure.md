# Captured asymmetric quad: placement folds

September 9, 2026. Reproducible observation, not a new meshing solution.

Source: `fe/fixtures/patch-cases/asymmetric-curved-quad-20260909.json`.
The Fe oracle reconstructs the authored geometry and matches its admission
flag and pole-margin witness (0.0351913869 within 2e-6).

Run from `fe/tools/quilting-fe-fixtures`:

```
cargo test --release --lib --features fe-oracle captured_asymmetric_quad_quality_probe -- --nocapture
```

The test passed, including 20 mesh observations. Passing means the experiment
ran and the geometry witness matched; it does NOT mean the meshes are valid.
The run took 173.86 seconds including Fe compilation and Wasm instantiation;
this is not a generation-time benchmark. The preceding Rust build took 2m33s.

## Decisive ablation

Four-triangle fan, fixed captured center, requested outer LoD 4, automatic
interior counts, **1,432 triangles in every row**:

| Placement | Nonpositive UV triangles | Shape below 0.1 | Mean shape | Area coefficient of variation |
|---|---:|---:|---:|---:|
| Affine/unwarped | 0 | 173 | 0.2714 | 3.5906 |
| Edge correction only | 90 | 160 | 0.3941 | 1.4605 |
| Edge correction + area slices | 198 | 66 | 0.5104 | 0.9927 |

The last two modes keep the same edge laws and counts; only the interior area
slices differ. Affine placement deliberately changes the boundary sampling law,
so it is a validity baseline, not an equivalent boundary-spacing solution.

At LoD 3 the same fan has 376 triangles: 0, 22 and 50 nonpositive UV triangles
respectively. Both diagonals also exhibit reversed triangles under correction.
Every sampled surface position and 3D triangle-quality evaluation was admitted.

The existing diagnostic counts signed UV area <= 0, including nonfinite
orientation, rather than measuring a global intersection arrangement. It cannot
prove complete coverage. These many failures nevertheless rule out accepting
the current correction on this case merely because mean shape and area balance
improve. Shape measures do not retain orientation, so they can reward folds.

## Scope and next gate

The experiment uses the production Fe evaluator, placement and packed atlas
generator, with deterministic seed42, also used by the browser worker. The1
passed to `pool::create_jobs` is its epoch, not its seed; the earlier report
misidentified it. This is the same authored surface and policies, not a
bit-identical replay of the captured triangles at a different LoD.
All runs here are at LoD 3/4, not the captured LoD 8. No extrapolated fold count
for the browser mesh is claimed.

The midpoint-square affine comparison has no nonpositive UV triangles at either
level, but does not yet implement the required curved boundary uniformization.
At level 4 it uses 1,088 triangles versus 1,432 for the fan; it is not a matched
budget victory or permission to drop boundary requirements.

Next: isolate where the edge-only extension first loses orientation, then admit
an interior map only with fold/coverage evidence. Preserve shared boundary laws;
do not fix this by independently moving seam samples or silently disabling the
requested edge spacing. Keep the affine reference available and label the
current correction experimental. More LoD alone is not the acceptance gate.

Raw local run: `/laboratory/quilting/scratch/captured-quad-quality-oracle-20260909.log`.
