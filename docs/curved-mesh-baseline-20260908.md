# Actual curved atlas mesh: first bounded baseline

This is a measurement checkpoint, not acceptance of the sampling method.
Production Fe generates packed primaries, decodes permutations/winding, applies
the configured viewer placement, evaluates the surface at each triangle vertex,
and measures those finite straight 3D triangles. Rust runs the Wasm and checks
the report's consistency; it does not implement the sampler.

Command (from `fe/tools/quilting-fe-fixtures`):

```sh
cargo test --release --features fe-oracle curved_atlas_mesh_quality_baseline -- --nocapture
```

Receipt: `/laboratory/quilting/scratch/curved-mesh-baseline-20260908-r2.log`.
The preceding receipt failed on unsupported f32 compound assignments; this run
uses ordinary addition/assignment and passes. Total test time was 80.95 seconds,
including Fe compilation/validation and Wasm setup, NOT atlas generation time.

## What was compared

Seed 42, outer level 3, automatic interior counts, neutral skews/warps. Two
geometries: the symmetric default (bulge .6), and bulge 2.4 with the final active
corner moved by (-.5, 0, .7). The latter is p3 for triangles and p2 for quads;
the separate fixture regression verifies the actual corner and interior move.

Both existing measured-spacing settings were exercised. This changes boundary
placement as well as interior placement; it is NOT an interior-only ablation
under identical boundary positions. In the enabled circular mode, production
placement uses the domain-density segment law and its edge correction. In the
disabled mode it uses reference density maps. This receipt does not establish
equivalence for free weights, arbitrary post-warps or other map policies.

All 16 runs reported zero invalid finite triangles and zero nonpositive UV
triangle orientations. That is not proof of coverage, absence of intersections,
boundary fidelity, geometric approximation accuracy or robustness elsewhere.

For the asymmetric fixture, shape quality is in [0,1], with 1 equilateral.
Area CV is standard deviation / mean of the finite triangles' areas.

| Layout | Triangles | Minimum shape off → on | Mean shape off → on | Area CV off → on |
|---|---:|---:|---:|---:|
| Triangle | 94 | .178627 → .184330 | .410390 → .628023 | .846227 → .401120 |
| Quad diagonal 02 | 272 | .245792 → .212968 | .600027 → .668791 | 1.120331 → .898046 |
| Quad diagonal 13 | 272 | .104948 → .118336 | .526296 → .576932 | 1.133714 → .872115 |
| Quad four-fan | 376 | .161176 → .197701 | .575996 → .700381 | 1.029512 → .706970 |

The worse tail on diagonal 02 contradicts an unqualified quality-improvement
claim even though its average improves. Diagonals are count-matched here; the
fan is not. The triangle case is smaller than the planned 256/1024 budget study.

## Cost and limits

Each audit call generated its required tiles and measured the mesh in 9.9–40.2
ms in release Wasmtime on this machine. These are single observations, not
browser-worker or frame timings, and include generation and diagnostics together.
The audit presently evaluates three vertices per triangle (282–1128 calls),
including repeated shared vertices. The reported counter covers these surface
evaluations only, not internal density computations or authoring. It is a
diagnostic baseline, not the final unique-vertex cached measurement path.

Pending: low-tail quantiles and worst-cell locations, larger/mixed budgets,
shared-boundary witnesses, crossing/coverage checks, independent approximation
error, cost separation, pathological replay and interactive common-scale A/B.
Then compare fixed-boundary topology candidates at matched budgets. Do not use
this passing integrity test as a permissive quality gate for those experiments.
