# Interior placement with fixed boundaries

The `interior_area_correction` checkbox compares two extensions of the SAME
measured edge law. It applies when measured spacing is active on an admitted
circular patch. Counts, atlas keys, coarse layout, edge targets and final
composition warp are unchanged. The default remains enabled.

- Enabled: use the existing two density slices, then correct their boundary
  displacement to meet every child edge.
- Disabled: use the identity as the interior base, then extend those same edge
  displacements with the existing partition weights. This is the edge-only
  baseline, not naive UV sampling with a boundary discontinuity.

Both call the same boundary bypass. The correction's weights vanish on the
other edges. This construction does not prove the interior map is fold-free.

## First receipt

Release Wasm test `curved_interior_comparison_preserves_boundaries_and_counts`
passed in 84.81 seconds including compilation and setup. Log:
`/laboratory/quilting/scratch/interior-ab-20260908.log`.

For triangular, both quad diagonals and four-fan layouts, at two bulges, all
257 dyadic positions on each child edge agree exactly between modes: 13,878
boundary comparisons, including a displaced fan center (.37,.61). These checks
compare the modes; they do not replace cross-child reversal or coverage tests.
The mesh comparisons use center (.5,.5), seed42, outer LoD3, automatic interior
counts and the active corner displacement (-.5,0,.7). Each paired count agrees.

At bulge 2.4, enabled → disabled:

| Layout | Triangles | Minimum shape | Mean shape | Area CV |
|---|---:|---:|---:|---:|
| Triangle | 94 | .184330 → .165307 | .628023 → .571151 | .401120 → .528121 |
| Diagonal 02 | 272 | .212968 → .234460 | .668791 → .661456 | .898046 → 1.124134 |
| Diagonal 13 | 272 | .118336 → .091798 | .576932 → .606767 | .872115 → 1.147676 |
| Four-fan | 376 | .197701 → .261038 | .700381 → .691109 | .706970 → .775009 |

All sixteen meshes have zero invalid finite triangles and zero nonpositive UV
triangle orientations. Diagonal13 edge-only has one triangle with shape<.1.
Neither method uniformly dominates. Fan versus diagonal is not count-matched.
These are observations, not accepted quality thresholds or approximation bounds.

Measured calls (generation plus diagnostics) took 10–38ms in release Wasmtime;
not browser frame times. The diagnostic still evaluates repeated triangle
vertices. The test ran before a spelling-only rename of the resolved scene
flag to `apply_area_slices`, needed to avoid ambiguity with the authored UI
parameter during actor-state projection. Browser verification follows the build.
