# Captured triangle: unchanged boundaries, folded tessellation

September 9, 2026. Fixture:
`fe/fixtures/patch-cases/curved-triangle-edge-only-20260909.json`.
The user reported overlap and edges apparently ceasing to be arcs when interior
area correction was enabled.

## Measured result

The release Fe/Wasm probe `captured_boundary_arcs_and_triangle_folds_probe`
passes. The captured triangle's admission/pole witness matches the browser:
admitted, pole margin approximately 0.018001802.

For this triangle and the captured asymmetric quad, all 257 dyadic samples on
every child edge, including endpoints, have exactly equal UV and 3D positions
with interior area correction on and off. Evaluations are admitted. This tests
the shared production placement/evaluation code in Wasm, not GPU execution or
the entire continuous curve. It does not independently fit circles or test all
possible authoring states.

The actual fine triangle audit gives:

| LoD | Triangles | Area correction | Nonpositive UV triangles | Shape < 0.1 | Minimum shape |
|---|---:|---|---:|---:|---:|
| 3 | 94 | off | 9 | 6 | 0.030602 |
| 3 | 94 | on | 1 | 1 | 0.076279 |
| 4 | 358 | off | 39 | 1 | 0.047365 |
| 4 | 358 | on | 15 | 13 | 0.002053 |

There were no rejected surface samples in these mesh audits. Shape is an
unsigned 3D measure; it cannot certify orientation. Nonpositive UV triangles
include reversed/degenerate parameter triangles and are a validity failure,
even if average shape improves.

These runs use the production Fe evaluator and packed primaries, seed 42,
at LoDs 3/4. The captured browser used seed 1 at LoD7. Therefore this is a
reproduction of the authored geometry and placement policy, **not** an exact
inventory of the triangles visible in the screenshot. Compilation plus the
test took 170.30 seconds; that is not an atlas-generation benchmark.

## Consequence

Changing the boundary curve is not necessary to produce the observed class of
failure: this geometry already exhibits parameter-space folding with unchanged
boundary samples. Which fine triangles caused the visible non-arc silhouette
still needs GPU/screenshot correlation. Do not claim that disabling correction,
raising LoD, or choosing the better average score fixes it.

Continue the bounded surface-aware coarse-mesh experiment with actual fine-mesh
validity gates. Preserve boundary ownership and use this case as a regression.
