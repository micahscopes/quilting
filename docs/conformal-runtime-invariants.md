# Conformal formal-to-runtime invariants

This file records the trust boundary between the kernel-checked Lean model in
`formal/` and the finite-precision Hyperscape/Hyperscope runtime.  It is an
implementation map, not a claim that Rust or the shaders are extracted from
Lean.

## Layer boundary

The ordinary glTF node tree, conformal coordinate frames, and oriented region
semantics are separate structures:

1. A glTF node tree owns entities, meshes, skins, and ordinary TRS animation.
2. A conformal frame forest maps local coordinates toward one ambient
   Euclidean frame.  It has one parent per frame so the path is unambiguous.
3. Round walls form complementary open sides.  An anchor chooses which side is
   interpreted as the active inside; changing the anchor changes orientation
   and chamber coordinates, not the underlying wall/contact skeleton.
4. Tracking and projection constraints refer to entities and frames without
   becoming parents in either graph.

An unrestricted multi-parent conformal DAG is intentionally deferred.  Two
paths between the same frames need not compose to the same map; deciding what
to do with that holonomy is a semantic feature, not a graph implementation
detail.

## Theorem-to-test map

| Runtime invariant | Lean evidence | Runtime evidence required |
| --- | --- | --- |
| Translation, nonzero scale, orthogonal maps, and inversion preserve compactified round sides | `RoundSideAutomorphism.translation`, `.scale`, `.orthogonal`, `.inversion` in `formal/ConformalMereology/SphericalInversion.lean` | Generator unit tests; sphere/plane image golden cases; Blender/glTF round trip |
| Compactified inversion is total and continuous, including pole and infinity | `continuous_extendedInversion`, `extendedInversionHomeomorph` | Runtime uses a documented finite pole sentinel; CPU/GPU thresholds and bounded-output parity tests must agree. The sentinel approximates, rather than proves, the compactified point at infinity |
| Generator composition may be used by the renderer | `IsOpenRoundSide.image_extendedInversionEquiv` and the generator preservation results | `ConformalTransformChain` composes in application order and collapses to the same `Mobius` coefficients used by the shader |
| Changing coordinate charts does not change the represented ambient point | Frame equivalences compose with their inverses; this is the runtime use of the automorphism layer, not a new incidence theorem | Path points stay in one reference frame and are converted to each timed active frame; ECS tests compare ambient coordinates immediately before/after enter, re-anchor, and exit events |
| Reorientation changes a signed Gram matrix by `D G D` | `gramMatrix_reorient` and `rank_singleWallGramFlip_sub_le_two` in `formal/ConformalMereology.lean` | Wall-orientation tests preserve absolute contact classification; a single flip changes only its row/column |
| Absolute inversive separation does not select its oriented branch | `inversivelySeparated_iff_externallySeparated_or_nested` | Runtime reports external separation versus nesting only when signed orientation/radius data is available |
| A finite laminar wall family has an explicit background chamber | `FiniteLaminarFamily.chamberModel` | Chamber fixtures use `WithTop`-style background identity and never silently discard the outside chamber |
| Payload totals are zeta coordinates and chamber payloads are recovered by incidence Möbius inversion without a global bottom | `zetaTransform_mobiusTransform`, `mobiusTransform_zetaTransform` | Integer/rational golden fixtures compare direct chamber accumulation with zeta then Möbius recovery |
| Honest anchor transport is semantic chamber reassignment, not reversal of wall labels | `ChamberReassignment`, the `Z_new * R * M_old` law, and `honest_reanchor_differs_from_naive` | Preserve the three-wall/four-chamber regression: honest totals `(14,12,8)` differ from the naive `(7,6,4)` result |
| Laminar incidence inversion is sparse | `mu_eq_identityKernel_sub_coverKernel` | Aggregate invalidation touches changed covers/chambers; geometric occlusion remains separately conservative |

## Numeric conventions

- Runtime quaternion layout is `(w, x, y, z)`. glTF rotation input is
  `(x, y, z, w)` and must be converted at the loader boundary.
- `ConformalTransformChain.generators` is in application order.
- Orientation parity counts every reversing generator: sphere reflection and
  negative uniform scale are odd; positive scale, rotation, and translation
  are even.
- A frame's chain maps local coordinates to its parent's coordinates.
- `Mobius::compose(self, other)` means apply `other`, then `self`.
- `SINGULARITY_NORM_SQ`, `SINGULARITY_SENTINEL`, and
  `POLE_PROXIMITY_NORM_SQ` remain shared CPU/GPU contracts. Their adversarial
  f32 tests are part of the conformance suite. `AFFINE_C_NORM_SQ` is only a
  CPU preprocessing threshold; the shader always evaluates the full
  differential so `c = 0` rotations and signed scales still transform
  directions and report their actual local stretch.
- General coefficient matrices are a render representation, not an authoring
  inverse API.  Inverse frame paths are constructed by reversing and inverting
  validated generator words.

## Reproducible baseline

```sh
cargo test -p quilting-core
cargo test -p hyperscape
cargo test -p quilting-gltf
cargo test --workspace
cargo check --target wasm32-unknown-unknown -p hyperscape
cargo check --target wasm32-unknown-unknown -p quilting-wasm
env -u NO_COLOR trunk build
nix shell nixpkgs#lean4 -c sh -c 'cd formal && lake build'
python -m unittest discover -s tools/blender_hyperscape/tests -p 'test_*.py' -v
blender --command extension validate tools/blender_hyperscape
blender --background --factory-startup --python-exit-code 1 --python tools/blender_hyperscape/tests/blender_roundtrip.py -- /tmp/hyperscape-roundtrip.glb
```

The final vertical slice must add golden tests for walls, chamber transport,
glTF interchange, ECS extraction, and Blender round trips to this baseline.
