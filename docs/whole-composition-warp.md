# Whole-composition redistribution

The composite viewer now declares an independent `composition_*` control group:
enabled, focal X/Y, concentration, twist, and a twist-falloff dropdown. Neutral
concentration is 1; neutral twist is 0. In a triangle, focal Y is the fraction of
the available height at focal X, keeping the focus inside the domain.

The order is: shared edge maps → child-local redistribution → assembly into
the full triangle/square → whole-composition redistribution → patch evaluation.
Internal seams can bend and move, but both neighboring uses apply the same map
to their canonical position. The outer domain boundary stays pointwise fixed.
This changes sampling, not the authored geometric patch or primary atlas.

`CompositionWarp` is an immutable value in the resident scene, so pending atlas
work cannot pair new controls with old child selection. No new buffer or JS
geometry path is introduced. `InteriorTwist` now accepts an ordinary Fe interval
law; existing callers retain quadratic falloff. The dropdown selects quadratic,
smoothstep, or smootherstep angle envelopes over the normalized polygon radius.

These are continuous boundary-preserving maps, not a promise of globally smooth
derivatives: polygon-radius sectors still have derivative changes. Nor does
continuous invertibility guarantee that coarse straight triangles remain
unfolded. Existing fold diagnostics remain relevant. Whole-domain warping also
changes interior density, so the reference-domain LoD estimate is not yet a
post-warp physical density guarantee.

Verification: release Fe source checking passed. The release Wasm composition
test passed in 51.41 seconds including compilation and its existing exhaustive
density/orientation checks. Added checks cover exact neutral behavior, fixed
outer boundaries, finite interior outputs and nontrivial movement of a point
on an internal quad diagonal across all three falloffs and both patch domains.
Browser acceptance is still pending at this checkpoint.
