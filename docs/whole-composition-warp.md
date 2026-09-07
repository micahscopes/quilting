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

A follow-up release Wasm run passed in 51.94 seconds. It additionally exercises
production placement on both sides of a reversed internal diagonal, with
nonidentity edge spacing, off-center focus, concentration 1.8 and twist 2.1.
All 257 dyadic boundary samples match bit-for-bit after whole-composition
warping for each of the three falloffs. This proves that tested seam, not an
all-topology or all-parameter finite-mesh coverage guarantee.

The initial release browser build exposed an upstream carrier limitation:
`prepare` rejects kernel argument 62 (`i1`) as an unsupported boolean storage
argument. Source checking and Wasm execution succeed. The policy intentionally
keeps real boolean/enum types; it is not flattened into demo-specific float
flags to bypass this. Sonatina fix `6bda54bb` now has five passing focused
regressions, including real GPU decoding. It represents storage arguments as
`u32` and decodes logical booleans inside the shader. The shared Fe release
CLI is rebuilding against this exact pushed revision.
The composite development server exited during this failed initial build;
the old tab may retain its last rendered view, but a reload is not available
until the corrected toolchain build is served. The separate arc server remains
independent. This is an unresolved delivery gate, not a completed browser feature.
