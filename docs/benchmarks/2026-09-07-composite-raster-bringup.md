# Compositional raster bring-up

The first explorer is an affine control: either square diagonal or a movable
four-child fan, sharing nine runtime-generated uniform triangle primaries at
LoDs 0–8. It is not the full mixed-edge composition method or a surface metric
uniformization result. Browser execution remains to be verified.

## Raster legality blocker

The release Fe CLI rejected the vertex stage with the generic
`allocation/private-memory/trap operations` diagnostic. Changing the child
record from an array to three named corners did not resolve it.

The producer capture stopped after graph normalization, before inlining. The
small IR snapshot showed checked arithmetic and a shared `unreachable` block in
the vertex root, not a large private allocation. Bounded draw-index arithmetic
was made explicitly modular, but constant-divisor guards still blocked it.

A minimal shared-mb2 regression reproduces the failure by adding only
`vertex_index % 3` to the existing typed raster fixture. Its denominator-zero
branch is provably dead. Before the change, this fails the raster safety gate.
Graph normalization previously ran CFG cleanup without constant propagation;
adding SCCP before that eligibility check allows the constant-divisor fixture.
The paired negative fixture with a potentially zero dynamic divisor still fails
closed. The release regression passed in 7.35 seconds; that is compiler-test
execution time, not atlas generation or browser startup.

Evidence in `/laboratory/quilting/scratch/`:

- `composite-raster-capture-20260907/`: producer events from the failing build.
- `composite-raster-ir-20260907/0002-vertices+shade-pre.sona`: small pre-normalization IR capture.
- `composite-raster-divisor-before-20260907.log`: failing minimal regression.
- `composite-raster-divisor-after-20260907.log`: positive and negative cases pass.

This diagnoses a pass-ordering defect, not a reason to remove runtime guards
generally. It does not establish that the complete explorer now builds or runs.

Shared mb2 commit: `2cd59ed28`. The broader authored-raster suite passed four
tests (Wasm source oracle, task families, nominal resident notifications, and
the new guard regression). Its fifth test could not obtain a native GPU
adapter; the suite is therefore not wholly green. Actual GPU verification must
use the available Chrome device. Log: `composite-raster-regressions-20260907.log`.
