# Shared spacing: partial browser acceptance

September7,2026. Release `fe web dev`, http://127.0.0.1:38331/, Chrome page65.
This receipt does not complete the delivery or recursive-atlas goal.

## Build identity

- Fe shared mb2: `94d3fe9f7`; Sonatina: `9ff01864`.
- Parent Wasm: `fe-render-236be90154f193ac.wasm`,358,989 bytes.
- WGSL: `fb56a15d3ab5ec56`, `2df18efe699cb9d6`, `50181b9cf65f7365`;
  total196,297 bytes across three passes. These are emitted sizes, not a
  claim that the remaining shader representation is optimally compact.
- Successful site log:
  `/laboratory/quilting/scratch/composite-spacing-fixed-site-20260907.log`.
- Runtime log:
  `/laboratory/quilting/scratch/composite-spacing-web-dev-20260907.log`.

## Checks actually performed

The four-triangle fan rendered with a fixed center(.5,.5), first with uniform
outer LoDs4/4/4/4 and then with1/6/6/1 (bottom/right/top/left). In the unequal
case, automatic interior LoD was enabled;24,396 vertices were drawn, with
status0 and pending0. Controls were exercised using their normal DOM input
events and the surface's public live/freeze methods.

Compensation on/off visibly changes spacing without changing the center. For
both states, a1018×1018 crop inside the square contained zero red fold-marker
pixels and zero exposed-background pixels. The red criterion was R>100 and
R>1.5G; the background criterion was R<35,G<30,B<30. This screen-space census
cannot rule out subpixel cracks, hidden overlaps, or all geometric folds.

Ten GPU-resident maps,170 floats/680bytes, were read using a temporary diagnostic
compute copy because production buffers correctly lack COPY_SRC usage. The
temporary buffers were destroyed. This probe is not shipped application code.
Against independently calculated16-bin midpoint integrals, maximum absolute
error was0.000004736193979226755; every map was strictly increasing. Totals:

`2,64,64,2,46.6690483,52.8085632,12.3743687,26.4116173,34.2946777,26.4116173`.

This verifies uploaded tables, not the full GPU inverse/interior-map function.
Exact warped shared-edge GPU fixtures remain required.

Release Wasm density/retention test passed in9.57s: all6,561 outer requests,
257 fractions each, preserved manual controls and coherent pending diagonal/fan
scenes. Log:
`/laboratory/quilting/scratch/composite-spacing-wasm-retention-20260907.log`.

The native authored-raster GPU gate also passed on AMD Radeon780M/RADV after
making the existing Vulkan loader/driver discoverable for that process only.
Log: `/laboratory/quilting/scratch/composite-native-gpu-loader-20260907.log`.
This complements the five previously passing raster tests; it is not a test of
the composition shader's full geometry.

## Remaining defects and limits

This list records the initial spacing checkpoint. The startup item is resolved
by the follow-up below; the other acceptance limits remain.

- The initial empty poster may outlive background generation. Entering live
  mode renders correctly; standalone activation/late-poster refresh needs fixing.
- Chrome MCP screenshot calls stalled. The actual poster canvas was inspected
  through MCP evaluation instead; pipeline creation alone was not accepted.
- Uniform sample spacing on curved/deformed surfaces is not established here.
  These are reference-domain density maps with piecewise-linear cumulative data.
- Shader size/performance, upload acknowledgments, staging allocations, extreme
  requests, moving-center pathologies and the broader UI remain to be measured
  and improved. The existing delivery checklist remains authoritative.

## Declarative startup follow-up

The standalone page now uses the supported `<fe-surface src="..." state="live">`
form. No imperative activation script was added. This exposed an upstream
custom-element upgrade race: initial `manifest` and scoped-task attribute
reactions started competing boots before `connectedCallback`, mixing Wasm/task
owners and producing memory-out-of-bounds errors. Shared mb2 `49c4abaae` makes
connection own initial boot; attribute reactions only reload an already booted
surface. All68 runtime unit tests pass, including automatic/manual upgrade and
later source replacement. This does not prove arbitrary rapid reconfiguration
or disconnected-in-flight ownership behavior.

Release CLI rebuilt, then the real page was checked through Chrome MCP without
calling `.live()` or editing controls. One measurement recorded `fe-ready` at
411.8ms and `fe-live` at452ms, both with4,296 vertices. This is navigation-to-
lifecycle timing with a warm browser shader cache and newly started workers,
not a cold-device GPU timestamp or the full165-tile generation time.
No console errors/warnings occurred. A prior fresh-load run's actual rendered
canvas had zero background/fold-marker pixels across the same1,036,324-pixel
interior crop. Canvas inspection followed automatic startup, not activation.

Current runtime asset: `fe-render-runtime-0ba5548668c8cb5d.js`.
Logs: `/laboratory/quilting/scratch/declarative-surface-upgrade-tests-20260907.log`
and `/laboratory/quilting/scratch/composite-declarative-live-web-dev-20260907.log`.
# Executed GPU/Wasm boundary agreement

`composition_gpu_oracle` is an authored Fe validation ingot, not a handwritten
WGSL substitute. Its compute actor calls the production `canonical_parameter`
and `EdgeSamples` storage view; the Wasm reference uses the shared generic
`OrientedIntervalMap` over owned CDFs. Both use canonical endpoint arithmetic
for physical placement. The Rust fixture only compiles, allocates, dispatches,
reads back and compares; it does not implement a delivered sampler.

Release execution on AMD Radeon 780M (RADV PHOENIX) passed thirteen density
fixtures, all ten boundaries, both directions and 257 sample coordinates:
66,820 queries / 267,280 scalar comparisons. The center was (0.17, 0.83).
Cases include uniform LoD0/8, 1/6/6/1, alternating 0/8, the single dense edge
0/0/0/8, rotated gradients and asymmetric 3/7/2/6. Output is prefilled with
NaNs, so missing writes fail the comparison.

- Maximum GPU/Wasm absolute error: 0.000000059604645 (tolerance 0.000002).
- Reversed GPU canonical parameters and physical positions: bit-identical.
- Probe artifacts: 10,972 bytes Wasm; 36,723 bytes WGSL.
- Test execution including Fe compilation: 11.35 seconds. The initial Rust
  harness/dependency build took 5m43s; neither is an atlas-generation benchmark.
- Evidence: `/laboratory/quilting/scratch/composition-gpu-agreement-20260907.log`.

This establishes agreement for the tested production boundary functions in a
compute probe, alongside the separately inspected raster demo. It is not yet
a capture of every final raster vertex, an all-permutation mesh audit, an
interior-distortion guarantee, or exact continuous-density integration. The
remaining coarse interiors should not be attributed to a boundary-backend
discrepancy without new evidence.

Reproduction uses the fixture crate's `raster-oracle` feature and test
`composition_gpu_boundary_positions_match_wasm`, in release mode. A real GPU
is required; this gate does not accept a GPU-skip result. On this Nix machine,
the successful run supplied the Vulkan loader through process-local
`LD_LIBRARY_PATH` and the Radeon ICD through `VK_DRIVER_FILES`.

# Checked boundary admission follow-up

`SquareBoundaryUse` now admits a square boundary once in host planning and
stores its canonical identity and traversal direction in one private word.
`ChildRequest` carries these witnesses into each resident child. GPU spacing
consumes them directly; the `None => 0` fallback has been removed. Invalid
triangles do not resolve to a request. Empty draw ranges carry valid inactive
topology rather than repeated corner zero.

The release Wasm gate adds all 216 triples over corner IDs 0..5 and an extreme
invalid ID, checking rejection of repeated/out-of-domain corners and agreement
of every admitted identity/direction with the original boundary. The existing
16.86 million warped samples and retained-snapshot checks still pass. Log:
`/laboratory/quilting/scratch/composite-admitted-boundary-tests-20260907.log`.

Release browser manifest `fe-render-c97b1107ab64a99b.json` rendered the default
fan and mixed outer LoDs 1/6/6/1. The latter drew 24,396 vertices. Inspection of
the 1,036,324-pixel interior found zero exposed-background pixels and zero red
fold-marker pixels. Large sparse-edge triangles remain; this is not a claim
of satisfactory density interpolation or a general mesh-coverage proof.

Cost tradeoff: emitted WGSL 158,987 -> 132,381 bytes; parent Wasm
358,989 -> 397,602 bytes. No measured runtime-speed claim. The development
server was restarted after its last-good-build retention failed to pick up the
corrected source; the inspected manifest is the corrected one.

Upstream limitation encountered: a boolean field in a host scene argument
failed compute lowering with `kernel arg 25 is i1; boolean storage-buffer
arguments are unsupported`. The boundary representation deliberately uses one
word for twenty directed identities, but general typed boolean ABI support
remains unfinished upstream. This representation is not evidence that the
general limitation has been fixed.

# Shared orientation follow-up

The composition now uses `quilting_patch::sampling_warp::OrientedIntervalMap`
over `InverseCumulativeMap`, instead of a demo-local reversal implementation.
The same generic law accepts owned Wasm CDFs and GPU storage views. Canonical
placement reverses the input only; local interior extension also reverses the
output. These are deliberately distinct operations.

Release Wasm acceptance checks all 6,561 outer-LoD combinations, all ten
undirected boundaries, and 257 dyadic samples per boundary (16,861,770 sample
checks), with the geometric center fixed at (0.17, 0.83). Reversed canonical
evaluations agree exactly; local reversal, endpoints and strict monotonicity
pass. This tests the represented CDF, not exact continuous uniformity or all
GPU/CPU positions. Evidence:
`/laboratory/quilting/scratch/composite-oriented-boundary-tests-20260907.log`.

The release development build compiled and served manifest
`fe-render-e55e0259d2528602.json`. Chrome page 65 entered live mode with 4,296
vertices and no console errors/warnings. Canvas inspection showed the complete
default square without visible holes or red fold markers. Total emitted WGSL
was 158,987 bytes (previous checkpoint 196,297); this is a size observation,
not an execution-speed measurement or a complete causal bloat diagnosis.

At that checkpoint, typed boundary admission was still pending; it is addressed
above. Nonidentity GPU/CPU position comparisons and broader mesh-distortion
checks remain pending.
