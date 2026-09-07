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
