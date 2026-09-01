# WebGPU image orientation and clear parity — 2026-09-01

## Question

The live WebGPU horse appeared upside down and its background looked different
from WebGL2. Logical draw-count parity alone could not distinguish an image
orientation bug from a camera/conformal-state difference, so this checkpoint
adds a same-frame image oracle.

## Method

`scripts/audit-webgpu-image-parity.mjs` opens an isolated Chrome target, pauses
animation, waits for a quiescent Rust LoD submission, and asks the application
to render one logical frame through both WebGL2 and WebGPU. It then verifies the
viewport, draw workload, shared clear policy, surface alpha mode, silhouette,
and RGB error before restoring the user's original tab.

The evidence buffers preserve the visible clear RGB while using alpha as a
coverage mask. Alpha is therefore not compared as presentation color: the
WebGL2 default framebuffer is opaque, while offscreen WebGPU evidence retains
fragment/fade coverage.

## Results

All runs used a 700 × 720 viewport and an `Rgba8Unorm` / `Opaque` WebGPU
surface. WebGL2 and WebGPU submitted identical draw calls, instances,
triangles, lines, and draw-sequence hashes.

| Scene | WebGL2 covered px | WebGPU covered px | coverage mismatch | RGB mean error | RGB pixels with delta > 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| paused horse, normals | 13,755 | 13,756 | 1 ppm | 5 ppm | 5 ppm |
| paused horse, PBR | 14,089 | 13,756 | 660 ppm | 285 ppm | 1,551 ppm |
| inverted paused horse, PBR | 191,479 | 191,091 | 769 ppm | 1,198 ppm | 1,339 ppm |
| inverted paused horse, spheroidal focus | 240,893 | 241,129 | 1,019 ppm | 729 ppm | 0 ppm |
| paused horse, PBR (final gate) | — | — | 662 ppm | 286 ppm | 1,545 ppm |
| paused horse, spheroidal focus (final gate) | — | — | 242 ppm | 145 ppm | 0 ppm |

The normals silhouette differs at one pixel out of 504,000. The inverted PBR
and focus cases remain below 0.14% of pixels with a large RGB delta. This rules
out a backend-wide vertical flip and provides a regression gate that would
reject one by orders of magnitude.

### Exact clear quantization

The first ordinary-PBR comparison exposed a smaller version of the reported
background difference. The shared blue clear value was `0.3`, exactly halfway
between RGBA8 codes 76 and 77. WebGL2 selected 76 while WebGPU selected 77, so
976,363 ppm of pixels differed even though nearly all of that difference was
one blue code. The canonical value is now the exact code point `77 / 255`.
The same scene subsequently measured 4,317 ppm mismatched pixels and 285 ppm
mean RGB error; empty background pixels agree exactly.

### Same-context classifier GL state

Repeated PBR evidence also found an intermittent stale `GL_INVALID_OPERATION`
before readback. A call-level isolated-Chromium trace localized it to the first
`drawArrays(POINTS)` in `LodCompute::compute_lods`. Optional joint and morph
sampler slots skipped `bindTexture` when their texture was absent, so a slot
could retain the pass-one render target and form an invalid framebuffer
feedback loop. The classifier now owns every sampler slot explicitly, binding
`None` for absent pose textures. One traced startup and three fresh untraced
startups retained no GL error after the fix.

### Static post-render readiness

A paused 94,628-face chess scene exposed a readiness-ordering bug that the
small horse did not: the render call published one valid WebGPU PBR frame and
its material resources, but `presentationArmed` had been computed from the
pre-render diagnostics snapshot. With no animation or input, that admitted
frame could remain hidden behind WebGL2 indefinitely.

Only the render loop may now ask `refreshWebGpuBackendDiagnostics(true)` to
re-evaluate support from the post-render residency snapshot before applying the
presentation policy. Inspector/audit refreshes retain the default `false`, so
observing diagnostics cannot seize presentation authority. The same chess
route subsequently reached two admitted WebGPU PBR frames with 94,628 resident
faces, 94,626 visible instances, exact logical workload parity, and zero frame
failure without camera input.

That successful cut exposed a separate textured-PBR parity gap: the chess
comparison measured 1,444 ppm silhouette mismatch but 31,840 ppm mean RGB
error and 105,468 ppm pixels with RGB delta above 16. Large regions sampled
black while the incumbent remained textured. The asset is three opaque,
double-sided materials over nine resident 4096 × 4096 textures, so this is not
the known transmission/blend exclusion. Portable-atlas mip/filtering and slot
sampling are the next diagnostic boundary; the result is recorded as a blocker,
not relaxed into the image tolerance. Further live diagnosis paused when a
separate project began intensive WebGPU testing against the shared browser.

## Decisions

- `quilting-core` owns the incumbent opaque clear color; WebGL2, WebGPU,
  focus composition, and evidence targets consume that one RGBA8-exact policy.
- WebGPU explicitly prefers an opaque surface alpha mode when the adapter
  supports it, with a deterministic supported-mode fallback.
- Image evidence waits for a quiet logical submission. Before this fence, the
  same URL could be sampled at either the initial 9,000-triangle frame or the
  asynchronously reconciled 70,948-triangle frame.
- The gate permits 5,000 ppm (0.5%) for silhouette and RGB metrics. This is
  intentionally tolerant of edge rasterization and floating-point shading
  differences while remaining far below a flipped or materially displaced
  image.

## Reproduction

```sh
node scripts/smoke-shared-render-clear.mjs
HYPERSCOPE_WEBGL_ERROR_URL='http://127.0.0.1:8888/?gfx=webgpu&lodimpl=rust' \
  node scripts/audit-webgl-error-state.mjs
node scripts/audit-webgpu-image-parity.mjs
```

Set `HYPERSCOPE_BACKEND_EVIDENCE_STYLE=pbr` to exercise PBR, and use
`HYPERSCOPE_BACKEND_EVIDENCE_FOCUS=preserve` with a focus-bearing URL to retain
the spheroidal focus postprocess. The three ppm limits can be tightened through
the `HYPERSCOPE_MAX_*` environment variables. Large assets can opt into a
longer bounded wait with `HYPERSCOPE_BACKEND_EVIDENCE_TIMEOUT_MS`; diagnostic
comparison of incumbent and device LOD paths can select
`HYPERSCOPE_BACKEND_EVIDENCE_LOD_IMPL=js|shadow|rust` without changing the
application's route defaults.
