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
| inverted paused horse, PBR | 191,479 | 191,091 | 769 ppm | 1,198 ppm | 1,339 ppm |
| inverted paused horse, spheroidal focus | 240,893 | 241,129 | 1,019 ppm | 729 ppm | 0 ppm |

The normals silhouette differs at one pixel out of 504,000. The inverted PBR
and focus cases remain below 0.14% of pixels with a large RGB delta. This rules
out a backend-wide vertical flip and provides a regression gate that would
reject one by orders of magnitude.

## Decisions

- `quilting-core` owns the incumbent opaque clear color; WebGL2, WebGPU,
  focus composition, and evidence targets consume that one policy.
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
node scripts/audit-webgpu-image-parity.mjs
```

Set `HYPERSCOPE_BACKEND_EVIDENCE_STYLE=pbr` to exercise PBR, and use
`HYPERSCOPE_BACKEND_EVIDENCE_FOCUS=preserve` with a focus-bearing URL to retain
the spheroidal focus postprocess. The three ppm limits can be tightened through
the `HYPERSCOPE_MAX_*` environment variables.
