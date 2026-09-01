# Textured resident WebGPU PBR — 2026-08-30

## Decision

The adaptive WebGPU layer already preserves one indirect draw boundary per
extracted material batch. It now binds that batch's resolved texture group
instead of slot zero and admits PBR only when every referenced image plus the
environment epoch is resident.

The direct source-root layer intentionally retains coarser atlas/permutation
parity buckets. It now resolves each root domain's material through one
baseline-WebGPU paged `texture_2d_array` atlas plus device-resident texture
descriptors and material-to-texture records. Arbitrary fully resident opaque
material identities can therefore share a GPU-driven atlas/parity bucket
without repeating vertices, splitting draws on the CPU, requiring native-only
binding arrays, or substituting missing images.

## Evidence

The native conformance scene now contains distinct untextured and blue-textured
resident root domains under one device-generated geometry plan. Its adaptive
replacement uses the untextured material, leaving the textured source root to
prove the portable resident shader path. The test asserts exact per-material
residency, executes both layers in one coherent submission, and requires the
readback's blue-channel energy to exceed its red-channel energy by more than
two-to-one. The latter would fail against the old material-zero placeholder.

The strict `quilting-webgpu` Clippy gate and the required Radeon 780M RADV
Vulkan `native_lod` test pass. Browser promotion remains opt-in and still uses
the shared capability predicates; live Chrome parity is a separate gate.

## Capability-gated presentation cut

The browser presentation path no longer treats every PBR request as
shadow-only. A retained scene may present PBR only when the authored basic-PBR
subset is supported, every referenced texture is exact, and the environment
bindings are resident. The WASM diagnostics expose that coherent fact as
`pbrPresentationReady`; the canvas becomes visible only after an actual PBR
surface frame has also been submitted. Diagnostic styles remain
unconditionally supported by the existing static predicate.

The 2026-08-31 browser regression found and removed a circular first-frame
gate: browser admission had required an already-presented PBR frame, while the
renderer requested PBR work only for one-shot headless evidence. A selected
live surface now requests capability-gated PBR continuously; a headless
pick/parity device still does not acquire that workload without an explicit
evidence request. Canvas promotion continues to require an actual submitted
frame whose retained style is exactly PBR.

An incumbent-required frame clears the retained presentation witness. This is
important when a previously valid PBR scene later enables unsupported focus
postprocessing: the browser falls back to WebGL2 instead of leaving a stale
WebGPU image above the current frame.

The Radeon conformance test proves both sides of the dynamic predicate before
and after environment residency, then executes the existing multi-material
textured raster proof. Exact WASM and browser-route smokes pass. A live Chrome
visual/parity capture is still required before changing the default graphics
backend.

## Portable multi-material direction

`wgpu 29` marks both dynamically indexed texture binding arrays and
GPU-counted multi-draw as native-only. Its browser implementation expands a
fixed multi-draw count into individual `drawIndexedIndirect` calls. A radix/run
compactor can therefore optimize native execution, but cannot by itself remove
the browser's material draw boundary without a readback or Cartesian command
expansion.

The shared path builds a deterministic paged `texture_2d_array` atlas at asset
upload time. Each stable texture slot retains its exact rectangle, dimensions,
layer, and wrap modes. Native byte uploads and browser `ImageBitmap` uploads
publish both the incumbent individual texture and the portable atlas;
allocation-preserving updates modify both. The resident shader performs exact
clamp, mirrored-repeat, or repeat addressing plus manual bilinear filtering via
`textureLoad`, then indexes the stable material records directly from the root
draw domain. Radeon conformance proves packed byte equality before and after an
in-place update, including sparse texture-table slots, and proves the
multi-material raster result. Exact WASM compilation and live Chrome promotion
remain separate gates.

The retained table reports the allocation actually created: individual and
portable mip counts, atlas extent and layers, occupied/source/allocated texels,
packing utilization, and the portable shader's manual filtering modes. This
first exposed that a fully resident table was still base-mip-only and therefore
was not evidence of minification parity with WebGL.

## Mip-safe minification cut — 2026-09-01

The static chess scene admitted an exact 94,626-instance resident-root workload
but measured 31,840 mean-RGB ppm from the incumbent image. Its nine authored
images are each 4096x4096. WebGL generated complete mip chains and selected them
with trilinear filtering; both WebGPU texture representations retained only
base mip zero. This is a concrete filtering-contract mismatch even though a
fresh live image gate is still required to prove how much of that scene's error
it explains.

The upload path now builds one box-filtered mip chain per source image. Every
mip rectangle is packed independently into the corresponding level of the
portable array atlas, so unrelated images cannot bleed together and long,
non-power-of-two images do not require power-of-two alignment padding. The
portable PBR shader computes the texture footprint from fragment derivatives,
performs exact wrap-aware bilinear reads within each logical mip rectangle, and
blends adjacent levels. Prepared/adaptive material batches and resident roots
now invoke that same portable sampler and bind the same four-resource atlas
table; the former twelve-binding individual texture/sampler layout and its
per-material placeholder bind groups have been removed.

Atlas planning evaluates every viable power-of-two page extent and selects the
lowest complete-mip allocation. For the chess-shaped case—nine 4096x4096
images under an 8192 device limit—it chooses nine exact 4096 layers rather than
three wasteful 8192 layers: 150,994,944 base texels and 201,326,589 texels over
the complete chains, both at 100% packing utilization. Diagnostics expose both
base and full-chain allocation so future context-loss investigations can
separate filtering correctness from residency pressure.

Native conformance retains the same 201,326,589 texels in individual source
textures, for 402,653,178 RGBA8 texels total: 1,610,612,712 bytes (about 1.50
GiB) before driver overhead. Browser upload no longer creates those sources:
each `ImageBitmap` copies directly into its packed mip-zero rectangle and one
reusable half-size scratch target box-filters all lower levels back into the
atlas. Its steady chess table is therefore 201,326,589 atlas texels, or
805,306,356 bytes (about 768 MiB), before driver overhead. The 2048-square
scratch target adds 4,194,304 transient texels, bounding the explicit upload
allocation at 205,520,893 texels, or 822,083,572 bytes (about 784 MiB), instead
of about 1.50 GiB. Diagnostics distinguish authored images from retained
individual images and report the direct-upload strategy, scratch allocation,
estimated upload peak, and allocations that actually survive publication.
Live context-loss validation remains pending.

The pure atlas, mip shader, resident-root shader, and WASM browser-target gates
pass without acquiring a GPU. Native raster/readback and live Chrome chess
parity remain deliberately pending while another project is exercising the
shared WebGPU device.

## Selected-face overlay

An ordinary selected face no longer invalidates an otherwise supported WebGPU
frame. The shared `PatchRenderFrame` carries the source-face selection as a
separate ABI field from material and node identity. Prepared adaptive patches
and source-indexed resident roots then issue one post-style triangle pass over
their existing indexed-indirect ranges. The stable source face survives QB
preparation and dyadic restriction; the highlight fragment entry point rejects
every other face and emits the incumbent cyan half-alpha overlay. Depth testing
remains `LessEqual`, while the overlay does not write depth.

This cut adds neither a face-ID attachment nor CPU map/readback. It does add an
extra indirect draw per resident geometry bucket or adaptive batch while a face
is selected; those attempts are intentionally reported in backend draw-call
diagnostics but remain outside the backend-neutral logical scene submission.
The hardware conformance test proves both prepared and resident command paths,
requires the highlighted resident image hash to differ from the unhighlighted
wire image, and passed on the Radeon 780M RADV Vulkan adapter. The shader ABI
tests, strict focused Clippy gate, and exact `wasm32-unknown-unknown`
`leptos-ui,webgpu-backend` build also pass. Live Chrome visual parity remains a
separate promotion gate.

## Diagnostic image oracle

The opt-in browser comparison is no longer hard-coded to re-render Normals.
For every WebGPU-supported diagnostic style—Matcap, Wire, Matcap+Wire, Normals,
LOD, and Stretch—the WebGL evidence target now renders the exact current Rust
`RenderStyle` and compares its workload and RGBA image with the same staged
WebGPU frame. Basic PBR retains its separate default-framebuffer capture and
dynamic residency gate. Diagnostic selected-face evidence now reruns the
incumbent pick-texture highlight into the same offscreen color target before
readback, so its complete composition is compared with WebGPU's retained
geometry-overlay pass. The incumbent PBR path now runs that same post-style
overlay before evidence capture, so highlighted PBR is covered as well. Focus
postprocessing remains explicitly excluded because it has no WebGPU
implementation yet.
