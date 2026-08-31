# WebGPU retained focus schedule and uniform table

## Question

Why did every focused WebGPU frame rebuild the same fullscreen-pass schedule
and upload the same padded uniform table when camera motion changed only the
scene rendered into the focus MRT?

## Finding

`FocusPostprocessTarget` already retained every intermediate texture, sampler,
bind group, pipeline, and uniform buffer. The encoder nevertheless rebuilt its
`PlannedFocusPass` vector, cleared and repopulated the aligned CPU uniform
words, and called `queue.write_buffer` on every frame.

The schedule depends only on the target dimensions and the validated
`FocusPostprocessPacket`. It does not depend on camera matrices, animation
pose, rendered scene pixels, or the output texture view.

Each target now retains an exact bitwise key for those schedule inputs plus the
planned passes and padded uniform words. An exact hit records the same render
passes against the current textures but performs no plan rebuild, CPU table
rewrite, or queue upload. A changed focus policy rebuilds and publishes the
complete table before it becomes the retained witness.

## Bounded traffic result

Let `P` be the scheduled fullscreen-pass count and `A` the device's aligned
uniform stride. The current default spheroidal policy has `P = 8`; a common
WebGPU alignment has `A = 256` bytes.

| Focus frame | Before | After |
| --- | ---: | ---: |
| First target/policy frame | `P × A` | `P × A` |
| Camera or animation only | `P × A` | `0` |
| Exact repeated policy | `P × A` | `0` |
| Focus-policy edit | `P × A` | `P × A` |

The fullscreen render passes still run because their source pixels change.
This checkpoint removes only redundant CPU planning and host-to-device uniform
traffic; it does not claim to cache a command buffer or postprocessed image.

`FocusPostprocessEncoding::plan_reused` and its actual
`uniform_upload_bytes`, plus target-lifetime
`FocusPostprocessMemoDiagnostics`, expose the boundary directly. Browser
diagnostics publish these as `focusPlanBuilds`, `focusPlanReuses`,
`focusUniformUploads`, `focusUniformReuses`, and
`focusUniformUploadBytes`.

## Verification

Zero-build source oracle:

```sh
node scripts/smoke-webgpu-focus-plan-memo.mjs
```

Compiler-only gates use one low-priority Cargo job:

```sh
cargo check -p quilting-webgpu --tests
cargo check -p quilting-wasm --target wasm32-unknown-unknown \
  --features leptos-ui,webgpu-backend
```

No linked WebGPU test binary, Trunk server, `wasm-pack`, binding generation, or
`wasm-opt` is involved.

## Live browser evidence

The subsequent live-PBR admission checkpoint made this path observable on the
selected browser surface. Shared Chromium ran the animated horse in PBR with
spheroidal focus for 15 seconds and reported 417 submitted WebGPU/focus frames,
zero frame failures, and no warning/error console entries. The retained policy
built once and reused 416 times; its uniform table uploaded once (2,048 bytes)
and reused 416 times. Camera/inversion motion occurred during the sample, so
the source image and scene state changed while the focus schedule remained
exactly stable. This is lifecycle/traffic evidence, not a frame-rate benchmark.
