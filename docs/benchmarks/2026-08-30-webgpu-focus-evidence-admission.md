# One-shot WebGPU focus evidence admission

Date: 2026-08-30

## Outcome

The explicit backend image oracle can now request a focus-enabled PBR frame.
The incumbent WebGL2 renderer captures its final composed framebuffer while the
headless WebGPU backend renders the same Rust `RenderFrame` packet into its
offscreen composed target. Existing frame identity, viewport, logical
submission, and RGBA8 image comparison checks remain unchanged.

Basic and focus PBR admission are intentionally disjoint:

- a basic PBR frame must not contain a focus packet;
- a focus PBR frame must contain a valid focus packet; and
- both paths require the same exact opaque material, texture, and environment
  subset before dispatch.

## First-use lifecycle

The request gate checks only prerequisites that must predate the frame: a
headless WebGPU device, retained focus pipelines, and environment residency.
It does not require an already extracted focus scene. That would deadlock a
fresh PBR-only startup because scene extraction and resident root/overlay
publication occur inside the requested frame. The frame boundary still demands
complete focus scene residency before it submits any GPU work.

## Promotion state

This opens only the deliberate one-shot evidence route. It does not advertise
focus presentation support to the browser UI and does not make WebGPU the
authoritative renderer. A human-triggered WebGL2/WebGPU comparison must pass on
a representative focus scene before the presentation capability is promoted.
No browser was launched, reloaded, or controlled as part of this cut.

## Verification

- exact `wasm32-unknown-unknown` check for `quilting-wasm` with
  `leptos-ui,webgpu-backend` passed;
- the 88-spec Hyperscope route/source oracle passed; and
- the Rust focus/basic PBR capability-disjointness test remains green in the
  native WebGPU suite.
