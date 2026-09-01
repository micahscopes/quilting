# WebGPU composed-scene LOD prefix refresh

Date: 2026-09-01

## Question

Can the WebGPU presentation keep static secondary assets' reconciled LOD on
the device while an animated primary asset receives a new pose and LOD epoch?

The prior path reclassified the complete composed model whenever the horse
advanced. That did not cross the CPU/GPU boundary, but it still repeated
classifier and reconciliation work for static assets.

## Contract

- Device authority starts only after a complete, visible scene epoch.
- Camera, projection, transform, tessellation-policy, and scene changes still
  require a complete epoch.
- An animation-only refresh may classify a strict source-face prefix only when
  that prefix is closed under prepared face adjacency.
- A partial refresh requires a complete resident baseline and unchanged edge
  grading.
- Classifier and reconciliation remain one GPU submission with no readback.
- The untouched suffix remains part of the same complete resident result.

## Evidence

`scripts/audit-webgpu-lod-prefix.mjs` opened an isolated Chromium target on the
final two-asset presentation cue with Rust scene, animation, LOD, and WebGPU
authority. The composed prepared model contained 4,432 faces; the animated
horse occupied the topology-closed first 984 faces.

After the authored 1.2-second camera/layer transition settled, a one-second
sample observed:

| Counter | Before | After |
| --- | ---: | ---: |
| presented WebGPU frames | 6 | 7 |
| all device LOD dispatches | 12 | 13 |
| complete-scene dispatches | 8 | 8 |
| primary-prefix dispatches | 4 | 5 |
| faces classified by the last dispatch | 984 | 984 |

The steady animation dispatch therefore visited 984 rather than 4,432 faces,
a 77.8% reduction in classified face records for this scene. The audit also
reported five resident assets, no WebGPU frame failure, no LOD or pose error,
no browser/Rust authority mismatch, and no console error.

The native GPU suffix-retention test compiles in
`crates/quilting-webgpu/tests/native_lod.rs`. This workstation's non-browser
test process did not expose a native graphics adapter during this run, so the
live runtime proof above uses Chromium's actual WebGPU device.

## Reproduction

With the user-owned Trunk server and debugging Chromium already running:

```sh
node scripts/smoke-webgpu-device-lod-authority.mjs
node scripts/smoke-webgpu-lod-single-submit.mjs
node scripts/audit-webgpu-lod-prefix.mjs
```

The browser audit creates and closes its own target, then reactivates the
pre-existing Hyperscope tab.
