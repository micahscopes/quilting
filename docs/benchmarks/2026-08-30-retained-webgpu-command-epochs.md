# Retained WebGPU command epochs

Date: 2026-08-30

## Outcome

High-rate render frames can now share the exact command allocation of a
low-rate `RenderCommandPlan`. `RenderFrame::from_command_plan` clones one
`Arc<[RenderCommand]>`; `execution_with_command_plan` proves the frame still
belongs to the plan by checking its validated scene revision, command-presence
key, and `Arc` identity. It performs no scene validation and constructs no
expected command vector. The original `RenderFrame::build` and
`RenderFrame::execution` remain available as the independent compatibility
oracle.

WebGPU's coherent `PatchRenderScene` now retains a `ValidatedRenderScene`
instead of an independently trusted snapshot. Scene upload validates before
publishing resources, and an in-place update validates its candidate before
issuing queue writes. The scene can create an exact command plan, and the
prepared-patch encoder accepts only a plan whose snapshot allocation is the
same retained epoch. The existing snapshot/frame encoder remains the rollback
path.

This removes two warm-frame costs from the planned prepared-patch path:

- validating immutable materials, batches, leaves, and suppression topology;
- allocating and filling the expected command vector before submission.

It does not claim GPU image parity. The local native test executable had no
usable graphics adapter and reported its existing explicit skip.

## Verification

- all 263 `quilting-core` library tests pass, including command-allocation
  identity, uniform-only frame changes, stale-plan rejection, and legacy
  mutation rejection;
- all 23 `quilting-webgpu` library tests pass;
- `quilting-renderer` and `quilting-webgpu --tests` pass a one-job native
  compile check;
- the retained-plan native workload compiles and starts, but skips raster
  execution because no native adapter is available in this shell;
- the exact `wasm32-unknown-unknown` check for `quilting-wasm` with
  `leptos-ui,webgpu-backend` and tests passes in 28.32 seconds at one Cargo job;
- no Trunk server, browser, `wasm-pack`, or `wasm-opt` process was launched.

## Remaining boundary

The planned encoder currently covers the coherent prepared-patch diagnostic
family, including composite matcap/wire and selection highlight ordering.
Focus-PBR, resident-root, and adaptive-overlay WebGPU entry points still admit
legacy frames directly. They should consume the same retained epoch in small
steps, retaining their legacy calls until browser image/workload parity is
recorded on the user-run server.
