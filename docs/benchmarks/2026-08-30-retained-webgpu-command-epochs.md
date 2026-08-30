# Retained WebGPU command epochs

Date: 2026-08-30

## Outcome

High-rate render frames can now share the exact command allocation of a
low-rate `RenderCommandPlan`. `RenderFrame::from_command_plan` clones one
`Arc<[RenderCommand]>` and retains the plan as private provenance. The ordinary
`RenderFrame::execution` method detects that provenance and proves the frame
still belongs to the plan by checking its validated scene allocation and
revision, command-presence key, and command `Arc` identity. It performs no scene
rescan and constructs no expected command vector. `RenderFrame::build` creates
an unplanned frame whose call to the same method remains the independent
full-validation compatibility oracle.

WebGPU's coherent `PatchRenderScene` now retains a `ValidatedRenderScene`
instead of an independently trusted snapshot. Scene upload validates before
publishing resources, and an in-place update validates its candidate before
issuing queue writes. The scene can create an exact command plan, and the
prepared-patch encoder accepts only a plan whose snapshot allocation is the
same retained epoch. The existing snapshot/frame encoder remains the rollback
path. Focus-PBR, resident-root, and adaptive-overlay composition now enter
through the same automatic admission seam. Root and adaptive paths calculate
logical submission evidence before encoding device work, instead of discovering
an invalid frame only after queue submission.

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
- `quilting-webgpu --tests` passes a one-job native compile check in 21.34
  seconds;
- the retained-plan native workload compiles and starts, but skips raster
  execution because no native adapter is available in this shell;
- the exact `wasm32-unknown-unknown` check for `quilting-wasm` with
  `leptos-ui,webgpu-backend` and tests passes in 1 minute 53 seconds at one
  low-priority Cargo job after briefly waiting on another workspace's shared
  package-cache lock;
- no Trunk server, browser, `wasm-pack`, or `wasm-opt` process was launched.

## Remaining boundary

The retained frame provenance now reaches the coherent prepared-patch diagnostic
family, focus-PBR scene/raw-field and postprocess, resident roots, and sparse
adaptive overlays without parallel plan-specific APIs. Browser promotion is
still gated on user-run image and workload parity; the unplanned
`RenderFrame::build` path remains the rollback oracle until that evidence is
recorded.
