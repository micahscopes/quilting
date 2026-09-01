# WebGPU device-loss epoch

The browser WebGPU backend now treats a lost `GPUDevice` as a terminal resource
epoch rather than a frame error. Every requested device receives a monotonic
epoch. Its loss callback can invalidate only that epoch, so a delayed callback
from an abandoned device cannot retire a newer replacement.

One accepted loss transition atomically releases the presentation surface,
offscreen and pick targets, atlas, PBR textures and environment, prepared model,
scene, focus resources, resident roots, and all pipelines before releasing the
device. It also clears current-frame, presentation-admission, pose, visibility,
and pick witnesses. Semantic application and scene authority remain outside
this device-local residency and WebGL2 remains the incumbent rollback renderer.

Diagnostics expose the device epoch, accepted losses, completed recoveries,
stale callbacks, stable reason/message, and explicit recovery eligibility. A
one-shot browser event schedules presentation-policy refresh on the next
microtask, causing `lost` to enter the same truthful fallback phase as `failed`.
The adapter deliberately does not request a replacement automatically: recovery
must replay the selected canvas and all asset residency, and repeated allocation
under GPU contention would otherwise thrash or falsely present an empty scene.

Static and CPU gates:

- all 34 `quilting-webgpu` unit tests pass;
- all eight `hyperscope-app` presentation-policy tests pass;
- the browser inline module passes Node syntax checking;
- `quilting-wasm` checks for `wasm32-unknown-unknown` with
  `leptos-ui,webgpu-backend`;
- `quilting-webgpu` passes Clippy with only the repository's explicitly
  acknowledged baseline lint classes allowed;
- the WASM unit-test artifact compiles, although executing its new lifecycle
  tests locally is blocked by the installed `wasm-bindgen-test-runner` schema
  version (runner 0.2.117 versus project 0.2.108).

No WebGPU adapter, device, surface, Chrome session, or native conformance run
was created for this gate because another workload was intentionally exercising
the shared GPU. Live loss injection and full asset replay remain a later,
isolated-context validation.
