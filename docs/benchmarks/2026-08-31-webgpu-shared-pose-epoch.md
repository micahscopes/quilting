# WebGPU shared pose epoch

## Question

After camera-only pose reuse was added to the render path, did device LOD
classification still upload the same joint/morph payload as rendering?

## Finding

Yes. Classification published the shared model pose on every LOD dispatch.
Because each new device LOD epoch retired `last_frame_input`, the following
resident render published the same model pose again and copied both vectors
into backend-owned comparison storage.

## Contract

`RenderPoseIdentity` now consists of the structural asset revision plus a
lossless packing of animation continuity epoch and local pose revision. Static
geometry uses pose revision zero. The classifier and renderer receive this
same identity and borrow the same renderer-owned pose payload.

The retained device state distinguishes:

- `Publish`: write shared joint/morph buffers and preparation-local uniforms;
- `PublishPreparation`: retain the shared pose and initialize only a newly
  retained patch/root/overlay uniform family; and
- `Reuse`: write neither kind of pose state.

Failed full publications clear their identity witness before writing, so a
later classifier cannot reuse a buffer whose final contents are uncertain.
Scene replacement invalidates only preparation-local witnesses; model or atlas
replacement invalidates the shared pose witness as well.

## Bounded traffic result

Let `J` and `M` mean the optional joint and morph queue writes (zero to two
writes total), and let `P` mean the small preparation-local joint-count writes.
Classifier dispatch metrics and subject rows remain necessary and are not
included here.

| Steady-state case | Before | After |
| --- | ---: | ---: |
| Camera-only LOD dispatch + resident render | `2(J+M) + P` | `0` pose writes |
| New animated pose + resident render | `2(J+M) + P` | `J+M` |
| Same pose, newly retained scene family | `J+M + P` | `P` |

The backend also no longer copies joint matrices or padded morph weights into
`last_*` comparison vectors. Morph padding is performed only for a real dynamic
publication and reuses one scratch allocation.

Runtime diagnostics expose `classifierPoseUploads/Reuses`,
`fallbackPoseUploads/Initializations/Reuses`,
`residentPoseUploads/Initializations/Reuses`, the exact retained pose identity,
and both preparation-ready witnesses.

## Verification

The zero-build source oracle is:

```sh
node scripts/smoke-webgpu-pose-upload-memo.mjs
```

Bounded checks used one low-priority Cargo job:

```sh
cargo check -p quilting-webgpu --lib
cargo check -p quilting-webgpu --tests
cargo check -p quilting-wasm --target wasm32-unknown-unknown \
  --features leptos-ui,webgpu-backend
cargo test -p quilting-core timed_pose_identity_includes_continuity_epoch --lib
```

No Trunk server, `wasm-pack`, binding generation, or `wasm-opt` was launched.
