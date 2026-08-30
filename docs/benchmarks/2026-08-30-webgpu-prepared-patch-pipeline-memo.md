# WebGPU prepared-patch pipeline memo

## Problem

The sparse adaptive overlay, ordinary diagnostic rendering, and focus graph
could request identical prepared-patch graphics families repeatedly. Shader
modules were already retained, but each request still recreated bind-group
layouts, pipeline layouts, and every requested style/winding pipeline.

## Boundary

`prepared_patch_pipeline_descriptors` now constructs the complete immutable
family before any WebGPU effect. The descriptor vector names:

- the prepared frame, record, atlas, visibility, batch, and material buffers;
- the six ordinary PBR texture/sampler channels and environment group;
- vertex layout, topology, winding, culling, depth/write, blend, and
  multisample state;
- ordinary color and optional focus raw-field attachments; and
- the requested style order, including one reusable line pipeline and two
  winding variants for each triangle pass.

`LodClassifierDevice` retains the lowered family in a device-scoped
`DeviceMemo`. An exact hit clones reference-counted `wgpu` handles and does not
revisit shader lowering. A failed construction is not published. Texture
formats outside the backend-neutral portable set use the prior uncached path,
so an unsupported format is never disguised by an incomplete memo key.

## Evidence

- 22 `quilting-webgpu` library tests pass;
- pure tests cover the 13-pipeline full family, PBR and diagnostic layout
  separation, the twelve ordinary material-texture bindings, focus MRT,
  single wire descriptor, highlight depth policy, invalid style/focus/sample
  requests, and format-sensitive memo identity;
- shared lowering tests prove the complete vertex/fixed-state/target mapping
  and reject noncontiguous vertex slots before a device effect;
- one semantic-plan test fixes the exact PBR, normals, matcap, LOD, stretch,
  wire, and highlight order; its 13 descriptor variants; focus entry point;
  line reuse; and highlight depth policy for both descriptors and runtime;
- the native conformance test compiles a one-miss/one-hit prepared focus-family
  assertion and verifies that the hit does not revisit shader lowering (the
  GPU body skips when the host exposes no adapter); and
- the exact Hyperscope WASM feature set type-checks with two Cargo jobs and
  without invoking Trunk, `wasm-pack`, or `wasm-opt`.

The serialized browser diagnostic packet now exposes prepared-patch hit, miss,
failed-creation, invalidation, and resident-entry counters beside the shader,
focus, and resident-root memo counters.

## Remaining work

This establishes functional identity and resource reuse; it is not visual
parity evidence. A user-triggered WebGL2/WebGPU image comparison over focus,
resident roots, and sparse adaptive leaves remains the next promotion gate.
