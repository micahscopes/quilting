# Exact WebGL entry-binding provenance

Date: 2026-08-30

## Outcome

The WebGL program memo no longer keys a selected shader entry with a shared
superset of every resource declared in its source module. Each immutable
binding plan now retains:

- the original WGSL variable name and `(group, binding, stage)`;
- whether an opaque resource originated as a sampled texture or sampler;
- the emitted GLSL uniform/block name; and
- the WebGL UBO binding point or texture unit applied after link.

`quilting-shaders` reflects resources reachable from one composed Naga entry
point. Renderer tests compare that reflection with every primary WebGL program
and both transform-feedback programs. The browser-only blur and highlight
programs use the same canonical source-interface projection and reflection
check. A changed source coordinate or stage is therefore a program-cache miss
even if Naga happens to emit the same mutable GL name.

## Cruft removed

The old shared vertex plan attached `face_data_tex` and
`suppressed_face_tex` to ordinary rendering even though `vs_main` cannot reach
them. Entry-specific plans now retain:

- render: frame/joint UBOs plus skinning and morph textures;
- current-pose preparation: frame/joint UBOs plus skinning, morph, and source
  face textures; and
- visibility: only the frame UBO and suppression mask.

The selected legacy PBR entry also cannot reach its declared sheen-E LUT or
blurred-scene texture pairs. Those inactive assignments were removed from
program identity. Runtime drawing and shader coordinates were not changed;
WebGL previously ignored all of these absent uniforms after link.

## Portability boundary

Exact provenance reports these cross-stage coordinate collisions:

| Program | Conflicting `(group, binding)` slots |
|---|---|
| matcap | `(0, 1)` |
| wire | `(0, 1)` |
| PBR | `(0, 1)`, `(0, 2)`, `(0, 3)` |
| normals, stretch, pick | none |
| preparation, visibility | none |
| blur, highlight | none |

No automatic WebGPU layout is synthesized from an incompatible legacy plan.
The next portable-binding step is to renumber distinct logical resources and
attach complete buffer/texture/sampler policy, with the existing WebGL plan
retained until browser image parity is demonstrated.

## Verification

- 26 `quilting-shaders` library tests passed, including reachable-resource
  filtering and missing-entry failure;
- 79 `quilting-renderer` library tests passed, including reflection equality,
  exact conflict coordinates, canonical ordering, duplicate-site rejection,
  compute-stage rejection, and transform-feedback interfaces;
- the exact `wasm32-unknown-unknown` check for `quilting-wasm` with
  `leptos-ui,webgpu-backend` and tests passed, including the auxiliary program
  descriptors;
- no Trunk server, browser, WASM optimizer, or release build was launched.
