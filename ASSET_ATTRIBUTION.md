# Bundled asset attribution and release status

Quilting source code is licensed under `MIT OR Apache-2.0`; see
[`LICENSE-MIT`](LICENSE-MIT) and [`LICENSE-APACHE`](LICENSE-APACHE). This file
tracks separately authored data bundled by the Hyperscope demo. It is a
provenance record, not legal advice.

## Cleared bundled assets

### Ant

`ant.glb` embeds the following source metadata:

- **ANT**, by
  [MAXDESIGN-3D](https://sketchfab.com/MAXDESIGN)
- Source:
  [Sketchfab model `dab7080251674ef98fc83b7604be2ffc`](https://sketchfab.com/3d-models/ant-dab7080251674ef98fc83b7604be2ffc)
- License:
  [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

Keep this attribution with redistributed copies.

### Environment maps

The environment maps are Poly Haven HDRIs. Poly Haven publishes its assets
under CC0 and permits redistribution; attribution is optional and appreciated.
See the [Poly Haven asset license](https://polyhaven.com/license).

- `envmaps/rosendal_plains_1_1k.hdr`: [Rosendal Plains 1](https://polyhaven.com/a/rosendal_plains_1), photography by Dimitrios Savva and processing by Jarod Guest.
- `envmaps/rogland_clear_night_2k.hdr`: [Rogland Clear Night](https://polyhaven.com/a/rogland_clear_night), by Greg Zaal.
- `envmaps/ticknock_04_1k.hdr`: [Ticknock 04](https://polyhaven.com/a/ticknock_04), by Savva Zakharov.

### Hyperscape Blender fixture

`examples/hyperscape-blender-demo.glb` is a project-authored test and
presentation fixture. It is covered by the repository's `MIT OR Apache-2.0`
license.

## Known restricted assets

### Animated horse

`horse.glb` is the three.js Horse example model, credited to
[Mirada](https://mirada.com/) for
[ROME — 3 Dreams of Black](https://rome.mrdoob.com/). The repository copy is
byte-for-byte identical to three.js's
[`Horse.glb` at commit `db75a3b`](https://github.com/mrdoob/three.js/blob/db75a3b38f940ffdd367bba42db943d0bb9ba4a5/examples/models/gltf/Horse.glb):

```text
SHA-256  bebaa4a60ba373317e25bf20f049f26ad0f5c86d4731ab67d46eb8c93c920947
```

The model's earlier ROME source records the asset license as
[Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-nc-sa/3.0/).
That statement is preserved in
[`horse.js` before three.js converted it to GLB](https://github.com/mrdoob/three.js/blob/4a4123206826429d154d1df9e9ef74560ae13dcd/examples/models/animated/horse.js),
and three.js continues to show the Mirada/ROME credit on its
[Horse example](https://github.com/mrdoob/three.js/blob/dev/examples/webgl_morphtargets_horse.html).

This license applies to the horse asset, not to Quilting's source code. Keep
the attribution and license link with redistributed copies, do not include it
in a commercial-use or MIT/Apache-only asset bundle, and follow the license's
ShareAlike terms for adaptations. The strict preflight remains red until a
release deliberately chooses a compatible noncommercial mixed-license bundle
or replaces/excludes this model.

## Unresolved before public asset redistribution

### Matcaps

The following images contain no author, source, copyright, or license metadata,
and their introducing commit records no provenance. The original Claude
session records that these files already existed when the user asked the agent
to add a dropdown for the images in the `matcaps` folder; the agent only wired
them into the renderer. No earlier local session or shell-history entry found
by the audit records how they were created or obtained. They are therefore
user-supplied assets with unrecorded source/license, not agent-generated assets:

- `matcaps/aqua.png` — SHA-256
  `c004b5b5c55a1fc5663c2b75168301c4ee3b269858299aed9b9eda9c98792483`
- `matcaps/citric-acid.png` — SHA-256
  `10993506abf888aa339b174a0862bd3229d8bba02853977db0d8974c0f698fe7`
- `matcaps/golden-soft.png` — SHA-256
  `c53d27ce80180c20e88f451e6346fc86f56ea7abdbb6d82131a85011d91b651e`
- `matcaps/soft-studio.png` — SHA-256
  `04de93e8794c3d45161745044e354b5ee4ecd8a639e846f7e6381c29057f6ed6`

Clear or replace them before publishing a bundle that contains them. The
renderer has a procedural matcap fallback, so these files are not necessary to
demonstrate the rendering technique.

## Untracked local assets

`local-glbs/` is intentionally ignored except for `.gitkeep`. Files copied from
that directory into `dist/` are not covered by this record and must not enter a
release archive without their own provenance and redistribution review.
