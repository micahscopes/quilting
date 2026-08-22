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

## Unresolved before public asset redistribution

### Animated horse

Repository history traces `horse.glb` to the three.js Horse example. The
three.js example credits the model to [Mirada](https://mirada.com/) from
[ROME](https://rome.mrdoob.com/), and the local file is a normalized,
GLB-converted derivative. The GLB itself contains only a
`THREE.GLTFExporter` generator tag. Although the three.js code repository is
MIT-licensed, no explicit license for this model asset has been located in the
file, this repository's history, or the upstream example page.

Do not treat the horse as cleared for a public downloadable bundle. Obtain the
model license or replace it with an explicitly redistributable animated
fixture. Local demonstration and code testing are a separate release decision.

### Matcaps

The following images contain no author, source, copyright, or license metadata,
and their introducing commit records no provenance:

- `matcaps/aqua.png`
- `matcaps/citric-acid.png`
- `matcaps/golden-soft.png`
- `matcaps/soft-studio.png`

Clear or replace them before publishing a bundle that contains them. The
renderer has a procedural matcap fallback, so these files are not necessary to
demonstrate the rendering technique.

## Untracked local assets

`local-glbs/` is intentionally ignored except for `.gitkeep`. Files copied from
that directory into `dist/` are not covered by this record and must not enter a
release archive without their own provenance and redistribution review.
