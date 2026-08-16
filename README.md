# Quilting

Quilting renders triangle meshes under conformal (Möbius) transformations in
real time in the browser. Sphere inversions, reflections, and other
angle-preserving maps of 3-space are applied to a loaded glTF model and
evaluated live, with PBR materials intact.

The interesting constraint is that WebGL2 has no geometry shaders and no
tessellation shaders, so there is no way to manufacture sub-triangle detail on
the GPU at draw time. A Möbius transform bends straight edges into arcs, which
makes a flat triangle exactly the wrong primitive — you need more geometry
precisely where the transform curves space the most, and you need it without a
tessellator.

Quilting's answer is to precompute the tessellation. Every triangle of the
input mesh is stamped with a "quilt patch": a small triangular sub-mesh pulled
from an atlas of pre-built patterns, keyed by the level of detail wanted along
each of its three edges. Because adjacent faces share an edge, they resolve to
the same LOD on that edge, and their patch boundaries line up vertex for
vertex, so the seams close without T-junction cracks. The atlas stores only
sorted LOD triples and recovers the rest by permuting barycentric coordinates,
which keeps it small.

The curvature itself lives in the vertex shader. Faces are drawn as instances
carrying quaternionic Bézier control data, and the Möbius transform enters as
four quaternion coefficients in a uniform block. The shader folds the transform
into the rational Bézier evaluation and derives the normal analytically. The
practical consequence: dragging a Möbius slider changes four uniforms and
nothing else — no re-tessellation, no buffer re-upload, no CPU geometry work
per frame.

The longer-term direction is **Hyperscope**, a production-quality glTF viewer
and environment, with a conformal scene graph and composable glTF elements.

`SPEC.md` documents the design in full: quaternion conventions, the LOD and
hysteresis rules, the instance data layout, atlas construction, and the known
limitations.

The formal/runtime trust boundary for the conformal scene work is documented
in [`docs/conformal-runtime-invariants.md`](docs/conformal-runtime-invariants.md),
and the graph/interchange contract is in
[`docs/conformal-scene-model.md`](docs/conformal-scene-model.md).

## Status

This is a research codebase under active development. Interfaces move without
warning, some subsystems described in `SPEC.md` are historical rather than
shipping (the 4D spacetime slicer was removed), and the renderer has known
rough edges — see §9 of the spec. It is not packaged for downstream use yet.
Two of the crates are already standalone and reusable, though; see the crate
map below.

## Prerequisites

- A Rust toolchain (stable), plus the WebAssembly target:

  ```sh
  rustup target add wasm32-unknown-unknown
  ```

- [`trunk`](https://trunkrs.dev/) to build and serve the pages:

  ```sh
  cargo install --locked trunk
  ```

- [`wasm-pack`](https://rustwasm.github.io/wasm-pack/), which Trunk invokes as
  a pre-build hook to produce the WASM bundle:

  ```sh
  cargo install --locked wasm-pack
  ```

## Quickstart

Hyperscope, the main viewer:

```sh
trunk serve
```

then open <http://localhost:8093>.

The Remesh Lab, a workbench for the mesh-to-QB-patch fitting pipeline:

```sh
trunk serve --config Trunk-lab.toml
```

then open <http://localhost:9090>.

To work on the native crates without the browser in the loop, a plain
`cargo build` and `cargo test` cover the whole workspace except
`quilting-wasm`, which is wasm32-only. Check that one with:

```sh
cargo check --target wasm32-unknown-unknown -p quilting-wasm
```

## Assets

`horse.glb` and `ant.glb` are tracked in the repository and are what the demo
pages load by default. Several larger glTF sample models used for material and
transmission testing are not tracked, to keep the clone small. Fetch them on
demand:

```sh
scripts/fetch-assets.sh
```

## Crate map

| Crate | Role |
| --- | --- |
| `hyperscape` | Lightweight Bevy ECS/app/time game layer for conformal frames, anchors, paths, cross-frame constraints, chamber state, and Hyperscope extraction (no Bevy renderer) |
| `quilting-core` | Conformal math, tessellation atlas, LOD selection, QB evaluation, instance layout |
| `quilting-mesh` | Half-edge mesh structure; source of the shared-edge invariant that makes stitching work |
| `quilting-shaders` | Modular WGSL shader library, compiled to GLSL ES 300 via naga |
| `quilting-renderer` | WebGL2/OpenGL renderer over `glow` — buffers, UBOs, VAO cache, draw calls |
| `quilting-gltf` | glTF 2.0 loading: meshes, materials, textures, animations, skins |
| `quilting-remesh` | VSA clustering and QB patch fitting — turns a dense mesh into curved patches |
| `quilting-wasm` | WASM entry point and JS interop; wasm32-only |
| `trunk-stub` | Placeholder cdylib so Trunk has a crate to build; the real WASM comes from the wasm-pack hook |
| `fuzzy-vision` | **Standalone.** JFA-based variable per-pixel blur for WebGL2/OpenGL — depth of field, conformal blur, transmission roughness, bloom. No dependency on the rest of the workspace. |
| `distressed-blue-noise` | **Standalone.** Variable-density Poisson-disk sampling over triangles and rectangles. No dependency on the rest of the workspace. |

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
