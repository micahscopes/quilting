# Classic Quilting M2 frozen-atlas wire proof

Date: 2026-09-02

This slice replaces the one-triangle M1 silhouette with one real, frozen
Quilting atlas patch. It is still deliberately bounded: topology is generated
ahead of time from the checked direct-atlas fixture, while Fe owns the QB
surface evaluation and WebGPU raster program.

## Render contract

- fixture: `direct-seed42-k2-4-8.cqa`
- requested edge densities: `[2,4,8]`
- canonical orientation only; the six S3 permutations are not yet a browser
  coverage claim
- 21 indexed atlas samples
- 26 constrained-Delaunay triangles
- 78 vertices after deterministic triangle-list expansion
- analytic QB position and normal at every atlas sample
- triangle-local barycentrics carried through the typed raster interface for
  the black wire diagnostic

The visual wire is a raster diagnostic over the atlas's actual triangles. It
is not a decorative overlay and it does not substitute a regular grid for the
frozen atlas topology.

## Browser constraint discovered

The first generated selector was a linear chain over all 78 expanded vertices.
Rust, Fe/Wasm, Naga, and llvmpipe accepted it, but Chrome rejected the emitted
WGSL because its statement nesting exceeded the browser limit. The generator
now emits a balanced decision tree, reducing lookup nesting from linear to
logarithmic depth. A Rust regression test bounds the committed generated
source's nesting, and Chrome shader-module validation remains a required
browser gate.

## Development loop

`fe web dev` watches by default:

```sh
.toolchains/fe/target/release/fe web dev \
  --port 8766 fe/web/classic-quilting/index.html
```

Use `--poll-ms` to tune its source polling/debounce interval. `--no-watch` is
the explicit compile-once mode. The release adapter remains a separate final
artifact gate; it is not needed for the edit/reload loop.

## Next boundary

This proof does not yet provide runtime topology, permutation selection,
interactive controls, GPU atlas generation, or GPU-resident adaptive sampling
and triangulation. Those remain later slices after this fixed path stays green.
