# Atlas mesh LoD lab

An Fe-owned WebGPU terrain demo built from a 8 x 6 quad sheet (96 triangular
macro-patch instances) and the checked Quilting LoD atlas.

- Hover moves a screen-space focus field. Each macro edge derives its dyadic
  density from its projected midpoint, making independently selected adjacent
  atlas patches conform at their shared boundary.
- Drag orbits; wheel zooms. Noise amplitude, frequency, domain warp, detail,
  speed and phase are ordinary reactive Fe parameters.
- Three OKLCH hue stops, hue-path policy, lightness, chroma, bias and contrast
  compose the reusable `quilting_oklch` API.
- Four-sample depth rasterization covers silhouettes. Internal topology uses
  a projected differential and triangle altitude for constant-pixel analytic
  wire coverage, with depth-cued perceptual lighting.

## Current compiler/runtime seams

The one draw uses
`Instanced<TriangleList<ATLAS_VERTEX_LIMIT>, PATCH_INSTANCES>` and the isolated
vertex signature `(vertex_index: u32, instance_index: u32)`. This is the only
place that depends on instanced raster lowering.

`animation_clock` consumes typed `AnimationFrame` timestamps and otherwise
advances from interaction timestamps. The current resident browser scheduler
requires pending application input before presenting and therefore cannot yet
drive an idle, zero-input animation loop. The source intentionally contains no
demo JavaScript workaround; once that host contract lands, the clock seam is
already Fe-authored.
