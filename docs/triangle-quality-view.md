# Triangle shape view

In the composition demo, `color_view` selects ordinary surface shading, a
fixed triangle-shape scale, or highlighting of shape below 0.1. The metric is
`4 sqrt(3) area / sum(edge_length_squared)`, evaluated on the three actual
world-space vertices already used for the face normal. One is equilateral;
zero is collapsed. It is not an approximation-error or pixel-density measure.

Warm-to-cool OKLCH colors use the same 0–1 scale for every mesh. Diagnostic
fill is unlit, so camera rotation does not alter the scale. Optional wires
still darken edges. Magenta marks an unadmitted visible measurement; a patch
discarded before rasterization cannot be shown by this view. Numerical invalid
counts and coverage diagnostics remain necessary.

This is the first visible diagnostic, not the complete fixed-boundary A/B,
worst-cell selection, or matched-budget comparison interface.

## Browser validation finding

Chrome rejected the initial shade module at line 1611:

```wgsl
_e50 <= 340282350000000000000000000000000000000f
```

`GPUShaderModule.getCompilationInfo()` returned “value cannot be represented
as 'f32'”. Artifact: `fe-render-15657836e51f8c6c.wgsl`, manifest
`fe-render-68ec09b322d998cc.json`. The module had already passed the source
build, so source checking alone did not catch this browser incompatibility.

The finite-area admission now uses IEEE `area - area == 0`: all finite values
pass, infinity/NaN fail, without approximating the upper limit. The release
Wasm regression includes nonfinite and overflowing-area inputs. Upstream should
add an extremal f32 literal browser regression and inspect WGSL serialization;
the precise responsible lowering/writer stage is not yet established.

## Served checkpoint

Release `fe web dev` on port 38331 rebuilt in 82.0 seconds after the fix.
Chrome MCP loaded manifest `fe-render-07098ef3bb9fddd9.json`; direct compilation
of its shade module returned no messages. Selecting curved quad and triangle
shape through the native dropdowns produced `path webgpu`, pending 0 and four
16-segment spokes. Total emitted WGSL: 613,468 bytes across four modules; Wasm:
1,204,666 bytes. These are artifact/build figures, not runtime speed claims.
The finite-area release Wasm regression passed in 20.64 seconds including setup.
Screenshot capture has not yet completed, so visual acceptance remains pending.
