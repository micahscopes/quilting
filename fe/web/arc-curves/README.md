# Geometric arc explorer

Serve from the Quilting Fe root with the release compiler on `mb2`:

```sh
.toolchains/fe/target/release/fe web dev fe/web/arc-curves/index.html --port 8780
```

The Fe actor lives in `fe/ingots/demos/arc_curve_explorer`. It constructs an arc
from two endpoints and an on-arc point at parameter 1/2, sharing the construction
between Euclidean and isotropic Clifford metrics. Choose a planar or orbitable
spatial view. Drag the colored handles; drag empty space to orbit. The wheel
moves a selected handle in depth, otherwise it changes camera distance.

Rendering uses independently instanced four-vertex triangle strips, native 4×
MSAA, depth testing and straight-alpha blending. Width and horizon fade are
separate controls. This is ordinary transparency, not OIT. The renderer rejects
intervals whose denominator may cross zero rather than connecting across a pole.

This first explorer is degree one in Clifford weights. Freely edited
higher-degree curves and exact circle-preserving degree elevation are not yet
implemented. Arc construction also does not solve a whole quad patch's boundary
compatibility constraints.

Checks on 2026-09-04: Fe typechecking and browser WebGPU rendering; an injected
pointer-event transition selected and moved the middle handle without changing
either endpoint. Chrome readback inspection verified both the initial circular
arc and the collinear pole case A=(0,0,0), B=(1,0,0), M=(2,0,0): two separate
finite branches, with no false connecting segment. This is not yet a complete
physical-input or pathological-curve acceptance suite. The shared color-packing
Wasm oracle checks every alpha byte and clamping while preserving RGB.
