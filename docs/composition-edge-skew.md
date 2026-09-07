# Fixed-edge density redistribution

The composition explorer's ten edge skews belong to canonical boundaries, not
to the children that happen to use them. Adjacent children use the same table
with opposite direction. Geometric corners and the fan center do not move.

For a sampled cumulative measure F with total T, a positive skew s gives:

`G(t) = T B_s(F(t) / T)`, where `B_s(u) = s u / (1 + (s - 1) u)`.

G has the same total mass and fixed endpoints as F. Equal increments of G
concentrate samples toward the canonical first endpoint when s>1, and toward
the second when s<1. The neutral s=1 bypass is exact. This changes the requested
distribution, not the edge segment count or the geometric midpoint.

`RedistributedCumulative<M,L>` in `quilting_patch::sampling_warp` expresses the
measure transformation independently of the square, storage or execution
backend. The viewer uses the existing projective interval law and stores17
samples per edge. Its inversion is exact for that represented piecewise-linear
table, not for the unsampled analytic G. Density compensation must be enabled;
turning it off bypasses both the original compensation and these skews.

Canonical directions are bottom0→1, right1→2, top2→3, left0→3, diagonals0→2
and1→3, and spokes0/1/2/3→4. Corners0–3 run counterclockwise from bottom-left;
4 is the authored center. Interior extension reuses `TriangleEdgeMaps`; it does
not make a coarse straight triangulation immune to folds or poor aspect ratios.

The visible scene owns a named `EdgeSkews` record. Its revision changes when a
law changes; pending geometry retains the previous scene and laws together.
GPU tables are produced from that retained snapshot, never newer live controls.
No extra shader, JavaScript sampling code or external generated asset is used.

## Executed checks (September 7, 2026)

- Release Wasm composition policy/retention gate:25.07s. Added all10 edges at
  skews0.25/1/4, monotone tables and inverted samples, exact retained total,
  exact neutral bypass, correct concentration direction and reversed shared
  sample equality. All10 resolved counts remain unchanged. Retained-scene
  checks cover skew revision changes and preserving old laws while pending.
- Real AMD Radeon780M/RADV GPU probe:13 density cases ×3 skew profiles ×10
  edges ×2 directions ×257 samples =200,460 queries /801,840 scalar comparisons.
  Maximum Wasm/GPU difference5.9604645e-8; reversed GPU canonical parameters and
  physical positions are bit-identical. Test runtime12.10s; separate release
  harness compilation3m30s. No elevated device limits or substitute shaders.
- Chrome65 served `fe-render-48d316f5ea50b953.json` from release `fe web dev`
  on38331. Parent Wasm587,321 bytes, WGSL145,268 bytes, still3 passes.
  Initial site build73.858s; these are compilation metrics, not atlas timing.
- All10 native skew inputs drove their intended Fe state values independently.
  Uniform-LoD4 fan kept4,296 vertices and16 segments on each spoke; center stayed
  (.5,.5). Switching just spoke0 from1 to4 incremented spacing revision to2
  and left all other skew inputs at1.
- Compared actual canvas pixels before/after that edit:141,579 changed pixels
  in a1018×1018 interior crop, zero red fold pixels and zero exposed-background
  pixels. The screenshot visibly redistributes samples toward corner0. This
  representative check is not an exhaustive finite-mesh fold/quality guarantee.
- Document and viewport height both1281; no application console errors. One
  Canvas2D readback-performance warning came from diagnostic pixel inspection.
  Surface restored to live; final view leaves spoke0 at2, other skews neutral.

Logs in `/laboratory/quilting/scratch/`:
`composition-edge-skew-final-tests-20260907.log`,
`composition-edge-skew-gpu-tests-20260907.log`,
`composite-edge-skew-web-dev-20260907.log`.

## Upstream observation

At shared mb2 `a7c0804ef`, surface-state projection rejects a `[f32;10]` member
with `unsupported canonical primitive Array`. The named domain record is useful
here independently, but this remains a generic projection limitation. Future
array support should preserve fixed-length typed layout and field identity
through projection and include native/Wasm/GPU round-trip tests. Do not address
it with browser-side flattening or generated application field declarations.
