# Two triangles with measured boundary spacing

Run from the worktree root:

```sh
.toolchains/fe/target/release/fe web dev fe/web/two-triangle-uniformization/index.html --port 8778 --color never
```

This is a separate explorer, not a replacement for the paired fan or quad warp
pages. It restricts the original quad parameter domain to two triangles along
00→11. It does not fit two new lower-degree surfaces: every rendered point and
normal still comes from the original analytical patch. Each half uses one
permutation-aware resident triangle atlas tile, through LoD 8.

## What the controls mean

- `edge_spacing`: requested maximum arc length per edge segment, in surface
  units, not pixels. Edge LoD is the smallest power of two meeting this target,
  subject to `max_lod` (default 7, range 0–8).
- `uniformize`: compare native parameter sampling with inverse cumulative
  length sampling. LoDs stay length-based in both modes.
- Four weights, optional automatic surface-family constraint resolution,
  orbit controls, wire toggle and wire width follow the shared explorers.

Five status strips run bottom-to-top: bottom, right, top, left, diagonal.
Green means the sampled length checks pass and the target is not capped;
amber means the 512/1024-segment polyline lengths differ by more than 0.01%;
red means invalid sampled geometry or a capped density request. Green is **not**
a certification of uniform interior triangles, absence of unsampled poles, or
an upper bound on integration error. Red surface shading separately indicates
weights outside the vector-valued surface family.

## Boundary authority and numerical scope

One GPU dispatch integrates five independently owned edge records. Each record
has a cumulative length table, coarse/fine totals, validity, a parameter
conditioning bias, and its source weights. Camera-only redraws reuse it.
A following dispatch selects the two atlas keys and their indirect draw count.
The rendering path performs no readback, CPU tessellation, or per-frame upload
of generated geometry.

The integration grid is preconditioned by endpoint weights (and their square
root ratio on the diagonal); sampled positions are still evaluated on the
original surface. Inverting the cumulative table is exact for its interpolated
CDF, **not exact analytical arc length**. The coarse/fine comparison is a
diagnostic, not a quadrature certificate.

The shared diagonal has one length, one LoD, one table, and one canonical
parameter direction. Boundary points bypass the interior map's reversed
arithmetic: either half calls the same diagonal map with the same dyadic input.
Interior coordinates use the shared monotone edge-map composition. This keeps
the edge rules independent but does not guarantee good straight-triangle
quality on a highly distorted surface.

## Evidence and remaining limits

The release Wasm test
`sampled_length_inverse_wasm_is_monotone_and_lod_tracks_curved_length`
checks monotonic inversion, endpoints, zero-length intervals, the error of a
sampled quadratic CDF, and capped dyadic LoD selection. Existing patch and
boundary-warp oracles remain applicable; this is not yet an independent
end-to-end numerical certificate for the GPU's five surface curves.

Chromium renders the default and an approximately separable extreme with
weights near 0.02 / 0.4 / 0.4 / 8. The extreme still produces poor interior
triangles. This explorer isolates that remaining problem instead of claiming
that boundary uniformization fixes it. Numerical boundary convergence, actual
atlas seam replay, and alternative interior extensions deserve focused follow-up.

Shared implementation lives in `quilting_patch::arc_length`,
`quilting_patch::sampling_warp::TriangleParameterMap`, and
`quilting_patch_view::arc_length`. The existing projective-map renderer delegates
to the same parameter-map renderer used here, retaining its depth, shading,
pixel-width wires, and 4× MSAA behavior.
