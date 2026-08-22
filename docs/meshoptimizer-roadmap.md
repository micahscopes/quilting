# Meshoptimizer integration roadmap

Meshoptimizer is useful here, but at three different layers that should not be
conflated: final index ordering, source-asset preprocessing, and future spatial
clusters. It does not replace the adaptive triangular atlas or the rational QB
surface model.

## Safe near-term uses

1. Optimize each completed canonical atlas patch for vertex-cache locality,
   then apply a vertex-fetch remap to its barycentric vertex range. The final
   triangle order must be established first, and line indices must be rebuilt
   from the reordered triangles. This preserves patch geometry, S3
   permutation semantics, edge samples, and face identity while reducing
   vertex invocations/fetch scatter. Meshoptimizer documents cache optimization
   before fetch optimization in its
   [core pipeline](https://github.com/zeux/meshoptimizer#core-pipeline); the
   current Rust crate exposes both
   [operations](https://docs.rs/meshopt/latest/meshopt/optimize/index.html).
2. Experiment with full-attribute source remapping within one glTF primitive
   domain. Position, normal, UV, joints, weights, morph deltas, material/node
   ownership, and stable source-face IDs must move together. Position-only
   welding across UV, normal, skin, material, or non-manifold seams is not
   permitted.
3. Treat meshopt-compressed glTF as an offline/file-size path. Decoder support
   must be explicit at the loader boundary; compressed buffer views cannot be
   handed to the current glTF importer as ordinary bytes.

Overdraw reordering is conditional. The upstream documentation explicitly
recommends measuring it by renderer and hardware, and notes that tile-based
mobile GPUs may not benefit. Quilting is often vertex-heavy and conformally
view-dependent, so cache/fetch measurements come first.

## WebGPU cluster use

Meshlets are a good future coarse indexing and scheduling unit: cluster IDs can
feed visibility compaction, streaming, and indirect draw generation without
changing the canonical per-edge tessellation contract. The hierarchical
cluster workflow described by meshoptimizer uses small connected triangle
clusters and preserves group boundaries while simplifying
([overview](https://zeux.io/2025/09/30/billions-of-triangles-in-minutes/)).
Those reported multi-fold construction gains are for an offline billion-scale
hierarchy build, not a prediction for Quilting's runtime atlas or frame time.

Ordinary meshlet spheres, AABBs, and normal cones are not sufficient culling
proofs for this renderer. A valid cluster bound must enclose:

- the animated and affine-posed source domain;
- the complete rational QB interior, not only source vertices;
- the complete Möbius image when its denominator stays away from zero; and
- pole/denominator failure cases conservatively, by refusing to cull.

The first implementation should therefore use meshlets only as cluster
membership. Quilting-owned bound generation remains authoritative.

## Weighted conformal fitting (deferred)

Meshoptimizer clustering can eventually seed connected regions for weighted
rational-QB fitting, but its public clustering score is not the fitting
objective. A Quilting fitter would additionally minimize positional and normal
error, preserve material/UV/skin/sharp/non-manifold domains, regularize the
quaternion denominator, retain pole-safe conformal bounds, and enforce shared
edge weights for watertight C0 continuity.

The current shipping target remains one fixed quadratic rational patch complex
plus adaptive per-edge tessellation. Higher-order rational triangular Bézier
surfaces remain a possible future fitting mode, but they add control data,
continuity constraints, denominator/pole analysis, shader cost, and bounding
complexity. They should be justified by measured fit quality after the
quadratic pipeline is stable, not introduced into the renderer preemptively.
