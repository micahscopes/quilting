# Surface-aware coarse mesh filled by atlas triangles

This is an experiment in progress, not a replacement renderer or a claim of
Delaunay triangulation. It follows the failure of fixed connectivity plus
interior warping on the captured asymmetric quad.

## First gate: connectivity independent of placement

`quilting_atlas::coarse_mesh::CoarseMesh` holds at most sixteen parameter-domain
triangles. It starts from either square diagonal or one triangle. Strictly
interior insertion replaces one face by three; existing vertex IDs and all
outer edges remain unchanged. Proposals on an edge, outside a face, outside the
Q14 domain or beyond capacity are rejected without mutation. Counts and storage
are private so callers cannot construct an inconsistent active range.

The composition oracle exposes snapshots after bounded centroid insertions.
The independent Rust/Wasm check measures positive integer orientation, total
area, opposite interior-edge incidence, unchanged outer edges, retained vertex
coordinates, capacity and invalid-index behavior. These checks concern coarse
parameter connectivity. They do not prove fine-mesh coverage after spacing
correction, surface approximation, or absence of surface self-intersection.

Run from `fe/tools/quilting-fe-fixtures`:

```
cargo test --release --lib --features fe-oracle \
  coarse_connectivity_preserves_coverage_boundaries_and_vertex_identity -- --nocapture
```

## Next gates

1. Use surface measurements to propose relocation, local flips and insertion;
   exact parameter predicates remain the connectivity admission check. Reuse
   the existing pullback/triangle metrics, rather than assuming UV distances
   describe surface distances. Keep work explicitly bounded.
2. Resolve shared edge counts and spacing once per owner. Fill coarse faces
   using the existing atlas, including its winding and permutation handling.
3. Reject actual finite-mesh folds and missing coverage before comparing shape,
   area allocation and approximation error against the best fixed candidate at
   matched budgets. Include the captured triangle and asymmetric quad.
4. Only promote a useful, valid candidate to the viewer. Report unmet demand
   when sixteen coarse faces do not suffice; do not hide it with more LoD.

Centroid insertion is currently a connectivity test stimulus, **not** the
surface-aware sampling algorithm. No boundary is subdivided in this first
slice. Later boundary subdivision must reference existing shared master samples.
