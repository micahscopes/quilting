# quilting-round-index

`quilting-round-index` is Quilting's backend-neutral spatial-query layer for
conformal scenes. It indexes conservative cluster carriers as oriented open
sphere or plane sides, and queries them with finite intersections of the same
kind of sides. Conformal generator words can push carriers or pull queries
without rebuilding hierarchy topology.

Animated scenes update leaf bounds and refit parent bounds bottom-up. Cache
entries should be keyed by `(TopologyKey, PoseKey, NodeId)`: topology revisions
own stable cluster membership, while pose keys identify the animated bound
snapshot. The crate does not evaluate animation or QB surfaces itself.

Mesh and patch addresses remain in posed source coordinates. Walking, physics,
frustum construction, and proximity distances instead use ordinary Euclidean
geometry in the active post-Möbius chart. Build the sphere/frustum there, pull
the resulting `RoundQuery` back through the conformal generator word, and then
traverse the source index. The API intentionally provides no source-Euclidean
nearest-neighbour or distance query.

`RoundQuery::from_view_projection` converts a column-major WebGL
view-projection matrix directly into those six output-chart half-spaces.
Invalid or degenerate matrices fail closed (no query) rather than manufacturing
a culling proof.

`StaticPatchIndex` is the first renderer-facing adapter. It builds a balanced
hierarchy over conservative source-space bounds for three-control QB patches.
Ordinary patches with one common quaternion weight use a tight sphere around
their control triangle. Rational patches use a finite norm bound only when the
convex hull of their denominator controls is proved clear of zero. Invalid or
potentially singular patches enter an explicit `always_candidates` lane: an
uncertain bound can cost performance, but cannot make a patch disappear.

This adapter is intentionally rest-pose only. Do not use it to cull active
animation until the caller supplies conservative pose envelopes or refits leaf
bounds for the current `PoseKey`. A renderer should first run it as shadow
telemetry against its authoritative visibility path and require zero false
negatives before allowing it to affect drawing.

The crate intentionally returns `IntersectsOrUnknown` whenever a numerical or
geometric case is not proved safe to prune. See the crate-level Rust
documentation and `formal/ConformalMereology/RoundSideIndex.lean` for the
corresponding incidence theorem.
