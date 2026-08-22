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

The crate intentionally returns `IntersectsOrUnknown` whenever a numerical or
geometric case is not proved safe to prune. See the crate-level Rust
documentation and `formal/ConformalMereology/RoundSideIndex.lean` for the
corresponding incidence theorem.
