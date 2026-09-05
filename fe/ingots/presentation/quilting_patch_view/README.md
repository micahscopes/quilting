# Shared patch display

Shared Fe rendering functions for the Four-Patch Center and Two Triangles,
Two Fans explorers. Callers select the atlas key, parameter triangle, and
positive projective sampling map. The renderer evaluates the original Clifford
patch at the mapped coordinates, keeping sample placement separate from shape.

This ingot owns atlas lookup/permutation, the current authored demonstration
patch, analytic normals, common depth shading, and pixel-width triangle wires.
Callers retain topology, draw preparation, camera policy, and reactive state.
It has no browser-specific JavaScript or geometry readback.

The evaluator is explicitly Clifford-specific today; the parameter-domain and
projective-map operations in `quilting_patch` remain algebra-independent. The
resident fixture provider still lives in `classic_quilting_lod`; extracting
that asset provider is separate from the sampling/rendering semantics.
