# Ingot organization

Every ingot has one semantic role and dependencies point inward:

- `foundation/` — coordinate domains, scalar/vector values, admission rules;
- `algebra/` — reusable algebraic representations and conditioned operations;
- `geometry/` — curves, patches, and geometric algorithms built on those
  representations;
- `validation/` — stable exported oracles used by independent hosts;
- `demos/` — bounded visual compositions and teaching programs.

Production ingots never depend on `validation` or `demos`. Experimental patch
families receive distinct ingot names and cannot replace the production QB
path without explicit parity, cost, and stability evidence.

Member manifests use workspace dependencies instead of directory-relative
paths. Add a member to `../fe.toml`, then declare sibling dependencies as
`name = true`. Only the workspace root may name the external Fe `core` and
`std` checkout.

Principal next additions:

- `algebra/quilting_cga` for typed conformal authoring and spherical controls;
- `geometry/quilting_clifford_bezier` for the Krasauskas–Zubė surface lane,
  with QB retained as a shared restriction/baseline oracle; and
- separate validation/demo ingots for those experiments, so their dependencies
  and generated cost remain visible.
