//! Core math and geometry for Quilting: rendering triangle meshes under
//! conformal (Möbius) transformations of R³.
//!
//! The technique, following Krasauskas & Zubė, identifies R³ with the
//! imaginary quaternions, so a Möbius map is `F(x) = (ax + b)(cx + d)⁻¹` for
//! quaternion coefficients. Each mesh face becomes a quaternionic-Bézier
//! triangle patch whose weights encode the local conformal stretch; the GPU
//! evaluates the rational quaternion form per vertex, so animating a transform
//! costs four uniform quaternions rather than a buffer rebuild.
//!
//! # Where things live
//!
//! - [`quaternion`] — [`Quat`] and [`Mobius`], the arithmetic everything rests on.
//! - [`conformal`] — serializable generator words and a validated conformal
//!   frame forest.  Generator words retain the authoring operations needed for
//!   exact inverse chains and preserve-world re-anchoring.
//! - [`mereology`] — framed round walls, complementary open sides, contact
//!   classification, and sparse anchor-orientation state.
//! - [`incidence`] — finite-poset zeta/Möbius payload coordinates and sparse
//!   cover structure, without assuming a global bottom element.
//! - [`patch`] — [`QBTriPatch`], the QB surface and its Möbius transform rule.
//! - [`atlas`] — [`TessellationAtlas`], pre-tessellated sub-meshes keyed by
//!   edge-LOD triple. WebGL2 has no tessellation shader, so patch geometry is
//!   baked once and stamped per face at draw time.
//! - [`permutation`] — S3 canonicalization ([`canonical_form`]) that lets one
//!   atlas entry serve all six orderings of an LOD triple.
//! - [`evaluate`] — per-face LOD computation and [`FaceInstance`] construction.
//! - [`instance_layout`] — normative definition of the GPU instance buffer.
//!   The Rust packer, the renderer's VAO setup and the WGSL shader must all
//!   agree with it; nothing should restate a stride or offset.
//! - [`batch`] — grouping faces into instanced draw calls.
//! - [`render`] — backend-neutral scene snapshots and frame commands shared by
//!   WebGL2 and WebGPU implementations.
//! - [`render_pipeline`] — immutable shader, binding-layout, and pipeline
//!   descriptions suitable for functional planning and backend memoization.
//! - [`source_bounds`] — backend-neutral post-model, pre-conformal bounds for
//!   selection, focus fitting, spatial indexing, and navigation scale.
//!
//! # Invariants worth knowing before editing
//!
//! Adjacent faces must agree on a shared edge's LOD or the tessellation
//! T-junctions and cracks visibly. That is enforced structurally: edge LODs
//! are stored per *canonical edge index* from the half-edge mesh, so both
//! faces read the same slot. LODs are also always powers of two, because the
//! atlas is only built for power-of-two triples. `SPEC.md` §8 has the full list.

pub mod triangle;
pub mod quaternion;
pub mod conformal;
pub mod mereology;
pub mod incidence;
pub mod interpolation;
pub mod sampling;
pub mod subdivide;
pub mod delaunay;
pub mod mesh;
pub mod permutation;
pub mod atlas;
pub mod instance_layout;
pub mod patch;
pub mod polytope4;
pub mod shapes;
pub mod evaluate;
pub mod conformal_lod;
pub mod batch;
pub mod educational;
pub mod render;
pub mod render_pipeline;
pub mod source_bounds;
pub mod screen_metric;
pub mod screen_partition;

pub use atlas::TessellationAtlas;
pub use evaluate::FaceInstance;
pub use patch::{QBPatchDomain, QBTriPatch, RestrictedQBTriPatch};
pub use permutation::canonical_form;
pub use quaternion::{Mobius, Quat};
pub use conformal::{
    ConformalError, ConformalFrame, ConformalFrameForest, ConformalGenerator,
    ConformalTransformChain, FrameId,
};
pub use mereology::{
    AnchorState, OpenRoundSide, RoundSideOrientation, RoundWall, RoundWallGeometry,
    RoundWallRelation, RoundWallSet, TangencyKind, WallId,
};
pub use incidence::{
    honest_three_wall_reanchor, naive_three_wall_reversal, FinitePoset, PosetError,
};
