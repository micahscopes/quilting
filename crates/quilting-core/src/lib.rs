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
pub mod interpolation;
pub mod sampling;
pub mod subdivide;
pub mod delaunay;
pub mod mesh;
pub mod permutation;
pub mod atlas;
pub mod instance_layout;
pub mod patch;
pub mod shapes;
pub mod evaluate;
pub mod batch;

pub use atlas::TessellationAtlas;
pub use evaluate::FaceInstance;
pub use patch::QBTriPatch;
pub use permutation::canonical_form;
pub use quaternion::{Mobius, Quat};
