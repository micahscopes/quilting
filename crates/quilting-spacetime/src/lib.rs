pub mod trajectory;
pub mod hyper_mesh;
pub mod slicer;
pub mod synthesize;

pub use trajectory::{HermiteSegment, VertexTrajectory};
pub use hyper_mesh::HyperMesh;
pub use slicer::{HyperplaneSlicer, SliceLayer, SliceResult};
