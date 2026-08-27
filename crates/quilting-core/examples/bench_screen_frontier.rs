//! Deterministic construction oracle for a large all-root adaptive frontier.
//!
//! This intentionally uses disconnected source triangles: it exercises the
//! source-sized validation, line, and corner-index work without conflating the
//! measurement with glTF decoding or camera-dependent partitioning.

use std::time::Instant;

use quilting_core::patch::QBPatchDomain;
use quilting_core::screen_leaf_lod::{
    ScreenMeshLeafFrontier, ScreenMeshLeafTopology, ScreenMeshTopologyCache,
};
use quilting_core::screen_partition::ScreenPatchLeafId;
use quilting_mesh::HalfEdgeMesh;

fn main() {
    let face_count = std::env::args()
        .nth(1)
        .map(|value| {
            value
                .parse::<usize>()
                .expect("face count must be an integer")
        })
        .unwrap_or(94_628);
    let repetitions = std::env::args()
        .nth(2)
        .map(|value| {
            value
                .parse::<usize>()
                .expect("repetitions must be an integer")
        })
        .unwrap_or(5)
        .max(1);
    let vertex_count = face_count
        .checked_mul(3)
        .and_then(|count| u32::try_from(count).ok())
        .expect("face count exceeds the u32 source-mesh limit");
    let faces = (0..face_count)
        .map(|face| {
            let first = u32::try_from(face * 3).expect("validated vertex identity");
            [first, first + 1, first + 2]
        })
        .collect::<Vec<_>>();
    let leaves = (0..face_count)
        .map(|face| ScreenMeshLeafTopology {
            source_face: u32::try_from(face).expect("face count exceeds u32"),
            id: ScreenPatchLeafId::ROOT,
            domain: QBPatchDomain::FULL,
        })
        .collect::<Vec<_>>();

    let topology_start = Instant::now();
    let mesh = HalfEdgeMesh::from_triangles(vertex_count, &faces);
    let topology = ScreenMeshTopologyCache::from_half_edge_mesh(&mesh)
        .expect("disconnected triangle topology must be valid");
    let topology_ms = topology_start.elapsed().as_secs_f64() * 1_000.0;

    let mut frontier_samples_ms = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let frontier_start = Instant::now();
        let frontier = ScreenMeshLeafFrontier::build(&leaves, &topology)
            .expect("all-root frontier must be valid");
        assert_eq!(frontier.leaves().len(), face_count);
        frontier_samples_ms.push(frontier_start.elapsed().as_secs_f64() * 1_000.0);
    }
    let mut sorted_samples = frontier_samples_ms.clone();
    sorted_samples.sort_by(f64::total_cmp);
    let frontier_median_ms = sorted_samples[sorted_samples.len() / 2];

    println!(
        "faces={face_count} repetitions={repetitions} topology_ms={topology_ms:.3} \
         frontier_median_ms={frontier_median_ms:.3} frontier_samples_ms={frontier_samples_ms:?}",
    );
}
