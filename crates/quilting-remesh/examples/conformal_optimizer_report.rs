//! Reproducible report for the isolated conformal/QB optimizer prototype.
//!
//! Run the self-contained path with:
//! `cargo run -p quilting-remesh --release --example conformal_optimizer_report`
//!
//! Add `--features meshopt-prototype` to compare meshoptimizer-seeded growth.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use quilting_core::patch::QBTriPatch;
use quilting_core::quaternion::{Mobius, Quat};
use quilting_remesh::conformal_optimizer::{
    cluster_connected, score_patch_complex, ClusterConfig, ClusterInput, ConformalProbe, FitSample,
    FitScore, FitScoreConfig,
};
use quilting_remesh::linear_fit::{linear_global_fit_full, LinearFitConfig, Sample};
use quilting_remesh::roundtrip::{sphere_patches, tessellate_patch};
use quilting_remesh::test_shapes;

fn main() {
    println!("cluster_fixture,method,faces,clusters,max_faces,max_vertices,avg_boundary_edges,disconnected_clusters,runtime_us");
    let (sphere_positions, sphere_faces) = test_shapes::sphere(2);
    report_clusters("sphere", &sphere_positions, &sphere_faces, None);

    let segments = 32;
    let rings = 8;
    let (cylinder_positions, cylinder_faces) = test_shapes::cylinder(segments, rings, 2.0, 1.0);
    let side_faces = rings * segments * 2;
    let mut domains = vec![0; cylinder_faces.len()];
    domains[side_faces..side_faces + segments].fill(1);
    domains[side_faces + segments..].fill(2);
    report_clusters(
        "cylinder",
        &cylinder_positions,
        &cylinder_faces,
        Some(&domains),
    );

    report_fit();
}

fn report_clusters(
    fixture: &str,
    positions: &[[f64; 3]],
    faces: &[[usize; 3]],
    domains: Option<&[u32]>,
) {
    let ids: Vec<u64> = (0..faces.len())
        .map(|face| 1_000_003 + face as u64 * 31)
        .collect();
    let config = ClusterConfig {
        max_triangles: 32,
        max_vertices: 32,
        ..Default::default()
    };
    let input = ClusterInput {
        positions,
        triangles: faces,
        source_face_ids: &ids,
        face_domains: domains.unwrap_or(&[]),
        locked_edges: &[],
    };

    let start = Instant::now();
    let connected = cluster_connected(&input, &config).unwrap();
    print_cluster_row(
        fixture,
        "connected-qb",
        faces.len(),
        &connected,
        start.elapsed(),
    );

    #[cfg(feature = "meshopt-prototype")]
    {
        let start = Instant::now();
        let seeded =
            quilting_remesh::conformal_optimizer::cluster_meshopt_seeded(&input, &config).unwrap();
        print_cluster_row(
            fixture,
            "meshopt-seeded-qb",
            faces.len(),
            &seeded,
            start.elapsed(),
        );
    }

    // Face-buffer chunks are the deliberately simple baseline: they have the
    // same face budget but no topological guarantee.
    let start = Instant::now();
    let chunks: Vec<&[[usize; 3]]> = faces.chunks(config.max_triangles).collect();
    let max_vertices = chunks
        .iter()
        .map(|chunk| {
            chunk
                .iter()
                .flatten()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
        })
        .max()
        .unwrap_or(0);
    let elapsed = start.elapsed();
    let disconnected = disconnected_face_chunks(faces, config.max_triangles);
    println!(
        "{fixture},face-order-chunks,{},{},{},{},n/a,{},{}",
        faces.len(),
        chunks.len(),
        config.max_triangles.min(faces.len()),
        max_vertices,
        disconnected,
        elapsed.as_micros(),
    );
}

fn disconnected_face_chunks(faces: &[[usize; 3]], chunk_size: usize) -> usize {
    faces
        .chunks(chunk_size)
        .filter(|chunk| {
            if chunk.is_empty() {
                return false;
            }
            let edge_set = |face: &[usize; 3]| -> BTreeSet<(usize, usize)> {
                (0..3)
                    .map(|local| {
                        let a = face[local];
                        let b = face[(local + 1) % 3];
                        (a.min(b), a.max(b))
                    })
                    .collect()
            };
            let all_edges: Vec<BTreeSet<(usize, usize)>> = chunk.iter().map(edge_set).collect();
            let mut reached = BTreeSet::new();
            let mut frontier = vec![0usize];
            while let Some(face) = frontier.pop() {
                if !reached.insert(face) {
                    continue;
                }
                for candidate in 0..chunk.len() {
                    if !reached.contains(&candidate)
                        && !all_edges[face].is_disjoint(&all_edges[candidate])
                    {
                        frontier.push(candidate);
                    }
                }
            }
            reached.len() != chunk.len()
        })
        .count()
}

fn print_cluster_row(
    fixture: &str,
    method: &str,
    face_count: usize,
    set: &quilting_remesh::conformal_optimizer::ClusterSet,
    elapsed: std::time::Duration,
) {
    let max_faces = set
        .clusters
        .iter()
        .map(|cluster| cluster.source_faces.len())
        .max()
        .unwrap_or(0);
    let max_vertices = set
        .clusters
        .iter()
        .map(|cluster| cluster.vertices.len())
        .max()
        .unwrap_or(0);
    let avg_boundary = set
        .clusters
        .iter()
        .map(|cluster| cluster.boundary_edges.len())
        .sum::<usize>() as f64
        / set.clusters.len().max(1) as f64;
    println!(
        "{fixture},{method},{face_count},{},{max_faces},{max_vertices},{avg_boundary:.3},0,{}",
        set.clusters.len(),
        elapsed.as_micros(),
    );
}

fn report_fit() {
    let truth = sphere_patches(2.5);
    let faces = vec![
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    let mut coarse_positions = vec![[0.0; 3]; 6];
    for (patch, face) in truth.iter().zip(&faces) {
        for local in 0..3 {
            coarse_positions[face[local]] = patch.positions[local].to_point();
        }
    }

    let mut samples = Vec::new();
    let mut linear_samples = Vec::new();
    for (patch_index, patch) in truth.iter().enumerate() {
        let tessellation = tessellate_patch(patch, 8);
        for sample_index in 0..tessellation.positions.len() {
            let barycentric = tessellation.bary[sample_index];
            samples.push(FitSample {
                patch_index,
                barycentric,
                target_position: tessellation.positions[sample_index],
                target_normal: tessellation.normals[sample_index],
            });
            linear_samples.push(Sample {
                face_index: patch_index,
                bary: barycentric,
                target: tessellation.positions[sample_index],
            });
        }
    }

    let fit_start = Instant::now();
    let fit = linear_global_fit_full(
        &coarse_positions,
        &faces,
        &linear_samples,
        &LinearFitConfig {
            tikhonov: 1e-10,
            ..LinearFitConfig::default()
        },
    )
    .expect("exact shared QB fixture fit");
    let fit_elapsed = fit_start.elapsed();
    let flat: Vec<QBTriPatch> = faces
        .iter()
        .map(|face| {
            QBTriPatch::flat(
                coarse_positions[face[0]],
                coarse_positions[face[1]],
                coarse_positions[face[2]],
            )
        })
        .collect();
    let probes = vec![
        ConformalProbe {
            name: "scale-8".into(),
            transform: Mobius::scale(8.0),
        },
        ConformalProbe {
            name: "near-pole-reflection".into(),
            transform: Mobius::sphere_reflection(Quat::from_point(2.0, -1.0, 3.0), 1.5),
        },
    ];
    let config = FitScoreConfig::default();
    let score_start = Instant::now();
    let fitted_score =
        score_patch_complex(&fit.patches, &faces, &samples, &probes, &config).unwrap();
    let fitted_score_elapsed = score_start.elapsed();
    let score_start = Instant::now();
    let flat_score = score_patch_complex(&flat, &faces, &samples, &probes, &config).unwrap();
    let flat_score_elapsed = score_start.elapsed();

    println!();
    println!("fit_fixture,method,patches,rms,max,normal_rms_deg,boundary_max,min_rel_denom,fit_us,score_us");
    print_fit_row(
        "sphere-qb",
        "shared-linear-qb",
        &fitted_score,
        fit.patches.len(),
        fit_elapsed.as_micros(),
        fitted_score_elapsed.as_micros(),
    );
    print_fit_row(
        "sphere-qb",
        "flat-baseline",
        &flat_score,
        flat.len(),
        0,
        flat_score_elapsed.as_micros(),
    );
    println!("active_chart,method,rms,max,normal_rms_deg,boundary_max,min_rel_denom,peak_dilation,pole_near");
    print_probe_rows("shared-linear-qb", &fitted_score);
    print_probe_rows("flat-baseline", &flat_score);
}

fn print_fit_row(
    fixture: &str,
    method: &str,
    score: &FitScore,
    patch_count: usize,
    fit_us: u128,
    score_us: u128,
) {
    println!(
        "{fixture},{method},{patch_count},{:.9},{:.9},{:.6},{:.3e},{:.6},{fit_us},{score_us}",
        score.source.position_rms,
        score.source.position_max,
        score.source.normal_rms_degrees,
        score.boundary.max_gap,
        score.weights.min_relative_denominator,
    );
}

fn print_probe_rows(method: &str, score: &FitScore) {
    let by_name: BTreeMap<&str, _> = score
        .conformal_probes
        .iter()
        .map(|probe| (probe.name.as_str(), probe))
        .collect();
    for (name, probe) in by_name {
        println!(
            "{name},{method},{:.9},{:.9},{:.6},{:.3e},{:.6},{:.3e},{}",
            probe.error.position_rms,
            probe.error.position_max,
            probe.error.normal_rms_degrees,
            probe.boundary.max_gap,
            probe.weights.min_relative_denominator,
            probe.peak_local_dilation,
            probe.pole_near_samples,
        );
    }
}
