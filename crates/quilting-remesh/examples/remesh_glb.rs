/// CLI tool to test the remeshing pipeline.
/// Usage:
///   cargo run --example remesh_glb -- <file.glb> [target_patches]
///   cargo run --example remesh_glb -- --sphere [subdivisions] [target_patches]
///   cargo run --example remesh_glb -- --cylinder [segments] [rings] [target_patches]
///   cargo run --example remesh_glb -- --voronoi [subdivisions] [num_cells] [target_patches]

use quilting_remesh::geometry;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let (positions, faces, normals, uvs) = if args.len() > 1 && args[1] == "--sphere" {
        let subdivisions = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
        println!("Generating icosphere (subdivisions={})", subdivisions);
        let (p, f) = generate_sphere(subdivisions);
        let n = quilting_remesh::geometry::compute_vertex_normals(&p, &f);
        println!("Generated: {} vertices, {} faces", p.len(), f.len());
        (p, f, Some(n), None::<Vec<[f64; 2]>>)
    } else if args.len() > 1 && args[1] == "--cylinder" {
        let segments = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
        let rings = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
        println!("Generating cylinder (segments={}, rings={})", segments, rings);
        let (p, f) = generate_cylinder(segments, rings, 1.0, 0.3);
        let n = quilting_remesh::geometry::compute_vertex_normals(&p, &f);
        println!("Generated: {} vertices, {} faces", p.len(), f.len());
        (p, f, Some(n), None::<Vec<[f64; 2]>>)
    } else if args.len() > 1 && args[1] == "--voronoi" {
        let subdivisions = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
        let num_cells = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
        println!("Generating Voronoi displacement sphere (subdivisions={}, cells={})", subdivisions, num_cells);
        let (p, f) = generate_voronoi_sphere(subdivisions, num_cells, 42);
        let n = quilting_remesh::geometry::compute_vertex_normals(&p, &f);
        println!("Generated: {} vertices, {} faces", p.len(), f.len());
        (p, f, Some(n), None::<Vec<[f64; 2]>>)
    } else if args.len() > 1 {
        let path = &args[1];
        println!("Loading {}", path);
        let data = std::fs::read(path).expect("failed to read file");
        let scene = quilting_gltf::load_gltf(&data).expect("failed to parse glTF");
        let mesh = &scene.meshes[0];
        let prim = &mesh.primitives[0];
        let normals = prim.normals.clone();
        let uvs = prim.uvs.as_ref().map(|u| u.iter().map(|v| [v[0], v[1]]).collect());
        println!("Loaded: {} vertices, {} faces", prim.positions.len(), prim.triangles.len());
        (prim.positions.clone(), prim.triangles.clone(), normals, uvs)
    } else {
        println!("Usage:");
        println!("  cargo run --example remesh_glb -- <file.glb> [target_patches]");
        println!("  cargo run --example remesh_glb -- --voronoi [subdivisions] [num_cells] [target_patches]");
        return;
    };

    let target = match args.get(1).map(|s| s.as_str()) {
        Some("--sphere") => args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20),
        Some("--cylinder") => args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20),
        Some("--voronoi") => args.get(4).and_then(|s| s.parse().ok()).unwrap_or(50),
        _ => args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500),
    };

    println!("\nRunning QEM simplification (target_patches={})...", target);
    let start = std::time::Instant::now();
    let result = quilting_remesh::remesh_simplified(
        &positions,
        &faces,
        target,
    );
    let elapsed = start.elapsed();

    match result {
        Ok(result) => {
            let s = &result.stats;
            println!("\n=== Remesh Results ===");
            println!("Time:              {:.2?}", elapsed);
            println!("Original faces:    {}", s.original_faces);
            println!("Clusters:          {}", s.num_clusters);
            println!("Fitted patches:    {}", s.num_patches);
            println!("Reduction ratio:   {:.1}x", s.reduction_ratio);
            println!("Skipped clusters:  {}", s.num_skipped);
            println!("Flipped patches:   {}", s.num_flipped);
            println!();
            println!("Position error:");
            println!("  RMS:             {:.6}", s.avg_position_error);
            println!("  Max:             {:.6}", s.max_position_error);
            println!();
            println!("Normal error:");
            println!("  RMS:             {:.2}°", s.avg_normal_error_degrees);
            println!("  Max:             {:.2}°", s.max_normal_error_degrees);
            println!();

            // Per-cluster error histogram
            let radius = quilting_remesh::geometry::bounding_radius(&positions);
            println!("Bounding radius:   {:.4}", radius);
            println!("Relative max err:  {:.4}%", 100.0 * s.max_position_error / radius);

            // Cluster size distribution
            let mut cluster_sizes: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
            for &label in &result.face_cluster_ids {
                *cluster_sizes.entry(label).or_insert(0) += 1;
            }
            let sizes: Vec<usize> = cluster_sizes.values().copied().collect();
            let min_size = sizes.iter().copied().min().unwrap_or(0);
            let max_size = sizes.iter().copied().max().unwrap_or(0);
            let avg_size = sizes.iter().sum::<usize>() as f64 / sizes.len().max(1) as f64;
            println!("\nCluster sizes:");
            println!("  Min:             {}", min_size);
            println!("  Max:             {}", max_size);
            println!("  Avg:             {:.1}", avg_size);

            // Weight deviation from identity
            let mut max_weight_dev = 0.0_f64;
            let mut sum_weight_dev = 0.0;
            for patch in &result.patches {
                for w in &patch.weights {
                    let dev = ((w.w - 1.0).powi(2) + w.x.powi(2) + w.y.powi(2) + w.z.powi(2)).sqrt();
                    max_weight_dev = max_weight_dev.max(dev);
                    sum_weight_dev += dev;
                }
            }
            let n_weights = (result.patches.len() * 3).max(1);
            // Normal error histogram
            let mut buckets = [0usize; 5]; // [0-10°, 10-30°, 30-60°, 60-90°, 90-180°]
            for &ne in &s.per_patch_normal_error {
                let idx = if ne < 10.0 { 0 } else if ne < 30.0 { 1 } else if ne < 60.0 { 2 } else if ne < 90.0 { 3 } else { 4 };
                buckets[idx] += 1;
            }
            let total_p = s.per_patch_normal_error.len().max(1);
            println!("\nNormal error distribution:");
            println!("  0-10°:   {:>5} ({:.1}%)", buckets[0], 100.0 * buckets[0] as f64 / total_p as f64);
            println!("  10-30°:  {:>5} ({:.1}%)", buckets[1], 100.0 * buckets[1] as f64 / total_p as f64);
            println!("  30-60°:  {:>5} ({:.1}%)", buckets[2], 100.0 * buckets[2] as f64 / total_p as f64);
            println!("  60-90°:  {:>5} ({:.1}%)", buckets[3], 100.0 * buckets[3] as f64 / total_p as f64);
            println!("  90-180°: {:>5} ({:.1}%)", buckets[4], 100.0 * buckets[4] as f64 / total_p as f64);

            println!("\nWeight deviation from identity:");
            println!("  Avg:             {:.6}", sum_weight_dev / n_weights as f64);
            println!("  Max:             {:.6}", max_weight_dev);

            // Export original mesh as OBJ (skip for very large meshes)
            if positions.len() < 10000 {
                let _ = export_original_obj("original_mesh.obj", &positions, &faces);
                println!("\nExported original:     original_mesh.obj");
            }

            // Export tessellated QB patches (subdivide each patch for smooth rendering)
            let _ = export_tessellated_obj("remeshed_tessellated.obj", &result.patches, 4);
            println!("Exported tessellated:  remeshed_tessellated.obj");

            // Export OBJ of control mesh
            let obj_path = "remeshed_control.obj";
            if let Err(e) = export_obj(obj_path, &result.patches) {
                eprintln!("Failed to write OBJ: {}", e);
            } else {
                println!("\nExported control mesh: {}", obj_path);
            }
        }
        Err(e) => {
            eprintln!("Remesh failed: {}", e);
        }
    }
}

/// Export QB patches tessellated at given subdivision level.
/// Each patch gets subdivided into sub_level^2 triangles.
fn export_tessellated_obj(path: &str, patches: &[quilting_core::patch::QBTriPatch], sub_level: usize) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "# Tessellated QB patches")?;
    writeln!(f, "# {} patches, sub_level={}", patches.len(), sub_level)?;

    let mut vertex_count = 0;
    for patch in patches {
        // Generate vertices on a triangular grid
        let n = sub_level + 1;
        let mut verts = Vec::new();
        for i in 0..n {
            for j in 0..(n - i) {
                let u = i as f64 / sub_level as f64;
                let v = j as f64 / sub_level as f64;
                let p = patch.eval(u, v);
                let pt = p.to_point();
                writeln!(f, "v {:.6} {:.6} {:.6}", pt[0], pt[1], pt[2])?;
                verts.push(vertex_count + 1); // OBJ is 1-indexed
                vertex_count += 1;
            }
        }

        // Generate triangle indices
        let mut idx = 0;
        for i in 0..sub_level {
            let row_len = n - i;
            let next_row_len = n - i - 1;
            for j in 0..(row_len - 1) {
                // Upper triangle
                let a = idx + j;
                let b = idx + j + 1;
                let c = idx + row_len + j;
                if c < verts.len() {
                    writeln!(f, "f {} {} {}", verts[a], verts[b], verts[c])?;
                }
                // Lower triangle (if not at the end)
                if j + 1 < next_row_len {
                    let d = idx + row_len + j + 1;
                    if d < verts.len() {
                        writeln!(f, "f {} {} {}", verts[b], verts[d], verts[c])?;
                    }
                }
            }
            idx += row_len;
        }
    }

    Ok(())
}

fn export_original_obj(path: &str, positions: &[[f64; 3]], faces: &[[usize; 3]]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "# Original mesh")?;
    writeln!(f, "# {} vertices, {} faces", positions.len(), faces.len())?;
    for p in positions {
        writeln!(f, "v {:.6} {:.6} {:.6}", p[0], p[1], p[2])?;
    }
    for face in faces {
        writeln!(f, "f {} {} {}", face[0] + 1, face[1] + 1, face[2] + 1)?;
    }
    Ok(())
}

fn export_obj(path: &str, patches: &[quilting_core::patch::QBTriPatch]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "# Remeshed QB control mesh")?;
    writeln!(f, "# {} patches", patches.len())?;

    // Merge vertices that are at the same position (shared corners)
    let mut unique_verts: Vec<[f64; 3]> = Vec::new();
    let mut vert_map: std::collections::HashMap<[i64; 3], usize> = std::collections::HashMap::new();
    let mut face_indices: Vec<[usize; 3]> = Vec::new();

    for patch in patches {
        let mut face = [0usize; 3];
        for (j, pos) in patch.positions.iter().enumerate() {
            let p = pos.to_point();
            // Quantize to merge nearby vertices (1e-6 precision)
            let key = [
                (p[0] * 1e6).round() as i64,
                (p[1] * 1e6).round() as i64,
                (p[2] * 1e6).round() as i64,
            ];
            let idx = match vert_map.get(&key) {
                Some(&idx) => idx,
                None => {
                    let idx = unique_verts.len();
                    unique_verts.push(p);
                    vert_map.insert(key, idx);
                    idx
                }
            };
            face[j] = idx;
        }
        face_indices.push(face);
    }

    for v in &unique_verts {
        writeln!(f, "v {:.6} {:.6} {:.6}", v[0], v[1], v[2])?;
    }
    for face in &face_indices {
        writeln!(f, "f {} {} {}", face[0] + 1, face[1] + 1, face[2] + 1)?;
    }

    let shared = patches.len() * 3 - unique_verts.len();
    writeln!(f, "# {} unique vertices ({} shared)", unique_verts.len(), shared)?;

    Ok(())
}

/// Generate a subdivided icosphere with Voronoi cell displacement.
fn generate_voronoi_sphere(subdivisions: u32, num_cells: usize, seed: u64) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    use quilting_remesh::geometry;

    // Start with icosahedron
    let (mut positions, mut faces) = quilting_core::shapes::icosahedron();

    // Subdivide
    for _ in 0..subdivisions {
        let (new_pos, new_faces) = subdivide_sphere(&positions, &faces);
        positions = new_pos;
        faces = new_faces;
    }

    // Normalize all vertices onto unit sphere
    for p in &mut positions {
        let len = geometry::vec3_len(*p);
        if len > 1e-10 {
            *p = geometry::vec3_scale(*p, 1.0 / len);
        }
    }

    // Generate random Voronoi cell centers on the sphere
    use rand::SeedableRng;
    use rand::Rng;
    let mut rng = rand_pcg::Pcg64::seed_from_u64(seed);
    let cell_centers: Vec<[f64; 3]> = (0..num_cells).map(|_| {
        // Random point on unit sphere via rejection sampling
        loop {
            let x: f64 = rng.gen_range(-1.0..1.0);
            let y: f64 = rng.gen_range(-1.0..1.0);
            let z: f64 = rng.gen_range(-1.0..1.0);
            let len = (x * x + y * y + z * z).sqrt();
            if len > 0.01 && len < 1.0 {
                return [x / len, y / len, z / len];
            }
        }
    }).collect();

    // Assign displacement based on distance to nearest Voronoi cell center
    // Cells get alternating heights for visual contrast
    let cell_heights: Vec<f64> = (0..num_cells).map(|i| {
        if i % 3 == 0 { 0.15 } else if i % 3 == 1 { 0.0 } else { 0.08 }
    }).collect();

    for p in &mut positions {
        // Find nearest and second-nearest cell centers in a single pass
        let mut min_dist = f64::MAX;
        let mut second_dist = f64::MAX;
        let mut nearest = 0;
        for (ci, center) in cell_centers.iter().enumerate() {
            let dot = geometry::vec3_dot(*p, *center).clamp(-1.0, 1.0);
            let dist = dot.acos();
            if dist < min_dist {
                second_dist = min_dist;
                min_dist = dist;
                nearest = ci;
            } else if dist < second_dist {
                second_dist = dist;
            }
        }

        let displacement = cell_heights[nearest];

        let boundary_factor = if second_dist < f64::MAX {
            let edge_dist = second_dist - min_dist;
            (edge_dist * 20.0).min(1.0)
        } else {
            1.0
        };

        let r = 1.0 + displacement * boundary_factor;
        *p = geometry::vec3_scale(*p, r);
    }

    (positions, faces)
}

/// Subdivide a triangle mesh on a sphere (midpoint subdivision + project to sphere).
fn subdivide_sphere(positions: &[[f64; 3]], faces: &[[usize; 3]]) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    use quilting_remesh::geometry;
    use std::collections::HashMap;

    let mut new_positions = positions.to_vec();
    let mut new_faces = Vec::with_capacity(faces.len() * 4);
    let mut midpoint_cache: HashMap<(usize, usize), usize> = HashMap::new();

    let mut get_midpoint = |a: usize, b: usize, positions: &mut Vec<[f64; 3]>| -> usize {
        let key = (a.min(b), a.max(b));
        if let Some(&idx) = midpoint_cache.get(&key) {
            return idx;
        }
        let mid = geometry::vec3_scale(
            geometry::vec3_add(positions[a], positions[b]),
            0.5,
        );
        let idx = positions.len();
        positions.push(mid);
        midpoint_cache.insert(key, idx);
        idx
    };

    for face in faces {
        let a = face[0];
        let b = face[1];
        let c = face[2];
        let ab = get_midpoint(a, b, &mut new_positions);
        let bc = get_midpoint(b, c, &mut new_positions);
        let ca = get_midpoint(c, a, &mut new_positions);

        new_faces.push([a, ab, ca]);
        new_faces.push([b, bc, ab]);
        new_faces.push([c, ca, bc]);
        new_faces.push([ab, bc, ca]);
    }

    (new_positions, new_faces)
}

/// Generate a unit icosphere with given subdivision level.
fn generate_sphere(subdivisions: u32) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let (mut positions, mut faces) = quilting_core::shapes::icosahedron();
    for _ in 0..subdivisions {
        let (new_pos, new_faces) = subdivide_sphere(&positions, &faces);
        positions = new_pos;
        faces = new_faces;
    }
    // Project onto unit sphere
    for p in &mut positions {
        let len = geometry::vec3_len(*p);
        if len > 1e-10 { *p = geometry::vec3_scale(*p, 1.0 / len); }
    }
    (positions, faces)
}

/// Generate a closed cylinder along the Y axis.
/// `height` = total length, `radius` = cross-section radius.
/// Capped at both ends.
fn generate_cylinder(segments: usize, rings: usize, height: f64, radius: f64) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut positions = Vec::new();
    let mut faces = Vec::new();

    // Generate ring vertices
    for ri in 0..=rings {
        let y = -height / 2.0 + height * (ri as f64 / rings as f64);
        for si in 0..segments {
            let angle = 2.0 * std::f64::consts::PI * (si as f64 / segments as f64);
            positions.push([radius * angle.cos(), y, radius * angle.sin()]);
        }
    }

    // Side faces: connect adjacent rings
    for ri in 0..rings {
        for si in 0..segments {
            let next_si = (si + 1) % segments;
            let a = ri * segments + si;
            let b = ri * segments + next_si;
            let c = (ri + 1) * segments + next_si;
            let d = (ri + 1) * segments + si;
            faces.push([a, b, c]);
            faces.push([a, c, d]);
        }
    }

    // Bottom cap
    let bottom_center = positions.len();
    positions.push([0.0, -height / 2.0, 0.0]);
    for si in 0..segments {
        let next_si = (si + 1) % segments;
        faces.push([bottom_center, next_si, si]); // reversed winding for bottom
    }

    // Top cap
    let top_center = positions.len();
    positions.push([0.0, height / 2.0, 0.0]);
    let top_ring_start = rings * segments;
    for si in 0..segments {
        let next_si = (si + 1) % segments;
        faces.push([top_center, top_ring_start + si, top_ring_start + next_si]);
    }

    (positions, faces)
}
