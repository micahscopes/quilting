/// Hyperplane slicing of 4D meshes.
///
/// Given a HyperMesh (triangle mesh x time) and a hyperplane in R^4,
/// compute the intersection as a 3D triangle mesh. The hyperplane is
/// defined by dot(normal, x) = offset where x = (x, y, z, t).

use std::collections::{HashMap, VecDeque};

use crate::hyper_mesh::HyperMesh;

/// Defines a hyperplane in R^4 via dot(normal, x) = offset.
pub struct HyperplaneSlicer {
    /// Hyperplane normal in R^4 (nx, ny, nz, nt).
    pub normal: [f64; 4],
    /// Hyperplane offset: dot(normal, x) = offset.
    pub offset: f64,
}

/// One connected layer of the slice result.
pub struct SliceLayer {
    /// 3D vertex positions.
    pub positions: Vec<[f64; 3]>,
    /// Per-vertex time values (when each vertex was "sampled").
    pub times: Vec<f64>,
    /// Triangle indices.
    pub faces: Vec<[u32; 3]>,
}

/// Complete slice result -- possibly multiple disconnected layers.
pub struct SliceResult {
    pub layers: Vec<SliceLayer>,
}

impl HyperplaneSlicer {
    pub fn new(normal: [f64; 4], offset: f64) -> Self {
        Self { normal, offset }
    }

    /// Pure time slice: normal = (0,0,0,1), offset = t.
    pub fn at_time(t: f64) -> Self {
        Self {
            normal: [0.0, 0.0, 0.0, 1.0],
            offset: t,
        }
    }

    /// Tilted slice: mixing a spatial direction with time.
    ///
    /// The normal is (spatial_dir * sin(tilt), cos(tilt)) and offset shifts
    /// along the time axis.
    pub fn tilted(spatial_dir: [f64; 3], tilt_angle: f64, time_offset: f64) -> Self {
        let sin_t = tilt_angle.sin();
        let cos_t = tilt_angle.cos();
        let normal = [
            spatial_dir[0] * sin_t,
            spatial_dir[1] * sin_t,
            spatial_dir[2] * sin_t,
            cos_t,
        ];
        Self {
            normal,
            offset: time_offset * cos_t,
        }
    }

    /// Slice a HyperMesh, returning all intersection layers.
    ///
    /// Algorithm:
    /// 1. For each vertex, find all hyperplane intersections along its trajectory.
    /// 2. For each face, match vertex intersections into coherent triangles.
    /// 3. Group connected triangles into separate layers.
    pub fn slice(&self, mesh: &HyperMesh) -> SliceResult {
        // Step 1: per-vertex intersections.
        // If the solver finds no roots but the slice is within the trajectory's
        // time range, fall back to evaluating at the nearest time. This handles
        // boundary cases where the cubic solver misses roots at segment endpoints.
        let vertex_hits: Vec<Vec<(f64, [f64; 3])>> = mesh
            .trajectories
            .iter()
            .map(|traj| traj.intersect_hyperplane(self.normal, self.offset))
            .collect();

        // Step 2: for each face, assemble triangles from vertex intersections.
        // We build a global vertex buffer and face list, deduplicating vertices
        // by (vertex_index, hit_index) pairs.
        let mut positions: Vec<[f64; 3]> = Vec::new();
        let mut times: Vec<f64> = Vec::new();
        let mut faces: Vec<[u32; 3]> = Vec::new();

        // Map from (original_vertex_idx, hit_idx) -> output vertex index
        let mut vertex_map: HashMap<(u32, usize), u32> = HashMap::new();

        let mut get_or_insert_vertex =
            |vertex_map: &mut HashMap<(u32, usize), u32>,
             positions: &mut Vec<[f64; 3]>,
             times: &mut Vec<f64>,
             v_idx: u32,
             hit_idx: usize,
             t: f64,
             pos: [f64; 3]|
             -> u32 {
                let key = (v_idx, hit_idx);
                if let Some(&idx) = vertex_map.get(&key) {
                    return idx;
                }
                let idx = positions.len() as u32;
                positions.push(pos);
                times.push(t);
                vertex_map.insert(key, idx);
                idx
            };

        for face in &mesh.faces {
            let [v0, v1, v2] = *face;
            let h0 = &vertex_hits[v0 as usize];
            let h1 = &vertex_hits[v1 as usize];
            let h2 = &vertex_hits[v2 as usize];

            if h0.is_empty() || h1.is_empty() || h2.is_empty() {
                continue;
            }

            // Simple case: each vertex has exactly 1 hit -- emit one triangle,
            // but only if the intersection times are temporally coherent.
            if h0.len() == 1 && h1.len() == 1 && h2.len() == 1 {
                let t_min_face = h0[0].0.min(h1[0].0).min(h2[0].0);
                let t_max_face = h0[0].0.max(h1[0].0).max(h2[0].0);
                // Skip if vertices span too much time — the triangle would
                // stretch across multiple rotation periods
                if t_max_face - t_min_face > mesh.period * 0.5 {
                    continue;
                }
                let i0 = get_or_insert_vertex(
                    &mut vertex_map,
                    &mut positions,
                    &mut times,
                    v0,
                    0,
                    h0[0].0,
                    h0[0].1,
                );
                let i1 = get_or_insert_vertex(
                    &mut vertex_map,
                    &mut positions,
                    &mut times,
                    v1,
                    0,
                    h1[0].0,
                    h1[0].1,
                );
                let i2 = get_or_insert_vertex(
                    &mut vertex_map,
                    &mut positions,
                    &mut times,
                    v2,
                    0,
                    h2[0].0,
                    h2[0].1,
                );
                faces.push([i0, i1, i2]);
                continue;
            }

            // Multi-intersection case: match by temporal proximity.
            // For each combination of hits, check if they're in a compatible
            // time window (all times within the same keyframe interval).
            let hits = [
                (v0, h0.as_slice()),
                (v1, h1.as_slice()),
                (v2, h2.as_slice()),
            ];
            let max_combos = hits[0].1.len() * hits[1].1.len() * hits[2].1.len();
            if max_combos > 1000 {
                // Pathological case, skip to avoid blowup
                continue;
            }

            for (i, &(t0, pos0)) in hits[0].1.iter().enumerate() {
                for (j, &(t1, pos1)) in hits[1].1.iter().enumerate() {
                    for (k, &(t2, pos2)) in hits[2].1.iter().enumerate() {
                        // Check temporal coherence: all three intersection times
                        // should be "close" -- within the same animation segment.
                        let t_min = t0.min(t1).min(t2);
                        let t_max = t0.max(t1).max(t2);
                        let (anim_start, anim_end) = mesh.time_range();
                        let anim_len = (anim_end - anim_start).max(1e-10);
                        let threshold = anim_len * 0.5; // generous threshold

                        if t_max - t_min > threshold {
                            continue;
                        }

                        let i0 = get_or_insert_vertex(
                            &mut vertex_map,
                            &mut positions,
                            &mut times,
                            hits[0].0,
                            i,
                            t0,
                            pos0,
                        );
                        let i1 = get_or_insert_vertex(
                            &mut vertex_map,
                            &mut positions,
                            &mut times,
                            hits[1].0,
                            j,
                            t1,
                            pos1,
                        );
                        let i2 = get_or_insert_vertex(
                            &mut vertex_map,
                            &mut positions,
                            &mut times,
                            hits[2].0,
                            k,
                            t2,
                            pos2,
                        );
                        faces.push([i0, i1, i2]);
                    }
                }
            }
        }

        if faces.is_empty() {
            return SliceResult { layers: vec![] };
        }

        // Step 3: group connected triangles into layers via flood fill.
        let layers = group_into_layers(&positions, &times, &faces);

        SliceResult { layers }
    }

    /// Marching slicer: intersect hyperplane with each (face × keyframe) prism.
    ///
    /// Decomposes each triangular prism into 3 tetrahedra and uses the standard
    /// marching tetrahedra case table. Correct at any tilt angle.
    pub fn slice_marching(&self, mesh: &HyperMesh) -> SliceResult {
        let mut positions: Vec<[f64; 3]> = Vec::new();
        let mut times: Vec<f64> = Vec::new();
        let mut faces: Vec<[u32; 3]> = Vec::new();

        // Deduplicate vertices by (position hash) to merge shared edges
        let mut vert_map: HashMap<[i64; 4], u32> = HashMap::new();

        let mut add_vert = |pos: [f64; 3], t: f64,
                            positions: &mut Vec<[f64; 3]>,
                            times: &mut Vec<f64>,
                            vert_map: &mut HashMap<[i64; 4], u32>| -> u32 {
            // Quantize to ~1e-8 resolution for dedup
            let key = [
                (pos[0] * 1e8) as i64,
                (pos[1] * 1e8) as i64,
                (pos[2] * 1e8) as i64,
                (t * 1e8) as i64,
            ];
            if let Some(&idx) = vert_map.get(&key) {
                return idx;
            }
            let idx = positions.len() as u32;
            positions.push(pos);
            times.push(t);
            vert_map.insert(key, idx);
            idx
        };

        // For each face, iterate over all keyframe segments
        for face in &mesh.faces {
            let [v0, v1, v2] = *face;
            let traj0 = &mesh.trajectories[v0 as usize];
            let traj1 = &mesh.trajectories[v1 as usize];
            let traj2 = &mesh.trajectories[v2 as usize];

            // All trajectories should have the same segment count (after padding)
            let n_segs = traj0.segments.len().min(traj1.segments.len()).min(traj2.segments.len());

            for si in 0..n_segs {
                let seg0 = &traj0.segments[si];
                let seg1 = &traj1.segments[si];
                let seg2 = &traj2.segments[si];

                // 6 prism vertices: bottom (t_start) and top (t_end)
                let t_bot = seg0.t_start;
                let t_top = seg0.t_end;
                let b0 = seg0.pos_start;
                let b1 = seg1.pos_start;
                let b2 = seg2.pos_start;
                let t0 = seg0.pos_end;
                let t1 = seg1.pos_end;
                let t2 = seg2.pos_end;

                // Signed distance to hyperplane for each vertex
                let n = self.normal;
                let o = self.offset;
                let d = |pos: [f64; 3], t: f64| -> f64 {
                    n[0]*pos[0] + n[1]*pos[1] + n[2]*pos[2] + n[3]*t - o
                };
                let db0 = d(b0, t_bot);
                let db1 = d(b1, t_bot);
                let db2 = d(b2, t_bot);
                let dt0 = d(t0, t_top);
                let dt1 = d(t1, t_top);
                let dt2 = d(t2, t_top);

                // Quick reject: all same sign = no intersection
                let all_pos = db0 > 0.0 && db1 > 0.0 && db2 > 0.0
                           && dt0 > 0.0 && dt1 > 0.0 && dt2 > 0.0;
                let all_neg = db0 < 0.0 && db1 < 0.0 && db2 < 0.0
                           && dt0 < 0.0 && dt1 < 0.0 && dt2 < 0.0;
                if all_pos || all_neg {
                    continue;
                }

                // Decompose prism into 3 tetrahedra:
                // Tet 0: b0, b1, b2, t0
                // Tet 1: b1, b2, t0, t1
                // Tet 2: b2, t0, t1, t2
                let prism_verts = [b0, b1, b2, t0, t1, t2];
                let prism_times = [t_bot, t_bot, t_bot, t_top, t_top, t_top];
                let prism_dists = [db0, db1, db2, dt0, dt1, dt2];

                let tets: [[usize; 4]; 3] = [
                    [0, 1, 2, 3],
                    [1, 2, 3, 4],
                    [2, 3, 4, 5],
                ];

                for tet in &tets {
                    let d0 = prism_dists[tet[0]];
                    let d1 = prism_dists[tet[1]];
                    let d2 = prism_dists[tet[2]];
                    let d3 = prism_dists[tet[3]];

                    // Classify vertices: 4-bit case index
                    let case = ((d0 > 0.0) as u8)
                             | (((d1 > 0.0) as u8) << 1)
                             | (((d2 > 0.0) as u8) << 2)
                             | (((d3 > 0.0) as u8) << 3);

                    // Skip no-intersection cases
                    if case == 0 || case == 15 {
                        continue;
                    }

                    // Interpolate edge crossing point
                    let lerp_edge = |a: usize, b: usize| -> (f64, [f64; 3]) {
                        let da = prism_dists[tet[a]];
                        let db = prism_dists[tet[b]];
                        let t_param = da / (da - db);
                        let pa = prism_verts[tet[a]];
                        let pb = prism_verts[tet[b]];
                        let ta = prism_times[tet[a]];
                        let tb = prism_times[tet[b]];
                        let pos = [
                            pa[0] + t_param * (pb[0] - pa[0]),
                            pa[1] + t_param * (pb[1] - pa[1]),
                            pa[2] + t_param * (pb[2] - pa[2]),
                        ];
                        let time = ta + t_param * (tb - ta);
                        (time, pos)
                    };

                    // Marching tet case table: edges are 01, 02, 03, 12, 13, 23
                    // Each case produces 1 or 2 triangles from edge crossings
                    let edge_tris: &[&[(usize, usize)]] = match case {
                        // 1 vertex inside: 1 triangle
                        1  => &[&[(0,1), (0,2), (0,3)]],
                        2  => &[&[(0,1), (1,3), (1,2)]],
                        4  => &[&[(0,2), (2,3), (1,2)]],  // fixed winding
                        8  => &[&[(0,3), (1,3), (2,3)]],
                        // 3 vertices inside (complement): 1 triangle, reversed
                        14 => &[&[(0,1), (0,3), (0,2)]],
                        13 => &[&[(0,1), (1,2), (1,3)]],
                        11 => &[&[(0,2), (1,2), (2,3)]],
                        7  => &[&[(0,3), (2,3), (1,3)]],
                        // 2 vertices inside: 2 triangles (quad split)
                        3  => &[&[(0,2), (0,3), (1,2)], &[(1,2), (0,3), (1,3)]],
                        5  => &[&[(0,1), (0,3), (1,2)], &[(1,2), (0,3), (2,3)]],
                        6  => &[&[(0,1), (0,2), (1,3)], &[(0,2), (2,3), (1,3)]],
                        9  => &[&[(0,1), (1,3), (0,2)], &[(0,2), (1,3), (2,3)]],
                        10 => &[&[(0,1), (0,3), (1,2)], &[(1,2), (0,3), (2,3)]],
                        12 => &[&[(0,2), (0,3), (1,2)], &[(1,2), (0,3), (1,3)]],
                        _ => continue,
                    };

                    for tri_edges in edge_tris {
                        if tri_edges.len() < 3 { continue; }
                        let (t_a, p_a) = lerp_edge(tri_edges[0].0, tri_edges[0].1);
                        let (t_b, p_b) = lerp_edge(tri_edges[1].0, tri_edges[1].1);
                        let (t_c, p_c) = lerp_edge(tri_edges[2].0, tri_edges[2].1);

                        let ia = add_vert(p_a, t_a, &mut positions, &mut times, &mut vert_map);
                        let ib = add_vert(p_b, t_b, &mut positions, &mut times, &mut vert_map);
                        let ic = add_vert(p_c, t_c, &mut positions, &mut times, &mut vert_map);

                        if ia != ib && ib != ic && ia != ic {
                            faces.push([ia, ib, ic]);
                        }
                    }
                }
            }
        }

        if faces.is_empty() {
            return SliceResult { layers: vec![] };
        }

        let layers = group_into_layers(&positions, &times, &faces);
        SliceResult { layers }
    }
}

/// Group triangles into connected components by shared vertices.
fn group_into_layers(
    positions: &[[f64; 3]],
    times: &[f64],
    faces: &[[u32; 3]],
) -> Vec<SliceLayer> {
    let num_faces = faces.len();
    if num_faces == 0 {
        return vec![];
    }

    // Build vertex -> face adjacency
    let mut vert_to_faces: HashMap<u32, Vec<usize>> = HashMap::new();
    for (fi, face) in faces.iter().enumerate() {
        for &v in face {
            vert_to_faces.entry(v).or_default().push(fi);
        }
    }

    // BFS flood fill to find connected components
    let mut visited = vec![false; num_faces];
    let mut layers = Vec::new();

    for start_face in 0..num_faces {
        if visited[start_face] {
            continue;
        }

        let mut component_faces: Vec<usize> = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start_face);
        visited[start_face] = true;

        while let Some(fi) = queue.pop_front() {
            component_faces.push(fi);

            // Find all faces sharing a vertex with this face
            for &v in &faces[fi] {
                if let Some(adj_faces) = vert_to_faces.get(&v) {
                    for &adj_fi in adj_faces {
                        if !visited[adj_fi] {
                            visited[adj_fi] = true;
                            queue.push_back(adj_fi);
                        }
                    }
                }
            }
        }

        // Build the layer: remap vertices to a compact range
        let mut old_to_new: HashMap<u32, u32> = HashMap::new();
        let mut layer_positions = Vec::new();
        let mut layer_times = Vec::new();
        let mut layer_faces = Vec::new();

        for &fi in &component_faces {
            let mut new_face = [0u32; 3];
            for (i, &v) in faces[fi].iter().enumerate() {
                let new_v = if let Some(&nv) = old_to_new.get(&v) {
                    nv
                } else {
                    let nv = layer_positions.len() as u32;
                    layer_positions.push(positions[v as usize]);
                    layer_times.push(times[v as usize]);
                    old_to_new.insert(v, nv);
                    nv
                };
                new_face[i] = new_v;
            }
            layer_faces.push(new_face);
        }

        layers.push(SliceLayer {
            positions: layer_positions,
            times: layer_times,
            faces: layer_faces,
        });
    }

    layers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesize;

    #[test]
    fn time_slice_rotating_cube_at_t0() {
        let mesh = synthesize::rotating_cube(1.0, std::f64::consts::TAU, 16);
        let slicer = HyperplaneSlicer::at_time(0.0);
        let result = slicer.slice(&mesh);

        assert!(!result.layers.is_empty(), "Should have at least one layer");

        let total_faces: usize = result.layers.iter().map(|l| l.faces.len()).sum();
        // Subdivided cube has many more than 12 faces
        assert!(total_faces >= 12, "Should have at least 12 triangles at t=0, got {}", total_faces);
    }

    #[test]
    fn time_slice_rotating_cube_at_half() {
        let mesh = synthesize::rotating_cube(1.0, std::f64::consts::TAU, 16);
        let slicer = HyperplaneSlicer::at_time(0.5);
        let result = slicer.slice(&mesh);

        assert!(!result.layers.is_empty(), "Should have at least one layer");

        let total_faces: usize = result.layers.iter().map(|l| l.faces.len()).sum();
        assert!(total_faces >= 12, "Rotated cube should have at least 12 triangles, got {}", total_faces);
    }

    #[test]
    fn tilted_slice_breathing_sphere() {
        let mesh = synthesize::breathing_sphere(2.0, 1.0, 0.3, 2);
        let slicer = HyperplaneSlicer::tilted([1.0, 0.0, 0.0], 0.3, 1.0);
        let result = slicer.slice(&mesh);

        // Should produce some geometry
        let total_faces: usize = result.layers.iter().map(|l| l.faces.len()).sum();
        assert!(total_faces > 0, "Tilted slice should produce triangles");

        // Check manifold-ness: every vertex should appear in at least one face
        for layer in &result.layers {
            let mut used = vec![false; layer.positions.len()];
            for face in &layer.faces {
                for &v in face {
                    used[v as usize] = true;
                }
            }
            assert!(
                used.iter().all(|&u| u),
                "All vertices should be referenced by faces"
            );
        }
    }

    #[test]
    fn at_time_constructor() {
        let slicer = HyperplaneSlicer::at_time(2.5);
        assert!((slicer.normal[3] - 1.0).abs() < 1e-12);
        assert!((slicer.offset - 2.5).abs() < 1e-12);
    }

    #[test]
    fn empty_slice_outside_range() {
        let mesh = synthesize::rotating_cube(1.0, 1.0, 4);
        let slicer = HyperplaneSlicer::at_time(100.0);
        let result = slicer.slice(&mesh);
        let total_faces: usize = result.layers.iter().map(|l| l.faces.len()).sum();
        assert_eq!(total_faces, 0, "No geometry outside time range");
    }
}
