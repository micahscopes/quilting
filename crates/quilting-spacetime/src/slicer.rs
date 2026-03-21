/// Hyperplane slicing of 4D meshes.
///
/// Given a HyperMesh (triangle mesh x time) and a hyperplane in R^4,
/// compute the intersection as a 3D triangle mesh. The hyperplane is
/// defined by dot(normal, x) = offset where x = (x, y, z, t).

use std::collections::{HashMap, VecDeque};

use crate::hyper_mesh::HyperMesh;
use quilting_core::quaternion::{Quat, Mobius};

/// How to embed the time dimension in 4D.
#[derive(Clone, Copy)]
pub enum TimeEmbedding {
    /// Linear: q = (t, x, y, z). Time is the real part of the quaternion.
    Linear,
    /// Toroidal: time wraps in a circle of radius R in the (w, z) plane.
    /// q = (R·cos(2πt/period), x, y, R·sin(2πt/period))
    /// Produces closed surfaces when sliced — no boundary artifacts.
    Toroidal { radius: f64, period: f64 },
}

/// Defines a hyperplane in R^4 via dot(normal, x) = offset.
pub struct HyperplaneSlicer {
    /// Hyperplane normal in R^4 (nx, ny, nz, nt).
    pub normal: [f64; 4],
    /// Hyperplane offset: dot(normal, x) = offset.
    pub offset: f64,
    /// How to embed time in 4D.
    pub time_embedding: TimeEmbedding,
}

/// One connected layer of the slice result.
pub struct SliceLayer {
    /// 3D vertex positions (F(q).xyz after Möbius, or just q.xyz without).
    pub positions: Vec<[f64; 3]>,
    /// Per-vertex time values (when each vertex was "sampled").
    pub times: Vec<f64>,
    /// Per-vertex conformal weights from 4D Möbius: w = (c·q + d).
    /// When no 4D transform, these are all [1,0,0,0] (identity weight).
    pub weights: Vec<[f64; 4]>,
    /// Triangle indices.
    pub faces: Vec<[u32; 3]>,
}

/// Complete slice result -- possibly multiple disconnected layers.
pub struct SliceResult {
    pub layers: Vec<SliceLayer>,
}

impl HyperplaneSlicer {
    pub fn new(normal: [f64; 4], offset: f64) -> Self {
        Self { normal, offset, time_embedding: TimeEmbedding::Linear }
    }

    pub fn with_toroidal(mut self, radius: f64, period: f64) -> Self {
        self.time_embedding = TimeEmbedding::Toroidal { radius, period };
        self
    }

    /// Pure time slice: normal = (0,0,0,1), offset = t.
    pub fn at_time(t: f64) -> Self {
        Self {
            normal: [0.0, 0.0, 0.0, 1.0],
            offset: t,
            time_embedding: TimeEmbedding::Linear,
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
            time_embedding: TimeEmbedding::Linear,
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
        let default_weights: Vec<[f64; 4]> = vec![[1.0, 0.0, 0.0, 0.0]; positions.len()];
        let layers = group_into_layers(&positions, &times, &default_weights, &faces);

        SliceResult { layers }
    }

    /// Marching slicer: intersect hyperplane with each (face × keyframe) prism.
    ///
    /// Decomposes each triangular prism into 3 tetrahedra and uses the standard
    /// marching tetrahedra case table. Correct at any tilt angle.
    pub fn slice_marching(&self, mesh: &HyperMesh) -> SliceResult {
        self.slice_marching_4d(mesh, None)
    }

    /// Marching slicer with optional 4D Möbius transform applied BEFORE slicing.
    ///
    /// The Möbius acts on full quaternions q = (t, x, y, z) — the real part is time.
    /// This conformally deforms the 4D worldsheet before the hyperplane cuts it.
    pub fn slice_marching_4d(&self, mesh: &HyperMesh, transform_4d: Option<&Mobius>) -> SliceResult {
        let mut positions: Vec<[f64; 3]> = Vec::new();
        let mut times: Vec<f64> = Vec::new();
        let mut weights: Vec<[f64; 4]> = Vec::new();
        let mut faces: Vec<[u32; 3]> = Vec::new();

        let mut vert_map: HashMap<[i64; 4], u32> = HashMap::new();

        let add_vert = |pos: [f64; 3], t: f64, w: [f64; 4],
                        positions: &mut Vec<[f64; 3]>,
                        times: &mut Vec<f64>,
                        weights: &mut Vec<[f64; 4]>,
                        vert_map: &mut HashMap<[i64; 4], u32>| -> u32 {
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
            weights.push(w);
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

            // Time-range culling: skip segments far from the hyperplane.
            // Disabled when 4D transform is active — the transform remaps time
            // so pre-transform segment times don't predict post-transform intersection.
            let (seg_start, seg_end) = if transform_4d.is_some() || matches!(self.time_embedding, TimeEmbedding::Toroidal { .. }) {
                (0, n_segs)
            } else {
                let nt = self.normal[3];
                let spatial_mag = (self.normal[0].powi(2) + self.normal[1].powi(2) + self.normal[2].powi(2)).sqrt();
                let (t_lo, t_hi) = if nt.abs() > 0.01 {
                    let t_center = self.offset / nt;
                    let t_radius = 4.0 * spatial_mag / nt.abs();
                    (t_center - t_radius, t_center + t_radius)
                } else {
                    (f64::NEG_INFINITY, f64::INFINITY)
                };
                let start = traj0.segments.partition_point(|s| s.t_end < t_lo);
                let end = traj0.segments.partition_point(|s| s.t_start <= t_hi).min(n_segs);
                (start, end)
            };

            for si in seg_start..seg_end {
                let seg0 = &traj0.segments[si];
                let seg1 = &traj1.segments[si];
                let seg2 = &traj2.segments[si];

                // Original spatial positions (for output — what gets rendered)
                let spatial_pos: [[f64; 3]; 6] = [
                    seg0.pos_start, seg1.pos_start, seg2.pos_start,
                    seg0.pos_end,   seg1.pos_end,   seg2.pos_end,
                ];
                let spatial_t: [f64; 6] = [
                    seg0.t_start, seg0.t_start, seg0.t_start,
                    seg0.t_end,   seg0.t_end,   seg0.t_end,
                ];

                // Embedded coordinates (for hyperplane distance test).
                // Toroidal: maps time to a circle in (w, z) plane.
                // Linear: time goes straight into w.
                let mut embed_pos: [[f64; 3]; 6] = spatial_pos;
                let mut embed_t: [f64; 6] = spatial_t;
                let mut prism_weights: [[f64; 4]; 6] = [[1.0, 0.0, 0.0, 0.0]; 6];

                match self.time_embedding {
                    TimeEmbedding::Linear => {}
                    TimeEmbedding::Toroidal { radius, period } => {
                        for i in 0..6 {
                            let theta = std::f64::consts::TAU * spatial_t[i] / period;
                            // Torus circle in (w, z) plane:
                            // w = R·cos θ, z_embed = R·sin θ
                            // Spatial x, y unchanged. Spatial z replaced by torus sin.
                            embed_t[i] = radius * theta.cos();      // w component
                            embed_pos[i][2] = radius * theta.sin(); // z component = torus sin
                            // x, y stay as spatial
                        }
                    }
                }

                // Apply 4D Möbius if active
                if let Some(m) = transform_4d {
                    for i in 0..6 {
                        let q = Quat::new(embed_t[i], embed_pos[i][0], embed_pos[i][1], embed_pos[i][2]);
                        let w = m.transform_weight(q, Quat::ONE);
                        prism_weights[i] = [w.w, w.x, w.y, w.z];
                        let q_prime = m.apply(q);
                        embed_pos[i] = [q_prime.x, q_prime.y, q_prime.z];
                        embed_t[i] = q_prime.w;
                    }
                }

                // Distance uses EMBEDDED coordinates
                let n = self.normal;
                let o = self.offset;
                let d_arr: [f64; 6] = std::array::from_fn(|i| {
                    n[0]*embed_pos[i][0] + n[1]*embed_pos[i][1] + n[2]*embed_pos[i][2] + n[3]*embed_t[i] - o
                });
                let db0 = d_arr[0];
                let db1 = d_arr[1];
                let db2 = d_arr[2];
                let dt0 = d_arr[3];
                let dt1 = d_arr[4];
                let dt2 = d_arr[5];

                // Quick reject: all same sign = no intersection
                let all_pos = db0 > 0.0 && db1 > 0.0 && db2 > 0.0
                           && dt0 > 0.0 && dt1 > 0.0 && dt2 > 0.0;
                let all_neg = db0 < 0.0 && db1 < 0.0 && db2 < 0.0
                           && dt0 < 0.0 && dt1 < 0.0 && dt2 < 0.0;
                if all_pos || all_neg {
                    continue;
                }

                // Direct prism-plane intersection.
                // 9 prism edges: 3 bottom, 3 top, 3 vertical.
                // Find all edge crossings, form a convex polygon, fan-triangulate.
                // Use SPATIAL positions for output, EMBEDDED distances for intersection
                let prism_verts = spatial_pos;
                let prism_times_arr = spatial_t;
                let prism_dists = d_arr;

                // 9 edges of a triangular prism
                let edges: [(usize, usize); 9] = [
                    (0,1), (1,2), (2,0),  // bottom triangle
                    (3,4), (4,5), (5,3),  // top triangle
                    (0,3), (1,4), (2,5),  // vertical edges
                ];

                // (time, position, weight) per crossing
                let mut crossings: Vec<(f64, [f64; 3], [f64; 4])> = Vec::new();

                for &(a, b) in &edges {
                    let da = prism_dists[a];
                    let db = prism_dists[b];
                    if (da > 0.0) != (db > 0.0) {
                        let t_param = da / (da - db);
                        let pa = prism_verts[a];
                        let pb = prism_verts[b];
                        let ta = prism_times_arr[a];
                        let tb = prism_times_arr[b];
                        let wa = prism_weights[a];
                        let wb = prism_weights[b];
                        let pos = [
                            pa[0] + t_param * (pb[0] - pa[0]),
                            pa[1] + t_param * (pb[1] - pa[1]),
                            pa[2] + t_param * (pb[2] - pa[2]),
                        ];
                        let time = ta + t_param * (tb - ta);
                        let weight = [
                            wa[0] + t_param * (wb[0] - wa[0]),
                            wa[1] + t_param * (wb[1] - wa[1]),
                            wa[2] + t_param * (wb[2] - wa[2]),
                            wa[3] + t_param * (wb[3] - wa[3]),
                        ];
                        crossings.push((time, pos, weight));
                    }
                }

                if crossings.len() < 3 {
                    continue;
                }

                // Sort crossings around the polygon centroid so fan
                // triangulation produces correct non-overlapping triangles.
                let cx = crossings.iter().map(|c| c.1[0]).sum::<f64>() / crossings.len() as f64;
                let cy = crossings.iter().map(|c| c.1[1]).sum::<f64>() / crossings.len() as f64;
                let cz = crossings.iter().map(|c| c.1[2]).sum::<f64>() / crossings.len() as f64;

                // Compute polygon normal from first two edges
                let e1 = [crossings[1].1[0]-crossings[0].1[0],
                          crossings[1].1[1]-crossings[0].1[1],
                          crossings[1].1[2]-crossings[0].1[2]];
                let e2 = [crossings[2].1[0]-crossings[0].1[0],
                          crossings[2].1[1]-crossings[0].1[1],
                          crossings[2].1[2]-crossings[0].1[2]];
                let pn = [e1[1]*e2[2]-e1[2]*e2[1],
                          e1[2]*e2[0]-e1[0]*e2[2],
                          e1[0]*e2[1]-e1[1]*e2[0]];

                // Reference direction from centroid to first crossing
                let ref_dir = [crossings[0].1[0]-cx, crossings[0].1[1]-cy, crossings[0].1[2]-cz];

                // Sort by angle around the normal axis
                crossings.sort_by(|a, b| {
                    let da = [a.1[0]-cx, a.1[1]-cy, a.1[2]-cz];
                    let db = [b.1[0]-cx, b.1[1]-cy, b.1[2]-cz];
                    let angle_a = {
                        let dot = da[0]*ref_dir[0]+da[1]*ref_dir[1]+da[2]*ref_dir[2];
                        let cross = pn[0]*(da[1]*ref_dir[2]-da[2]*ref_dir[1])
                                  + pn[1]*(da[2]*ref_dir[0]-da[0]*ref_dir[2])
                                  + pn[2]*(da[0]*ref_dir[1]-da[1]*ref_dir[0]);
                        cross.atan2(dot)
                    };
                    let angle_b = {
                        let dot = db[0]*ref_dir[0]+db[1]*ref_dir[1]+db[2]*ref_dir[2];
                        let cross = pn[0]*(db[1]*ref_dir[2]-db[2]*ref_dir[1])
                                  + pn[1]*(db[2]*ref_dir[0]-db[0]*ref_dir[2])
                                  + pn[2]*(db[0]*ref_dir[1]-db[1]*ref_dir[0]);
                        cross.atan2(dot)
                    };
                    angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
                });

                // Fan triangulate
                let i0 = add_vert(crossings[0].1, crossings[0].0, crossings[0].2,
                    &mut positions, &mut times, &mut weights, &mut vert_map);
                for j in 1..crossings.len() - 1 {
                    let i1 = add_vert(crossings[j].1, crossings[j].0, crossings[j].2,
                        &mut positions, &mut times, &mut weights, &mut vert_map);
                    let i2 = add_vert(crossings[j+1].1, crossings[j+1].0, crossings[j+1].2,
                        &mut positions, &mut times, &mut weights, &mut vert_map);
                    if i0 != i1 && i1 != i2 && i0 != i2 {
                        faces.push([i0, i1, i2]);
                    }
                }
            }
        }

        if faces.is_empty() {
            return SliceResult { layers: vec![] };
        }

        let layers = group_into_layers(&positions, &times, &weights, &faces);
        SliceResult { layers }
    }
}

/// Group triangles into connected components by shared vertices.
fn group_into_layers(
    positions: &[[f64; 3]],
    times: &[f64],
    weights: &[[f64; 4]],
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
        let mut layer_weights = Vec::new();
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
                    layer_weights.push(weights[v as usize]);
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
            weights: layer_weights,
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
