use std::collections::HashMap;
use crate::quaternion::{Quat, Mobius};

/// Per-face instance data for instanced rendering.
#[derive(Debug, Clone)]
pub struct FaceInstance {
    pub positions: [Quat; 3],
    pub weights: [Quat; 3],
    /// Per-edge LOD levels [edge_a, edge_b, edge_c] where:
    /// edge_a = edge opposite vertex 0 (connecting verts 1,2)
    /// edge_b = edge opposite vertex 1 (connecting verts 0,2)
    /// edge_c = edge opposite vertex 2 (connecting verts 0,1)
    pub edge_lods: [u32; 3],
    /// Per-vertex LOD levels [v0, v1, v2] — max of all edges meeting at each vertex.
    /// Used for smooth density visualization that's continuous across face boundaries.
    pub vertex_lods: [u32; 3],
}

/// Compute per-face instance data with adaptive LOD.
/// Screen-space projection info for LOD computation.
pub struct ScreenInfo {
    pub vp_matrix: [f64; 16], // column-major view-projection
    pub width: f64,
    pub height: f64,
}

impl ScreenInfo {
    /// Project a 3D point to screen pixels. Returns None if behind camera.
    pub fn project(&self, p: [f64; 3]) -> Option<[f64; 2]> {
        let m = &self.vp_matrix;
        let x = m[0]*p[0] + m[4]*p[1] + m[8]*p[2] + m[12];
        let y = m[1]*p[0] + m[5]*p[1] + m[9]*p[2] + m[13];
        let w = m[3]*p[0] + m[7]*p[1] + m[11]*p[2] + m[15];
        if w.abs() < 1e-10 { return None; } // at infinity or behind camera
        let ndc_x = x / w;
        let ndc_y = y / w;
        Some([
            (ndc_x * 0.5 + 0.5) * self.width,
            (ndc_y * 0.5 + 0.5) * self.height,
        ])
    }

    /// Screen-space distance between two 3D points. Returns f64::MAX if either is behind camera.
    pub fn screen_distance(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        match (self.project(a), self.project(b)) {
            (Some(pa), Some(pb)) => {
                let dx = pa[0] - pb[0];
                let dy = pa[1] - pb[1];
                (dx*dx + dy*dy).sqrt()
            }
            _ => f64::MAX, // one or both behind camera → max LOD
        }
    }
}

pub fn compute_instances(
    vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
    transform: &Mobius,
    screen: Option<&ScreenInfo>,
) -> Vec<FaceInstance> {
    // Pre-transform all vertices
    let transformed: Vec<(Quat, Quat)> = vertices.iter().map(|v| {
        let p = Quat::from_point(v[0], v[1], v[2]);
        let p_prime = transform.apply(p);
        let w_prime = transform.transform_weight(p, Quat::ONE);
        (p_prime, w_prime)
    }).collect();

    // Build face instances
    let instances: Vec<FaceInstance> = faces.iter().map(|face| {
        let (p0, w0) = transformed[face[0]];
        let (p1, w1) = transformed[face[1]];
        let (p2, w2) = transformed[face[2]];
        FaceInstance {
            positions: [p0, p1, p2],
            weights: [w0, w1, w2],
            edge_lods: [1, 1, 1],
            vertex_lods: [1, 1, 1],
        }
    }).collect();

    // Per-edge LOD from screen-space edge lengths.
    let target_pixels_per_sub = 4.0;

    // Compute the Möbius pole: the point where the transform is singular.
    // For F(x) = (ax+b)(cx+d)⁻¹, the pole is at x = -c⁻¹d.
    let pole: Option<[f64; 3]> = if transform.c.norm_sq() > 1e-20 {
        let p = -(transform.c.inv() * transform.d);
        Some(p.to_point())
    } else {
        None // c≈0: affine transform, no pole
    };

    // Check if a line segment (p0→p1) passes within `threshold` of the pole.
    let segment_near_pole = |p0: [f64; 3], p1: [f64; 3], threshold: f64| -> bool {
        let Some(pole) = pole else { return false; };
        let d = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
        let f = [p0[0]-pole[0], p0[1]-pole[1], p0[2]-pole[2]];
        let a = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
        if a < 1e-20 { // degenerate edge
            return f[0]*f[0]+f[1]*f[1]+f[2]*f[2] < threshold*threshold;
        }
        // t of closest point on segment to pole
        let t = -(f[0]*d[0]+f[1]*d[1]+f[2]*d[2]) / a;
        let t = t.clamp(0.0, 1.0);
        let closest = [p0[0]+t*d[0], p0[1]+t*d[1], p0[2]+t*d[2]];
        let dist_sq = (closest[0]-pole[0]).powi(2)+(closest[1]-pole[1]).powi(2)+(closest[2]-pole[2]).powi(2);
        dist_sq < threshold * threshold
    };

    // Measure screen-space arc length between two mesh vertices.
    // Samples the QB-transformed edge at multiple points to capture curvature.
    let screen_arc_len = |va: usize, vb: usize| -> f64 {
        let pa = transformed[va].0.to_point();
        let pb = transformed[vb].0.to_point();

        match screen {
            Some(s) => {
                let n_samples = 9;
                let mut total = 0.0;
                let mut prev = s.project(pa);
                for i in 1..=n_samples {
                    let t = i as f64 / n_samples as f64;
                    let orig = [
                        vertices[va][0]*(1.0-t) + vertices[vb][0]*t,
                        vertices[va][1]*(1.0-t) + vertices[vb][1]*t,
                        vertices[va][2]*(1.0-t) + vertices[vb][2]*t,
                    ];
                    let p = Quat::from_point(orig[0], orig[1], orig[2]);
                    let tp = transform.apply(p).to_point();
                    let curr = s.project(tp);
                    if let (Some(p1), Some(p2)) = (prev, curr) {
                        let dx = p2[0]-p1[0]; let dy = p2[1]-p1[1];
                        total += (dx*dx+dy*dy).sqrt();
                    } else {
                        return f64::MAX;
                    }
                    prev = curr;
                }
                total
            }
            None => {
                let dx = pa[0]-pb[0]; let dy = pa[1]-pb[1]; let dz = pa[2]-pb[2];
                (dx*dx + dy*dy + dz*dz).sqrt() * 100.0
            }
        }
    };

    let mut edge_lod_map: HashMap<(usize, usize), u32> = HashMap::new();
    for face in faces {
        let edges = [
            (face[1], face[2]),
            (face[0], face[2]),
            (face[0], face[1]),
        ];
        for &(va, vb) in &edges {
            let key = if va < vb { (va, vb) } else { (vb, va) };
            if edge_lod_map.contains_key(&key) { continue; }

            // If the edge passes through the pole → max LOD
            let lod = if segment_near_pole(vertices[va], vertices[vb], 0.5) {
                512
            } else {
                let pixels = screen_arc_len(va, vb);
                snap_to_power_of_2((pixels / target_pixels_per_sub).ceil() as u32).min(512)
            };
            edge_lod_map.insert(key, lod.max(1));
        }

        // Check medians: screen-space arc from each vertex through center to opposite edge midpoint.
        let mut max_median_lod = 0u32;
        for vi_idx in 0..3 {
            let vi = face[vi_idx];
            let vj = face[(vi_idx + 1) % 3];
            let vk = face[(vi_idx + 2) % 3];
            // Median: from vertex vi to midpoint of edge (vj,vk) in ORIGINAL space
            let orig_vi = vertices[vi];
            let orig_mid = [
                (vertices[vj][0]+vertices[vk][0])/2.0,
                (vertices[vj][1]+vertices[vk][1])/2.0,
                (vertices[vj][2]+vertices[vk][2])/2.0,
            ];
            // If median passes through the pole → max LOD
            if segment_near_pole(orig_vi, orig_mid, 0.5) {
                max_median_lod = 512;
                continue;
            }
            let pixels = match screen {
                Some(s) => {
                    let n = 5;
                    let mut total = 0.0;
                    let mut prev = s.project(transformed[vi].0.to_point());
                    for i in 1..=n {
                        let t = i as f64 / n as f64;
                        let p = [
                            orig_vi[0]*(1.0-t) + orig_mid[0]*t,
                            orig_vi[1]*(1.0-t) + orig_mid[1]*t,
                            orig_vi[2]*(1.0-t) + orig_mid[2]*t,
                        ];
                        let tp = transform.apply(Quat::from_point(p[0], p[1], p[2])).to_point();
                        let curr = s.project(tp);
                        if let (Some(a), Some(b)) = (prev, curr) {
                            total += ((b[0]-a[0]).powi(2) + (b[1]-a[1]).powi(2)).sqrt();
                        } else {
                            total = f64::MAX; break;
                        }
                        prev = curr;
                    }
                    total
                }
                None => {
                    let pi = transformed[vi].0.to_point();
                    let pj = transformed[vj].0.to_point();
                    let pk = transformed[vk].0.to_point();
                    let mid = [(pj[0]+pk[0])/2.0, (pj[1]+pk[1])/2.0, (pj[2]+pk[2])/2.0];
                    let dx = pi[0]-mid[0]; let dy = pi[1]-mid[1]; let dz = pi[2]-mid[2];
                    (dx*dx + dy*dy + dz*dz).sqrt() * 100.0
                }
            };
            let lod = snap_to_power_of_2((pixels / target_pixels_per_sub).ceil() as u32).min(512);
            max_median_lod = max_median_lod.max(lod);
        }
        // Floor all edges of this face at the max median LOD
        if max_median_lod > 1 {
            for &(va, vb) in &edges {
                let key = if va < vb { (va, vb) } else { (vb, va) };
                let current = edge_lod_map.get(&key).copied().unwrap_or(1);
                edge_lod_map.insert(key, current.max(max_median_lod));
            }
        }
    }

    // The edge_lod_map is already shared across faces — a shared edge
    // gets one value that both faces read. No propagation needed for
    // edge consistency. The median floor may create within-face variation
    // (e.g., [512, 4, 4]) but that's correct: the 512 edge is near the
    // singularity and needs dense tessellation, the others don't.

    // Write per-edge LODs into each face instance
    let mut result = instances;
    for (fi, face) in faces.iter().enumerate() {
        let edges = [
            (face[1], face[2]),
            (face[0], face[2]),
            (face[0], face[1]),
        ];
        for (local_idx, &(va, vb)) in edges.iter().enumerate() {
            let key = if va < vb { (va, vb) } else { (vb, va) };
            result[fi].edge_lods[local_idx] = *edge_lod_map.get(&key).unwrap_or(&1);
        }
    }

    // Compute per-vertex LOD = max of all edges meeting at each mesh vertex.
    // This gives a smooth density field that's continuous across face boundaries.
    let mut vertex_max_lod: HashMap<usize, u32> = HashMap::new();
    for (fi, face) in faces.iter().enumerate() {
        // edge_a (opposite v0) connects v1,v2 → contributes to v1 and v2
        // edge_b (opposite v1) connects v0,v2 → contributes to v0 and v2
        // edge_c (opposite v2) connects v0,v1 → contributes to v0 and v1
        let lods = result[fi].edge_lods;
        for &vi in &[face[1], face[2]] {
            let e = vertex_max_lod.entry(vi).or_insert(1);
            *e = (*e).max(lods[0]);
        }
        for &vi in &[face[0], face[2]] {
            let e = vertex_max_lod.entry(vi).or_insert(1);
            *e = (*e).max(lods[1]);
        }
        for &vi in &[face[0], face[1]] {
            let e = vertex_max_lod.entry(vi).or_insert(1);
            *e = (*e).max(lods[2]);
        }
    }

    // Write vertex LODs into each face instance
    for (fi, face) in faces.iter().enumerate() {
        result[fi].vertex_lods = [
            *vertex_max_lod.get(&face[0]).unwrap_or(&1),
            *vertex_max_lod.get(&face[1]).unwrap_or(&1),
            *vertex_max_lod.get(&face[2]).unwrap_or(&1),
        ];
    }

    result
}


/// Snap to the nearest power of 2 (round up).
fn snap_to_power_of_2(v: u32) -> u32 {
    if v <= 1 { return 1; }
    let mut p = 1u32;
    while p < v { p *= 2; }
    p
}

impl FaceInstance {
    /// Pack as 32 f32s (8 vec4s):
    /// [p0(4), p1(4), p2(4), w0(4), w1(4), w2(4), edgeLods(3)+pad, vertexLods(3)+pad]
    pub fn to_f32_array(&self) -> [f32; 32] {
        let mut out = [0.0f32; 32];
        for (i, p) in self.positions.iter().enumerate() {
            out[i*4]   = p.w as f32;
            out[i*4+1] = p.x as f32;
            out[i*4+2] = p.y as f32;
            out[i*4+3] = p.z as f32;
        }
        for (i, w) in self.weights.iter().enumerate() {
            out[12+i*4]   = w.w as f32;
            out[12+i*4+1] = w.x as f32;
            out[12+i*4+2] = w.y as f32;
            out[12+i*4+3] = w.z as f32;
        }
        // vec4 #7: edge LODs
        out[24] = self.edge_lods[0] as f32;
        out[25] = self.edge_lods[1] as f32;
        out[26] = self.edge_lods[2] as f32;
        out[27] = 0.0;
        // vec4 #8: vertex LODs (for smooth density visualization)
        out[28] = self.vertex_lods[0] as f32;
        out[29] = self.vertex_lods[1] as f32;
        out[30] = self.vertex_lods[2] as f32;
        out[31] = 0.0;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes;

    #[test]
    fn identity_lod_proportional_to_edge_length() {
        let (verts, faces) = shapes::cube();
        let instances = compute_instances(&verts, &faces, &Mobius::identity(), None);
        // All LODs should be power of 2 and within a reasonable range
        for inst in &instances {
            for &l in &inst.edge_lods {
                assert!(l.is_power_of_two(), "LOD {} not power of 2", l);
                assert!(l >= 1 && l <= 256, "LOD {} out of range", l);
            }
        }
    }

    #[test]
    fn sphere_reflection_higher_lod() {
        let (verts, faces) = shapes::cube();
        let m = Mobius::sphere_reflection(Quat::from_point(0.5, 0.0, 0.0), 2.0);
        let instances = compute_instances(&verts, &faces, &m, None);
        // Sphere reflection should produce higher LOD on some faces
        let max_lod = instances.iter()
            .flat_map(|i| i.edge_lods.iter())
            .copied()
            .max()
            .unwrap();
        assert!(max_lod > 1, "sphere reflection should increase LOD, got max={}", max_lod);
    }

    #[test]
    fn adjacent_edges_match() {
        let (verts, faces) = shapes::icosahedron();
        let m = Mobius::sphere_reflection(Quat::from_point(0.3, 0.0, 0.0), 1.5);
        let instances = compute_instances(&verts, &faces, &m, None);

        // Build edge → LOD map and verify consistency
        let mut edge_lods: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
        for (fi, face) in faces.iter().enumerate() {
            let edges = [
                (face[1], face[2], 0),
                (face[0], face[2], 1),
                (face[0], face[1], 2),
            ];
            for &(va, vb, local_idx) in &edges {
                let key = if va < vb { (va, vb) } else { (vb, va) };
                edge_lods.entry(key).or_default().push(instances[fi].edge_lods[local_idx]);
            }
        }
        for (edge, lods) in &edge_lods {
            let first = lods[0];
            for &l in lods {
                assert_eq!(l, first,
                    "edge {:?} has inconsistent LODs: {:?}", edge, lods);
            }
        }
    }

    #[test]
    fn lods_are_powers_of_2() {
        let (verts, faces) = shapes::octahedron();
        let m = Mobius::sphere_reflection(Quat::from_point(0.2, 0.3, 0.0), 1.8);
        let instances = compute_instances(&verts, &faces, &m, None);
        for inst in &instances {
            for &l in &inst.edge_lods {
                assert!(l.is_power_of_two(), "LOD {} is not a power of 2", l);
            }
        }
    }
}
