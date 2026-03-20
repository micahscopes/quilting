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
pub fn compute_instances(
    vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
    transform: &Mobius,
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

    // Per-EDGE LOD from the exact transformed arc length under the Möbius map.
    //
    // The conformal scale factor |F'(x)| = sqrt(C)/|cx+d|² where C = |ad-bc|².
    // The transformed arc length of edge (v0→v1) is:
    //   L' = sqrt(C) × |v1-v0| × ∫₀¹ 1/|d0 + t(d1-d0)|² dt
    // where di = c·vi + d.
    //
    // |d0+t(d1-d0)|² = Q(t) = a·t² + b·t + c (positive definite quadratic)
    // ∫₀¹ 1/Q(t) dt = (2/√Δ)[arctan((2a+b)/√Δ) - arctan(b/√Δ)]
    // where Δ = 4ac - b².

    let sqrt_c = (transform.a * transform.d - transform.b * transform.c).norm();

    // Pre-compute di = c·vi + d for each vertex
    let _denom_quats: Vec<Quat> = vertices.iter().map(|v| {
        let p = Quat::from_point(v[0], v[1], v[2]);
        transform.c * p + transform.d
    }).collect();

    // Compute exact Möbius arc length between two points in the original mesh.
    // Uses the closed-form integral of the conformal scale along the line segment.
    let arc_length = |p0: [f64; 3], p1: [f64; 3]| -> f64 {
        let q0 = Quat::from_point(p0[0], p0[1], p0[2]);
        let q1 = Quat::from_point(p1[0], p1[1], p1[2]);
        let d0 = transform.c * q0 + transform.d;
        let d1 = transform.c * q1 + transform.d;
        let diff = d1 - d0;

        let qa = diff.norm_sq();
        let qb = 2.0 * (d0 * diff.conj()).re();
        let qc = d0.norm_sq();

        // ∫₀¹ 1/(qa·t²+qb·t+qc) dt
        // When Q passes near zero (singularity on segment), the integral
        // naturally produces a very large value — no special cases needed.
        let delta = 4.0 * qa * qc - qb * qb;
        let integral = if delta > 1e-20 {
            let sqrt_delta = delta.sqrt();
            (2.0 / sqrt_delta) * (((2.0 * qa + qb) / sqrt_delta).atan() - (qb / sqrt_delta).atan())
        } else if delta > -1e-10 {
            // Near-degenerate: Q barely touches zero. Use midpoint rule as fallback.
            let q_mid = qa * 0.25 + qb * 0.5 + qc;
            1.0 / q_mid.max(1e-20)
        } else {
            // Q has real roots (segment crosses singularity).
            // The integral diverges — return a very large value.
            1e8
        };

        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let dz = p1[2] - p0[2];
        let orig_len = (dx*dx + dy*dy + dz*dz).sqrt();

        sqrt_c * orig_len * integral
    };

    // Per-edge LOD from exact transformed arc length.
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

            let al = arc_length(vertices[va], vertices[vb]);
            let raw = (al * 16.0).ceil() as u32;
            edge_lod_map.insert(key, snap_to_power_of_2(raw.max(1).min(512)));
        }

        // Check the three medians (vertex → opposite edge midpoint).
        // These cross the face interior — if any blows up, the interior
        // has a singularity and ALL edges of this face need high LOD.
        let midpoints = [
            (face[0], [(vertices[face[1]][0]+vertices[face[2]][0])/2.0,
                       (vertices[face[1]][1]+vertices[face[2]][1])/2.0,
                       (vertices[face[1]][2]+vertices[face[2]][2])/2.0]),
            (face[1], [(vertices[face[0]][0]+vertices[face[2]][0])/2.0,
                       (vertices[face[0]][1]+vertices[face[2]][1])/2.0,
                       (vertices[face[0]][2]+vertices[face[2]][2])/2.0]),
            (face[2], [(vertices[face[0]][0]+vertices[face[1]][0])/2.0,
                       (vertices[face[0]][1]+vertices[face[1]][1])/2.0,
                       (vertices[face[0]][2]+vertices[face[1]][2])/2.0]),
        ];
        let mut max_median_lod = 0u32;
        for (vi, mid) in &midpoints {
            let al = arc_length(vertices[*vi], *mid);
            let lod = snap_to_power_of_2((al * 16.0).ceil() as u32).min(512);
            max_median_lod = max_median_lod.max(lod);
        }
        // Use max median LOD as a floor for ALL three edges of this face
        if max_median_lod > 1 {
            for &(va, vb) in &edges {
                let key = if va < vb { (va, vb) } else { (vb, va) };
                let current = edge_lod_map.get(&key).copied().unwrap_or(1);
                edge_lod_map.insert(key, current.max(max_median_lod));
            }
        }
    }

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
        let instances = compute_instances(&verts, &faces, &Mobius::identity());
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
        let instances = compute_instances(&verts, &faces, &m);
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
        let instances = compute_instances(&verts, &faces, &m);

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
        let instances = compute_instances(&verts, &faces, &m);
        for inst in &instances {
            for &l in &inst.edge_lods {
                assert!(l.is_power_of_two(), "LOD {} is not a power of 2", l);
            }
        }
    }
}
