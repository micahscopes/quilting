use std::collections::HashMap;
use crate::quaternion::{Quat, Mobius};
use crate::patch::QBTriPatch;

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

    // Estimate per-face LOD from scale + curvature + singularity proximity
    let face_lods: Vec<u32> = faces.iter().enumerate().map(|(fi, face)| {
        let scale_lod = estimate_face_lod_from_scale(&instances[fi], transform, vertices, face);
        let curvature_lod = estimate_face_curvature(&instances[fi]);
        scale_lod.max(curvature_lod)
    }).collect();

    // Build edge adjacency: edge (min_vi, max_vi) → list of (face_idx, local_edge_idx)
    let mut edge_faces: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for (fi, face) in faces.iter().enumerate() {
        // 3 edges per face. local_edge_idx 0 = opposite vert 0 = edge(v1,v2)
        let edges = [
            (face[1], face[2], 0), // edge a: opposite v0
            (face[0], face[2], 1), // edge b: opposite v1
            (face[0], face[1], 2), // edge c: opposite v2
        ];
        for &(va, vb, local_idx) in &edges {
            let key = if va < vb { (va, vb) } else { (vb, va) };
            edge_faces.entry(key).or_default().push((fi, local_idx));
        }
    }

    // Assign per-edge LOD = max of adjacent face LODs, snapped to power of 2
    let mut result = instances;
    for (_edge, face_refs) in &edge_faces {
        let max_lod = face_refs.iter()
            .map(|&(fi, _)| face_lods[fi])
            .max()
            .unwrap_or(1);
        let snapped = snap_to_power_of_2(max_lod);
        for &(fi, local_idx) in face_refs {
            result[fi].edge_lods[local_idx] = snapped;
        }
    }

    // Boundary edges (only one face) — use that face's LOD
    for (fi, face) in faces.iter().enumerate() {
        let edges = [
            (face[1], face[2], 0),
            (face[0], face[2], 1),
            (face[0], face[1], 2),
        ];
        for &(va, vb, local_idx) in &edges {
            let key = if va < vb { (va, vb) } else { (vb, va) };
            if edge_faces.get(&key).map_or(true, |v| v.len() == 1) {
                result[fi].edge_lods[local_idx] = snap_to_power_of_2(face_lods[fi]);
            }
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

/// Estimate face LOD from QB surface curvature (H/L ratio).
fn estimate_face_curvature(inst: &FaceInstance) -> u32 {
    let [p0, p1, p2] = inst.positions;
    let [w0, w1, w2] = inst.weights;

    let e01 = (p0 - p1).norm();
    let e02 = (p0 - p2).norm();
    let e12 = (p1 - p2).norm();
    let l = e01.max(e02).max(e12);
    if l < 1e-12 { return 1; }

    let flat_mid = (p0 + p1 + p2) * (1.0 / 3.0);
    let patch = QBTriPatch::new([p0, p1, p2], [w0, w1, w2]);
    let qb_mid = patch.eval(1.0 / 3.0, 1.0 / 3.0);
    let h = (qb_mid - Quat::new(flat_mid.w, flat_mid.x, flat_mid.y, flat_mid.z)).norm();

    let raw_lod = (64.0 * h / l).ceil() as u32;
    raw_lod.max(1).min(512)
}

/// Estimate face LOD from transformed edge lengths + singularity proximity.
fn estimate_face_lod_from_scale(inst: &FaceInstance, transform: &Mobius, orig_verts: &[[f64; 3]], face: &[usize; 3]) -> u32 {
    let [p0, p1, p2] = inst.positions;
    let e01 = (p0 - p1).norm();
    let e02 = (p0 - p2).norm();
    let e12 = (p1 - p2).norm();
    let max_edge = e01.max(e02).max(e12);

    // Edge-length LOD: 16 subdivisions per unit
    let edge_lod = max_edge * 16.0;

    // Singularity LOD: 1/|cx+d|² blows up near the singularity.
    // Use the min |cx+d|² across all face vertices + centroid.
    let mut min_denom_sq = f64::MAX;
    for &vi in face {
        let p = Quat::from_point(orig_verts[vi][0], orig_verts[vi][1], orig_verts[vi][2]);
        let denom = transform.c * p + transform.d;
        min_denom_sq = min_denom_sq.min(denom.norm_sq());
    }
    // Singularity LOD: inversely proportional to min |cx+d|
    // At |cx+d|=1 → LOD 4, at |cx+d|=0.1 → LOD 40, at |cx+d|=0.01 → LOD 400
    let singularity_lod = 4.0 / min_denom_sq.sqrt().max(1e-10);

    let raw_lod = edge_lod.max(singularity_lod).ceil() as u32;
    raw_lod.max(1).min(512)
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
    fn identity_uniform_lod() {
        let (verts, faces) = shapes::cube();
        let instances = compute_instances(&verts, &faces, &Mobius::identity());
        // Identity: all faces should have the same LOD (uniform)
        let first = instances[0].edge_lods;
        for inst in &instances {
            assert_eq!(inst.edge_lods, first,
                "identity should have uniform LOD, got {:?}", inst.edge_lods);
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
